#!/usr/bin/env python3
"""
Estimate the water plane by sampling depth directly ON the water surface.

Companion to estimate_water_plane.py (which samples the wall just above the
waterline). Here you choose a box (ROI) on the water and a frame/time range:

  1. Lay an evenly spaced grid of sample pixels over the ROI (default 20 x 20).
  2. Read each pixel's 3D point in every frame; the SDK's confidence threshold
     turns unreliable readings into NaN. Take each pixel's median over frames,
     and drop pixels that were valid in too few frames (--min-valid).
  3. Two independent planes from those points:
       IMU plane     normal = gravity from the accelerometer,
                     d = median over points of (normal . point)
       Fitted plane  fitted to the points alone (no IMU), using RANSAC so
                     reflections and glints don't tilt it
  4. Compare them. If the depth data really follows the water surface, the
     fitted plane's tilt matches gravity to within a fraction of a degree.
     (A uniform depth scale error moves the fitted plane up/down but keeps its
     tilt, so agreement checks the SHAPE of the depth, not absolute distance.)

Plane form, same as estimate_water_plane.py:  normal . X = d
    normal  unit vector pointing DOWN, camera frame (x right, y down, z forward)
    d       camera height above the water, metres

Until zed_real.py is available, run with --stub:
    python estimate_water_surface.py --stub --roi 200,620,1720,1000 --end-frame 60
"""

import argparse
import json
from pathlib import Path

import numpy as np

from estimate_water_plane import (camera_tilt, fit_plane, gravity_down, median_over_frames,
                                  open_camera, read_points)


def parse_grid(text: str) -> tuple[int, int]:
    """'20x20' -> (20, 20) as (columns, rows); a single number '25' -> (25, 25)."""
    parts = text.lower().replace(",", "x").split("x")
    cols, rows = (int(parts[0]), int(parts[0])) if len(parts) == 1 else (int(parts[0]), int(parts[1]))
    if cols < 1 or rows < 1:
        raise ValueError("grid sizes must be at least 1")
    return cols, rows


def grid_pixels(roi: str, cols: int, rows: int, width: int, height: int) -> np.ndarray:
    """Evenly spaced (u, v) pixels over the box 'x1,y1,x2,y2'. Returns (K, 2) ints;
    K = cols * rows unless the box is too small for that many distinct pixels."""
    x1, y1, x2, y2 = (int(v) for v in roi.split(","))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        raise ValueError(f"ROI {roi} is outside the {width}x{height} image")
    us = np.round(np.linspace(x1, x2 - 1, cols)).astype(int)
    vs = np.round(np.linspace(y1, y2 - 1, rows)).astype(int)
    uu, vv = np.meshgrid(us, vs)                          # every (column, row) combination
    return np.unique(np.stack([uu.ravel(), vv.ravel()], axis=1), axis=0)


def fit_plane_ransac(points: np.ndarray, threshold: float = 0.03, iterations: int = 500,
                     seed: int = 0) -> tuple[np.ndarray, float, np.ndarray]:
    """Plane through 3D points, ignoring outliers, without using gravity.

    RANSAC: many times, pick 3 random points, make the plane through them, and
    count how many points lie within `threshold` metres of it. Keep the plane
    most points agree with, then refit it to all of those "inliers" with least
    squares (SVD: the normal is the direction the points vary least in).

    Returns (unit normal, d, inlier mask) with normal . X = d.
    """
    rng = np.random.default_rng(seed)
    n_pts = len(points)
    best = np.zeros(n_pts, bool)
    for _ in range(iterations):
        a, b, c = points[rng.choice(n_pts, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:                                 # the 3 points were in a line
            continue
        normal /= length
        inliers = np.abs(points @ normal - normal @ a) < threshold
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 3:
        raise RuntimeError("RANSAC found no plane; try a larger --ransac-threshold")
    centroid = points[best].mean(axis=0)
    _, _, vt = np.linalg.svd(points[best] - centroid)
    normal = vt[-1]                                       # least-variance direction
    return normal, float(normal @ centroid), best


def compare_normals(fitted: np.ndarray, imu_down: np.ndarray) -> dict:
    """How far the fitted plane's tilt is from gravity, in total and split
    into pitch (front-back) and roll (side-to-side), degrees."""
    angle = np.degrees(np.arccos(np.clip(fitted @ imu_down, -1, 1)))
    t_fit, t_imu = camera_tilt(fitted), camera_tilt(imu_down)
    return {"angle_deg": float(angle),
            "pitch_diff_deg": t_fit["pitch_deg"] - t_imu["pitch_deg"],
            "roll_diff_deg": t_fit["roll_deg"] - t_imu["roll_deg"]}


def estimate_surface(cam, info, roi: str, frames: range, grid=(20, 20), min_valid=0.3,
                     ransac_threshold=0.03, origin_pixel=None) -> dict:
    """Run the whole method. Returns the result dict written to the JSON file."""
    grid_px = grid_pixels(roi, *grid, info.width, info.height)
    pixels = grid_px if origin_pixel is None else np.vstack([grid_px, [origin_pixel]])
    stack, accel = read_points(cam, pixels, frames)
    med, counts = median_over_frames(stack, min_valid)
    pts_all, origin_pt = med[:len(grid_px)], (None if origin_pixel is None else med[-1])

    ok = ~np.isnan(pts_all).any(axis=1)
    if ok.sum() < 3:
        raise RuntimeError(f"Only {ok.sum()} grid points had valid depth; check the ROI is on "
                           "the water, or loosen --confidence-threshold / --min-valid.")
    pts = pts_all[ok]

    # A) IMU plane: gravity gives the tilt, the median height gives d.
    down, g_spread = gravity_down(accel, cam.imu_to_camera_rotation())
    imu = fit_plane(down, pts)

    # B) Fitted plane: from the points alone, oriented to point down like gravity.
    normal, d_fit, inlier = fit_plane_ransac(pts, ransac_threshold)
    if normal @ down < 0:
        normal, d_fit = -normal, -d_fit
    resid = pts[inlier] @ normal - d_fit

    n_frames = len(stack)
    result = {
        "imu_plane": {"normal": down.tolist(), "d": imu["d"],
                      "height_spread_std_m": imu["height_std_m"],
                      "height_spread_mad_m": imu["height_mad_m"],
                      "camera_tilt": camera_tilt(down),
                      "gravity_sample_spread_deg": g_spread},
        "fitted_plane": {"normal": normal.tolist(), "d": d_fit,
                         "rms_residual_m": float(np.sqrt(np.mean(resid ** 2))),
                         "inlier_fraction": float(inlier.mean()),
                         "camera_tilt": camera_tilt(normal)},
        "comparison": {**compare_normals(normal, down), "d_diff_m": d_fit - imu["d"]},
        "quality": {"points_used": int(ok.sum()), "points_sampled": len(grid_px),
                    "valid_depth_fraction": float(counts[:len(grid_px)].sum() / (n_frames * len(grid_px))),
                    "frames_read": n_frames},
        "plane_equation": "normal . X = d  (X in camera frame; normal points down; d = camera height above water)",
        "frame": "left camera: x right, y down, z forward",
        "units": "metres",
        "inputs": {"roi": roi, "grid": list(grid), "frames": [frames.start, frames.stop],
                   "min_valid": min_valid, "ransac_threshold_m": ransac_threshold,
                   "confidence_threshold": getattr(cam, "confidence_threshold", None)},
        "samples": [{"pixel": px.tolist(), "point_m": p.tolist(), "height_m": float(h),
                     "valid_frames": int(c), "fit_inlier": bool(i)}
                    for px, p, h, c, i in zip(grid_px[ok], pts, imu["heights"],
                                              counts[:len(grid_px)][ok], inlier)],
    }
    if origin_pixel is not None:
        if np.isnan(origin_pt).any():
            raise RuntimeError("No valid depth at --origin-pixel; pick a textured, visible spot.")
        h = imu["d"] - origin_pt @ down
        result["origin"] = {"pixel": list(origin_pixel), "point_m": origin_pt.tolist(),
                            "height_above_water_m": float(h),
                            "on_water_m": (origin_pt + h * down).tolist()}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Estimate the water plane from depth sampled on the water surface.")
    p.add_argument("--svo", help="ZED .svo2 recording (ignored with --stub)")
    p.add_argument("--stub", action="store_true", help="use the fake camera in zed_stub.py")
    p.add_argument("--roi", required=True, help="box on the water to sample: x1,y1,x2,y2 (pixels)")
    p.add_argument("--grid", default="20x20",
                   help="sample grid as COLUMNSxROWS, e.g. 30x10; a single number means square")
    start = p.add_mutually_exclusive_group()
    start.add_argument("--start-frame", type=int, help="first frame (inclusive); default 0")
    start.add_argument("--start-time", type=float, help="start in seconds")
    end = p.add_mutually_exclusive_group(required=True)
    end.add_argument("--end-frame", type=int, help="last frame (exclusive)")
    end.add_argument("--end-time", type=float, help="end in seconds")
    p.add_argument("--confidence-threshold", type=float, default=50,
                   help="SDK depth confidence threshold (1-100, lower = stricter)")
    p.add_argument("--min-valid", type=float, default=0.3,
                   help="drop grid points with valid depth in fewer than this fraction of frames")
    p.add_argument("--ransac-threshold", type=float, default=0.03,
                   help="max distance (m) from the fitted plane to count as an inlier")
    p.add_argument("--origin-pixel", help="u,v of a marked reference point (optional)")
    p.add_argument("--output", default="plane_surface.json")
    a = p.parse_args(argv)
    if not a.stub and not a.svo:
        p.error("--svo is required unless --stub is used")
    grid = parse_grid(a.grid)
    origin = [int(v) for v in a.origin_pixel.split(",")] if a.origin_pixel else None

    cam, info = open_camera(a.stub, a.svo, a.confidence_threshold)
    try:
        fps = info.fps or 30.0
        f0 = int(round(a.start_time * fps)) if a.start_time is not None else (a.start_frame or 0)
        f1 = int(round(a.end_time * fps)) if a.end_time is not None else a.end_frame
        if f1 <= f0:
            p.error(f"empty frame range [{f0}, {f1})")
        res = estimate_surface(cam, info, a.roi, range(f0, f1), grid, a.min_valid,
                               a.ransac_threshold, origin)
    finally:
        cam.close()
    res["inputs"].update({"svo": "stub" if a.stub else a.svo, "fps": fps})
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "w") as f:
        json.dump(res, f, indent=1)

    imu, fit, cmp_, q = res["imu_plane"], res["fitted_plane"], res["comparison"], res["quality"]
    print(f"Frames {f0}-{f1} ({f0 / fps:.2f}-{f1 / fps:.2f} s), grid {grid[0]}x{grid[1]}: "
          f"{q['points_used']}/{q['points_sampled']} points, {q['valid_depth_fraction']:.0%} valid depth")
    print(f"IMU plane:    d = {imu['d']:.4f} m, height spread {imu['height_spread_mad_m'] * 100:.2f} cm (MAD), "
          f"tilt pitch {imu['camera_tilt']['pitch_deg']:+.2f} / roll {imu['camera_tilt']['roll_deg']:+.2f} deg")
    print(f"Fitted plane: d = {fit['d']:.4f} m, RMS {fit['rms_residual_m'] * 100:.2f} cm, "
          f"{fit['inlier_fraction']:.0%} inliers, tilt pitch {fit['camera_tilt']['pitch_deg']:+.2f} / "
          f"roll {fit['camera_tilt']['roll_deg']:+.2f} deg")
    print(f"Fit vs IMU:   {cmp_['angle_deg']:.3f} deg apart (pitch {cmp_['pitch_diff_deg']:+.3f}, "
          f"roll {cmp_['roll_diff_deg']:+.3f}), d differs by {cmp_['d_diff_m'] * 100:+.2f} cm")
    if "origin" in res:
        o = res["origin"]
        print(f"Origin: {o['height_above_water_m']:.4f} m above water; on water at "
              f"{np.round(o['on_water_m'], 4).tolist()}")
    print(f"Saved {a.output}")


if __name__ == "__main__":
    main()
