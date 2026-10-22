#!/usr/bin/env python3
"""
Interactive 3D view of estimated water planes relative to the camera.

Reads one or more JSON results and writes a single HTML file you can open in
any web browser (rotate: drag, zoom: scroll, hover: point details):
    plane.json          from estimate_water_plane.py   (waterline method)
    plane_surface.json  from estimate_water_surface.py (IMU plane + fitted plane)

Shown:
    * the camera at the origin with its x / y / z axes
    * each plane as a see-through patch, with its "down" arrow (normal)
    * a dashed line from the camera straight down to each plane, labelled with d
    * the sample points each plane was built from (surface method: red = points
      the fitted plane rejected as outliers)
    * the origin mark and the point below it on the water, if present

Display axes: the camera frame has y pointing DOWN, which would draw the scene
upside-down, so for display only we use  X = x (right), Y = z (forward),
Z = -y (up). Hover labels show the original camera-frame coordinates.

Usage:
    python visual_planes.py out_test/plane.json out_test/plane_surface.json -o planes.html
Needs:  pip install plotly
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

COLORS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]


def to_display(p) -> np.ndarray:
    """Camera frame (x right, y down, z forward) -> display (X right, Y forward, Z up)."""
    p = np.atleast_2d(np.asarray(p, float))
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], axis=1)


def load_planes(path) -> list[dict]:
    """Read a result file and return a list of planes, each a dict with
    label, normal, d, sample points, and (optionally) inlier flags and origin."""
    data = json.load(open(path))
    name = Path(path).name
    samples = data.get("samples", [])
    pts = np.array([s["point_m"] for s in samples], float).reshape(-1, 3)
    origin = data.get("origin")
    if "plane" in data:                                    # estimate_water_plane.py
        return [{"label": f"{name}: waterline method", "normal": data["plane"]["normal"],
                 "d": data["plane"]["d"], "points": pts, "inlier": None, "origin": origin}]
    if "imu_plane" in data:                                # estimate_water_surface.py
        inl = np.array([s.get("fit_inlier", True) for s in samples], bool)
        return [{"label": f"{name}: IMU plane", "normal": data["imu_plane"]["normal"],
                 "d": data["imu_plane"]["d"], "points": pts, "inlier": None, "origin": origin},
                {"label": f"{name}: fitted plane", "normal": data["fitted_plane"]["normal"],
                 "d": data["fitted_plane"]["d"], "points": pts, "inlier": inl, "origin": None,
                 "hide_points": True}]               # same points as the IMU plane; drawn once
    raise ValueError(f"{path}: not a plane result (no 'plane' or 'imu_plane' key)")


def plane_patch(normal, d, points, margin=0.5, default_size=4.0):
    """Corner grid (2 x 2) of a rectangle lying on the plane normal . X = d,
    sized to cover the sample points plus a margin. Returns (xs, ys, zs) in
    camera coordinates and the patch centre."""
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    # Two directions lying in the plane: e1 roughly "right", e2 roughly "forward".
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, n)
    if len(points):
        centroid = points.mean(axis=0)
        centre = centroid - (centroid @ n - d) * n        # centroid moved onto the plane
        a, b = (points - centre) @ e1, (points - centre) @ e2
        a_rng = (a.min() - margin, a.max() + margin)
        b_rng = (b.min() - margin, b.max() + margin)
    else:
        centre = d * n                                    # point straight below the camera
        a_rng = b_rng = (-default_size / 2, default_size / 2)
    corners = np.array([[centre + i * e1 + j * e2 for i in a_rng] for j in b_rng])   # (2, 2, 3)
    return corners, centre


def arrow(fig, start, end, color, name, width=6, show_legend=False, group=None):
    """Line plus cone head from start to end (camera coordinates)."""
    s, e = to_display(start)[0], to_display(end)[0]
    fig.add_trace(go.Scatter3d(x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]], mode="lines",
                               line=dict(color=color, width=width), name=name,
                               showlegend=show_legend, legendgroup=group, hoverinfo="name"))
    v = e - s
    fig.add_trace(go.Cone(x=[e[0]], y=[e[1]], z=[e[2]], u=[v[0]], v=[v[1]], w=[v[2]],
                          sizemode="absolute", sizeref=0.12, anchor="tip", showscale=False,
                          colorscale=[[0, color], [1, color]], name=name, showlegend=False,
                          legendgroup=group, hoverinfo="name"))


def build_figure(planes: list[dict], axis_len=0.5) -> go.Figure:
    fig = go.Figure()

    # Camera at the origin, with its own axes (labelled in camera terms).
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["camera"],
                               textposition="top center", marker=dict(size=6, color="black"),
                               name="camera", hoverinfo="name"))
    for vec, color, label in [((axis_len, 0, 0), "red", "camera x (right)"),
                              ((0, axis_len, 0), "green", "camera y (down)"),
                              ((0, 0, axis_len), "blue", "camera z (forward)")]:
        arrow(fig, (0, 0, 0), vec, color, label, width=4, show_legend=True, group="axes")

    for i, pl in enumerate(planes):
        color, group = COLORS[i % len(COLORS)], pl["label"]
        n = np.asarray(pl["normal"], float)
        n /= np.linalg.norm(n)
        corners, centre = plane_patch(n, pl["d"], pl["points"])
        c = to_display(corners.reshape(-1, 3)).reshape(2, 2, 3)
        fig.add_trace(go.Surface(x=c[..., 0], y=c[..., 1], z=c[..., 2], opacity=0.35,
                                 colorscale=[[0, color], [1, color]], showscale=False,
                                 name=pl["label"], showlegend=True, legendgroup=group,
                                 hovertemplate=f"{pl['label']}<br>d = {pl['d']:.4f} m<extra></extra>"))

        # "Down" arrow on the plane, and the camera-to-plane distance d.
        arrow(fig, centre - 0.6 * n, centre, color, f"{pl['label']} normal", group=group)
        foot = pl["d"] * n
        f = to_display([[0, 0, 0], foot])
        fig.add_trace(go.Scatter3d(x=f[:, 0], y=f[:, 1], z=f[:, 2], mode="lines+text",
                                   line=dict(color=color, width=3, dash="dash"),
                                   text=["", f"d = {pl['d']:.3f} m"], textposition="bottom center",
                                   name=f"{pl['label']}: d", showlegend=False, legendgroup=group,
                                   hoverinfo="name"))

        # Sample points (hover shows camera-frame coordinates and height).
        pts = pl["points"]
        if len(pts) and not pl.get("hide_points"):
            disp = to_display(pts)
            heights = pts @ n
            hover = [f"camera xyz = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m<br>"
                     f"height along normal = {h:.4f} m" for p, h in zip(pts, heights)]
            fig.add_trace(go.Scatter3d(x=disp[:, 0], y=disp[:, 1], z=disp[:, 2], mode="markers",
                                       marker=dict(size=3, color=color), text=hover,
                                       hoverinfo="text", name=f"{pl['label']}: samples",
                                       legendgroup=group))
        if pl["inlier"] is not None and (~pl["inlier"]).any():
            out = to_display(pts[~pl["inlier"]])
            fig.add_trace(go.Scatter3d(x=out[:, 0], y=out[:, 1], z=out[:, 2], mode="markers",
                                       marker=dict(size=5, color="red", symbol="x"),
                                       name=f"{pl['label']}: rejected outliers", legendgroup=group))

        # Origin mark and the point directly below it on the water.
        o = pl.get("origin")
        if o:
            seg = to_display([o["point_m"], o["on_water_m"]])
            fig.add_trace(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines+markers+text",
                                       line=dict(color="black", width=3), marker=dict(size=5, color="black"),
                                       text=["origin mark", f"{o['height_above_water_m']:.3f} m above water"],
                                       textposition="top center", name="origin", legendgroup=group))

    fig.update_layout(
        title="Estimated water planes (camera at origin)",
        scene=dict(xaxis_title="X right (m)", yaxis_title="Y forward (m)", zaxis_title="Z up (m)",
                   aspectmode="data", camera=dict(eye=dict(x=-1.7, y=-2.6, z=1.4))),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0))
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Interactive 3D view of estimated water planes.")
    p.add_argument("results", nargs="+", help="plane.json / plane_surface.json files")
    p.add_argument("-o", "--output", default="planes.html", help="HTML file to write")
    a = p.parse_args(argv)
    planes = [pl for path in a.results for pl in load_planes(path)]
    fig = build_figure(planes)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(a.output, include_plotlyjs=True)
    for pl in planes:
        print(f"{pl['label']}: d = {pl['d']:.4f} m, normal = {np.round(pl['normal'], 4).tolist()}")
    print(f"Wrote {a.output} - open it in a web browser")


if __name__ == "__main__":
    main()
