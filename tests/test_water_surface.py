"""Tests for estimate_water_surface.py against the known stub scene.
Run with:  python -m pytest tests -q"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import estimate_water_surface as ws  # noqa: E402
from zed_stub import StubCamera, StubScene  # noqa: E402

ROI = "200,620,1720,1000"   # on the water, below the stub's waterline (rows ~560-580)


def run(scene=None, **kw):
    scene = scene or StubScene()
    cam = StubCamera(scene)
    return ws.estimate_surface(cam, cam.open(), ROI, range(0, 60), **kw), scene


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.asarray(a) @ np.asarray(b), -1, 1)))


def test_parse_grid():
    assert ws.parse_grid("20x20") == (20, 20)
    assert ws.parse_grid("30x10") == (30, 10)
    assert ws.parse_grid("25") == (25, 25)
    with pytest.raises(ValueError):
        ws.parse_grid("0x5")


def test_grid_pixels_count_and_bounds():
    px = ws.grid_pixels("100,200,500,400", 20, 10, 1920, 1080)
    assert len(px) == 200
    assert px[:, 0].min() == 100 and px[:, 0].max() == 499
    assert px[:, 1].min() == 200 and px[:, 1].max() == 399
    with pytest.raises(ValueError, match="outside"):
        ws.grid_pixels("0,0,2000,100", 5, 5, 1920, 1080)


def test_imu_plane_matches_truth():
    res, scene = run()
    assert abs(res["imu_plane"]["d"] - scene.camera_height) < 0.01
    assert angle_deg(res["imu_plane"]["normal"], scene.down()) < 0.1


def test_fitted_plane_matches_truth_and_imu():
    res, scene = run()
    assert angle_deg(res["fitted_plane"]["normal"], scene.down()) < 0.1
    assert abs(res["fitted_plane"]["d"] - scene.camera_height) < 0.01
    assert res["comparison"]["angle_deg"] < 0.1
    assert res["fitted_plane"]["normal"][1] > 0.9          # oriented down, like gravity


def test_ransac_ignores_persistent_reflections():
    res, scene = run(StubScene(reflection_fraction=0.3))
    assert angle_deg(res["fitted_plane"]["normal"], scene.down()) < 0.1
    assert res["fitted_plane"]["inlier_fraction"] < 0.9     # it did reject the reflections
    assert abs(res["imu_plane"]["d"] - scene.camera_height) < 0.01


def test_comparison_detects_imu_bias():
    # A biased accelerometer tilts the IMU plane; the fitted plane is unaffected,
    # so the comparison should report the disagreement.
    res, _ = run(StubScene(accel_bias=(0.0, 0.0, 0.17)))   # ~1 deg of tilt error
    assert res["comparison"]["angle_deg"] > 0.5


def test_custom_grid_size():
    res, _ = run(grid=(8, 5))
    assert res["quality"]["points_sampled"] == 40
    assert res["inputs"]["grid"] == [8, 5]


def test_no_water_depth_is_reported():
    with pytest.raises(RuntimeError, match="valid depth"):
        run(StubScene(water_depth=False))


def test_cli_time_range_and_grid(tmp_path):
    out = tmp_path / "plane_surface.json"
    ws.main(["--stub", "--roi", ROI, "--end-time", "1", "--grid", "10x6", "--output", str(out)])
    res = json.load(open(out))
    assert res["inputs"]["frames"] == [0, 30] and res["quality"]["points_sampled"] == 60
    assert set(res) >= {"imu_plane", "fitted_plane", "comparison", "quality", "samples"}
