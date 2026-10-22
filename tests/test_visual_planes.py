"""Tests for visual_planes.py. Run with:  python -m pytest tests -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("plotly")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import estimate_water_plane as wp  # noqa: E402
import estimate_water_surface as ws  # noqa: E402
import visual_planes as vp  # noqa: E402
from zed_stub import StubCamera, write_waterline  # noqa: E402


def test_display_axes_put_up_on_z():
    # camera 'down' (+y) must display as negative Z; forward (+z) as +Y
    np.testing.assert_allclose(vp.to_display([0, 1, 0]), [[0, 0, -1]])
    np.testing.assert_allclose(vp.to_display([0, 0, 2]), [[0, 2, 0]])


def test_plane_patch_lies_on_plane():
    n = np.array([0.0, 1.0, 0.1]) / np.linalg.norm([0.0, 1.0, 0.1])
    pts = np.array([[x, 0.5, z] for x in (-1, 1) for z in (3, 6)], float)
    corners, centre = vp.plane_patch(n, 0.5, pts)
    np.testing.assert_allclose(corners.reshape(-1, 3) @ n, 0.5, atol=1e-9)
    np.testing.assert_allclose(centre @ n, 0.5, atol=1e-9)


def test_end_to_end_html(tmp_path):
    import json
    waterline = write_waterline(tmp_path / "waterline.json")
    cam = StubCamera()
    json.dump(wp.estimate(cam, cam.open(), waterline, origin_pixel=(300, 540)),
              open(tmp_path / "plane.json", "w"))
    cam = StubCamera()
    json.dump(ws.estimate_surface(cam, cam.open(), "200,620,1720,1000", range(0, 30)),
              open(tmp_path / "plane_surface.json", "w"))

    planes = vp.load_planes(tmp_path / "plane.json") + vp.load_planes(tmp_path / "plane_surface.json")
    assert [p["label"].split(": ")[1] for p in planes] == ["waterline method", "IMU plane", "fitted plane"]

    out = tmp_path / "planes.html"
    vp.main([str(tmp_path / "plane.json"), str(tmp_path / "plane_surface.json"), "-o", str(out)])
    assert out.exists() and out.stat().st_size > 100_000


def test_rejects_unknown_json(tmp_path):
    (tmp_path / "x.json").write_text('{"something": 1}')
    with pytest.raises(ValueError, match="not a plane result"):
        vp.load_planes(tmp_path / "x.json")
