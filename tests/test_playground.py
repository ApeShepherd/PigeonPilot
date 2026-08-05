"""Unit tests for playground geometry helpers (no BindsNET / GUI)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pigeonpilot.paths import home_heading_bin, trajectory_points
from pigeonpilot.playground import (
    level_from_segments,
    polyline_to_segments,
    predicted_home_ray,
    resample_polyline,
)
from pigeonpilot.snn import bin_to_heading_deg, circular_bin_error, heading_to_unit


def test_polyline_to_segments_snaps_east():
    # East = +x = 90°
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    segs = polyline_to_segments(pts, min_segment_distance=0.01)
    assert len(segs) == 1
    assert segs[0].heading_deg == pytest.approx(90.0)
    assert segs[0].distance == pytest.approx(4.0)


def test_polyline_merges_same_heading_and_turns():
    # East then North
    pts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]])
    segs = polyline_to_segments(pts, min_segment_distance=0.01)
    assert len(segs) == 2
    assert segs[0].heading_deg == pytest.approx(90.0)
    assert segs[1].heading_deg == pytest.approx(0.0)
    assert segs[0].distance == pytest.approx(2.0)
    assert segs[1].distance == pytest.approx(3.0)


def test_level_from_segments_home_is_negated_end():
    segs = polyline_to_segments([[0, 0], [0, 2]], min_segment_distance=0.01)
    level = level_from_segments(segs)
    assert level.end_xy[1] == pytest.approx(2.0)
    assert level.home_xy[1] == pytest.approx(-2.0)
    assert home_heading_bin(level) == 18  # south home after north flight


def test_predicted_home_ray_uses_true_length():
    segs = polyline_to_segments([[0, 0], [0, 2]], min_segment_distance=0.01)
    level = level_from_segments(segs)
    release, delta = predicted_home_ray(level, pred_bin=18)  # south
    assert release[1] == pytest.approx(2.0)
    assert np.linalg.norm(delta) == pytest.approx(2.0)
    assert delta[1] == pytest.approx(-2.0)


def test_bin_to_heading_and_circular_error():
    assert bin_to_heading_deg(0) == 0.0
    assert bin_to_heading_deg(18) == 180.0
    bins_err, deg_err = circular_bin_error(18, 19)
    assert bins_err == 1 and deg_err == 10.0


def test_heading_to_unit_compass():
    np.testing.assert_allclose(heading_to_unit(0.0), [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(heading_to_unit(90.0), [1.0, 0.0], atol=1e-9)


def test_resample_polyline_endpoints():
    pts = np.array([[0.0, 0.0], [0.0, 4.0]])
    frames = resample_polyline(pts, 5)
    assert frames.shape == (5, 2)
    np.testing.assert_allclose(frames[0], [0.0, 0.0])
    np.testing.assert_allclose(frames[-1], [0.0, 4.0])


def test_list_and_resolve_runs(tmp_path):
    from pigeonpilot.snn import list_runs, resolve_run_dir, set_latest

    root = tmp_path / "checkpoints"
    run_a = root / "n1000_smoke"
    run_b = root / "n10000_jury"
    for folder, n in ((run_a, 1000), (run_b, 10000)):
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text(
            json.dumps({"name": folder.name, "config": {"n_reservoir": n}, "metrics": {}}),
            encoding="utf-8",
        )

    set_latest("n10000_jury", root=root)
    runs = list_runs(root=root)
    assert {r["name"] for r in runs} == {"n1000_smoke", "n10000_jury"}
    assert next(r for r in runs if r["name"] == "n10000_jury")["is_latest"]
    assert resolve_run_dir("latest", root=root) == run_b
    assert resolve_run_dir("n1000_smoke", root=root) == run_a
