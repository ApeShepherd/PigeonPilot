"""Freehand stroke → model-ready level (no BindsNET, no GUI).

The regression these tests exist for: raw pointer samples used to be snapped
one by one, so hand tremor was integrated into the route length and a stroke
the user saw as ~6 units long reached the encoder as 15–53 units with 100+
heading changes — far outside anything the reservoir was trained on.
"""

from __future__ import annotations

import numpy as np
import pytest

from resources.drawing import (
    merge_heading_blocks,
    polyline_to_segments,
    prepare_drawn_path,
    scale_segments,
    simplify_polyline,
    training_band,
)
from resources.paths import Segment, trajectory_points


def jittery_arc(sigma: float = 0.15, n: int = 200, scale: float = 5.0) -> np.ndarray:
    """A hand-drawn-looking stroke: smooth arc plus per-sample tremor."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, n)
    arc = np.stack([scale * np.sin(2.5 * t), scale * t], axis=1)
    return arc + rng.normal(0.0, sigma, arc.shape)


def bearing_deg(point) -> float:
    return float(np.degrees(np.arctan2(point[0], point[1])) % 360.0)


def angle_gap(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


# --------------------------------------------------------------------- RDP


def test_simplify_keeps_endpoints():
    pts = jittery_arc()
    out = simplify_polyline(pts, epsilon=0.5)
    np.testing.assert_allclose(out[0], pts[0])
    np.testing.assert_allclose(out[-1], pts[-1])
    assert len(out) < len(pts)


def test_simplify_collapses_a_straight_line():
    pts = np.stack([np.zeros(50), np.linspace(0.0, 5.0, 50)], axis=1)
    assert len(simplify_polyline(pts, epsilon=0.01)) == 2


def test_simplify_is_monotone_in_epsilon():
    pts = jittery_arc()
    counts = [len(simplify_polyline(pts, epsilon=e)) for e in (0.05, 0.2, 0.8, 2.0)]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------- snapping


def test_polyline_to_segments_keeps_every_step():
    """The old filter silently dropped short steps; total length must survive."""
    pts = jittery_arc(sigma=0.05)
    segments = polyline_to_segments(pts)
    walked = float(np.linalg.norm(np.diff(np.vstack([np.zeros(2), pts]), axis=0), axis=1).sum())
    assert sum(s.distance for s in segments) == pytest.approx(walked, rel=1e-9)


def test_dense_smooth_stroke_still_yields_segments():
    """Regression: fine sampling used to fall entirely below the step filter."""
    t = np.linspace(0.0, 1.0, 200)
    pts = np.stack([5 * np.sin(2.5 * t), 5 * t], axis=1)
    assert len(polyline_to_segments(pts)) > 0


def test_polyline_snaps_and_merges():
    segments = polyline_to_segments([[0, 0], [2, 0], [4, 0], [4, 3]])
    assert [s.heading_deg for s in segments] == [90.0, 0.0]
    assert segments[0].distance == pytest.approx(4.0)
    assert segments[1].distance == pytest.approx(3.0)


def test_merge_heading_blocks_collapses_same_bin():
    segments = (
        Segment(heading_deg=0.0, distance=1.0),
        Segment(heading_deg=0.0, distance=2.0),
        Segment(heading_deg=90.0, distance=1.0),
    )
    assert merge_heading_blocks(segments) == ((0, 3.0), (27, 1.0))


def test_scale_segments_preserves_headings():
    segments = polyline_to_segments([[0, 0], [2, 0], [2, 3]])
    scaled = scale_segments(segments, 0.5)
    assert [s.heading_deg for s in scaled] == [s.heading_deg for s in segments]
    assert sum(s.distance for s in scaled) == pytest.approx(
        0.5 * sum(s.distance for s in segments)
    )


# ------------------------------------------------------------ training band


def test_training_band_matches_the_curriculum():
    """Guard rails are measured from the curriculum, not hard-coded."""
    from resources.curriculum import generate_curriculum_dataset

    band = training_band()
    levels = generate_curriculum_dataset(seed=42)
    totals = [sum(s.distance for s in lv.segments) for lv in levels]
    blocks = [len(merge_heading_blocks(lv.segments)) for lv in levels]
    assert band.min_total_distance == pytest.approx(min(totals))
    assert band.max_total_distance == pytest.approx(max(totals))
    assert band.max_blocks == max(blocks)


# --------------------------------------------------------- drawn path guards


@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.15, 0.3])
def test_drawn_path_lands_inside_the_training_band(sigma):
    band = training_band()
    drawn = prepare_drawn_path(jittery_arc(sigma=sigma))
    assert drawn is not None
    assert drawn.n_blocks <= band.max_blocks
    assert band.min_total_distance <= drawn.total_distance <= band.max_total_distance * (1 + 1e-9)
    assert drawn.in_distribution


@pytest.mark.parametrize("scale", [0.4, 1.0, 3.0, 8.0])
def test_scaling_preserves_the_drawn_direction(scale):
    """Normalisation may change length, never the shape the user drew."""
    pts = jittery_arc(sigma=0.08, scale=scale)
    drawn = prepare_drawn_path(pts)
    assert drawn is not None
    assert angle_gap(bearing_deg(pts[-1]), bearing_deg(drawn.level.end_xy)) <= 15.0


def test_uniform_scale_does_not_change_the_route_shape():
    small = prepare_drawn_path(jittery_arc(sigma=0.08, scale=1.0))
    large = prepare_drawn_path(jittery_arc(sigma=0.08, scale=1.0) * 4.0)
    assert small is not None and large is not None
    assert angle_gap(bearing_deg(small.level.end_xy), bearing_deg(large.level.end_xy)) <= 15.0


def test_long_stroke_is_scaled_down_and_reported():
    drawn = prepare_drawn_path(jittery_arc(sigma=0.1) * 4.0)
    assert drawn is not None
    assert drawn.scale_factor < 1.0
    assert drawn.raw_total_distance > drawn.total_distance


def test_stroke_already_in_band_is_left_alone():
    drawn = prepare_drawn_path([[0, 0], [0, 4]])
    assert drawn is not None
    assert drawn.scale_factor == pytest.approx(1.0)
    assert drawn.total_distance == pytest.approx(4.0)


def test_geometry_is_self_consistent():
    drawn = prepare_drawn_path(jittery_arc(sigma=0.1))
    assert drawn is not None
    np.testing.assert_allclose(drawn.snapped_points[-1], drawn.level.end_xy, atol=1e-9)
    np.testing.assert_allclose(drawn.level.home_xy, -np.asarray(drawn.level.end_xy), atol=1e-9)
    np.testing.assert_allclose(
        trajectory_points(drawn.segments)[-1], drawn.level.end_xy, atol=1e-9
    )


def test_block_budget_is_respected_even_for_a_scribble():
    t = np.linspace(0.0, 1.0, 400)
    scribble = np.stack([7 * np.sin(9 * t), 7 * t], axis=1)
    drawn = prepare_drawn_path(scribble)
    assert drawn is not None
    assert drawn.n_blocks <= training_band().max_blocks


def test_a_stray_click_is_refused_not_inflated():
    assert prepare_drawn_path([[0.0, 0.0], [0.0001, 0.0]]) is None
    assert prepare_drawn_path([[0.0, 0.0]]) is None
    assert prepare_drawn_path([]) is None


def test_step_count_stays_within_the_trained_horizon():
    """Encoder steps must not exceed what the reservoir saw during training."""
    velocity = 0.009148043867160583
    drawn = prepare_drawn_path(jittery_arc(sigma=0.2) * 5.0, velocity=velocity)
    assert drawn is not None
    assert drawn.n_steps <= 1010
