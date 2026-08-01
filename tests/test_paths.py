"""Unit tests for path geometry and single-level generation."""

from __future__ import annotations

import numpy as np

from pigeonpilot.paths import (
    STYLES,
    Segment,
    generate_level,
    home_vector,
    displacement_vector,
    snap_heading,
    trajectory_points,
)


def _circ_diff_deg(a: float, b: float) -> float:
    """Signed shortest turn from heading a to b in (-180, 180]."""
    return ((b - a + 180.0) % 360.0) - 180.0


def test_style_names():
    assert STYLES == ("linear", "turning", "curved")


def test_snap_heading_36_bins():
    assert snap_heading(4, 36) == 0.0
    assert snap_heading(6, 36) == 10.0


def test_single_segment_north_is_plus_y():
    """Compass: 0° = North → +y."""
    segments = (Segment(heading_deg=0.0, distance=1.0),)
    np.testing.assert_allclose(displacement_vector(segments), [0, 1], atol=1e-9)
    np.testing.assert_allclose(home_vector(segments), [0, -1], atol=1e-9)


def test_ten_steps_north_y_is_plus_ten():
    segments = (Segment(heading_deg=0.0, distance=10.0),)
    np.testing.assert_allclose(displacement_vector(segments), [0.0, 10.0], atol=1e-9)


def test_single_segment_east_is_plus_x():
    """Compass: 90° = East → +x."""
    segments = (Segment(heading_deg=90.0, distance=1.0),)
    np.testing.assert_allclose(displacement_vector(segments), [1, 0], atol=1e-9)
    np.testing.assert_allclose(home_vector(segments), [-1, 0], atol=1e-9)


def test_generated_level_home_xy_is_negation_of_end_xy():
    """generate_level stores home/end once; home_xy == -end_xy."""
    level = generate_level(style="turning", n_segments=4, seed=7, turning_scale="gentle")
    assert np.allclose(level.home_xy, (-level.end_xy[0], -level.end_xy[1]))
    end = displacement_vector(level.segments)
    np.testing.assert_allclose(level.end_xy, end, atol=1e-9)
    np.testing.assert_allclose(level.home_xy, -end, atol=1e-9)


def test_linear_is_truly_straight():
    level = generate_level(style="linear", n_segments=3, seed=0)
    assert len({seg.heading_deg for seg in level.segments}) == 1


def test_turning_gentle_jumps_are_small():
    """Gentle turns use 2–4 bin jumps → 20°–40° on a 36-bin grid."""
    heading_bins = 36
    allowed = {20.0, 30.0, 40.0}
    for seed in range(40):
        level = generate_level(
            style="turning",
            n_segments=5,
            seed=seed,
            heading_bins=heading_bins,
            turning_scale="gentle",
        )
        headings = [seg.heading_deg for seg in level.segments]
        for prev, cur in zip(headings, headings[1:]):
            jump = abs(_circ_diff_deg(prev, cur))
            assert jump in allowed
            assert jump > 0.0


def test_turning_sharp_jumps_are_large():
    """Sharp turns use ~90°–120° jumps on a 36-bin grid."""
    heading_bins = 36
    allowed = {90.0, 120.0}
    for seed in range(40):
        level = generate_level(
            style="turning",
            n_segments=5,
            seed=seed,
            heading_bins=heading_bins,
            turning_scale="sharp",
        )
        headings = [seg.heading_deg for seg in level.segments]
        for prev, cur in zip(headings, headings[1:]):
            jump = abs(_circ_diff_deg(prev, cur))
            assert jump in allowed
            assert jump > 0.0


def test_curved_mild_oscillates_not_straight():
    """Mild curved paths are not a single constant heading."""
    heading_bins = 36
    for seed in range(20):
        level = generate_level(
            style="curved",
            n_segments=6,
            seed=seed,
            curved_mode="mild",
            heading_bins=heading_bins,
        )
        headings = [seg.heading_deg for seg in level.segments]
        assert len(set(headings)) >= 2
        assert all(snap_heading(h, heading_bins) == h for h in headings)


def test_curved_arc_sweeps_monotonically():
    """Arc curved paths keep a consistent turn direction between steps."""
    for seed in range(20):
        level = generate_level(
            style="curved",
            n_segments=8,
            seed=seed,
            curved_mode="arc",
            heading_bins=36,
        )
        headings = [seg.heading_deg for seg in level.segments]
        diffs = [_circ_diff_deg(a, b) for a, b in zip(headings, headings[1:])]
        nonzero = [d for d in diffs if abs(d) > 1e-9]
        assert nonzero
        signs = {np.sign(d) for d in nonzero}
        assert len(signs) == 1


def test_trajectory_points_starts_at_home():
    segments = (
        Segment(heading_deg=0.0, distance=1.0),
        Segment(heading_deg=90.0, distance=2.0),
    )
    pts = trajectory_points(segments)
    assert pts.shape == (3, 2)
    np.testing.assert_allclose(pts[0], [0.0, 0.0])
    np.testing.assert_allclose(pts[-1], displacement_vector(segments))


def test_generate_level_rejects_bad_args():
    import pytest

    with pytest.raises(ValueError, match="Unknown style"):
        generate_level(style="diagonal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_segments"):
        generate_level(n_segments=0)
