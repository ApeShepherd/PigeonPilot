"""Unit tests for path geometry and single-level generation."""

from __future__ import annotations

import numpy as np
import pytest

from resources.paths import (
    DIFFICULTIES,
    STYLES,
    Segment,
    generate_level,
    home_heading_bin,
    home_heading_deg,
    home_vector,
    displacement_vector,
    snap_heading,
    trajectory_points,
)


def _circ_diff_deg(a: float, b: float) -> float:
    """Signed shortest turn from heading a to b in (-180, 180]."""
    return ((b - a + 180.0) % 360.0) - 180.0


def test_style_names():
    assert STYLES == ("linear", "turning", "zigzag", "curved")


def test_difficulty_ladder_order():
    assert DIFFICULTIES == ("easy", "medium", "hard", "expert", "extreme")


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


def test_single_segment_south_is_minus_y():
    """Compass: 180° = South → -y."""
    segments = (Segment(heading_deg=180.0, distance=1.0),)
    np.testing.assert_allclose(displacement_vector(segments), [0, -1], atol=1e-9)
    np.testing.assert_allclose(home_vector(segments), [0, 1], atol=1e-9)


def test_single_segment_west_is_minus_x():
    """Compass: 270° = West → -x."""
    segments = (Segment(heading_deg=270.0, distance=1.0),)
    np.testing.assert_allclose(displacement_vector(segments), [-1, 0], atol=1e-9)
    np.testing.assert_allclose(home_vector(segments), [1, 0], atol=1e-9)


def test_home_heading_helpers_south_home_after_north_flight():
    level = generate_level(
        style="linear",
        n_segments=2,
        seed=0,
        base_heading_deg=0.0,
        distance_range=(1.0, 1.0),
    )
    assert abs(home_heading_deg(level.home_xy) - 180.0) < 1e-6
    assert home_heading_bin(level) == 18


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


def test_linear_respects_base_heading():
    level = generate_level(
        style="linear",
        n_segments=3,
        seed=0,
        base_heading_deg=90.0,
        distance_range=(1.0, 1.0),
    )
    assert all(seg.heading_deg == 90.0 for seg in level.segments)


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


def test_zigzag_alternates_signs():
    """Zigzag jumps must strictly alternate left/right."""
    for scale in ("gentle", "sharp"):
        for seed in range(40):
            level = generate_level(
                style="zigzag",
                n_segments=6,
                seed=seed,
                zigzag_scale=scale,  # type: ignore[arg-type]
            )
            headings = [seg.heading_deg for seg in level.segments]
            diffs = [_circ_diff_deg(a, b) for a, b in zip(headings, headings[1:])]
            assert all(abs(d) > 1e-9 for d in diffs)
            signs = [np.sign(d) for d in diffs]
            for prev, cur in zip(signs, signs[1:]):
                assert prev == -cur


def test_zigzag_sharp_jumps_exceed_gentle():
    gentle_jumps: list[float] = []
    sharp_jumps: list[float] = []
    for seed in range(30):
        for scale, bucket in (("gentle", gentle_jumps), ("sharp", sharp_jumps)):
            level = generate_level(
                style="zigzag",
                n_segments=5,
                seed=seed,
                zigzag_scale=scale,  # type: ignore[arg-type]
            )
            headings = [seg.heading_deg for seg in level.segments]
            bucket.extend(abs(_circ_diff_deg(a, b)) for a, b in zip(headings, headings[1:]))
    # Zigzag gentle ≈ 50–70°; sharp ≈ 90–120°
    assert min(gentle_jumps) >= 50.0 - 1e-6
    assert max(gentle_jumps) <= 70.0 + 1e-6
    assert min(sharp_jumps) >= 90.0 - 1e-6
    assert max(gentle_jumps) < min(sharp_jumps)


def test_zigzag_gentle_jumps_are_visible():
    """Zigzag gentle uses 5–7 bin jumps → 50°–70° (not the tiny turning gentle)."""
    allowed = {50.0, 60.0, 70.0}
    for seed in range(40):
        level = generate_level(
            style="zigzag",
            n_segments=6,
            seed=seed,
            zigzag_scale="gentle",
        )
        headings = [seg.heading_deg for seg in level.segments]
        for prev, cur in zip(headings, headings[1:]):
            assert abs(_circ_diff_deg(prev, cur)) in allowed


def test_zigzag_min_segments_edge():
    level = generate_level(style="zigzag", n_segments=2, seed=0, zigzag_scale="gentle")
    assert len(level.segments) == 2
    d = _circ_diff_deg(level.segments[0].heading_deg, level.segments[1].heading_deg)
    assert abs(d) > 0.0


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
    with pytest.raises(ValueError, match="Unknown style"):
        generate_level(style="diagonal")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_segments"):
        generate_level(n_segments=0)
