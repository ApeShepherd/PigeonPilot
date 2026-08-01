"""Unit tests for rate-coding spike encoding."""

from __future__ import annotations

import numpy as np

from pigeonpilot.encoding import (
    compass_bin,
    encode_level,
    encode_segments,
    heading_to_bin,
    plan_encoding,
    segment_n_steps,
)
from pigeonpilot.paths import Segment, generate_level


def test_compass_bin_is_world_frame():
    assert compass_bin(0.0) == 0
    assert compass_bin(90.0) == 9
    assert compass_bin(180.0) == 18
    assert compass_bin(270.0) == 27


def test_heading_to_bin_north_aligned_body_ring():
    """Active bin = body neuron pointing at North (not travel-direction label)."""
    assert heading_to_bin(0.0) == 0  # face North (heading 0°) → bin 0 points North
    assert heading_to_bin(90.0) == 27  # face East (heading 90°) → bin 27 points North
    assert heading_to_bin(180.0) == 18  # face South (heading 180°) → bin 18 points North
    assert heading_to_bin(270.0) == 9  # face West (heading 270°) → bin 9 points North #


def test_segment_n_steps_defaults_map_distance_one_to_one():
    assert segment_n_steps(10.0) == 10
    assert segment_n_steps(0.4) == 1  # rounds / floor to at least 1
    assert segment_n_steps(0.0) == 1


def test_ten_units_north_activates_only_bin_zero():
    segments = (Segment(heading_deg=0.0, distance=10.0),)
    spikes = encode_segments(segments)
    assert spikes.shape == (10, 36)
    assert spikes.dtype == np.float32
    np.testing.assert_array_equal(spikes.sum(axis=0), np.eye(36)[0] * 10)
    assert spikes.sum() == 10.0


def test_west_activates_bin_nine():
    """Heading 270° = West → body neuron 9 points North and receives spikes."""
    segments = (Segment(heading_deg=270.0, distance=4.0),)
    spikes = encode_segments(segments)
    assert spikes.shape == (4, 36)
    np.testing.assert_array_equal(spikes.sum(axis=0), np.eye(36)[9] * 4)


def test_north_then_east_concatenates_bins():
    segments = (
        Segment(heading_deg=0.0, distance=3.0),
        Segment(heading_deg=90.0, distance=2.0),
    )
    spikes = encode_segments(segments)
    assert spikes.shape == (5, 36)
    assert np.all(spikes[:3, 0] == 1.0)
    assert np.all(spikes[3:, 27] == 1.0)  # East → north-pointing body bin 27
    assert np.all(spikes[3:, 0] == 0.0)


def test_plan_encoding_covers_timeline():
    segments = (
        Segment(heading_deg=0.0, distance=3.0),
        Segment(heading_deg=270.0, distance=2.0),
    )
    plan = plan_encoding(segments)
    assert [b.bin_idx for b in plan] == [0, 9]
    assert plan[0].start == 0 and plan[0].end == 3
    assert plan[1].start == 3 and plan[1].end == 5
    spikes = encode_segments(segments)
    assert spikes.shape[0] == plan[-1].end


def test_encode_level_shape_and_min_duration():
    level = generate_level(style="linear", n_segments=3, seed=0)
    spikes = encode_level(level)
    assert spikes.ndim == 2
    assert spikes.shape[1] == 36
    assert spikes.shape[0] >= len(level.segments)


def test_encode_is_reproducible():
    level = generate_level(style="turning", n_segments=4, seed=7, turning_scale="gentle")
    a = encode_level(level, velocity=1.0, dt=1.0)
    b = encode_level(level, velocity=1.0, dt=1.0)
    np.testing.assert_array_equal(a, b)


def test_segment_n_steps_scales_with_dt():
    assert segment_n_steps(1.0, velocity=1.0, dt=0.25) == 4
    assert segment_n_steps(1.0, velocity=2.0, dt=0.25) == 2


def test_segment_n_steps_rejects_bad_args():
    import pytest

    with pytest.raises(ValueError, match="velocity"):
        segment_n_steps(1.0, velocity=0.0)
    with pytest.raises(ValueError, match="dt"):
        segment_n_steps(1.0, dt=-0.1)
    with pytest.raises(ValueError, match="distance"):
        segment_n_steps(-1.0)


def test_plan_encoding_rejects_empty():
    import pytest

    with pytest.raises(ValueError, match="non-empty"):
        plan_encoding(())


def test_heading_bins_other_than_default():
    segments = (Segment(heading_deg=0.0, distance=2.0),)
    spikes = encode_segments(segments, heading_bins=18)
    assert spikes.shape == (2, 18)
    assert spikes[:, 0].sum() == 2.0
