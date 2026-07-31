"""Unit tests for path geometry, curriculum, and epoch shuffling."""

from __future__ import annotations

from collections import Counter

import numpy as np

from pigeonpilot.paths import (
    DIFFICULTIES,
    DIFFICULTY_SPECS,
    STYLES,
    Segment,
    epoch_order,
    generate_curriculum_dataset,
    generate_dataset,
    generate_level,
    home_vector,
    displacement_vector,
    iter_training_indices,
    snap_heading,
    split_dataset,
    summarize_dataset,
)


def test_style_names():
    assert STYLES == ("linear", "turning", "curved")


def test_snap_heading_36_bins():
    assert snap_heading(4, 36) == 0.0
    assert snap_heading(6, 36) == 10.0


def test_single_segment_east_home_vector():
    segments = (Segment(heading_deg=0.0, distance=1.0),)
    np.testing.assert_allclose(displacement_vector(segments), [1, 0], atol=1e-9)
    np.testing.assert_allclose(home_vector(segments), [-1, 0], atol=1e-9)


def test_linear_is_truly_straight():
    level = generate_level(style="linear", n_segments=3, seed=0)
    assert len({seg.heading_deg for seg in level.segments}) == 1


def test_curriculum_progressive_styles_and_counts():
    data = generate_curriculum_dataset(seed=42)
    expected = sum(sum(DIFFICULTY_SPECS[d]["counts"].values()) for d in DIFFICULTIES)
    assert len(data) == expected == 150

    summary = summarize_dataset(data)
    assert summary["by_difficulty"] == {"easy": 60, "medium": 60, "hard": 30}

    easy = {lv.style for lv in data if lv.difficulty == "easy"}
    medium = {lv.style for lv in data if lv.difficulty == "medium"}
    hard = {lv.style for lv in data if lv.difficulty == "hard"}
    assert easy == {"linear", "turning"}
    assert medium == {"turning", "curved"}
    assert hard == {"turning", "curved"}
    assert "linear" not in medium | hard


def test_hard_curved_is_long_arc():
    data = generate_curriculum_dataset(seed=0)
    hard_curved = [lv for lv in data if lv.difficulty == "hard" and lv.style == "curved"]
    assert hard_curved
    assert all(6 <= len(lv.segments) <= 8 for lv in hard_curved)


def test_split_and_schedule():
    from pigeonpilot.paths import iter_train_schedule

    data = generate_curriculum_dataset(seed=1)
    train, test = split_dataset(data, train_frac=0.8, seed=0)
    assert len(train) + len(test) == len(data)
    schedule = list(
        iter_train_schedule(
            train,
            mode="curriculum",
            epochs_per_difficulty={"easy": 1, "medium": 1, "hard": 1},
            seed=0,
        )
    )
    phases = [p for p, _, _, _ in schedule]
    assert phases == sorted(phases, key=lambda p: {"easy": 0, "medium": 1, "hard": 2}[p])


def test_epoch_order_is_permutation():
    order = epoch_order(10, epoch=3, seed=0)
    assert sorted(order.tolist()) == list(range(10))
    counts = Counter(idx for _, idx in iter_training_indices(5, n_epochs=20, seed=0))
    assert all(counts[i] == 20 for i in range(5))


def test_shared_limits_are_square():
    from pigeonpilot.viz import compute_xy_limits

    data = generate_curriculum_dataset(seed=2)
    xlim, ylim = compute_xy_limits(data)
    assert xlim == ylim
    assert abs(xlim[0] + xlim[1]) < 1e-9


def test_grid_zooms_to_given_levels():
    import matplotlib

    matplotlib.use("Agg")
    from pigeonpilot.viz import plot_levels_grid, compute_xy_limits

    data = generate_curriculum_dataset(seed=42)
    easy = [lv for lv in data if lv.difficulty == "easy"]
    assert compute_xy_limits(data)[0][1] > 6.0

    fig = plot_levels_grid(easy, title="easy")
    ax = next(a for a in fig.axes if a.has_data())
    assert ax.get_xlim()[1] < 3.0
    assert ax.get_xlim() == ax.get_ylim()
