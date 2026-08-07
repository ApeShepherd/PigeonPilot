"""Unit tests for curriculum config, datasets, and training schedules."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pytest

from resources.curriculum import (
    DEFAULT_EPOCHS_PER_DIFFICULTY,
    DEFAULT_MIXED_EPOCHS_AFTER,
    DIFFICULTY_SPECS,
    LINEAR_REPEATS_PER_BIN,
    STYLE_LABELS,
    curriculum_level_count,
    epoch_order,
    format_curriculum_table,
    format_style_labels_table,
    generate_curriculum_dataset,
    iter_train_schedule,
    iter_training_indices,
    split_dataset,
    summarize_dataset,
)
from resources.encoding import encode_level
from resources.paths import (
    DEFAULT_HEADING_BINS,
    DIFFICULTIES,
    STYLES,
    home_heading_bin,
)


EXPECTED_N = 720


def test_style_labels_ssot():
    assert set(STYLE_LABELS) == set(STYLES)
    assert STYLE_LABELS["linear"] == "straight / direct"
    assert STYLE_LABELS["turning"] == "heading jumps (turns)"
    assert STYLE_LABELS["zigzag"] == "alternating left/right zigzags"
    assert STYLE_LABELS["curved"] == "wave or sweeping arc"


def test_curriculum_ssot_helpers():
    assert curriculum_level_count() == EXPECTED_N
    assert DEFAULT_EPOCHS_PER_DIFFICULTY == {
        "easy": 15,
        "medium": 30,
        "hard": 25,
        "expert": 20,
        "extreme": 20,
    }
    assert DEFAULT_MIXED_EPOCHS_AFTER == 6
    assert DEFAULT_EPOCHS_PER_DIFFICULTY["easy"] < DEFAULT_EPOCHS_PER_DIFFICULTY["medium"]
    table = format_curriculum_table()
    assert f"| **easy** | {DIFFICULTY_SPECS['easy']['counts']['linear']} |" in table
    assert f"| **medium** | {DIFFICULTY_SPECS['medium']['counts']['turning']} |" in table
    assert f"| **hard** | {DIFFICULTY_SPECS['hard']['counts']['zigzag']} |" in table
    assert f"| **expert** | {DIFFICULTY_SPECS['expert']['counts']['zigzag']} |" in table
    assert f"| **extreme** | {DIFFICULTY_SPECS['extreme']['counts']['curved']} |" in table
    assert f"Total = {EXPECTED_N} levels" in table
    assert DIFFICULTY_SPECS["easy"]["skill"] in table
    styles = format_style_labels_table()
    assert "`zigzag`" in styles
    assert STYLE_LABELS["zigzag"] in styles


def test_level_home_xy_is_negation_of_end_xy():
    """home_xy == -end_xy for every curriculum level."""
    data = generate_curriculum_dataset(seed=42)
    for level in data:
        assert np.allclose(
            level.home_xy,
            (-level.end_xy[0], -level.end_xy[1]),
        )


def test_curriculum_monostyle_stages_and_counts():
    data = generate_curriculum_dataset(seed=42)
    expected = curriculum_level_count()
    assert len(data) == expected == EXPECTED_N

    summary = summarize_dataset(data)
    assert summary["by_difficulty"] == {
        d: sum(DIFFICULTY_SPECS[d]["counts"].values()) for d in DIFFICULTIES
    }
    assert summary["by_style"]["linear"] == 180
    assert summary["by_style"]["turning"] == 135
    assert summary["by_style"]["zigzag"] == 270
    assert summary["by_style"]["curved"] == 135

    for difficulty in DIFFICULTIES:
        styles = {lv.style for lv in data if lv.difficulty == difficulty}
        assert styles == set(DIFFICULTY_SPECS[difficulty]["counts"])


def test_easy_linear_covers_all_home_bins():
    data = generate_curriculum_dataset(seed=42)
    easy = [lv for lv in data if lv.difficulty == "easy"]
    assert len(easy) == LINEAR_REPEATS_PER_BIN * DEFAULT_HEADING_BINS
    assert all(lv.style == "linear" for lv in easy)
    bins = {home_heading_bin(lv) for lv in easy}
    assert bins == set(range(DEFAULT_HEADING_BINS))


def test_hard_and_expert_are_zigzag_with_correct_scale():
    data = generate_curriculum_dataset(seed=0)
    hard = [lv for lv in data if lv.difficulty == "hard"]
    expert = [lv for lv in data if lv.difficulty == "expert"]
    assert hard and all(lv.style == "zigzag" for lv in hard)
    assert expert and all(lv.style == "zigzag" for lv in expert)
    assert all(5 <= len(lv.segments) <= 7 for lv in hard + expert)


def test_extreme_is_arc_only():
    """Extreme stage is sweeping arcs only — no mild waves mixed in."""
    data = generate_curriculum_dataset(seed=0)
    extreme = [lv for lv in data if lv.difficulty == "extreme"]
    assert len(extreme) == 135
    assert all(lv.style == "curved" for lv in extreme)
    assert all(6 <= len(lv.segments) <= 8 for lv in extreme)
    assert "curved_modes" not in DIFFICULTY_SPECS["extreme"]
    assert DIFFICULTY_SPECS["extreme"]["curved_mode"] == "arc"

    # Arc paths keep a consistent turn direction between steps.
    def circ_diff(a: float, b: float) -> float:
        return ((b - a + 180.0) % 360.0) - 180.0

    for lv in extreme[:40]:
        headings = [seg.heading_deg for seg in lv.segments]
        diffs = [circ_diff(a, b) for a, b in zip(headings, headings[1:])]
        nonzero = [d for d in diffs if abs(d) > 1e-9]
        assert nonzero
        signs = {np.sign(d) for d in nonzero}
        assert len(signs) == 1


def test_split_preserves_all_levels():
    data = generate_curriculum_dataset(seed=1)
    train, test = split_dataset(data, train_frac=0.8, seed=0)
    assert len(train) + len(test) == len(data)
    assert {lv.level_id for lv in train} | {lv.level_id for lv in test} == {
        lv.level_id for lv in data
    }
    assert not {lv.level_id for lv in train} & {lv.level_id for lv in test}


def test_split_train_covers_all_easy_home_bins():
    data = generate_curriculum_dataset(seed=1)
    train, test = split_dataset(data, train_frac=0.8, seed=0)
    easy_train = [lv for lv in train if lv.difficulty == "easy"]
    assert {home_heading_bin(lv) for lv in easy_train} == set(range(DEFAULT_HEADING_BINS))
    assert any(lv.difficulty == d for d in DIFFICULTIES for lv in test)


def test_stratified_split_both_sides_nonempty_when_group_ge_5():
    """
    For every (difficulty, style) group with size >= 5, train and test
    must both be nonempty.
    """
    data = generate_curriculum_dataset(seed=1)
    train, test = split_dataset(data, train_frac=0.8, seed=0)

    by_key_all: dict[tuple[str, str], list] = defaultdict(list)
    by_key_train: dict[tuple[str, str], list] = defaultdict(list)
    by_key_test: dict[tuple[str, str], list] = defaultdict(list)
    for lv in data:
        by_key_all[(lv.difficulty, lv.style)].append(lv)
    for lv in train:
        by_key_train[(lv.difficulty, lv.style)].append(lv)
    for lv in test:
        by_key_test[(lv.difficulty, lv.style)].append(lv)

    for key, group in by_key_all.items():
        if len(group) < 5:
            continue
        assert by_key_train[key], f"train empty for {key} (n={len(group)})"
        assert by_key_test[key], f"test empty for {key} (n={len(group)})"
        assert len(by_key_train[key]) + len(by_key_test[key]) == len(group)


def test_curriculum_schedule_phase_order():
    data = generate_curriculum_dataset(seed=1)
    train, _test = split_dataset(data, train_frac=0.8, seed=0)
    epochs = {d: 1 for d in DIFFICULTIES}
    schedule = list(
        iter_train_schedule(
            train,
            mode="curriculum",
            epochs_per_difficulty=epochs,
            n_epochs_mixed_after=0,
            seed=0,
        )
    )
    phases = [p for p, _, _, _ in schedule]
    rank = {d: i for i, d in enumerate(DIFFICULTIES)}
    assert phases == sorted(phases, key=lambda p: rank[p])


def test_curriculum_schedule_mixed_after():
    data = generate_curriculum_dataset(seed=1)
    train, _test = split_dataset(data, train_frac=0.8, seed=0)
    epochs = {d: 1 for d in DIFFICULTIES}
    schedule = list(
        iter_train_schedule(
            train,
            mode="curriculum",
            epochs_per_difficulty=epochs,
            n_epochs_mixed_after=2,
            seed=0,
        )
    )
    phases = [p for p, _, _, _ in schedule]
    assert "mixed" in phases
    first_mixed = next(i for i, p in enumerate(phases) if p == "mixed")
    assert all(p != "mixed" for p in phases[:first_mixed])
    assert all(p == "mixed" for p in phases[first_mixed:])
    mixed_epochs = {ep for phase, ep, _, _ in schedule if phase == "mixed"}
    assert mixed_epochs == {0, 1}


def test_mixed_schedule_shuffles_full_train_each_epoch():
    data = generate_curriculum_dataset(seed=1)
    train, _test = split_dataset(data, train_frac=0.8, seed=0)
    n_epochs = 3
    schedule = list(
        iter_train_schedule(
            train,
            mode="mixed",
            n_epochs_mixed=n_epochs,
            seed=0,
        )
    )

    assert len(schedule) == n_epochs * len(train)
    assert all(phase == "mixed" for phase, *_ in schedule)

    for epoch in range(n_epochs):
        rows = [(idx, lv) for phase, ep, idx, lv in schedule if ep == epoch]
        idxs = [idx for idx, _ in rows]
        assert sorted(idxs) == list(range(len(train)))
        for idx, lv in rows:
            assert lv.level_id == train[idx].level_id


def test_epoch_order_is_permutation():
    order = epoch_order(10, epoch=3, seed=0)
    assert sorted(order.tolist()) == list(range(10))
    counts = Counter(idx for _, idx in iter_training_indices(5, n_epochs=20, seed=0))
    assert all(counts[i] == 20 for i in range(5))


def test_split_dataset_rejects_bad_train_frac():
    data = generate_curriculum_dataset(seed=0)
    with pytest.raises(ValueError, match="train_frac"):
        split_dataset(data, train_frac=0.0)
    with pytest.raises(ValueError, match="train_frac"):
        split_dataset(data, train_frac=1.0)


def test_encode_zigzag_level_smoke():
    data = generate_curriculum_dataset(seed=0)
    zig = next(lv for lv in data if lv.style == "zigzag")
    spikes = encode_level(zig, dt=1.0, velocity=1.0)
    assert spikes.ndim == 2
    assert spikes.shape[1] == DEFAULT_HEADING_BINS
    assert spikes.shape[0] > 0
