"""Unit tests for curriculum config, datasets, and training schedules."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from pigeonpilot.curriculum import (
    DEFAULT_EPOCHS_PER_DIFFICULTY,
    DIFFICULTY_SPECS,
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
from pigeonpilot.paths import DIFFICULTIES, STYLES


def test_style_labels_ssot():
    assert set(STYLE_LABELS) == set(STYLES)
    assert STYLE_LABELS["linear"] == "straight / direct"
    assert STYLE_LABELS["turning"] == "heading jumps (turns)"
    assert STYLE_LABELS["curved"] == "wave or sweeping arc"


def test_curriculum_ssot_helpers():
    assert curriculum_level_count() == 150
    assert DEFAULT_EPOCHS_PER_DIFFICULTY == {"easy": 40, "medium": 30, "hard": 30}
    table = format_curriculum_table()
    assert "| **easy** | 60 |" in table
    assert "| **medium** | 60 |" in table
    assert "| **hard** | 30 |" in table
    assert "Total = 150 levels" in table
    assert DIFFICULTY_SPECS["easy"]["skill"] in table
    styles = format_style_labels_table()
    assert "`linear`" in styles
    assert STYLE_LABELS["linear"] in styles


def test_level_home_xy_is_negation_of_end_xy():
    """home_xy == -end_xy for every curriculum level."""
    data = generate_curriculum_dataset(seed=42)
    for level in data:
        assert np.allclose(
            level.home_xy,
            (-level.end_xy[0], -level.end_xy[1]),
        )


def test_curriculum_progressive_styles_and_counts():
    data = generate_curriculum_dataset(seed=42)
    expected = curriculum_level_count()
    assert len(data) == expected == 150

    summary = summarize_dataset(data)
    assert summary["by_difficulty"] == {
        d: sum(DIFFICULTY_SPECS[d]["counts"].values()) for d in DIFFICULTIES
    }

    easy = {lv.style for lv in data if lv.difficulty == "easy"}
    medium = {lv.style for lv in data if lv.difficulty == "medium"}
    hard = {lv.style for lv in data if lv.difficulty == "hard"}
    assert easy == set(DIFFICULTY_SPECS["easy"]["counts"])
    assert medium == set(DIFFICULTY_SPECS["medium"]["counts"])
    assert hard == set(DIFFICULTY_SPECS["hard"]["counts"])
    assert "linear" not in medium | hard


def test_hard_curved_is_long_arc():
    data = generate_curriculum_dataset(seed=0)
    hard_curved = [lv for lv in data if lv.difficulty == "hard" and lv.style == "curved"]
    assert hard_curved
    assert all(6 <= len(lv.segments) <= 8 for lv in hard_curved)


def test_split_preserves_all_levels():
    data = generate_curriculum_dataset(seed=1)
    train, test = split_dataset(data, train_frac=0.8, seed=0)
    assert len(train) + len(test) == len(data)
    assert {lv.level_id for lv in train} | {lv.level_id for lv in test} == {
        lv.level_id for lv in data
    }
    assert not {lv.level_id for lv in train} & {lv.level_id for lv in test}


def test_stratified_split_both_sides_nonempty_when_group_ge_5():
    """
    For every (difficulty, style) group with size >= 5, train and test
    must both be nonempty (current split_dataset contract).
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
