"""
Curriculum config, datasets, and training schedules for PigeonPilot.

Depends on ``paths`` (domain / generation). Never imported by ``paths``.
"""

from __future__ import annotations

from typing import Any, Iterator, Literal, Sequence

import numpy as np

from .paths import (
    DEFAULT_HEADING_BINS,
    DIFFICULTIES,
    STYLES,
    Difficulty,
    DifficultySpec,
    Level,
    Style,
    generate_level,
)

STYLE_LABELS: dict[Style, str] = {
    "linear": "straight / direct",
    "turning": "heading jumps (turns)",
    "curved": "wave or sweeping arc",
}

DIFFICULTY_SPECS: dict[Difficulty, DifficultySpec] = {
    "easy": {
        "counts": {"linear": 30, "turning": 30},
        "n_segments": (2, 3),
        "distance_range": (0.4, 1.0),
        "turning_scale": "gentle",
        "curved_mode": "mild",
        "skill": "distance + mild turn",
    },
    "medium": {
        "counts": {"turning": 30, "curved": 30},
        "n_segments": (4, 5),
        "distance_range": (0.5, 1.4),
        "turning_scale": "sharp",
        "curved_mode": "mild",
        "skill": "many turns/curves (no linear)",
    },
    "hard": {
        "counts": {"turning": 15, "curved": 15},
        "n_segments": (6, 8),
        "distance_range": (0.7, 1.8),
        "turning_scale": "sharp",
        "curved_mode": "arc",
        "skill": "long memory + circle-like arcs",
    },
}

DEFAULT_EPOCHS_PER_DIFFICULTY: dict[Difficulty, int] = {
    "easy": 40,
    "medium": 30,
    "hard": 30,
}

ScheduleMode = Literal["mixed", "curriculum"]
SEED_STRIDE = 1_000_003
# Keep ≥1 held-out sample when a (difficulty, style) group is large enough.
MIN_GROUP_SIZE_FOR_HELD_OUT_TEST = 5


def curriculum_level_count(
    specs: dict[Difficulty, DifficultySpec] | None = None,
) -> int:
    """Total levels implied by ``DIFFICULTY_SPECS`` (or an override)."""
    specs = specs or DIFFICULTY_SPECS
    return sum(sum(cfg["counts"].values()) for cfg in specs.values())


def _format_families_used(cfg: DifficultySpec) -> str:
    parts: list[str] = []
    for style in STYLES:
        if style not in cfg["counts"]:
            continue
        if style == "turning":
            parts.append(f"{cfg['turning_scale']} `{style}`")
        elif style == "curved":
            parts.append(f"{cfg['curved_mode']} `{style}`")
        else:
            parts.append(f"`{style}`")
    return " + ".join(parts)


def format_curriculum_table(
    specs: dict[Difficulty, DifficultySpec] | None = None,
    train_frac: float = 0.8,
) -> str:
    """
    Markdown table of the curriculum SSOT for notebooks / docs.

    Renders cleanly via ``IPython.display.Markdown(...)``.
    """
    specs = specs or DIFFICULTY_SPECS
    lines = [
        "| Difficulty | Count | Families used | Segments | Skill |",
        "|------------|------:|---------------|----------|-------|",
    ]
    for difficulty in DIFFICULTIES:
        cfg = specs[difficulty]
        count = sum(cfg["counts"].values())
        n_lo, n_hi = cfg["n_segments"]
        lines.append(
            f"| **{difficulty}** | {count} | {_format_families_used(cfg)} "
            f"| {n_lo}–{n_hi} | {cfg['skill']} |"
        )
    total = curriculum_level_count(specs)
    train_pct = int(round(train_frac * 100))
    test_pct = 100 - train_pct
    lines.append("")
    lines.append(
        f"**Total = {total} levels.** Then ~{train_pct}% train / ~{test_pct}% test."
    )
    return "\n".join(lines)


def format_style_labels_table() -> str:
    """Markdown table of trajectory-family display labels (SSOT)."""
    lines = [
        "| Name | Meaning |",
        "|------|---------|",
    ]
    for style in STYLES:
        lines.append(f"| `{style}` | {STYLE_LABELS[style]} |")
    return "\n".join(lines)


def generate_curriculum_dataset(
    seed: int = 42,
    heading_bins: int = DEFAULT_HEADING_BINS,
    specs: dict[Difficulty, DifficultySpec] | None = None,
) -> list[Level]:
    """
    Build easy → medium → hard with *different skills*, not just longer lines.

    Default size comes from ``DIFFICULTY_SPECS`` (see ``curriculum_level_count``).
    """
    specs = specs or DIFFICULTY_SPECS
    rng = np.random.default_rng(seed)
    levels: list[Level] = []
    level_id = 0

    for difficulty in DIFFICULTIES:
        cfg = specs[difficulty]
        n_lo, n_hi = cfg["n_segments"]
        counts = cfg["counts"]
        for style, n_style in counts.items():
            for _ in range(int(n_style)):
                n_segments = int(rng.integers(n_lo, n_hi + 1))
                child_seed = int(rng.integers(0, 2**31 - 1))
                levels.append(
                    generate_level(
                        style=style,
                        n_segments=n_segments,
                        seed=child_seed,
                        heading_bins=heading_bins,
                        distance_range=cfg["distance_range"],
                        level_id=level_id,
                        difficulty=difficulty,
                        curved_mode=cfg["curved_mode"],
                        turning_scale=cfg["turning_scale"],
                    )
                )
                level_id += 1
    return levels


def split_dataset(
    levels: Sequence[Level],
    train_frac: float = 0.8,
    seed: int = 0,
) -> tuple[list[Level], list[Level]]:
    """
    Stratified train/test split by (difficulty, style).

    Prevents 'it only memorized these 24 paths' when evaluating.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")

    rng = np.random.default_rng(seed)
    train: list[Level] = []
    test: list[Level] = []

    groups: dict[tuple[Difficulty, Style], list[Level]] = {}
    for lv in levels:
        groups.setdefault((lv.difficulty, lv.style), []).append(lv)

    for key in sorted(groups.keys()):
        group = list(groups[key])
        rng.shuffle(group)
        n_train = max(1, int(round(len(group) * train_frac)))
        if len(group) >= MIN_GROUP_SIZE_FOR_HELD_OUT_TEST:
            n_train = min(n_train, len(group) - 1)
        train.extend(group[:n_train])
        test.extend(group[n_train:])

    train.sort(key=lambda lv: lv.level_id)
    test.sort(key=lambda lv: lv.level_id)
    return train, test


def summarize_dataset(levels: Sequence[Level]) -> dict[str, Any]:
    """Counts per difficulty and style — handy for notebook prints."""
    summary: dict[str, Any] = {
        "total": len(levels),
        "by_difficulty": {},
        "by_style": {},
    }
    for diff in DIFFICULTIES:
        summary["by_difficulty"][diff] = sum(1 for lv in levels if lv.difficulty == diff)
    for style in STYLES:
        summary["by_style"][style] = sum(1 for lv in levels if lv.style == style)
    return summary


def epoch_order(n_levels: int, epoch: int, seed: int = 0) -> np.ndarray:
    """Permutation of level indices for one epoch (reproducible)."""
    rng = np.random.default_rng(seed + epoch * SEED_STRIDE)
    return rng.permutation(n_levels)


def iter_training_indices(
    n_levels: int,
    n_epochs: int = 100,
    seed: int = 0,
) -> Iterator[tuple[int, int]]:
    """
    Yield (epoch, level_index) with a fresh shuffle every epoch (mixed mode).

    After n_epochs, each index appeared exactly n_epochs times.
    """
    for epoch in range(n_epochs):
        for level_index in epoch_order(n_levels, epoch=epoch, seed=seed):
            yield epoch, int(level_index)


def iter_train_schedule(
    train_levels: Sequence[Level],
    mode: ScheduleMode = "curriculum",
    epochs_per_difficulty: dict[Difficulty, int] | None = None,
    n_epochs_mixed: int = 100,
    seed: int = 0,
) -> Iterator[tuple[str, int, int, Level]]:
    """
    Training presentation order over the train set.

    Parameters
    ----------
    mode:
        - "curriculum": train easy first, then medium, then hard
          (each phase shuffles only that difficulty).
        - "mixed": every epoch shuffles all train levels together.
    epochs_per_difficulty:
        Only for curriculum mode. Defaults to ``DEFAULT_EPOCHS_PER_DIFFICULTY``.
    n_epochs_mixed:
        Only for mixed mode.

    Yields
    ------
    (phase_name, phase_epoch, train_index, level)
        train_index is the index into ``train_levels``.
    """
    if mode == "mixed":
        for epoch, idx in iter_training_indices(len(train_levels), n_epochs_mixed, seed):
            yield "mixed", epoch, idx, train_levels[idx]
        return

    epochs_per_difficulty = epochs_per_difficulty or DEFAULT_EPOCHS_PER_DIFFICULTY

    by_diff: dict[Difficulty, list[int]] = {d: [] for d in DIFFICULTIES}
    for i, lv in enumerate(train_levels):
        by_diff[lv.difficulty].append(i)

    global_phase_epoch = 0
    for difficulty in DIFFICULTIES:
        idxs = by_diff[difficulty]
        if not idxs:
            continue
        n_epochs = int(epochs_per_difficulty.get(difficulty, 0))
        for ep in range(n_epochs):
            order = epoch_order(len(idxs), epoch=global_phase_epoch, seed=seed)
            for local in order:
                train_index = idxs[int(local)]
                yield difficulty, ep, train_index, train_levels[train_index]
            global_phase_epoch += 1
