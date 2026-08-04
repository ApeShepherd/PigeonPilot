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
    FULL_CIRCLE_DEG,
    STYLES,
    CurvedMode,
    Difficulty,
    DifficultySpec,
    Level,
    Style,
    generate_level,
    home_heading_bin,
)

STYLE_LABELS: dict[Style, str] = {
    "linear": "straight / direct",
    "turning": "heading jumps (turns)",
    "zigzag": "alternating left/right zigzags",
    "curved": "wave or sweeping arc",
}

# Monostyle skill ladder: linear → gentle turn → zigzag×2 → curved.
# easy is short (sanity / bin coverage); real PI starts at medium.
DIFFICULTY_SPECS: dict[Difficulty, DifficultySpec] = {
    "easy": {
        "counts": {"linear": 180},
        "n_segments": (2, 4),
        "distance_range": (0.5, 1.2),
        "turning_scale": "gentle",
        "zigzag_scale": "gentle",
        "curved_mode": "mild",
        "skill": "heading inversion / distance (warm-up)",
    },
    "medium": {
        "counts": {"turning": 135},
        "n_segments": (3, 5),
        "distance_range": (0.5, 1.3),
        "turning_scale": "gentle",
        "zigzag_scale": "gentle",
        "curved_mode": "mild",
        "skill": "gentle vector addition (first real PI)",
    },
    "hard": {
        "counts": {"zigzag": 135},
        "n_segments": (5, 7),
        "distance_range": (0.5, 1.2),
        "turning_scale": "gentle",
        "zigzag_scale": "gentle",
        "curved_mode": "arc",
        "skill": "visible zigzag / moderate cancellation (~50–70°)",
    },
    "expert": {
        "counts": {"zigzag": 135},
        "n_segments": (5, 7),
        "distance_range": (0.5, 1.2),
        "turning_scale": "sharp",
        "zigzag_scale": "sharp",
        "curved_mode": "arc",
        "skill": "sharp zigzag / strong cancellation (~90–120°)",
    },
    "extreme": {
        "counts": {"curved": 135},
        "n_segments": (6, 8),
        "distance_range": (0.5, 1.3),
        "turning_scale": "sharp",
        "zigzag_scale": "sharp",
        "curved_mode": "arc",
        "skill": "sweeping arcs only (long memory)",
    },
}

# Few epochs on easy (avoid +180° shortcut); more on first real PI stages.
DEFAULT_EPOCHS_PER_DIFFICULTY: dict[Difficulty, int] = {
    "easy": 15,
    "medium": 30,
    "hard": 25,
    "expert": 20,
    "extreme": 20,
}

# Short mixed rehearsal after the curriculum ladder (forgetting control).
DEFAULT_MIXED_EPOCHS_AFTER = 6

LINEAR_REPEATS_PER_BIN = 5  # 5 × 36 = 180 easy linear levels

ScheduleMode = Literal["mixed", "curriculum"]
SEED_STRIDE = 1_000_003
# Keep ≥1 held-out sample when a (difficulty, style[, home_bin]) group is large enough.
MIN_GROUP_SIZE_FOR_HELD_OUT_TEST = 5


def curriculum_level_count(
    specs: dict[Difficulty, DifficultySpec] | None = None,
) -> int:
    """Total levels implied by ``DIFFICULTY_SPECS`` (or an override).

    Parameters
    ----------
    specs :
        Optional override of ``DIFFICULTY_SPECS``.

    Returns
    -------
    int
        Sum of all style counts across difficulties.
    """
    specs = specs or DIFFICULTY_SPECS
    return sum(sum(cfg["counts"].values()) for cfg in specs.values())


def _format_families_used(cfg: DifficultySpec) -> str:
    parts: list[str] = []
    for style in STYLES:
        if style not in cfg["counts"]:
            continue
        if style == "turning":
            parts.append(f"{cfg['turning_scale']} `{style}`")
        elif style == "zigzag":
            parts.append(f"{cfg['zigzag_scale']} `{style}`")
        elif style == "curved":
            modes = cfg.get("curved_modes")
            if modes:
                mode_bits = " + ".join(f"{n} {m}" for m, n in modes.items())
                parts.append(f"`{style}` ({mode_bits})")
            else:
                parts.append(f"{cfg['curved_mode']} `{style}`")
        else:
            parts.append(f"`{style}`")
    return " + ".join(parts)


def format_curriculum_table(
    specs: dict[Difficulty, DifficultySpec] | None = None,
    train_frac: float = 0.8,
) -> str:
    """Markdown table of the curriculum SSOT for notebooks / docs.

    Renders cleanly via ``IPython.display.Markdown(...)``.

    Parameters
    ----------
    specs :
        Optional override of ``DIFFICULTY_SPECS``.
    train_frac :
        Assumed train share for the footer line (display only).

    Returns
    -------
    str
        GitHub-flavored Markdown table plus a total / split footer.
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
    """Markdown table of trajectory-family display labels (SSOT).

    Returns
    -------
    str
        Two-column Markdown table over ``STYLE_LABELS``.
    """
    lines = [
        "| Name | Meaning |",
        "|------|---------|",
    ]
    for style in STYLES:
        lines.append(f"| `{style}` | {STYLE_LABELS[style]} |")
    return "\n".join(lines)


def _append_generated_level(
    levels: list[Level],
    *,
    style: Style,
    n_segments: int,
    child_seed: int,
    heading_bins: int,
    cfg: DifficultySpec,
    difficulty: Difficulty,
    level_id: int,
    curved_mode: CurvedMode | None = None,
    base_heading_deg: float | None = None,
) -> None:
    levels.append(
        generate_level(
            style=style,
            n_segments=n_segments,
            seed=child_seed,
            heading_bins=heading_bins,
            distance_range=cfg["distance_range"],
            level_id=level_id,
            difficulty=difficulty,
            curved_mode=curved_mode if curved_mode is not None else cfg["curved_mode"],
            turning_scale=cfg["turning_scale"],
            zigzag_scale=cfg["zigzag_scale"],
            base_heading_deg=base_heading_deg,
        )
    )


def generate_curriculum_dataset(
    seed: int = 42,
    heading_bins: int = DEFAULT_HEADING_BINS,
    specs: dict[Difficulty, DifficultySpec] | None = None,
) -> list[Level]:
    """Build the monostyle skill ladder easy → … → extreme.

    Linear (easy) levels systematically cover every outbound heading bin
    (``LINEAR_REPEATS_PER_BIN`` repeats) so home-bin labels are complete.

    Parameters
    ----------
    seed :
        Master RNG seed (default ``42``). Child seeds for each ``generate_level``
        call are drawn from this generator so the full dataset is reproducible.
    heading_bins :
        Compass resolution forwarded to ``generate_level`` (``0°`` = North).
    specs :
        Optional override of ``DIFFICULTY_SPECS``.

    Returns
    -------
    list of Level
        Ordered by ``DIFFICULTIES``. Size from ``curriculum_level_count``.
    """
    specs = specs or DIFFICULTY_SPECS
    rng = np.random.default_rng(seed)
    levels: list[Level] = []
    level_id = 0
    step = FULL_CIRCLE_DEG / heading_bins

    for difficulty in DIFFICULTIES:
        cfg = specs[difficulty]
        n_lo, n_hi = cfg["n_segments"]
        counts = cfg["counts"]
        for style in STYLES:
            if style not in counts:
                continue
            n_style = int(counts[style])

            if style == "linear":
                # Systematic outbound headings: repeats × bins == n_style.
                if n_style != LINEAR_REPEATS_PER_BIN * heading_bins:
                    raise ValueError(
                        f"easy linear count must be "
                        f"{LINEAR_REPEATS_PER_BIN}×{heading_bins}="
                        f"{LINEAR_REPEATS_PER_BIN * heading_bins}, got {n_style}"
                    )
                for rep in range(LINEAR_REPEATS_PER_BIN):
                    for bin_i in range(heading_bins):
                        n_segments = int(rng.integers(n_lo, n_hi + 1))
                        child_seed = int(rng.integers(0, 2**31 - 1))
                        _append_generated_level(
                            levels,
                            style=style,
                            n_segments=n_segments,
                            child_seed=child_seed,
                            heading_bins=heading_bins,
                            cfg=cfg,
                            difficulty=difficulty,
                            level_id=level_id,
                            base_heading_deg=float(bin_i * step),
                        )
                        level_id += 1
                continue

            if style == "curved" and "curved_modes" in cfg:
                mode_counts = cfg["curved_modes"]
                if sum(mode_counts.values()) != n_style:
                    raise ValueError(
                        f"{difficulty} curved_modes sum "
                        f"{sum(mode_counts.values())} != counts {n_style}"
                    )
                for mode, n_mode in mode_counts.items():
                    for _ in range(int(n_mode)):
                        n_segments = int(rng.integers(n_lo, n_hi + 1))
                        child_seed = int(rng.integers(0, 2**31 - 1))
                        _append_generated_level(
                            levels,
                            style=style,
                            n_segments=n_segments,
                            child_seed=child_seed,
                            heading_bins=heading_bins,
                            cfg=cfg,
                            difficulty=difficulty,
                            level_id=level_id,
                            curved_mode=mode,
                        )
                        level_id += 1
                continue

            for _ in range(n_style):
                n_segments = int(rng.integers(n_lo, n_hi + 1))
                child_seed = int(rng.integers(0, 2**31 - 1))
                _append_generated_level(
                    levels,
                    style=style,
                    n_segments=n_segments,
                    child_seed=child_seed,
                    heading_bins=heading_bins,
                    cfg=cfg,
                    difficulty=difficulty,
                    level_id=level_id,
                )
                level_id += 1
    return levels


def split_dataset(
    levels: Sequence[Level],
    train_frac: float = 0.8,
    seed: int = 0,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> tuple[list[Level], list[Level]]:
    """Stratified train/test split by (difficulty, style, home_bin).

    Prevents "it only memorized these paths" when evaluating.
    Groups with size ``>= MIN_GROUP_SIZE_FOR_HELD_OUT_TEST`` keep at
    least one held-out test sample.

    Parameters
    ----------
    levels :
        Full curriculum (or any ``Level`` sequence).
    train_frac :
        Target train fraction in ``(0, 1)``.
    seed :
        Shuffle seed within each stratum.
    heading_bins :
        Bin resolution for ``home_heading_bin`` stratification.

    Returns
    -------
    train, test : list of Level
        Sorted by ``level_id`` within each split.

    Raises
    ------
    ValueError
        If ``train_frac`` is not in ``(0, 1)``.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")

    rng = np.random.default_rng(seed)
    train: list[Level] = []
    test: list[Level] = []

    groups: dict[tuple[Difficulty, Style, int], list[Level]] = {}
    for lv in levels:
        key = (lv.difficulty, lv.style, home_heading_bin(lv, heading_bins))
        groups.setdefault(key, []).append(lv)

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
    """Counts per difficulty and style — handy for notebook prints.

    Parameters
    ----------
    levels :
        Levels to summarize.

    Returns
    -------
    dict
        Keys ``total``, ``by_difficulty``, ``by_style``.
    """
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
    """Permutation of level indices for one epoch (reproducible).

    Parameters
    ----------
    n_levels :
        Length of the index set to shuffle.
    epoch :
        Epoch counter mixed into the RNG seed.
    seed :
        Base seed (``seed + epoch * SEED_STRIDE``).

    Returns
    -------
    np.ndarray
        Permutation of ``0 … n_levels-1``.
    """
    rng = np.random.default_rng(seed + epoch * SEED_STRIDE)
    return rng.permutation(n_levels)


def iter_training_indices(
    n_levels: int,
    n_epochs: int = 100,
    seed: int = 0,
) -> Iterator[tuple[int, int]]:
    """Yield ``(epoch, level_index)`` with a fresh shuffle every epoch.

    After ``n_epochs``, each index appeared exactly ``n_epochs`` times
    (mixed-mode helper).

    Parameters
    ----------
    n_levels :
        Size of the train index set.
    n_epochs :
        Number of full passes.
    seed :
        Base seed forwarded to ``epoch_order``.

    Yields
    ------
    epoch, level_index : tuple of int
        Epoch counter and shuffled train index.
    """
    for epoch in range(n_epochs):
        for level_index in epoch_order(n_levels, epoch=epoch, seed=seed):
            yield epoch, int(level_index)


def iter_train_schedule(
    train_levels: Sequence[Level],
    mode: ScheduleMode = "curriculum",
    epochs_per_difficulty: dict[Difficulty, int] | None = None,
    n_epochs_mixed: int = 100,
    n_epochs_mixed_after: int = 0,
    seed: int = 0,
) -> Iterator[tuple[str, int, int, Level]]:
    """Training presentation order over the train set.

    Parameters
    ----------
    train_levels :
        Train split (index space for yielded ``train_index``).
    mode :
        ``"curriculum"``: easy → … → extreme (shuffle within phase).
        ``"mixed"``: every epoch shuffles all train levels together.
    epochs_per_difficulty :
        Only for curriculum mode. Defaults to ``DEFAULT_EPOCHS_PER_DIFFICULTY``.
    n_epochs_mixed :
        Only for mixed mode.
    n_epochs_mixed_after :
        After curriculum phases, optionally run this many full mixed epochs
        (forgetting control). Ignored when ``mode="mixed"``.
    seed :
        Base seed for epoch shuffles.

    Yields
    ------
    phase_name, phase_epoch, train_index, level
        ``train_index`` indexes into ``train_levels``.
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

    if n_epochs_mixed_after > 0 and train_levels:
        for ep in range(int(n_epochs_mixed_after)):
            order = epoch_order(len(train_levels), epoch=global_phase_epoch, seed=seed)
            for train_index in order:
                yield "mixed", ep, int(train_index), train_levels[int(train_index)]
            global_phase_epoch += 1
