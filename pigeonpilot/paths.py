"""
Displacement paths for PigeonPilot.

A *level* is a displacement route made of heading/distance segments.
Home is the origin (0, 0). After the route, home_vector = -end_xy.

Two naming layers
-----------------
1) Trajectory family (shape of the path):
   - linear  : straight / direct
   - turning : sharp or gentle heading jumps
   - curved  : mild wave or sweeping arc

2) Difficulty tag (curriculum stage):
   - easy / medium / hard

Curriculum (skills build up):
  easy   (60): linear + gentle turning, 2–3 short segments
  medium (60): sharp turning + mild curved, 4–5 segments (no linear)
  hard   (30): sharp turning + arc curved, 6–8 longer segments
  Total = 150 levels, then ~80/20 train/test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal, Sequence

import numpy as np

Style = Literal["linear", "turning", "curved"]
Difficulty = Literal["easy", "medium", "hard"]
TurningScale = Literal["gentle", "sharp"]

STYLES: tuple[Style, ...] = ("linear", "turning", "curved")
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

STYLE_LABELS = {
    "linear": "linear",
    "turning": "turning",
    "curved": "curved",
}

DIFFICULTY_SPECS: dict[Difficulty, dict] = {
    "easy": {
        "counts": {"linear": 30, "turning": 30},
        "n_segments": (2, 3),
        "distance_range": (0.4, 1.0),
        "turning_scale": "gentle",
        "curved_mode": "mild",
    },
    "medium": {
        "counts": {"turning": 30, "curved": 30},
        "n_segments": (4, 5),
        "distance_range": (0.5, 1.4),
        "turning_scale": "sharp",
        "curved_mode": "mild",
    },
    "hard": {
        "counts": {"turning": 15, "curved": 15},
        "n_segments": (6, 8),
        "distance_range": (0.7, 1.8),
        "turning_scale": "sharp",
        "curved_mode": "arc",
    },
}

@dataclass(frozen=True)
class Segment:
    """One straight flight piece: compass heading + distance."""

    heading_deg: float  # 0 = East, 90 = North (math angle, degrees)
    distance: float


@dataclass(frozen=True)
class Level:
    """One full displacement trial with ground-truth vectors."""

    level_id: int
    style: Style
    segments: tuple[Segment, ...]
    difficulty: Difficulty = "easy"
    home_xy: tuple[float, float] = field(init=False)
    end_xy: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        points = trajectory_points(self.segments)
        end = points[-1]
        home = home_vector(self.segments)
        object.__setattr__(self, "end_xy", (float(end[0]), float(end[1])))
        object.__setattr__(self, "home_xy", (float(home[0]), float(home[1])))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _heading_to_unit(heading_deg: float) -> np.ndarray:
    """Convert degrees to a 2D unit vector (0° = +x / East, 90° = +y / North)."""
    rad = np.deg2rad(heading_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=float)


def snap_heading(heading_deg: float, heading_bins: int = 36) -> float:
    """Snap heading to the nearest bin (default: 10° steps for 36 input neurons)."""
    step = 360.0 / heading_bins
    snapped = round(heading_deg / step) * step
    return float(snapped % 360.0)


def trajectory_points(segments: Sequence[Segment]) -> np.ndarray:
    """Cumulative (x, y) positions. Shape (n_segments + 1, 2), start at home."""
    points = [np.zeros(2, dtype=float)]
    pos = np.zeros(2, dtype=float)
    for seg in segments:
        pos = pos + _heading_to_unit(seg.heading_deg) * seg.distance
        points.append(pos.copy())
    return np.stack(points, axis=0)


def displacement_vector(segments: Sequence[Segment]) -> np.ndarray:
    """Vector from home (0,0) to the release point."""
    return trajectory_points(segments)[-1].copy()


def home_vector(segments: Sequence[Segment]) -> np.ndarray:
    """Vector from the release point back home (= -displacement)."""
    return -displacement_vector(segments)


# ---------------------------------------------------------------------------
# Heading generators
# ---------------------------------------------------------------------------

def _sample_distance(rng: np.random.Generator, distance_range: tuple[float, float]) -> float:
    lo, hi = distance_range
    return float(rng.uniform(lo, hi))


def _generate_linear_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
) -> list[float]:
    """True straight line: same heading every segment."""
    step = 360.0 / heading_bins
    base = float(rng.integers(0, heading_bins) * step)
    return [base] * n_segments


def _generate_turning_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
    turning_scale: TurningScale = "sharp",
) -> list[float]:
    """Turns on the bin grid. gentle ≈ small bends; sharp ≈ ~90° jumps."""
    step = 360.0 / heading_bins
    headings = [float(rng.integers(0, heading_bins) * step)]
    if turning_scale == "gentle":
        # ~20–40° — still a turn, but easy to integrate
        jumps = [2, 3, 4, -2, -3, -4]
    else:
        # ~90–120°
        jumps = [
            heading_bins // 4,
            heading_bins // 3,
            -(heading_bins // 4),
            -(heading_bins // 3),
        ]
    for _ in range(n_segments - 1):
        jump = int(rng.choice(jumps))
        headings.append(snap_heading(headings[-1] + jump * step, heading_bins))
    return headings


def _generate_curved_mild_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
) -> list[float]:
    """Gentle oscillation around a main heading."""
    step = 360.0 / heading_bins
    main = float(rng.integers(0, heading_bins) * step)
    amp_bins = max(1, heading_bins // 12)  # ~30°
    headings = []
    for i in range(n_segments):
        sign = 1 if (i % 2 == 0) else -1
        jitter = sign * int(rng.integers(1, amp_bins + 1))
        headings.append(snap_heading(main + jitter * step, heading_bins))
    return headings


def _generate_curved_arc_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
) -> list[float]:
    """Heading sweeps steadily — arc / near-circular displacement (harder)."""
    step = 360.0 / heading_bins
    start_bin = int(rng.integers(0, heading_bins))
    # turn between ~180° and ~300° over the whole path
    total_turn_bins = int(rng.integers(heading_bins // 2, int(0.85 * heading_bins) + 1))
    sign = 1 if rng.random() < 0.5 else -1
    headings = []
    for i in range(n_segments):
        # progress 0..1 along the arc
        frac = i / max(n_segments - 1, 1)
        bin_i = start_bin + sign * int(round(frac * total_turn_bins))
        headings.append(snap_heading(bin_i * step, heading_bins))
    return headings


def _curved_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
    curved_mode: str,
) -> list[float]:
    if curved_mode == "arc":
        return _generate_curved_arc_headings(rng, n_segments, heading_bins)
    return _generate_curved_mild_headings(rng, n_segments, heading_bins)


# ---------------------------------------------------------------------------
# Public constructors
# ---------------------------------------------------------------------------

def generate_level(
    style: Style = "linear",
    n_segments: int = 3,
    seed: int | None = None,
    heading_bins: int = 36,
    distance_range: tuple[float, float] = (0.5, 1.5),
    level_id: int = 0,
    difficulty: Difficulty = "easy",
    curved_mode: str = "mild",
    turning_scale: TurningScale = "sharp",
) -> Level:
    """Create one displacement level."""
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}. Choose from {STYLES}.")
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    rng = np.random.default_rng(seed)
    if style == "linear":
        headings = _generate_linear_headings(rng, n_segments, heading_bins)
    elif style == "turning":
        headings = _generate_turning_headings(
            rng, n_segments, heading_bins, turning_scale=turning_scale
        )
    else:
        headings = _curved_headings(rng, n_segments, heading_bins, curved_mode)

    segments = tuple(
        Segment(heading_deg=h, distance=_sample_distance(rng, distance_range))
        for h in headings
    )
    return Level(
        level_id=level_id,
        style=style,
        segments=segments,
        difficulty=difficulty,
    )


def generate_dataset(
    n_per_style: int = 8,
    styles: Sequence[Style] = STYLES,
    n_segments: int = 3,
    seed: int = 42,
    heading_bins: int = 36,
    distance_range: tuple[float, float] = (0.5, 1.5),
    difficulty: Difficulty = "easy",
    curved_mode: str = "mild",
    turning_scale: TurningScale = "gentle",
) -> list[Level]:
    """Small single-difficulty helper (demos / unit tests)."""
    rng = np.random.default_rng(seed)
    levels: list[Level] = []
    level_id = 0
    for style in styles:
        for _ in range(n_per_style):
            child_seed = int(rng.integers(0, 2**31 - 1))
            levels.append(
                generate_level(
                    style=style,
                    n_segments=n_segments,
                    seed=child_seed,
                    heading_bins=heading_bins,
                    distance_range=distance_range,
                    level_id=level_id,
                    difficulty=difficulty,
                    curved_mode=curved_mode,
                    turning_scale=turning_scale,
                )
            )
            level_id += 1
    return levels


def generate_curriculum_dataset(
    seed: int = 42,
    heading_bins: int = 36,
    specs: dict[Difficulty, dict] | None = None,
) -> list[Level]:
    """
    Build easy → medium → hard with *different skills*, not just longer lines.

    Default: 60 easy + 60 medium + 30 hard = 150 levels.
    """
    specs = specs or DIFFICULTY_SPECS
    rng = np.random.default_rng(seed)
    levels: list[Level] = []
    level_id = 0

    for difficulty in DIFFICULTIES:
        cfg = specs[difficulty]
        n_lo, n_hi = cfg["n_segments"]
        counts: dict[str, int] = cfg["counts"]
        for style, n_style in counts.items():
            for _ in range(int(n_style)):
                child_seed = int(rng.integers(0, 2**31 - 1))
                child_rng = np.random.default_rng(child_seed)
                n_segments = int(child_rng.integers(n_lo, n_hi + 1))
                levels.append(
                    generate_level(
                        style=style,  # type: ignore[arg-type]
                        n_segments=n_segments,
                        seed=child_seed,
                        heading_bins=heading_bins,
                        distance_range=tuple(cfg["distance_range"]),
                        level_id=level_id,
                        difficulty=difficulty,
                        curved_mode=cfg.get("curved_mode", "mild"),
                        turning_scale=cfg.get("turning_scale", "sharp"),
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

    # group indices
    groups: dict[tuple[Difficulty, Style], list[Level]] = {}
    for lv in levels:
        groups.setdefault((lv.difficulty, lv.style), []).append(lv)

    for key in sorted(groups.keys()):
        group = list(groups[key])
        rng.shuffle(group)
        n_train = max(1, int(round(len(group) * train_frac)))
        # keep at least one test sample when group is large enough
        if len(group) >= 5:
            n_train = min(n_train, len(group) - 1)
        train.extend(group[:n_train])
        test.extend(group[n_train:])

    train.sort(key=lambda lv: lv.level_id)
    test.sort(key=lambda lv: lv.level_id)
    return train, test


def summarize_dataset(levels: Sequence[Level]) -> dict:
    """Counts per difficulty and style — handy for notebook prints."""
    summary: dict = {"total": len(levels), "by_difficulty": {}, "by_style": {}}
    for diff in DIFFICULTIES:
        summary["by_difficulty"][diff] = sum(1 for lv in levels if lv.difficulty == diff)
    for style in STYLES:
        summary["by_style"][style] = sum(1 for lv in levels if lv.style == style)
    return summary


# ---------------------------------------------------------------------------
# Epoch shuffling / curriculum schedule
# ---------------------------------------------------------------------------

ScheduleMode = Literal["mixed", "curriculum"]


def epoch_order(n_levels: int, epoch: int, seed: int = 0) -> np.ndarray:
    """Permutation of level indices for one epoch (reproducible)."""
    rng = np.random.default_rng(seed + epoch * 1_000_003)
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
        Only for curriculum mode. Default: easy=40, medium=30, hard=30.
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

    epochs_per_difficulty = epochs_per_difficulty or {
        "easy": 40,
        "medium": 30,
        "hard": 30,
    }

    # indices into train_levels grouped by difficulty
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
            rng = np.random.default_rng(seed + global_phase_epoch * 1_000_003)
            order = rng.permutation(len(idxs))
            for local in order:
                train_index = idxs[int(local)]
                yield difficulty, ep, train_index, train_levels[train_index]
            global_phase_epoch += 1
