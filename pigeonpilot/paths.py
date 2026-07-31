"""
Displacement paths for PigeonPilot — domain types, geometry, and single-level generation.

A *level* is a displacement route made of heading/distance segments.
Home is the origin (0, 0). After the route, home_vector = -end_xy.

Curriculum config, datasets, and training schedules live in ``curriculum.py``.
``paths`` must never import ``curriculum`` (one-way dependency).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict

import numpy as np

Style = Literal["linear", "turning", "curved"]
Difficulty = Literal["easy", "medium", "hard"]
TurningScale = Literal["gentle", "sharp"]
CurvedMode = Literal["mild", "arc"]

STYLES: tuple[Style, ...] = ("linear", "turning", "curved")
DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")

FULL_CIRCLE_DEG = 360.0
DEFAULT_HEADING_BINS = 36

# Turning / curved generator parameters (bin counts relative to heading_bins)
GENTLE_JUMP_BINS: tuple[int, ...] = (2, 3, 4)
SHARP_JUMP_QUARTER_DIV = 4  # ~90° on a full circle of bins
SHARP_JUMP_THIRD_DIV = 3  # ~120°
MILD_AMP_DIVISOR = 12  # amp ≈ heading_bins/12 (~30° at 36 bins)
ARC_TURN_MIN_RATIO = 0.5  # ≥ ~180° sweep
ARC_TURN_MAX_RATIO = 0.85  # ≤ ~306° sweep


class DifficultySpec(TypedDict):
    """Schema for one difficulty stage (concrete specs live in ``curriculum``)."""

    counts: dict[Style, int]
    n_segments: tuple[int, int]
    distance_range: tuple[float, float]
    turning_scale: TurningScale
    curved_mode: CurvedMode
    skill: str


@dataclass(frozen=True)
class Segment:
    """One straight flight piece: compass heading + distance."""

    heading_deg: float  # 0 = East, 90 = North (math angle, degrees)
    distance: float


@dataclass(frozen=True)
class Level:
    """One full displacement trial with ground-truth vectors (pure data)."""

    level_id: int
    style: Style
    segments: tuple[Segment, ...]
    end_xy: tuple[float, float]
    home_xy: tuple[float, float]
    difficulty: Difficulty = "easy"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _heading_step(heading_bins: int) -> float:
    """Degrees per heading bin."""
    return FULL_CIRCLE_DEG / heading_bins


def _heading_to_unit(heading_deg: float) -> np.ndarray:
    """Convert degrees to a 2D unit vector (0° = +x / East, 90° = +y / North)."""
    rad = np.deg2rad(heading_deg)
    return np.array([np.cos(rad), np.sin(rad)], dtype=float)


def snap_heading(heading_deg: float, heading_bins: int = DEFAULT_HEADING_BINS) -> float:
    """Snap heading to the nearest bin (default: 10° steps for 36 input neurons)."""
    step = _heading_step(heading_bins)
    snapped = round(heading_deg / step) * step
    return float(snapped % FULL_CIRCLE_DEG)


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
    step = _heading_step(heading_bins)
    base = float(rng.integers(0, heading_bins) * step)
    return [base] * n_segments


def _generate_turning_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
    turning_scale: TurningScale = "sharp",
) -> list[float]:
    """Turns on the bin grid. gentle ≈ small bends; sharp ≈ ~90° jumps."""
    step = _heading_step(heading_bins)
    headings = [float(rng.integers(0, heading_bins) * step)]
    if turning_scale == "gentle":
        # ~20–40° on a 36-bin grid
        jumps = list(GENTLE_JUMP_BINS) + [-b for b in GENTLE_JUMP_BINS]
    else:
        # ~90° / ~120° from quarter / third of the bin circle
        q = heading_bins // SHARP_JUMP_QUARTER_DIV
        t = heading_bins // SHARP_JUMP_THIRD_DIV
        jumps = [q, t, -q, -t]
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
    step = _heading_step(heading_bins)
    main = float(rng.integers(0, heading_bins) * step)
    amp_bins = max(1, heading_bins // MILD_AMP_DIVISOR)
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
    step = _heading_step(heading_bins)
    start_bin = int(rng.integers(0, heading_bins))
    lo = int(heading_bins * ARC_TURN_MIN_RATIO)
    hi = int(ARC_TURN_MAX_RATIO * heading_bins) + 1
    total_turn_bins = int(rng.integers(lo, hi))
    sign = 1 if rng.random() < 0.5 else -1
    headings = []
    for i in range(n_segments):
        frac = i / max(n_segments - 1, 1)
        bin_i = start_bin + sign * int(round(frac * total_turn_bins))
        headings.append(snap_heading(bin_i * step, heading_bins))
    return headings


def _curved_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
    curved_mode: CurvedMode,
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
    heading_bins: int = DEFAULT_HEADING_BINS,
    distance_range: tuple[float, float] = (0.5, 1.5),
    level_id: int = 0,
    difficulty: Difficulty = "easy",
    curved_mode: CurvedMode = "mild",
    turning_scale: TurningScale = "gentle",
) -> Level:
    """Create one displacement level (geometry computed once, then stored on Level)."""
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
    end = trajectory_points(segments)[-1]
    end_xy = (float(end[0]), float(end[1]))
    home_xy = (-end_xy[0], -end_xy[1])
    return Level(
        level_id=level_id,
        style=style,
        segments=segments,
        end_xy=end_xy,
        home_xy=home_xy,
        difficulty=difficulty,
    )
