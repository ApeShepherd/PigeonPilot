"""
Displacement paths for PigeonPilot — domain types, geometry, and single-level generation.

A *level* is a displacement route made of heading/distance segments.
Home is the origin (0, 0). After the route, home_vector = -end_xy.

Curriculum config, datasets, and training schedules live in ``curriculum.py``.
``paths`` must never import ``curriculum`` (one-way dependency).

Heading convention (frozen)
---------------------------
Compass / navigation degrees, **not** the mathematical polar angle:

- ``0°``   = North  (+y)
- ``90°``  = East   (+x)
- ``180°`` = South  (-y)
- ``270°`` = West   (-x)

Unit displacement::

    (dx, dy) = (sin(heading), cos(heading)) * distance

Default input resolution is 36 bins → 10° steps (one neuron per bin later).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, Sequence, TypedDict

import numpy as np

Style = Literal["linear", "turning", "zigzag", "curved"]
Difficulty = Literal["easy", "medium", "hard", "expert", "extreme"]
TurningScale = Literal["gentle", "sharp"]
ZigzagScale = Literal["gentle", "sharp"]
CurvedMode = Literal["mild", "arc"]

STYLES: tuple[Style, ...] = ("linear", "turning", "zigzag", "curved")
DIFFICULTIES: tuple[Difficulty, ...] = (
    "easy",
    "medium",
    "hard",
    "expert",
    "extreme",
)

FULL_CIRCLE_DEG = 360.0
DEFAULT_HEADING_BINS = 36

# Turning / zigzag / curved generator parameters (bin counts relative to heading_bins)
GENTLE_JUMP_BINS: tuple[int, ...] = (2, 3, 4)  # turning gentle ≈ 20–40°
ZIGZAG_GENTLE_JUMP_BINS: tuple[int, ...] = (5, 6, 7)  # zigzag gentle ≈ 50–70° (visible)
SHARP_JUMP_QUARTER_DIV = 4  # ~90° on a full circle of bins
SHARP_JUMP_THIRD_DIV = 3  # ~120°
MILD_AMP_DIVISOR = 12  # amp ≈ heading_bins/12 (~30° at 36 bins)
ARC_TURN_MIN_RATIO = 0.5  # ≥ ~180° sweep
ARC_TURN_MAX_RATIO = 0.85  # ≤ ~306° sweep


class DifficultySpec(TypedDict):
    """Schema for one difficulty stage (concrete specs live in ``curriculum``).

    Attributes
    ----------
    counts :
        How many levels of each trajectory ``Style`` to generate.
    n_segments :
        Inclusive ``(lo, hi)`` range for segment count per level.
    distance_range :
        Inclusive-style ``(lo, hi)`` uniform draw for segment lengths.
    turning_scale :
        Jump size for ``turning`` levels (``gentle`` / ``sharp``).
    zigzag_scale :
        Jump size for ``zigzag`` levels (``gentle`` / ``sharp``).
    curved_mode :
        Default shape for ``curved`` levels (``mild`` / ``arc``).
    skill :
        Short human-readable label for tables / notebooks.
    curved_modes :
        Optional split of curved counts by mode (e.g. mild + arc).
    """

    counts: dict[Style, int]
    n_segments: tuple[int, int]
    distance_range: tuple[float, float]
    turning_scale: TurningScale
    zigzag_scale: ZigzagScale
    curved_mode: CurvedMode
    skill: str
    curved_modes: NotRequired[dict[CurvedMode, int]]


@dataclass(frozen=True)
class Segment:
    """One straight flight piece: compass heading + distance.

    Attributes
    ----------
    heading_deg :
        Compass heading in degrees. ``0`` = North (+y), ``90`` = East (+x),
        ``180`` = South (-y), ``270`` = West (-x).
    distance :
        Path length along that heading (same units as coordinates).
    """

    heading_deg: float
    distance: float


@dataclass(frozen=True)
class Level:
    """One full displacement trial with ground-truth vectors (pure data).

    Attributes
    ----------
    level_id :
        Stable integer id within a dataset.
    style :
        Trajectory family (``linear`` / ``turning`` / ``zigzag`` / ``curved``).
    segments :
        Ordered flight pieces from home.
    end_xy :
        Release point ``(x, y)`` after integrating all segments.
    home_xy :
        Vector from release back home (``-end_xy``).
    difficulty :
        Curriculum stage tag.
    """

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
    """Convert compass degrees to a 2D unit vector.

    Parameters
    ----------
    heading_deg :
        Compass heading. ``0`` = North (+y), ``90`` = East (+x),
        ``180`` = South (-y), ``270`` = West (-x).

    Returns
    -------
    np.ndarray
        Shape ``(2,)``: ``[sin(θ), cos(θ)]`` → ``(x, y)``.
    """
    rad = np.deg2rad(heading_deg)
    return np.array([np.sin(rad), np.cos(rad)], dtype=float)


def snap_heading(heading_deg: float, heading_bins: int = DEFAULT_HEADING_BINS) -> float:
    """Snap heading to the nearest bin (default: 10° steps for 36 input neurons).

    Parameters
    ----------
    heading_deg :
        Raw compass heading in degrees.
    heading_bins :
        Number of equal bins over ``[0, 360)``. Default ``36`` → 10° resolution.

    Returns
    -------
    float
        Snapped heading in ``[0, 360)``.
    """
    step = _heading_step(heading_bins)
    snapped = round(heading_deg / step) * step
    return float(snapped % FULL_CIRCLE_DEG)


def trajectory_points(segments: Sequence[Segment]) -> np.ndarray:
    """Cumulative ``(x, y)`` positions along a route.

    Parameters
    ----------
    segments :
        Ordered flight pieces starting from home.

    Returns
    -------
    np.ndarray
        Shape ``(n_segments + 1, 2)``. Row 0 is home ``(0, 0)``.
    """
    points = [np.zeros(2, dtype=float)]
    pos = np.zeros(2, dtype=float)
    for seg in segments:
        pos = pos + _heading_to_unit(seg.heading_deg) * seg.distance
        points.append(pos.copy())
    return np.stack(points, axis=0)


def displacement_vector(segments: Sequence[Segment]) -> np.ndarray:
    """Vector from home ``(0, 0)`` to the release point.

    Parameters
    ----------
    segments :
        Ordered flight pieces.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` end position.
    """
    return trajectory_points(segments)[-1].copy()


def home_vector(segments: Sequence[Segment]) -> np.ndarray:
    """Vector from the release point back home (``-displacement``).

    Parameters
    ----------
    segments :
        Ordered flight pieces.

    Returns
    -------
    np.ndarray
        Shape ``(2,)`` home vector (Ridge-regression target).
    """
    return -displacement_vector(segments)


def home_heading_deg(home_xy: tuple[float, float] | Sequence[float]) -> float:
    """Compass heading of the home vector (``0°`` = North / +y)."""
    x, y = float(home_xy[0]), float(home_xy[1])
    return float((np.degrees(np.arctan2(x, y)) + FULL_CIRCLE_DEG) % FULL_CIRCLE_DEG)


def home_heading_bin(
    level: Level,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> int:
    """Snap ``level.home_xy`` to a heading bin in ``{0 … heading_bins-1}``."""
    step = _heading_step(heading_bins)
    return int(round(home_heading_deg(level.home_xy) / step)) % heading_bins


# ---------------------------------------------------------------------------
# Heading generators
# ---------------------------------------------------------------------------

def _sample_distance(rng: np.random.Generator, distance_range: tuple[float, float]) -> float:
    lo, hi = distance_range
    return float(rng.uniform(lo, hi))


def _jump_magnitudes(
    heading_bins: int,
    scale: TurningScale | ZigzagScale,
    *,
    family: Literal["turning", "zigzag"] = "turning",
) -> list[int]:
    """Absolute jump sizes in bins.

    Turning gentle ≈ 20–40°; zigzag gentle ≈ 50–70° (clearer cancellation);
    sharp ≈ 90–120° for both families.
    """
    if scale == "gentle":
        if family == "zigzag":
            return list(ZIGZAG_GENTLE_JUMP_BINS)
        return list(GENTLE_JUMP_BINS)
    q = heading_bins // SHARP_JUMP_QUARTER_DIV
    t = heading_bins // SHARP_JUMP_THIRD_DIV
    return [q, t]


def _generate_linear_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
    base_heading_deg: float | None = None,
) -> list[float]:
    """True straight line: same heading every segment."""
    step = _heading_step(heading_bins)
    if base_heading_deg is None:
        base = float(rng.integers(0, heading_bins) * step)
    else:
        base = snap_heading(base_heading_deg, heading_bins)
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
    mags = _jump_magnitudes(heading_bins, turning_scale, family="turning")
    jumps = mags + [-b for b in mags]
    for _ in range(n_segments - 1):
        jump = int(rng.choice(jumps))
        headings.append(snap_heading(headings[-1] + jump * step, heading_bins))
    return headings


def _generate_zigzag_headings(
    rng: np.random.Generator,
    n_segments: int,
    heading_bins: int,
    zigzag_scale: ZigzagScale = "gentle",
) -> list[float]:
    """Strictly alternating left/right jumps (cancellation skill)."""
    step = _heading_step(heading_bins)
    headings = [float(rng.integers(0, heading_bins) * step)]
    mags = _jump_magnitudes(heading_bins, zigzag_scale, family="zigzag")
    sign = 1 if rng.random() < 0.5 else -1
    for _ in range(n_segments - 1):
        mag = int(rng.choice(mags))
        headings.append(snap_heading(headings[-1] + sign * mag * step, heading_bins))
        sign = -sign
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
    seed: int = 0,
    heading_bins: int = DEFAULT_HEADING_BINS,
    distance_range: tuple[float, float] = (0.5, 1.5),
    level_id: int = 0,
    difficulty: Difficulty = "easy",
    curved_mode: CurvedMode = "mild",
    turning_scale: TurningScale = "gentle",
    zigzag_scale: ZigzagScale = "gentle",
    base_heading_deg: float | None = None,
) -> Level:
    """Create one displacement level (geometry computed once, then stored).

    Parameters
    ----------
    style :
        Trajectory family: ``linear``, ``turning``, ``zigzag``, or ``curved``.
    n_segments :
        Number of straight pieces (≥ 1).
    seed :
        RNG seed for reproducible headings/distances. Always an ``int``
        (never ``None``). Curriculum generation uses its own default ``42``
        and passes explicit child seeds here.
    heading_bins :
        Compass resolution. Default ``36`` → 10° bins (``0°`` = North).
    distance_range :
        Inclusive-style ``(lo, hi)`` uniform draw for each segment length.
    level_id :
        Stored on the returned ``Level``.
    difficulty :
        Curriculum stage tag stored on the level.
    curved_mode :
        Only for ``style="curved"``: ``mild`` or ``arc``.
    turning_scale :
        Only for ``style="turning"``: ``gentle`` or ``sharp``.
    zigzag_scale :
        Only for ``style="zigzag"``: ``gentle`` or ``sharp``.
    base_heading_deg :
        Optional fixed outbound heading for ``style="linear"`` (systematic
        label coverage). Ignored for other styles.

    Returns
    -------
    Level
        Immutable trial with ``end_xy`` and ``home_xy`` precomputed.

    Raises
    ------
    ValueError
        If ``style`` is unknown or ``n_segments < 1``.
    """
    if style not in STYLES:
        raise ValueError(f"Unknown style {style!r}. Choose from {STYLES}.")
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")

    rng = np.random.default_rng(seed)
    if style == "linear":
        headings = _generate_linear_headings(
            rng, n_segments, heading_bins, base_heading_deg=base_heading_deg
        )
    elif style == "turning":
        headings = _generate_turning_headings(
            rng, n_segments, heading_bins, turning_scale=turning_scale
        )
    elif style == "zigzag":
        headings = _generate_zigzag_headings(
            rng, n_segments, heading_bins, zigzag_scale=zigzag_scale
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
