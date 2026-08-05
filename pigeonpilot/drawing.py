"""
Freehand drawing → model-ready ``Level`` for the live playground.

Why this module exists
----------------------
A mouse polyline is not a curriculum route. Raw pointer samples carry hand
tremor, and ``playground.polyline_to_segments`` keeps the full length of every
sample step, so the noise is integrated into the path length: a stroke the user
perceives as ~6 units long arrives at the encoder as 15–53 units with 100+
heading changes. The reservoir was trained on routes of 1.08–9.15 total length
with at most 8 merged heading blocks, so such a stroke is far outside the
training distribution and the readout returns noise.

This module closes that gap in three steps, all of them reported back to the
caller so the UI can state what happened instead of silently "fixing" the input:

1. **Simplify** (Ramer–Douglas–Peucker) until the snapped route has at most
   ``max_blocks`` merged heading blocks — this removes tremor, not shape.
2. **Snap** each simplified step to the frozen 10° compass grid.
3. **Scale** the segment distances so the total route length lands inside the
   training band, and record the factor that was applied.

What the network actually sees is the sequence of *merged* heading blocks
(consecutive segments on the same bin are one continuous firing window), so the
distribution guards below are stated in terms of merged blocks, not raw
segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .encoding import heading_to_bin, segment_n_steps
from .paths import (
    DEFAULT_HEADING_BINS,
    FULL_CIRCLE_DEG,
    Difficulty,
    Level,
    Segment,
    Style,
    snap_heading,
    trajectory_points,
)

DEMO_LEVEL_ID = 900_001
DEFAULT_MAX_BLOCKS = 8
# A stray click must not be scaled up into a full route: below this stroke
# length (world units) the direction is noise, so we refuse instead of inventing.
MIN_STROKE_DISTANCE = 0.25
_MIN_POINT_SPACING = 1e-9
_EPSILON_GROWTH = 1.35
_EPSILON_MAX_ROUNDS = 60
# Scaling lands exactly on a band edge; absorb float round-off in the check.
_BAND_TOLERANCE = 1e-9


@dataclass(frozen=True)
class TrainingBand:
    """Envelope of the curriculum, measured on merged heading blocks.

    Attributes
    ----------
    min_total_distance, max_total_distance :
        Total route length of the shortest / longest curriculum level.
    median_total_distance :
        Median route length, used as the fallback target for degenerate strokes.
    max_release_distance :
        Farthest release point in the curriculum — the honest radius for the
        drawing canvas.
    max_blocks :
        Largest number of merged heading blocks seen in training.
    """

    min_total_distance: float
    median_total_distance: float
    max_total_distance: float
    max_release_distance: float
    max_blocks: int


@lru_cache(maxsize=4)
def training_band(seed: int = 42, heading_bins: int = DEFAULT_HEADING_BINS) -> TrainingBand:
    """Measure the curriculum envelope (~0.25 s, cached).

    Derived from ``generate_curriculum_dataset`` rather than hard-coded so the
    guard rails move automatically if the curriculum is ever retuned.
    """
    from .curriculum import generate_curriculum_dataset

    levels = generate_curriculum_dataset(seed=seed, heading_bins=heading_bins)
    totals = np.array([sum(s.distance for s in lv.segments) for lv in levels])
    releases = np.array([float(np.hypot(*lv.end_xy)) for lv in levels])
    blocks = np.array([len(merge_heading_blocks(lv.segments, heading_bins)) for lv in levels])
    return TrainingBand(
        min_total_distance=float(totals.min()),
        median_total_distance=float(np.median(totals)),
        max_total_distance=float(totals.max()),
        max_release_distance=float(releases.max()),
        max_blocks=int(blocks.max()),
    )


def merge_heading_blocks(
    segments: Sequence[Segment],
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> tuple[tuple[int, float], ...]:
    """Collapse segments into the ``(bin, distance)`` blocks the encoder emits.

    Consecutive segments that snap to the same body-ring neuron produce one
    continuous firing window, so they are indistinguishable to the network.
    """
    blocks: list[list[float]] = []
    for seg in segments:
        bin_idx = heading_to_bin(seg.heading_deg, heading_bins)
        if blocks and int(blocks[-1][0]) == bin_idx:
            blocks[-1][1] += float(seg.distance)
        else:
            blocks.append([float(bin_idx), float(seg.distance)])
    return tuple((int(b), float(d)) for b, d in blocks)


def simplify_polyline(points: np.ndarray | Sequence[Sequence[float]], epsilon: float) -> np.ndarray:
    """Ramer–Douglas–Peucker simplification (iterative, no recursion limit).

    Keeps every vertex whose perpendicular distance to the current chord exceeds
    ``epsilon``. Endpoints are always kept.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(pts) < 3 or epsilon <= 0.0:
        return pts.copy()

    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack: list[tuple[int, int]] = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        chord = pts[hi] - pts[lo]
        chord_len = float(np.linalg.norm(chord))
        span = pts[lo + 1 : hi] - pts[lo]
        if chord_len < _MIN_POINT_SPACING:
            dist = np.linalg.norm(span, axis=1)
        else:
            normal = np.array([-chord[1], chord[0]]) / chord_len
            dist = np.abs(span @ normal)
        local = int(np.argmax(dist))
        if float(dist[local]) > epsilon:
            split = lo + 1 + local
            keep[split] = True
            stack.append((lo, split))
            stack.append((split, hi))
    return pts[keep]


def polyline_to_segments(
    points: np.ndarray | Sequence[Sequence[float]],
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> tuple[Segment, ...]:
    """Snap every step of a polyline to the compass grid and merge equal headings.

    Unlike ``playground.polyline_to_segments`` this keeps the full length of the
    stroke: no step is dropped, so the snapped route stays anchored to what the
    user drew.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(pts) < 2:
        return ()
    if float(np.linalg.norm(pts[0])) > _MIN_POINT_SPACING:
        pts = np.vstack([np.zeros(2), pts])

    merged: list[list[float]] = []
    for i in range(1, len(pts)):
        delta = pts[i] - pts[i - 1]
        distance = float(np.linalg.norm(delta))
        if distance < _MIN_POINT_SPACING:
            continue
        raw_heading = (
            np.degrees(np.arctan2(delta[0], delta[1])) + FULL_CIRCLE_DEG
        ) % FULL_CIRCLE_DEG
        heading = float(snap_heading(raw_heading, heading_bins))
        if merged and merged[-1][0] == heading:
            merged[-1][1] += distance
        else:
            merged.append([heading, distance])
    return tuple(Segment(heading_deg=h, distance=d) for h, d in merged)


def scale_segments(segments: Sequence[Segment], factor: float) -> tuple[Segment, ...]:
    """Uniformly scale distances; headings (and therefore the shape) are untouched."""
    return tuple(
        Segment(heading_deg=s.heading_deg, distance=float(s.distance) * float(factor))
        for s in segments
    )


def level_from_segments(
    segments: Sequence[Segment],
    *,
    level_id: int = DEMO_LEVEL_ID,
    style: Style = "turning",
    difficulty: Difficulty = "medium",
) -> Level:
    """Build a ``Level`` from segments, home at the origin."""
    if not segments:
        raise ValueError("need at least one segment")
    end = trajectory_points(segments)[-1]
    end_xy = (float(end[0]), float(end[1]))
    return Level(
        level_id=level_id,
        style=style,
        segments=tuple(segments),
        end_xy=end_xy,
        home_xy=(-end_xy[0], -end_xy[1]),
        difficulty=difficulty,
    )


@dataclass(frozen=True)
class DrawnPath:
    """A user stroke turned into a model-ready level, with the audit trail.

    The UI is expected to surface ``scale_factor`` and ``in_distribution`` so a
    viewer can see that the stroke was normalised and by how much.

    Attributes
    ----------
    raw_points :
        Pointer samples as drawn, home prepended.
    simplified_points :
        Stroke after RDP, before snapping and scaling.
    snapped_points :
        Vertices of the route the network actually receives.
    segments :
        Final scaled + snapped segments.
    level :
        Ready for ``snn.predict_home_bin``.
    raw_total_distance, total_distance :
        Route length before / after scaling.
    scale_factor :
        Applied uniform scale (``1.0`` when the stroke already fit).
    epsilon :
        RDP tolerance that met the block budget.
    n_blocks :
        Merged heading blocks — what the encoder turns into firing windows.
    n_steps :
        Simulation timesteps the encoder will produce.
    in_distribution :
        Whether the final route sits inside the measured training band.
    """

    raw_points: np.ndarray
    simplified_points: np.ndarray
    snapped_points: np.ndarray
    segments: tuple[Segment, ...]
    level: Level
    raw_total_distance: float
    total_distance: float
    scale_factor: float
    epsilon: float
    n_blocks: int
    n_steps: int
    in_distribution: bool


def prepare_drawn_path(
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    level_id: int = DEMO_LEVEL_ID,
    heading_bins: int = DEFAULT_HEADING_BINS,
    max_blocks: int | None = None,
    band: TrainingBand | None = None,
    velocity: float = 1.0,
    dt: float = 1.0,
    min_stroke_distance: float = MIN_STROKE_DISTANCE,
) -> DrawnPath | None:
    """Turn pointer samples into a level inside the training distribution.

    Returns ``None`` when the stroke is too short to define a direction.

    Parameters
    ----------
    points :
        Pointer samples in world coordinates, ``(N, 2)``.
    max_blocks :
        Merged-heading-block budget. Defaults to the curriculum maximum.
    band :
        Pre-measured training envelope; measured on demand when omitted.
    velocity, dt :
        Encoder settings, used only to report ``n_steps``.
    min_stroke_distance :
        Reject strokes shorter than this instead of scaling them up.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        return None
    if float(np.linalg.norm(pts[0])) > _MIN_POINT_SPACING:
        pts = np.vstack([np.zeros(2), pts])

    keep = np.concatenate([[True], np.linalg.norm(np.diff(pts, axis=0), axis=1) > _MIN_POINT_SPACING])
    pts = pts[keep]
    if len(pts) < 2:
        return None

    band = band or training_band(heading_bins=heading_bins)
    budget = int(max_blocks if max_blocks is not None else band.max_blocks)

    # Grow the RDP tolerance until the *snapped* route fits the block budget.
    # Seeded from the stroke's own extent so it scales with the canvas.
    extent = float(np.abs(pts).max())
    epsilon = max(extent, 1.0) * 0.005
    simplified = pts
    segments = polyline_to_segments(pts, heading_bins)
    for _ in range(_EPSILON_MAX_ROUNDS):
        if segments and len(merge_heading_blocks(segments, heading_bins)) <= budget:
            break
        epsilon *= _EPSILON_GROWTH
        simplified = simplify_polyline(pts, epsilon)
        segments = polyline_to_segments(simplified, heading_bins)
    if not segments:
        return None

    raw_total = float(sum(s.distance for s in segments))
    if raw_total < max(float(min_stroke_distance), _MIN_POINT_SPACING):
        return None
    if raw_total > band.max_total_distance:
        factor = band.max_total_distance / raw_total
    elif raw_total < band.min_total_distance:
        factor = band.min_total_distance / raw_total
    else:
        factor = 1.0

    segments = scale_segments(segments, factor)
    level = level_from_segments(segments, level_id=level_id)
    blocks = merge_heading_blocks(segments, heading_bins)
    total = float(sum(s.distance for s in segments))
    n_steps = int(sum(segment_n_steps(d, velocity=velocity, dt=dt) for _, d in blocks))
    return DrawnPath(
        raw_points=pts,
        simplified_points=np.asarray(simplified, dtype=float),
        snapped_points=trajectory_points(segments),
        segments=segments,
        level=level,
        raw_total_distance=raw_total,
        total_distance=total,
        scale_factor=float(factor),
        epsilon=float(epsilon),
        n_blocks=len(blocks),
        n_steps=n_steps,
        in_distribution=(
            len(blocks) <= band.max_blocks
            and band.min_total_distance * (1.0 - _BAND_TOLERANCE) <= total
            and total <= band.max_total_distance * (1.0 + _BAND_TOLERANCE)
        ),
    )
