"""
Rate-coding spike trains for PigeonPilot — sensor layer before the reservoir.

Depends on ``paths`` (geometry / segments). Never imports ``curriculum``,
``viz``, Torch, or BindsNET.

Encoding rule (exposé)
----------------------
The pigeon always faces its travel direction. Input neurons are a **body-fixed**
ring of ``heading_bins`` (default 36 → 10°). Neuron / bin ``0`` sits at the
beak (forward). Spikes go to the bin that currently points at **geographic
North**:

    active_bin = (-compass_bin(heading)) mod n_bins

Examples (36 bins): face North → bin ``0``; turn 90° left to West → bin ``9``.

Constant velocity: firing **duration** ∝ distance. With ``velocity=1`` and
``dt=1``, distance 10 North → 10 timesteps with only bin ``0`` = ``1.0``.
Output shape is always ``(T, n_bins)`` float32.

Why "bin" and why ``0`` not ``1``
--------------------------------
A *bin* is one discrete direction bucket (= one input neuron). Arrays in Python
are **0-based**, so the first neuron is index ``0`` (people often say
"neuron 1" in everyday language — that is our bin ``0``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .paths import (
    DEFAULT_HEADING_BINS,
    FULL_CIRCLE_DEG,
    Level,
    Segment,
    snap_heading,
)


@dataclass(frozen=True)
class SpikeBlock:
    """One segment's contribution to the spike matrix (for tables / rasters).

    Attributes
    ----------
    start :
        Inclusive start timestep in the concatenated train.
    n_steps :
        Firing duration (``>= 1``).
    bin_idx :
        Active body-ring neuron (North-pointing).
    heading_deg :
        Travel / body heading for this segment.
    distance :
        Segment length.
    """

    start: int
    n_steps: int
    bin_idx: int
    heading_deg: float
    distance: float

    @property
    def end(self) -> int:
        """Exclusive end timestep."""
        return self.start + self.n_steps



def compass_bin(
    heading_deg: float,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> int:
    """World-frame compass index of a travel heading (``0`` = North, ``9`` = East).

    Parameters
    ----------
    heading_deg :
        Compass heading in degrees (``0`` = North, ``90`` = East).
    heading_bins :
        Number of equal bins over ``[0, 360)``. Default ``36`` → 10° steps.

    Returns
    -------
    int
        Index in ``[0, heading_bins)``.
    """
    snapped = snap_heading(heading_deg, heading_bins)
    step = FULL_CIRCLE_DEG / heading_bins
    return int(round(snapped / step)) % heading_bins


def heading_to_bin(
    heading_deg: float,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> int:
    """Map travel heading to the body-fixed neuron that points at North.

    The bird faces ``heading_deg``. Bin ``0`` is forward (beak). Bins increase
    clockwise on the body. The neuron currently aligned with geographic North
    receives the spikes::

        heading_to_bin(H) = (-compass_bin(H)) mod n_bins

    Parameters
    ----------
    heading_deg :
        Compass travel / body heading (``0`` = North, ``90`` = East,
        ``270`` = West).
    heading_bins :
        Number of body-ring neurons. Default ``36`` → 10° steps.

    Returns
    -------
    int
        Active input index in ``[0, heading_bins)``.
        North → ``0``; West (90° left) → ``9``; East → ``27`` (36 bins).
    """
    return (-compass_bin(heading_deg, heading_bins)) % heading_bins


def segment_n_steps(
    distance: float,
    *,
    velocity: float = 1.0,
    dt: float = 1.0,
) -> int:
    """Convert segment distance to discrete firing duration.

    Parameters
    ----------
    distance :
        Path length of one straight segment.
    velocity :
        Constant speed (distance per time unit). Must be ``> 0``.
    dt :
        Simulation timestep. Must be ``> 0``.

    Returns
    -------
    int
        Number of timesteps the heading bin stays active (at least 1).

    Raises
    ------
    ValueError
        If ``velocity`` or ``dt`` is not positive, or ``distance`` is negative.
    """
    if velocity <= 0.0:
        raise ValueError("velocity must be > 0")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if distance < 0.0:
        raise ValueError("distance must be >= 0")
    return max(1, int(round(distance / (velocity * dt))))


def plan_encoding(
    segments: Sequence[Segment],
    *,
    velocity: float = 1.0,
    dt: float = 1.0,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> list[SpikeBlock]:
    """Plan per-segment spike blocks (bin + duration) without building the matrix.

    Parameters
    ----------
    segments :
        Ordered flight pieces from home.
    velocity :
        Constant speed (distance per time unit).
    dt :
        Simulation timestep.
    heading_bins :
        Number of body-ring input neurons.

    Returns
    -------
    list of SpikeBlock
        Contiguous blocks covering ``[0, T)``.

    Raises
    ------
    ValueError
        If ``segments`` is empty.
    """
    if not segments:
        raise ValueError("segments must be non-empty")

    blocks: list[SpikeBlock] = []
    t = 0
    for seg in segments:
        n_steps = segment_n_steps(seg.distance, velocity=velocity, dt=dt)
        bin_idx = heading_to_bin(seg.heading_deg, heading_bins)
        blocks.append(
            SpikeBlock(
                start=t,
                n_steps=n_steps,
                bin_idx=bin_idx,
                heading_deg=seg.heading_deg,
                distance=seg.distance,
            )
        )
        t += n_steps
    return blocks


def encode_segments(
    segments: Sequence[Segment],
    *,
    velocity: float = 1.0,
    dt: float = 1.0,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> np.ndarray:
    """Encode a route as a deterministic one-hot spike matrix.

    Each segment activates the body-ring neuron that points at North
    (see ``heading_to_bin``) for ``segment_n_steps(distance)`` timesteps.

    Parameters
    ----------
    segments :
        Ordered flight pieces from home.
    velocity :
        Constant speed (distance per time unit).
    dt :
        Simulation timestep.
    heading_bins :
        Number of body-ring input neurons.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(T, heading_bins)``.

    Raises
    ------
    ValueError
        If ``segments`` is empty.
    """
    plan = plan_encoding(
        segments, velocity=velocity, dt=dt, heading_bins=heading_bins
    )
    blocks: list[np.ndarray] = []
    for block in plan:
        mat = np.zeros((block.n_steps, heading_bins), dtype=np.float32)
        mat[:, block.bin_idx] = 1.0
        blocks.append(mat)
    return np.concatenate(blocks, axis=0)


def encode_level(
    level: Level,
    *,
    velocity: float = 1.0,
    dt: float = 1.0,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> np.ndarray:
    """Encode a ``Level`` via its segments.

    Parameters
    ----------
    level :
        Displacement trial from ``paths.generate_level`` / curriculum.
    velocity :
        Constant speed (distance per time unit).
    dt :
        Simulation timestep.
    heading_bins :
        Number of body-ring input neurons.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(T, heading_bins)``.
    """
    return encode_segments(
        level.segments,
        velocity=velocity,
        dt=dt,
        heading_bins=heading_bins,
    )
