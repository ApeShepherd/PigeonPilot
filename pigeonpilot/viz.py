"""
Plot helpers for PigeonPilot — layered Matplotlib rendering.

``paths`` computes geometry; ``viz`` draws it in strict z-order layers.
``encoding`` supplies spike matrices / plans for raster and ring figures.
Grids always zoom to the levels they receive: square axes ±max(|x|,|y|)
around home.

One-way deps: ``viz`` → ``paths`` / ``encoding`` / ``curriculum``;
never the reverse.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .curriculum import STYLE_LABELS
from .encoding import SpikeBlock, encode_level, heading_to_bin, plan_encoding
from .paths import (
    DEFAULT_HEADING_BINS,
    DIFFICULTIES,
    FULL_CIRCLE_DEG,
    STYLES,
    Level,
    Segment,
    Style,
    trajectory_points,
)

# ---------------------------------------------------------------------------
# Global styling & z-order (never collide layers)
# ---------------------------------------------------------------------------

Z_ORDER: dict[str, int] = {
    "grid": 1,
    "chrome": 1,  # axis crosshairs under path/markers (must not cover home star)
    "trajectory": 2,
    "markers": 3,
    "vectors": 4,
}

COLORS: dict[str, str] = {
    "path": "black",
    "turn": "dimgray",
    "home": "green",
    "release": "red",
    "displacement": "darkorange",
    "home_vector": "royalblue",
    "axis": "gray",
}

DIFFICULTY_COLORS: dict[str, str] = {
    "easy": "tab:green",
    "medium": "tab:orange",
    "hard": "tab:red",
}

SIZES: dict[str, float] = {
    "path_lw": 2.5,
    "path_lw_grid": 3.5,
    "home": 220.0,
    "home_grid": 180.0,
    "home_release_map": 280.0,
    "release": 80.0,
    "release_grid": 90.0,
    "turn": 35.0,
    "turn_grid": 50.0,
    "release_point": 40.0,
    "quiver_width": 0.008,
}


def compute_xy_limits(
    levels: Sequence[Level | Sequence[Segment]],
    padding: float = 0.03,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Square window centered on home: both axes use ``(-half, +half)``.

    Parameters
    ----------
    levels :
        One or more levels / segment sequences whose trajectories define the zoom.
    padding :
        Relative margin beyond ``max(|x|, |y|)``.

    Returns
    -------
    xlim, ylim : tuple of float
        Identical square limits.
    """
    pts = [np.zeros(2)]
    for item in levels:
        segments = item.segments if isinstance(item, Level) else item
        pts.append(trajectory_points(segments))
    half = float(np.max(np.abs(np.vstack(pts))))
    half = max(half, 1e-6) * (1.0 + padding)
    return (-half, half), (-half, half)


def _as_segments(level: Level | Sequence[Segment]) -> tuple[Segment, ...]:
    if isinstance(level, Level):
        return level.segments
    return tuple(level)


def _default_title(level: Level | Sequence[Segment]) -> str:
    if isinstance(level, Level):
        family = STYLE_LABELS.get(level.style, level.style)
        return f"Level {level.level_id} · {level.difficulty} · {family}"
    return "Displacement path"


def _finalize_legend(ax: Axes) -> None:
    """Call legend exactly once with de-duplicated labels."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)


# ---------------------------------------------------------------------------
# Rendering layers (private) — every artist sets an explicit zorder
# ---------------------------------------------------------------------------

def _draw_chrome(
    ax: Axes,
    limits: tuple[tuple[float, float], tuple[float, float]],
    title: str,
) -> None:
    """Grid, axis chrome, locked equal aspect, and fixed limits."""
    xlim, ylim = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)

    ax.grid(True, alpha=0.3, zorder=Z_ORDER["grid"])
    ax.axhline(0, color=COLORS["axis"], lw=0.5, zorder=Z_ORDER["chrome"])
    ax.axvline(0, color=COLORS["axis"], lw=0.5, zorder=Z_ORDER["chrome"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)


def _draw_trajectory(
    ax: Axes,
    points: np.ndarray,
    *,
    path_lw: float = SIZES["path_lw"],
) -> None:
    """Path polyline only."""
    ax.plot(
        points[:, 0],
        points[:, 1],
        "-",
        color=COLORS["path"],
        lw=path_lw,
        label="path",
        zorder=Z_ORDER["trajectory"],
    )


def _draw_keypoints(
    ax: Axes,
    segments: Sequence[Segment],
    points: np.ndarray,
    *,
    home_size: float = SIZES["home"],
    release_size: float = SIZES["release"],
    turn_size: float = SIZES["turn"],
) -> None:
    """Turn vertices plus home / release markers."""
    end = points[-1]
    z = Z_ORDER["markers"]

    turn_xy = [
        points[i]
        for i in range(1, len(segments))
        if segments[i].heading_deg != segments[i - 1].heading_deg
    ]
    if turn_xy:
        turns = np.asarray(turn_xy)
        ax.scatter(
            turns[:, 0],
            turns[:, 1],
            s=turn_size,
            c=COLORS["turn"],
            zorder=z,
            label="turn",
        )

    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=home_size,
        c=COLORS["home"],
        zorder=z,
        label="home (start)",
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        marker="o",
        s=release_size,
        c=COLORS["release"],
        zorder=z,
        label="release (end)",
    )


def _draw_vectors(
    ax: Axes,
    points: np.ndarray,
    *,
    show_displacement: bool = True,
    show_home_vector: bool = True,
) -> None:
    """Displacement and/or home-vector quivers (from precomputed points)."""
    end = points[-1]
    z = Z_ORDER["vectors"]
    width = SIZES["quiver_width"]

    if show_displacement:
        ax.quiver(
            0.0,
            0.0,
            end[0],
            end[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=COLORS["displacement"],
            width=width,
            label="displacement vector",
            zorder=z,
        )
    if show_home_vector:
        ax.quiver(
            end[0],
            end[1],
            -end[0],
            -end[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=COLORS["home_vector"],
            width=width,
            label="home vector",
            zorder=z,
        )


# ---------------------------------------------------------------------------
# Public compositors
# ---------------------------------------------------------------------------

def plot_level(
    level: Level | Sequence[Segment],
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    show_markers: bool = True,
    show_vectors: bool = True,
    show_displacement: bool = True,
    show_home_vector: bool = True,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    show_legend: bool = True,
    path_lw: float = SIZES["path_lw"],
    home_size: float = SIZES["home"],
    release_size: float = SIZES["release"],
    turn_size: float = SIZES["turn"],
    padding: float = 0.03,
) -> Axes:
    """Composite one level: trajectory → markers → vectors → chrome → legend.

    Parameters
    ----------
    level :
        A ``Level`` or raw segment sequence.
    ax :
        Target axes; created if omitted.
    title :
        Override for the axes title (default from level metadata).
    show_markers :
        Draw home / release / turn markers.
    show_vectors :
        Master switch for displacement / home-vector quivers.
    show_displacement, show_home_vector :
        Refine which vectors appear when ``show_vectors`` is True.
    xlim, ylim :
        Fixed axis limits; auto square zoom if omitted.
    show_legend :
        Draw a de-duplicated legend.
    path_lw, home_size, release_size, turn_size :
        Artist sizes (defaults from ``SIZES``).
    padding :
        Relative margin when auto-computing limits.

    Returns
    -------
    Axes
        The axes that received the artists.
    """
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if xlim is None or ylim is None:
        auto_x, auto_y = compute_xy_limits([level], padding=padding)
        xlim = xlim or auto_x
        ylim = ylim or auto_y
    limits = (xlim, ylim)

    segments = _as_segments(level)
    points = trajectory_points(segments)

    _draw_trajectory(ax, points, path_lw=path_lw)
    if show_markers:
        _draw_keypoints(
            ax,
            segments,
            points,
            home_size=home_size,
            release_size=release_size,
            turn_size=turn_size,
        )
    if show_vectors:
        _draw_vectors(
            ax,
            points,
            show_displacement=show_displacement,
            show_home_vector=show_home_vector,
        )
    _draw_chrome(ax, limits, title or _default_title(level))

    if show_legend:
        _finalize_legend(ax)

    if created_fig:
        plt.tight_layout()
    return ax


def plot_levels_grid(
    levels: Sequence[Level],
    max_per_style: Optional[int] = None,
    max_cols: int = 5,
    panel_inches: float = 3.4,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    padding: float = 0.03,
) -> Figure:
    """Grid grouped by trajectory family (rows).

    Zoom is always derived from ``levels`` (square ±half around home).

    Parameters
    ----------
    levels :
        Levels to show (order within each style preserved).
    max_per_style :
        Optional cap per trajectory family.
    max_cols :
        Panels per row (must be ``>= 1``).
    panel_inches :
        Approximate width/height of one panel when ``figsize`` is omitted.
    figsize :
        Optional ``(width, height)`` in inches.
    title :
        Figure suptitle prefix.
    padding :
        Relative margin for shared limits.

    Returns
    -------
    Figure
        Matplotlib figure with the grid.

    Raises
    ------
    ValueError
        If ``max_cols < 1``.
    """
    if max_cols < 1:
        raise ValueError("max_cols must be >= 1")

    present_styles = [s for s in STYLES if any(lv.style == s for lv in levels)]
    if not present_styles:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title("No levels to plot")
        ax.axis("off")
        return fig

    xlim, ylim = compute_xy_limits(levels, padding=padding)
    half = float(xlim[1])

    by_style = {style: [lv for lv in levels if lv.style == style] for style in present_styles}
    row_chunks: list[tuple[Style, list[Level]]] = []
    for style in present_styles:
        examples = by_style[style]
        if max_per_style is not None:
            examples = examples[:max_per_style]
        for start in range(0, len(examples), max_cols):
            row_chunks.append((style, examples[start : start + max_cols]))

    n_rows = len(row_chunks)
    if figsize is None:
        # Small top band for the difficulty title — avoid a large empty gap.
        figsize = (panel_inches * max_cols, panel_inches * n_rows + 0.35)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=figsize, squeeze=False)
    for row, (style, chunk) in enumerate(row_chunks):
        family = STYLE_LABELS.get(style, style)
        for col in range(max_cols):
            ax = axes[row][col]
            if col >= len(chunk):
                ax.axis("off")
                continue
            lv = chunk[col]
            plot_level(
                lv,
                ax=ax,
                xlim=xlim,
                ylim=ylim,
                show_legend=False,
                show_markers=True,
                show_vectors=False,
                path_lw=SIZES["path_lw_grid"],
                home_size=SIZES["home_grid"],
                release_size=SIZES["release_grid"],
                turn_size=SIZES["turn_grid"],
                padding=padding,
            )
            ax.set_title(f"#{lv.level_id}", fontsize=11)
            if col > 0:
                ax.set_ylabel("")
        # Side label: same wording as §1.2 STYLE_LABELS, rotated to save width.
        axes[row][0].set_ylabel(
            f"{style}: {family}",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
            labelpad=10,
        )

    base = title or "Displacement levels"
    fig.suptitle(f"{base}  |  axes ±{half:.2f}", fontsize=14, fontweight="bold", y=0.995)
    # Leave only a slim top strip for the title; rotated labels need little left margin.
    fig.tight_layout(rect=[0.03, 0.0, 1.0, 0.97])
    return fig


def plot_release_points(
    levels: Sequence[Level],
    ax: Optional[Axes] = None,
    title: str = "Release points (where the pigeon ends up)",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> Axes:
    """Release-point scatter colored by difficulty.

    Parameters
    ----------
    levels :
        Levels whose ``end_xy`` points are plotted.
    ax :
        Target axes; created if omitted.
    title :
        Axes title.
    xlim, ylim :
        Fixed limits; auto square zoom if omitted.

    Returns
    -------
    Axes
        The axes that received the artists.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if xlim is None or ylim is None:
        auto_x, auto_y = compute_xy_limits(levels)
        xlim = xlim or auto_x
        ylim = ylim or auto_y
    limits = (xlim, ylim)

    z_markers = Z_ORDER["markers"]

    for diff in DIFFICULTIES:
        pts = np.array([lv.end_xy for lv in levels if lv.difficulty == diff], dtype=float)
        if len(pts) == 0:
            continue
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=DIFFICULTY_COLORS[diff],
            s=SIZES["release_point"],
            alpha=0.75,
            label=f"{diff} (n={len(pts)})",
            edgecolors="black",
            linewidths=0.3,
            zorder=z_markers,
        )

    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=SIZES["home_release_map"],
        c=COLORS["home"],
        zorder=z_markers,
        label="home",
    )

    _draw_chrome(ax, limits, title)
    _finalize_legend(ax)
    return ax


# ---------------------------------------------------------------------------
# Encoding views (path ↔ spike raster)
# ---------------------------------------------------------------------------

def _segment_colors(n: int) -> list[tuple]:
    """Distinct colors for path segments / matching raster ticks."""
    if n <= 0:
        return []
    cmap = colormaps["tab10" if n <= 10 else "tab20"]
    return [cmap(i % cmap.N) for i in range(n)]


def _cardinal_bin_indices(heading_bins: int = DEFAULT_HEADING_BINS) -> tuple[int, ...]:
    """Quarter-circle body-bin indices (beak / right / tail / left when facing North)."""
    q = heading_bins // 4
    return (0, q, 2 * q, 3 * q)


def plot_spike_raster(
    spikes: np.ndarray,
    ax: Optional[Axes] = None,
    *,
    plan: Optional[Sequence[SpikeBlock]] = None,
    title: str = "Spike raster (North-pointing body bin)",
    t_max: Optional[float] = None,
) -> Axes:
    """
    Black spike ticks (tutorial-style), optional colored segment bands.

    Parameters
    ----------
    spikes :
        Shape ``(T, n_bins)``.
    ax :
        Target axes; created if omitted.
    plan :
        Optional ``SpikeBlock`` sequence from ``plan_encoding`` — draws
        vertical segment boundaries and colors ticks by segment.
    title :
        Axes title.
    t_max :
        Shared right edge for the time axis (exclusive-style length).
        Use the longest trial's ``T`` so short routes stay visually
        comparable. Default: ``spikes.shape[0]``.

    Returns
    -------
    Axes
        The axes that received the raster.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.2))

    times, neurons = np.where(spikes > 0)
    if plan is not None and len(plan) > 0:
        colors = _segment_colors(len(plan))
        for block, color in zip(plan, colors):
            mask = (times >= block.start) & (times < block.end)
            ax.scatter(
                times[mask],
                neurons[mask],
                s=28,
                c=[color],
                marker="|",
                linewidths=1.6,
                zorder=3,
            )
            if block.start > 0:
                ax.axvline(block.start - 0.5, color="0.65", ls=":", lw=0.9, zorder=1)
    else:
        ax.scatter(times, neurons, s=22, c="black", marker="|", linewidths=1.4, zorder=3)

    n_bins = spikes.shape[1]
    right = float(t_max) if t_max is not None else float(spikes.shape[0])
    ax.set_xlim(-0.5, max(right - 0.5, 0.5))
    ax.set_ylim(-1.0, n_bins)
    ax.set_xlabel("time step")
    ax.set_ylabel("body bin (0 = beak)")
    ax.set_title(title)
    if n_bins == DEFAULT_HEADING_BINS:
        ax.set_yticks([*_cardinal_bin_indices(n_bins), n_bins - 1])
    else:
        ax.set_yticks([0, n_bins - 1])
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    return ax


def plot_level_encoding(
    level: Level,
    *,
    velocity: float = 1.0,
    dt: float = 0.25,
    figsize: tuple[float, float] = (12.5, 4.6),
    title: Optional[str] = None,
    t_max: Optional[float] = None,
) -> Figure:
    """Side-by-side: displacement path (segments colored) | spike raster.

    Parameters
    ----------
    level :
        Displacement trial to encode and draw.
    velocity :
        Constant speed forwarded to ``plan_encoding`` / ``encode_level``.
    dt :
        Simulation timestep. Defaults to ``0.25`` so duration bars are
        readable in demos; encoding unit tests use ``dt=1.0``.
    figsize :
        Figure size in inches.
    title :
        Optional override for the path panel title.
    t_max :
        Shared raster x-axis length (pass the longer trial's ``T`` when
        comparing levels).

    Returns
    -------
    Figure
        Two-panel figure (path | raster).
    """
    segments = level.segments
    plan = plan_encoding(segments, velocity=velocity, dt=dt)
    spikes = encode_level(level, velocity=velocity, dt=dt)
    colors = _segment_colors(len(segments))
    points = trajectory_points(segments)

    fig, (ax_path, ax_raster) = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.35]},
    )

    for i, color in enumerate(colors):
        p0, p1 = points[i], points[i + 1]
        ax_path.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            lw=SIZES["path_lw"],
            solid_capstyle="round",
            zorder=Z_ORDER["trajectory"],
        )
    ax_path.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=SIZES["home"],
        c=COLORS["home"],
        zorder=Z_ORDER["markers"],
        label="home",
    )
    ax_path.scatter(
        [points[-1, 0]],
        [points[-1, 1]],
        s=SIZES["release"],
        c=COLORS["release"],
        zorder=Z_ORDER["markers"],
        label="release",
    )
    limits = compute_xy_limits([level])
    head = title or _default_title(level)
    _draw_chrome(ax_path, limits, f"{head}\npath (color = segment)")
    _finalize_legend(ax_path)

    plot_spike_raster(
        spikes,
        ax=ax_raster,
        plan=plan,
        t_max=t_max,
        title=f"rate code  |  dt={dt:g}, v={velocity:g}  |  shape {tuple(spikes.shape)}",
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Body-ring / compass teaching figures
# ---------------------------------------------------------------------------

def _as_compass_polar(ax: Axes) -> None:
    """North up, angles clockwise (navigation convention)."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # Outer spine beyond bin-index labels (r≈0.93) so 0/9/18/27 are not clipped;
    # tick ``pad`` still keeps N/E/S/W close outside (avoids \"EW\" between panels).
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"])
    ax.tick_params(axis="x", labelsize=8, pad=1)


def _bin_world_deg(
    heading_deg: float,
    bin_idx: int,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> float:
    """World compass angle of a body-fixed bin when the beak faces ``heading_deg``."""
    step = FULL_CIRCLE_DEG / heading_bins
    return (heading_deg + bin_idx * step) % FULL_CIRCLE_DEG


def _draw_body_ring_world(
    ax: Axes,
    heading_deg: float,
    *,
    active_bin: Optional[int] = None,
    highlight_color: str = "crimson",
    heading_bins: int = DEFAULT_HEADING_BINS,
    show_beak: bool = True,
    dim: float = 0.35,
) -> None:
    """World-frame ring: geographic N fixed at top; body bins rotate with heading.

    The North-pointing neuron sits at world angle 0° and is highlighted when
    ``active_bin`` is set.
    """
    _as_compass_polar(ax)
    radii = np.full(heading_bins, 0.82)
    thetas = np.deg2rad(
        [_bin_world_deg(heading_deg, i, heading_bins) for i in range(heading_bins)]
    )

    # faint ring + all bins
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 0.82), color="0.85", lw=1.0, zorder=1)
    ax.scatter(thetas, radii, s=18, c="0.55", alpha=dim, zorder=2)

    # Cardinal body-bin dots slightly darker so labels map clearly
    cardinal_bins = _cardinal_bin_indices(heading_bins)
    card_theta = np.deg2rad(
        [_bin_world_deg(heading_deg, b, heading_bins) for b in cardinal_bins]
    )
    ax.scatter(
        card_theta,
        np.full(len(cardinal_bins), 0.82),
        s=28,
        c="0.28",
        alpha=min(1.0, dim + 0.45),
        zorder=3,
        edgecolors="0.15",
        linewidths=0.4,
    )

    for b in cardinal_bins:
        ang = np.deg2rad(_bin_world_deg(heading_deg, b, heading_bins))
        ax.text(
            ang,
            1.05,
            str(b),
            ha="center",
            va="center",
            fontsize=7,
            color="0.35",
            zorder=4,
        )

    if active_bin is not None:
        ang = np.deg2rad(_bin_world_deg(heading_deg, active_bin, heading_bins))
        ax.scatter([ang], [0.82], s=110, c=[highlight_color], zorder=5, edgecolors="k", lw=0.6)
        ax.annotate(
            f"bin {active_bin}",
            xy=(ang, 0.82),
            xytext=(ang, 0.45),
            textcoords="data",
            fontsize=8,
            ha="center",
            color=highlight_color,
            arrowprops=dict(arrowstyle="-", color=highlight_color, lw=0.8),
            zorder=6,
        )
        # North ray (stops short of the outer spine)
        ax.plot([0, 0], [0.15, 1.12], color="0.25", ls="--", lw=0.8, zorder=1)

    if show_beak:
        beak = np.deg2rad(heading_deg % FULL_CIRCLE_DEG)
        ax.annotate(
            "",
            xy=(beak, 0.72),
            xytext=(beak, 0.12),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.6),
            zorder=4,
        )
        ax.text(beak, 0.05, "beak", ha="center", va="top", fontsize=7)


def plot_body_ring_anatomy(figsize: tuple[float, float] = (11.5, 3.8)) -> Figure:
    """Three-panel legend: world compass · body bins · worked example (face West).

    Didactic order: where is North → how bins sit on the bird → which bin fires.

    Parameters
    ----------
    figsize :
        Figure size in inches.

    Returns
    -------
    Figure
        Three polar panels.
    """
    fig, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        subplot_kw={"projection": "polar"},
        gridspec_kw={"wspace": 0.55},
    )

    # (1) Geographic compass only
    ax0 = axes[0]
    _as_compass_polar(ax0)
    ax0.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, 0.75), color="0.8", lw=1.0)
    for ang, lab in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        ax0.annotate(
            "",
            xy=(np.deg2rad(ang), 0.75),
            xytext=(np.deg2rad(ang), 0.15),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        )
    ax0.set_title("1 · Geographic North is fixed", fontsize=10, pad=12)

    # (2) Body ring facing North: bin 0 at N, all 36 bins visible
    ax1 = axes[1]
    _draw_body_ring_world(
        ax1,
        heading_deg=0.0,
        active_bin=0,
        highlight_color=COLORS["home"],
        dim=0.55,
    )
    ax1.set_title("2 · Face North → bin 0 at N\n(beak = bin 0)", fontsize=10, pad=12)

    # (3) Face West: body rotated 90° left; bin 9 sits at North
    ax2 = axes[2]
    west = 270.0
    _draw_body_ring_world(
        ax2,
        heading_deg=west,
        active_bin=heading_to_bin(west),
        highlight_color="crimson",
        dim=0.4,
    )
    ax2.set_title("3 · Face West (90° left)\n→ bin 9 at North fires", fontsize=10, pad=12)

    step = FULL_CIRCLE_DEG / DEFAULT_HEADING_BINS
    fig.suptitle(
        f"Body ring ({DEFAULT_HEADING_BINS} × {step:g}°) · "
        "spikes = neuron currently pointing at geographic North",
        fontsize=11,
        y=1.05,
    )
    return fig


def plot_level_ring_frames(
    level: Level,
    *,
    velocity: float = 1.0,
    dt: float = 0.25,
    panel_inches: float = 2.35,
    canvas_slots: int = 7,
) -> Figure:
    """One polar frame per segment: fixed North, rotating body, active bin lit.

    Segment colors match ``plot_level_encoding`` / the spike raster.
    Short routes are centered on a fixed-width canvas (padding made even
    so short demos are not left-biased).

    Parameters
    ----------
    level :
        Displacement trial to animate frame-by-frame.
    velocity :
        Constant speed forwarded to ``plan_encoding``.
    dt :
        Simulation timestep (demo default ``0.25``; encoding tests use ``1.0``).
    panel_inches :
        Approximate width of one polar panel.
    canvas_slots :
        Minimum number of column slots (short levels are centered).

    Returns
    -------
    Figure
        One polar axes per segment (plus empty margin slots).
    """
    from matplotlib.gridspec import GridSpec

    plan = plan_encoding(level.segments, velocity=velocity, dt=dt)
    colors = _segment_colors(len(plan))
    n = len(plan)
    slots = max(canvas_slots, n)
    # Even leftover columns → equal left/right margin (fixes slight left bias).
    if (slots - n) % 2 == 1:
        slots += 1
    start = (slots - n) // 2

    fig = plt.figure(figsize=(panel_inches * slots, panel_inches + 0.55))
    gs = GridSpec(1, slots, figure=fig, wspace=0.55)

    for i, (block, color) in enumerate(zip(plan, colors)):
        ax = fig.add_subplot(gs[0, start + i], projection="polar")
        _draw_body_ring_world(
            ax,
            heading_deg=block.heading_deg,
            active_bin=block.bin_idx,
            highlight_color=color,
            dim=0.35,
        )
        ax.set_title(
            f"seg {i}\nH={block.heading_deg:.0f}° → bin {block.bin_idx}",
            fontsize=9,
            pad=10,
            color=color,
        )

    family = STYLE_LABELS.get(level.style, level.style)
    fig.suptitle(
        f"Level #{level.level_id} · {level.difficulty} · {family}  "
        f"| North fixed · body turns · highlighted bin fires",
        fontsize=11,
        y=1.02,
    )
    return fig
