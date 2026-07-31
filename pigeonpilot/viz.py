"""
Plot helpers for PigeonPilot — layered Matplotlib rendering.

`paths` computes geometry; `viz` draws it in strict z-order layers.
Grids always zoom to the levels they receive: square axes ±max(|x|,|y|) around home.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .curriculum import STYLE_LABELS
from .paths import (
    DIFFICULTIES,
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
    """Square window centered on home: both axes use (-half, +half)."""
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
    """
    Composite one level: trajectory → markers → vectors → chrome → legend.

    ``show_markers`` / ``show_vectors`` are the primary toggles.
    ``show_displacement`` / ``show_home_vector`` refine vectors when enabled.
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
    """
    Grid grouped by trajectory family (rows).

    Zoom is always derived from `levels` (square ±half around home).
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
    """Release-point map; limits from data first, then chrome."""
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
