"""
Plot helpers for PigeonPilot.

`paths.py` computes geometry; `viz.py` draws it.
Grids always zoom to the levels they receive: square axes ±max(|x|,|y|) around home.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .paths import (
    DIFFICULTIES,
    Level,
    STYLE_LABELS,
    STYLES,
    Segment,
    displacement_vector,
    home_vector,
    trajectory_points,
)


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


def _apply_limits(ax: Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)


def plot_level(
    level: Level | Sequence[Segment],
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    show_displacement: bool = True,
    show_home_vector: bool = True,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    show_legend: bool = True,
    path_lw: float = 2.5,
    home_size: float = 220,
    release_size: float = 80,
    turn_size: float = 35,
    padding: float = 0.03,
) -> Axes:
    """Plot one level in x/y."""
    if isinstance(level, Level):
        segments = level.segments
        family = STYLE_LABELS.get(level.style, level.style)
        default_title = f"Level {level.level_id} · {level.difficulty} · {family}"
    else:
        segments = level
        default_title = "Displacement path"

    points = trajectory_points(segments)
    end = points[-1]
    disp = displacement_vector(segments)
    home = home_vector(segments)

    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(points[:, 0], points[:, 1], "-", color="black", lw=path_lw, label="path")

    turn_xy = [
        points[i]
        for i in range(1, len(segments))
        if segments[i].heading_deg != segments[i - 1].heading_deg
    ]
    if turn_xy:
        turns = np.asarray(turn_xy)
        ax.scatter(
            turns[:, 0], turns[:, 1], s=turn_size, c="dimgray", zorder=4, label="turn"
        )

    ax.scatter(
        [0.0], [0.0], marker="*", s=home_size, c="green", zorder=5, label="home (start)"
    )
    ax.scatter(
        [end[0]],
        [end[1]],
        marker="o",
        s=release_size,
        c="red",
        zorder=5,
        label="release (end)",
    )

    if show_displacement:
        ax.quiver(
            0.0,
            0.0,
            disp[0],
            disp[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="darkorange",
            width=0.008,
            label="displacement vector",
        )
    if show_home_vector:
        ax.quiver(
            end[0],
            end[1],
            home[0],
            home[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="royalblue",
            width=0.008,
            label="home vector",
        )

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or default_title)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc="best", fontsize=8)

    if xlim is None or ylim is None:
        auto_x, auto_y = compute_xy_limits([level], padding=padding)
        xlim = xlim or auto_x
        ylim = ylim or auto_y
    _apply_limits(ax, xlim, ylim)

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
    row_chunks: list[tuple[str, list[Level]]] = []
    for style in present_styles:
        examples = by_style[style]
        if max_per_style is not None:
            examples = examples[:max_per_style]
        family = STYLE_LABELS.get(style, style)
        for start in range(0, len(examples), max_cols):
            row_chunks.append((family, examples[start : start + max_cols]))

    n_rows = len(row_chunks)
    if figsize is None:
        figsize = (panel_inches * max_cols, panel_inches * n_rows + 0.6)

    fig, axes = plt.subplots(n_rows, max_cols, figsize=figsize, squeeze=False)
    for row, (family, chunk) in enumerate(row_chunks):
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
                show_displacement=False,
                show_home_vector=False,
                path_lw=3.5,
                home_size=180,
                release_size=90,
                turn_size=50,
                padding=padding,
            )
            ax.set_title(f"#{lv.level_id}", fontsize=11)
            if col > 0:
                ax.set_ylabel("")
        axes[row][0].set_ylabel(
            f"family:\n{family}",
            fontsize=11,
            fontweight="bold",
            rotation=0,
            ha="right",
            va="center",
            labelpad=28,
        )

    base = title or "Displacement levels"
    fig.suptitle(f"{base}  |  axes ±{half:.2f}", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.06, 0.0, 1.0, 0.93])
    return fig


def plot_release_points(
    levels: Sequence[Level],
    ax: Optional[Axes] = None,
    title: str = "Release points (where the pigeon ends up)",
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
) -> Axes:
    """Release-point map; optional shared limits (defaults to square ±half)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    colors = {"easy": "tab:green", "medium": "tab:orange", "hard": "tab:red"}
    for diff in DIFFICULTIES:
        pts = np.array([lv.end_xy for lv in levels if lv.difficulty == diff], dtype=float)
        if len(pts) == 0:
            continue
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=colors[diff],
            s=40,
            alpha=0.75,
            label=f"{diff} (n={len(pts)})",
            edgecolors="black",
            linewidths=0.3,
        )

    ax.scatter([0.0], [0.0], marker="*", s=280, c="green", zorder=5, label="home")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    if xlim is None or ylim is None:
        xlim, ylim = compute_xy_limits(levels)
    _apply_limits(ax, xlim, ylim)
    return ax
