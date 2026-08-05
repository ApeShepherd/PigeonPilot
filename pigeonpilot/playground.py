"""
Interactive notebook playground — draw a displacement path, animate the pigeon,
run a frozen SNN checkpoint (A and/or B), animate the predicted home heading.

Uses the same Matplotlib chrome / colors as ``viz.plot_level``. Requires
``%matplotlib widget`` (ipympl) for mouse drawing in Jupyter / Cursor.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Annotation

from .paths import (
    DEFAULT_HEADING_BINS,
    FULL_CIRCLE_DEG,
    Difficulty,
    Level,
    Segment,
    Style,
    home_heading_bin,
    home_heading_deg,
    snap_heading,
    trajectory_points,
)
from .snn import (
    CheckpointBundle,
    bin_to_heading_deg,
    circular_bin_error,
    heading_to_unit,
    predict_home_bin,
)
from .viz import COLORS, SIZES, Z_ORDER, compute_xy_limits

ModelChoice = Literal["A", "B", "both"]
DEMO_LEVEL_ID = 900_001
PRED_COLOR = "magenta"
DOVE = "🕊️"


# ---------------------------------------------------------------------------
# Pure geometry (testable without Matplotlib / BindsNET)
# ---------------------------------------------------------------------------

def polyline_to_segments(
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    heading_bins: int = DEFAULT_HEADING_BINS,
    min_segment_distance: float = 0.05,
) -> tuple[Segment, ...]:
    """Snap a freehand polyline to discrete compass segments (10° by default).

    Consecutive samples with the same snapped heading are merged. Tiny steps
    below ``min_segment_distance`` are skipped so mouse jitter does not create
    zero-length bins.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")
    if len(pts) < 2:
        return ()

    # Force start at home so drawn paths match curriculum geometry.
    if float(np.linalg.norm(pts[0])) > 1e-9:
        pts = np.vstack([np.zeros(2), pts])

    raw: list[Segment] = []
    for i in range(1, len(pts)):
        delta = pts[i] - pts[i - 1]
        dist = float(np.linalg.norm(delta))
        if dist < min_segment_distance:
            continue
        heading = float(
            (np.degrees(np.arctan2(delta[0], delta[1])) + FULL_CIRCLE_DEG) % FULL_CIRCLE_DEG
        )
        raw.append(
            Segment(heading_deg=snap_heading(heading, heading_bins), distance=dist)
        )

    if not raw:
        return ()

    merged: list[Segment] = [raw[0]]
    for seg in raw[1:]:
        prev = merged[-1]
        if seg.heading_deg == prev.heading_deg:
            merged[-1] = Segment(
                heading_deg=prev.heading_deg,
                distance=prev.distance + seg.distance,
            )
        else:
            merged.append(seg)
    return tuple(merged)


def level_from_segments(
    segments: Sequence[Segment],
    *,
    level_id: int = DEMO_LEVEL_ID,
    style: Style = "turning",
    difficulty: Difficulty = "medium",
) -> Level:
    """Build a ``Level`` from snapped segments (home at origin)."""
    if not segments:
        raise ValueError("need at least one segment")
    points = trajectory_points(segments)
    end = points[-1]
    end_xy = (float(end[0]), float(end[1]))
    home_xy = (-end_xy[0], -end_xy[1])
    return Level(
        level_id=level_id,
        style=style,
        segments=tuple(segments),
        end_xy=end_xy,
        home_xy=home_xy,
        difficulty=difficulty,
    )


def resample_polyline(points: np.ndarray, n_frames: int) -> np.ndarray:
    """Evenly resample a polyline by arc length for animation frames."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    if len(pts) == 1 or n_frames <= 1:
        return np.repeat(pts[:1], max(n_frames, 1), axis=0)

    deltas = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(deltas, axis=1)
    total = float(seg_len.sum())
    if total < 1e-12:
        return np.repeat(pts[:1], n_frames, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, total, n_frames)
    out = np.empty((n_frames, 2), dtype=float)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = min(max(j, 0), len(seg_len) - 1)
        span = seg_len[j]
        alpha = 0.0 if span < 1e-12 else (t - cum[j]) / span
        out[i] = pts[j] + alpha * (pts[j + 1] - pts[j])
    return out


def predicted_home_ray(
    level: Level,
    pred_bin: int,
    *,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Release point + displacement along predicted heading with true home length."""
    release = np.asarray(level.end_xy, dtype=float)
    length = float(np.linalg.norm(level.home_xy))
    heading = bin_to_heading_deg(pred_bin, heading_bins)
    delta = heading_to_unit(heading) * length
    return release, delta


# ---------------------------------------------------------------------------
# Drawing chrome (viz-compatible)
# ---------------------------------------------------------------------------

def _draw_base_axes(ax: Axes, half: float, title: str) -> None:
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.grid(True, alpha=0.3, zorder=Z_ORDER["grid"])
    ax.axhline(0, color=COLORS["axis"], lw=0.5, zorder=Z_ORDER["chrome"])
    ax.axvline(0, color=COLORS["axis"], lw=0.5, zorder=Z_ORDER["chrome"])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=SIZES["home"],
        c=COLORS["home"],
        zorder=Z_ORDER["markers"],
        label="home (start)",
    )


def _place_dove(ax: Axes, x: float, y: float) -> Annotation:
    return ax.annotate(
        DOVE,
        xy=(x, y),
        fontsize=16,
        ha="center",
        va="center",
        zorder=Z_ORDER["vectors"] + 1,
    )


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

class PigeonPlayground:
    """Mouse-draw path → outbound animation → live readout → homebound animation."""

    def __init__(
        self,
        bundle: CheckpointBundle,
        *,
        model: ModelChoice = "A",
        field_half: float = 8.0,
        outbound_frames: int = 60,
        home_frames: int = 45,
        min_segment_distance: float = 0.08,
    ) -> None:
        self.bundle = bundle
        self.model = model
        self.field_half = float(field_half)
        self.outbound_frames = outbound_frames
        self.home_frames = home_frames
        self.min_segment_distance = min_segment_distance

        self._draw_pts: list[list[float]] = []
        self._drawing = False
        self._busy = False
        self._draft_line: Optional[Line2D] = None
        self._path_line: Optional[Line2D] = None
        self._dove: Optional[Annotation] = None
        self._status = None
        self._artists: list = []

        self.fig: Figure
        self.ax: Axes
        n_cols = 2 if model == "both" else 1
        self.fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6), squeeze=False)
        self.axes = list(axes[0])
        self.ax = self.axes[0]
        for i, ax in enumerate(self.axes):
            label = "A (fixed)" if (model == "A" or (model == "both" and i == 0)) else "B (STDP)"
            if model != "both":
                label = "A (fixed)" if model == "A" else "B (STDP)"
            _draw_base_axes(ax, self.field_half, f"PigeonPilot playground — {label}")
        self._status = self.fig.text(
            0.5,
            0.02,
            "Draw a path from home (star). Release mouse, then click Fly.",
            ha="center",
            fontsize=10,
        )
        self._draft_line, = self.ax.plot(
            [],
            [],
            "-",
            color="0.5",
            lw=1.5,
            zorder=Z_ORDER["trajectory"],
        )
        self._path_line, = self.ax.plot(
            [],
            [],
            "-",
            color=COLORS["path"],
            lw=SIZES["path_lw"],
            zorder=Z_ORDER["trajectory"],
            label="path",
        )
        self._dove = _place_dove(self.ax, 0.0, 0.0)

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        self._add_buttons()
        self.fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    def _set_status(self, text: str) -> None:
        if self._status is not None:
            self._status.set_text(text)
        self.fig.canvas.draw_idle()

    def _add_buttons(self) -> None:
        try:
            from IPython.display import display
            import ipywidgets as widgets
        except ImportError:
            self._set_status(
                "ipywidgets missing — call playground.clear() / playground.fly() from code."
            )
            return

        fly_btn = widgets.Button(description="Fly 🕊️", button_style="success")
        clear_btn = widgets.Button(description="Clear", button_style="warning")
        fly_btn.on_click(lambda _: self.fly())
        clear_btn.on_click(lambda _: self.clear())
        display(widgets.HBox([fly_btn, clear_btn]))

    def _on_press(self, event: MouseEvent) -> None:
        if self._busy or event.inaxes is not self.ax or event.xdata is None:
            return
        self._drawing = True
        self._draw_pts = [[0.0, 0.0], [float(event.xdata), float(event.ydata)]]
        self._update_draft()

    def _on_motion(self, event: MouseEvent) -> None:
        if not self._drawing or event.inaxes is not self.ax or event.xdata is None:
            return
        self._draw_pts.append([float(event.xdata), float(event.ydata)])
        self._update_draft()

    def _on_release(self, event: MouseEvent) -> None:
        if not self._drawing:
            return
        self._drawing = False
        if event.inaxes is self.ax and event.xdata is not None:
            self._draw_pts.append([float(event.xdata), float(event.ydata)])
        self._update_draft()
        self._set_status("Path captured — click Fly to displace & predict home.")

    def _update_draft(self) -> None:
        if not self._draw_pts:
            self._draft_line.set_data([], [])
        else:
            arr = np.asarray(self._draw_pts)
            self._draft_line.set_data(arr[:, 0], arr[:, 1])
        self.fig.canvas.draw_idle()

    def clear(self) -> None:
        self._busy = False
        self._drawing = False
        self._draw_pts = []
        self._draft_line.set_data([], [])
        self._path_line.set_data([], [])
        for artist in self._artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._artists.clear()
        if self._dove is not None:
            self._dove.set_position((0.0, 0.0))
        for ax in self.axes:
            ax.set_xlim(-self.field_half, self.field_half)
            ax.set_ylim(-self.field_half, self.field_half)
        self._set_status("Cleared — draw a new path from home.")

    def fly(self) -> None:
        if self._busy:
            return
        segments = polyline_to_segments(
            self._draw_pts,
            min_segment_distance=self.min_segment_distance,
        )
        if not segments:
            self._set_status("Need a longer path — draw farther from home.")
            return

        level = level_from_segments(segments)
        points = trajectory_points(level.segments)
        self._path_line.set_data(points[:, 0], points[:, 1])
        self._draft_line.set_data([], [])

        # Autoscale field if the path leaves the default window.
        (xlim, ylim) = compute_xy_limits([level], padding=0.08)
        half = max(abs(xlim[0]), abs(xlim[1]), self.field_half)
        for ax in self.axes:
            ax.set_xlim(-half, half)
            ax.set_ylim(-half, half)

        self._busy = True
        self._set_status("Displacing pigeon…")
        self._animate_along(points, self.outbound_frames, phase="outbound")

        true_bin = home_heading_bin(level)
        results: list[tuple[str, int, float]] = []

        models: list[tuple[str, Axes, object, object]]
        if self.model == "both":
            models = [
                ("A", self.axes[0], self.bundle.network_a, self.bundle.classifier_a),
                ("B", self.axes[1], self.bundle.network_b, self.bundle.classifier_b),
            ]
            # Sync path onto panel B
            self.axes[1].plot(
                points[:, 0],
                points[:, 1],
                "-",
                color=COLORS["path"],
                lw=SIZES["path_lw"],
                zorder=Z_ORDER["trajectory"],
            )
            self.axes[1].scatter(
                [level.end_xy[0]],
                [level.end_xy[1]],
                marker="o",
                s=SIZES["release"],
                c=COLORS["release"],
                zorder=Z_ORDER["markers"],
            )
        elif self.model == "A":
            models = [("A", self.ax, self.bundle.network_a, self.bundle.classifier_a)]
        else:
            models = [("B", self.ax, self.bundle.network_b, self.bundle.classifier_b)]

        # Release marker on main axis
        rel = self.ax.scatter(
            [level.end_xy[0]],
            [level.end_xy[1]],
            marker="o",
            s=SIZES["release"],
            c=COLORS["release"],
            zorder=Z_ORDER["markers"],
            label="release (end)",
        )
        self._artists.append(rel)

        for name, ax, net, clf in models:
            self._set_status(f"Inferring model {name} (reservoir may take a few seconds)…")
            self.fig.canvas.draw_idle()
            pred_bin = predict_home_bin(net, clf, level, self.bundle.config)
            _, err_deg = circular_bin_error(true_bin, pred_bin)
            results.append((name, pred_bin, err_deg))
            self._draw_home_overlay(ax, level, pred_bin)
            if name == models[0][0] or self.model != "both":
                release, delta = predicted_home_ray(level, pred_bin)
                home_pts = np.vstack([release, release + delta])
                self._animate_along(home_pts, self.home_frames, phase="home", ax=ax)

        true_h = home_heading_deg(level.home_xy)
        bits = [
            f"{name}: pred {bin_to_heading_deg(pb):.0f}° (err {err:.0f}°)"
            for name, pb, err in results
        ]
        self._set_status(
            f"True home {true_h:.0f}° (bin {true_bin})  |  " + "  ·  ".join(bits)
        )
        self._busy = False

    def _draw_home_overlay(self, ax: Axes, level: Level, pred_bin: int) -> None:
        end = np.asarray(level.end_xy, dtype=float)
        true_delta = np.asarray(level.home_xy, dtype=float)
        _, pred_delta = predicted_home_ray(level, pred_bin)
        width = SIZES["quiver_width"]
        q_true = ax.quiver(
            end[0],
            end[1],
            true_delta[0],
            true_delta[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=COLORS["home_vector"],
            width=width,
            label="true home",
            zorder=Z_ORDER["vectors"],
        )
        q_pred = ax.quiver(
            end[0],
            end[1],
            pred_delta[0],
            pred_delta[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=PRED_COLOR,
            width=width,
            label="predicted home",
            zorder=Z_ORDER["vectors"],
        )
        self._artists.extend([q_true, q_pred])
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)

    def _animate_along(
        self,
        points: np.ndarray,
        n_frames: int,
        *,
        phase: str,
        ax: Optional[Axes] = None,
    ) -> None:
        ax = ax or self.ax
        frames = resample_polyline(points, n_frames)
        dove = self._dove if ax is self.ax else _place_dove(ax, frames[0, 0], frames[0, 1])
        if ax is not self.ax:
            self._artists.append(dove)

        trail_x: list[float] = []
        trail_y: list[float] = []
        trail_color = COLORS["path"] if phase == "outbound" else PRED_COLOR
        (trail,) = ax.plot(
            [],
            [],
            "-",
            color=trail_color,
            lw=1.2,
            alpha=0.5,
            zorder=Z_ORDER["trajectory"],
        )
        self._artists.append(trail)

        for xy in frames:
            x, y = float(xy[0]), float(xy[1])
            trail_x.append(x)
            trail_y.append(y)
            trail.set_data(trail_x, trail_y)
            if isinstance(dove, Annotation):
                dove.xy = (x, y)
                dove.set_position((x, y))
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.02)


def launch_playground(
    bundle: CheckpointBundle,
    *,
    model: ModelChoice = "A",
    field_half: float = 8.0,
) -> PigeonPlayground:
    """Create and return an interactive playground bound to a loaded checkpoint."""
    return PigeonPlayground(bundle, model=model, field_half=field_half)
