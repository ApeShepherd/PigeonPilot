"""Unit tests for plotting helpers (zoom / square limits / layer toggles)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver

from pigeonpilot.curriculum import generate_curriculum_dataset
from pigeonpilot.paths import generate_level
from pigeonpilot.viz import compute_xy_limits, plot_level, plot_levels_grid


def test_shared_limits_are_square():
    data = generate_curriculum_dataset(seed=2)
    xlim, ylim = compute_xy_limits(data)
    assert xlim == ylim
    assert abs(xlim[0] + xlim[1]) < 1e-9


def test_grid_zooms_to_given_levels():
    data = generate_curriculum_dataset(seed=42)
    easy = [lv for lv in data if lv.difficulty == "easy"]
    full_half = compute_xy_limits(data)[0][1]
    easy_half = compute_xy_limits(easy)[0][1]
    assert easy_half < full_half

    fig = plot_levels_grid(easy, title="easy")
    ax = next(a for a in fig.axes if a.has_data())
    assert ax.get_xlim()[1] == easy_half
    assert ax.get_xlim() == ax.get_ylim()
    assert ax.get_xlim()[1] < full_half


def test_plot_level_aspect_locked_equal():
    level = generate_level(style="turning", n_segments=4, seed=0, turning_scale="gentle")
    fig, ax = plt.subplots()
    plot_level(level, ax=ax)
    assert ax.get_aspect() == 1.0 or ax.get_aspect() == "equal"
    assert ax.get_xlim() == ax.get_ylim()
    plt.close(fig)


def test_plot_level_toggle_flags_hide_vectors_and_markers():
    level = generate_level(style="turning", n_segments=5, seed=1, turning_scale="sharp")
    fig, ax = plt.subplots()
    plot_level(level, ax=ax, show_markers=False, show_vectors=False, show_legend=True)

    quivers = [c for c in ax.collections if isinstance(c, Quiver)]
    assert quivers == []

    _, labels = ax.get_legend_handles_labels()
    assert "path" in labels
    assert "displacement vector" not in labels
    assert "home vector" not in labels
    assert "home (start)" not in labels
    assert "release (end)" not in labels
    assert "turn" not in labels
    plt.close(fig)


def test_plot_level_show_vectors_without_markers():
    level = generate_level(style="linear", n_segments=3, seed=2)
    fig, ax = plt.subplots()
    plot_level(level, ax=ax, show_markers=False, show_vectors=True, show_legend=True)

    quivers = [c for c in ax.collections if isinstance(c, Quiver)]
    assert len(quivers) == 2

    _, labels = ax.get_legend_handles_labels()
    assert "displacement vector" in labels
    assert "home vector" in labels
    assert "home (start)" not in labels
    plt.close(fig)
