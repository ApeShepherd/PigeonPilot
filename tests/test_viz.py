"""Unit tests for plotting helpers (zoom / square limits / layer toggles)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver

from pigeonpilot.curriculum import generate_curriculum_dataset
from pigeonpilot.paths import generate_level
from pigeonpilot.viz import (
    compute_xy_limits,
    plot_body_ring_anatomy,
    plot_home_prediction_ring,
    plot_level,
    plot_level_encoding,
    plot_level_ring_frames,
    plot_levels_grid,
    plot_release_points,
    plot_spike_raster,
)


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


def test_plot_level_encoding_side_by_side():
    data = generate_curriculum_dataset(seed=42)
    level = next(lv for lv in data if lv.level_id == 142)
    fig = plot_level_encoding(level, dt=0.25)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_plot_spike_raster_smoke():
    from pigeonpilot.encoding import encode_segments, plan_encoding
    from pigeonpilot.paths import Segment

    segments = (
        Segment(heading_deg=0.0, distance=3.0),
        Segment(heading_deg=270.0, distance=2.0),
    )
    spikes = encode_segments(segments, dt=1.0)
    plan = plan_encoding(segments, dt=1.0)
    fig, ax = plt.subplots()
    plot_spike_raster(spikes, ax=ax, plan=plan)
    assert ax.get_ylabel()
    plt.close(fig)


def test_plot_body_ring_anatomy_smoke():
    fig = plot_body_ring_anatomy()
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_level_ring_frames_smoke():
    data = generate_curriculum_dataset(seed=42)
    level = next(lv for lv in data if lv.level_id == 34)
    fig = plot_level_ring_frames(level, dt=0.25)
    assert len(fig.axes) == len(level.segments)
    plt.close(fig)


def test_plot_home_prediction_ring_smoke():
    fig = plot_home_prediction_ring(true_bin=18, pred_bin=20)
    polar = [ax for ax in fig.axes if getattr(ax, "name", None) == "polar"]
    assert len(polar) == 2
    plt.close(fig)


def test_plot_release_points_smoke():
    data = generate_curriculum_dataset(seed=42)[:20]
    fig, ax = plt.subplots()
    plot_release_points(data, ax=ax)
    assert ax.get_xlim() == ax.get_ylim()
    plt.close(fig)


def test_plot_levels_grid_rejects_bad_max_cols():
    import pytest

    data = generate_curriculum_dataset(seed=0)[:3]
    with pytest.raises(ValueError, match="max_cols"):
        plot_levels_grid(data, max_cols=0)


def test_package_exports_public_api():
    import pigeonpilot as pp

    assert callable(pp.plot_body_ring_anatomy)
    assert callable(pp.plot_home_prediction_ring)
    assert callable(pp.encode_level)
    assert callable(pp.snap_heading)
    assert callable(pp.plan_encoding)
