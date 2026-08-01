"""
PigeonPilot — path integration with a spiking reservoir.

Research question (Variant B):
Does STDP in a recurrent LIF reservoir improve path integration
compared to a fixed (non-plastic) reservoir?
"""

from .curriculum import (
    DEFAULT_EPOCHS_PER_DIFFICULTY,
    DIFFICULTY_SPECS,
    STYLE_LABELS,
    curriculum_level_count,
    epoch_order,
    format_curriculum_table,
    format_style_labels_table,
    generate_curriculum_dataset,
    iter_train_schedule,
    iter_training_indices,
    split_dataset,
    summarize_dataset,
)
from .encoding import (
    SpikeBlock,
    compass_bin,
    encode_level,
    encode_segments,
    heading_to_bin,
    plan_encoding,
    segment_n_steps,
)
from .paths import (
    DIFFICULTIES,
    STYLES,
    DifficultySpec,
    Level,
    Segment,
    displacement_vector,
    generate_level,
    home_vector,
    snap_heading,
    trajectory_points,
)
from .viz import (
    compute_xy_limits,
    plot_body_ring_anatomy,
    plot_level,
    plot_level_encoding,
    plot_level_ring_frames,
    plot_levels_grid,
    plot_release_points,
    plot_spike_raster,
)

__all__ = [
    "Segment",
    "Level",
    "STYLES",
    "STYLE_LABELS",
    "DIFFICULTIES",
    "DIFFICULTY_SPECS",
    "DifficultySpec",
    "DEFAULT_EPOCHS_PER_DIFFICULTY",
    "curriculum_level_count",
    "format_curriculum_table",
    "format_style_labels_table",
    "generate_level",
    "generate_curriculum_dataset",
    "split_dataset",
    "summarize_dataset",
    "trajectory_points",
    "home_vector",
    "displacement_vector",
    "snap_heading",
    "compass_bin",
    "heading_to_bin",
    "segment_n_steps",
    "SpikeBlock",
    "plan_encoding",
    "encode_segments",
    "encode_level",
    "iter_training_indices",
    "iter_train_schedule",
    "epoch_order",
    "compute_xy_limits",
    "plot_level",
    "plot_levels_grid",
    "plot_release_points",
    "plot_spike_raster",
    "plot_level_encoding",
    "plot_body_ring_anatomy",
    "plot_level_ring_frames",
]
