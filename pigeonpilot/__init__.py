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
from .paths import (
    DIFFICULTIES,
    STYLES,
    DifficultySpec,
    Level,
    Segment,
    displacement_vector,
    generate_level,
    home_vector,
    trajectory_points,
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
    "iter_training_indices",
    "iter_train_schedule",
    "epoch_order",
]
