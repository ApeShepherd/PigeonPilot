"""
PigeonPilot — path integration with a spiking reservoir.

Research question (Variant B):
Does STDP in a recurrent LIF reservoir improve path integration
compared to a fixed (non-plastic) reservoir?
"""

from .paths import (
    Segment,
    Level,
    STYLES,
    STYLE_LABELS,
    DIFFICULTIES,
    DIFFICULTY_SPECS,
    generate_level,
    generate_dataset,
    generate_curriculum_dataset,
    split_dataset,
    summarize_dataset,
    trajectory_points,
    home_vector,
    displacement_vector,
    iter_training_indices,
    iter_train_schedule,
    epoch_order,
)

__all__ = [
    "Segment",
    "Level",
    "STYLES",
    "STYLE_LABELS",
    "DIFFICULTIES",
    "DIFFICULTY_SPECS",
    "generate_level",
    "generate_dataset",
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
