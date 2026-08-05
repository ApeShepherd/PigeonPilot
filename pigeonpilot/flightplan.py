"""
One drawn stroke → everything the live playground needs, computed up front.

The browser cannot animate what the kernel has not finished computing, and the
old playground animated *while* it computed, which is why the flight stuttered
and the spike raster could never stay in sync. Inference is cheap here (~30 ms
per model on the n=100 checkpoint), so this module runs the whole trial in one
go and hands the frontend a JSON-serialisable "flight plan": path geometry,
input and reservoir spike trains on the encoder clock, and the readout for both
pigeons.

Everything in the plan is measured, not illustrative. In particular:

- ``t_start`` / ``t_end`` per segment are the encoder's own firing windows, so a
  frontend that maps animation time to timesteps gets an exact bird↔raster sync.
- ``scores`` is the RidgeClassifier decision function over all 36 bins — the
  readout's actual margins, not a softmax invented for the plot.
- ``ensemble`` re-runs the trial under fresh Poisson draws. The encoding is
  stochastic, so a single draw is one sample from the model's output
  distribution; the circular mean and circular SD summarise that distribution
  using standard directional statistics.
- ``chance_error_deg`` is the mean circular error of a uniform random guess
  (90°), which is the reference the jury needs to read the numbers.

No Matplotlib, no widget code — this module is importable and testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from .drawing import DrawnPath, TrainingBand, prepare_drawn_path, training_band
from .paths import DEFAULT_HEADING_BINS, FULL_CIRCLE_DEG, Level, home_heading_bin, home_heading_deg
from .snn import (
    CheckpointBundle,
    ReservoirConfig,
    bin_to_heading_deg,
    circular_bin_error,
    encode_level_for_network,
    heading_to_unit,
    pool_reservoir_state,
    to_bindsnet_input,
)

# Extra Poisson re-draws per model. Off by default: the widget reports the
# single trial against the held-out error distribution instead, which answers
# "is this typical?" without another second of inference per click.
DEFAULT_ENSEMBLE_SIZE = 0
ENSEMBLE_SEED_STRIDE = 7919
# Mean |circular error| of a uniform random heading guess.
CHANCE_ERROR_DEG = 90.0
# Reported spread of a fully scattered ensemble; keeps the plan JSON-safe.
MAX_CIRCULAR_SD_DEG = 360.0
MODEL_LABELS = {"A": "A (fixed)", "B": "B (STDP)"}
REFERENCE_CACHE_NAME = "reference_errors.json"
REFERENCE_HIST_STEP_DEG = 15.0


@dataclass(frozen=True)
class TrialRun:
    """One pass of a level through one reservoir."""

    state: np.ndarray
    input_spikes: np.ndarray
    reservoir_spikes: np.ndarray


def run_trial_traced(net, level: Level, config: ReservoirConfig) -> TrialRun:
    """Run a level and keep the spike trains instead of only the pooled state.

    Mirrors ``snn.run_trial`` exactly (same encoding, same pooling, same reset)
    so the traced run and the plain run cannot drift apart.
    """
    spikes = encode_level_for_network(level, config)
    net.train(False)
    net.run(inputs={"Input": to_bindsnet_input(spikes)}, time=spikes.shape[0])
    monitor = net.monitors["Reservoir Spikes"].get("s")
    state = pool_reservoir_state(monitor)
    reservoir = monitor.squeeze(1).detach().cpu().numpy().astype(bool)
    net.reset_state_variables()
    return TrialRun(state=state, input_spikes=spikes, reservoir_spikes=reservoir)


def circular_summary(headings_deg: Sequence[float]) -> dict[str, float]:
    """Circular mean and circular SD of headings (directional statistics).

    ``circular_sd_deg`` follows ``sqrt(-2 ln R)`` with ``R`` the mean resultant
    length: 0 when the draws agree, growing as they spread. It is capped at
    ``MAX_CIRCULAR_SD_DEG`` so a fully scattered ensemble stays JSON-safe.
    """
    if len(headings_deg) == 0:
        return {"mean_deg": 0.0, "resultant": 0.0, "circular_sd_deg": 0.0}
    rad = np.deg2rad(np.asarray(headings_deg, dtype=float))
    vec = np.stack([np.sin(rad), np.cos(rad)]).mean(axis=1)
    resultant = float(np.hypot(vec[0], vec[1]))
    mean_deg = float(np.degrees(np.arctan2(vec[0], vec[1])) % FULL_CIRCLE_DEG)
    # atan2 round-off can land a hair below zero and wrap to 360°.
    if mean_deg > FULL_CIRCLE_DEG - 1e-9:
        mean_deg = 0.0
    # max(0, ...) keeps a unanimous ensemble at 0° rather than -0.0.
    sd_deg = (
        min(max(0.0, float(np.degrees(np.sqrt(-2.0 * np.log(resultant))))), MAX_CIRCULAR_SD_DEG)
        if resultant > 1e-12
        else MAX_CIRCULAR_SD_DEG
    )
    return {"mean_deg": mean_deg, "resultant": resultant, "circular_sd_deg": sd_deg}


def reference_errors(
    bundle: CheckpointBundle,
    *,
    models: Sequence[str] = ("A", "B"),
    seed: int = 42,
    train_frac: float = 0.8,
    cache: bool = True,
) -> dict[str, Any]:
    """Per-route error distribution on the held-out test split.

    A single drawn route is one sample from a wide distribution — on this
    checkpoint the per-route standard deviation is larger than the difference
    between the two models — so any single flight can rank B above A without
    that meaning anything. This function supplies the context that makes such a
    flight readable: the full error histogram per model and how often each model
    wins a head-to-head on individual routes.

    Evaluating 147 held-out routes takes a few seconds, so the result is cached
    as JSON next to the checkpoint (``bundle.source``) and reused.
    """
    import json
    from pathlib import Path

    # getattr: a bundle held over from an older kernel session has no ``source``.
    source = getattr(bundle, "source", None)
    cache_path: Path | None = None
    if cache and source is not None:
        cache_path = Path(source) / REFERENCE_CACHE_NAME
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("seed") == seed and set(cached.get("models", {})) >= set(models):
                    return cached
            except (json.JSONDecodeError, OSError):
                pass

    from .curriculum import generate_curriculum_dataset, split_dataset

    heading_bins = int(bundle.config.n_input)
    levels = generate_curriculum_dataset(seed=seed, heading_bins=heading_bins)
    _, test = split_dataset(levels, train_frac=train_frac, heading_bins=heading_bins)

    nets = {
        "A": (bundle.network_a, bundle.classifier_a),
        "B": (bundle.network_b, bundle.classifier_b),
    }
    edges = np.arange(0.0, 180.0 + REFERENCE_HIST_STEP_DEG, REFERENCE_HIST_STEP_DEG)
    per_model: dict[str, Any] = {}
    errors: dict[str, np.ndarray] = {}
    for name in models:
        if name not in nets:
            continue
        net, classifier = nets[name]
        values = []
        for level in test:
            state = run_trial_traced(net, level, bundle.config).state
            pred = int(classifier.predict(state.reshape(1, -1))[0])
            values.append(circular_bin_error(home_heading_bin(level, heading_bins), pred, heading_bins)[1])
        arr = np.asarray(values, dtype=float)
        errors[name] = arr
        per_model[name] = {
            "mean_deg": float(arr.mean()),
            "std_deg": float(arr.std()),
            "median_deg": float(np.median(arr)),
            "exact_acc": float((arr == 0.0).mean()),
            "errors": [float(v) for v in arr],
            "histogram": [int(v) for v in np.histogram(arr, bins=edges)[0]],
        }

    head_to_head: dict[str, float] = {}
    if len(errors) == 2:
        first, second = list(errors)
        a, b = errors[first], errors[second]
        head_to_head = {
            "pair": f"{second}<{first}",
            f"{first}_better": float((a < b).mean()),
            f"{second}_better": float((b < a).mean()),
            "tie": float((a == b).mean()),
        }

    result = {
        "seed": seed,
        "n": len(test),
        "hist_step_deg": REFERENCE_HIST_STEP_DEG,
        "hist_edges": [float(v) for v in edges],
        "models": per_model,
        "head_to_head": head_to_head,
    }
    if cache_path is not None:
        try:
            cache_path.write_text(json.dumps(result), encoding="utf-8")
        except OSError:
            pass
    return result


def _percentile_beaten(errors: Sequence[float], value: float) -> float:
    """Share of reference routes on which the model did *worse* than ``value``."""
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        return 0.0
    return float((arr > value).mean())


def _spike_pairs(matrix: np.ndarray, limit: int | None = None) -> list[list[int]]:
    """Sparse ``[[t, unit], ...]`` encoding of a boolean spike matrix."""
    times, units = np.nonzero(np.asarray(matrix) > 0)
    if limit is not None and len(times) > limit:
        pick = np.linspace(0, len(times) - 1, limit).astype(int)
        times, units = times[pick], units[pick]
    return [[int(t), int(u)] for t, u in zip(times, units)]


def _evaluate_model(
    name: str,
    net,
    classifier,
    level: Level,
    config: ReservoirConfig,
    true_bin: int,
    *,
    ensemble_size: int,
    heading_bins: int,
) -> dict[str, Any]:
    run = run_trial_traced(net, level, config)
    scores = classifier.decision_function(run.state.reshape(1, -1))[0]
    classes = np.asarray(classifier.classes_, dtype=int)
    pred_bin = int(classifier.predict(run.state.reshape(1, -1))[0])
    _, err_deg = circular_bin_error(true_bin, pred_bin, heading_bins)

    # Full 36-bin score profile, reindexed so position == bin index.
    profile = np.full(heading_bins, float(np.min(scores)), dtype=float)
    profile[classes % heading_bins] = scores

    result = {
        "name": name,
        "label": MODEL_LABELS.get(name, name),
        "pred_bin": pred_bin,
        "pred_heading_deg": bin_to_heading_deg(pred_bin, heading_bins),
        "error_deg": float(err_deg),
        "scores": [float(v) for v in profile],
        "reservoir_spikes": _spike_pairs(run.reservoir_spikes, limit=6000),
        "reservoir_spike_count": int(run.reservoir_spikes.sum()),
        "reservoir_active_units": int((run.reservoir_spikes.sum(axis=0) > 0).sum()),
    }
    if int(ensemble_size) <= 0:
        return result

    ensemble_bins: list[int] = []
    for i in range(int(ensemble_size)):
        seed_cfg = replace(config, encoding_seed=int(config.encoding_seed + (i + 1) * ENSEMBLE_SEED_STRIDE))
        state = run_trial_traced(net, level, seed_cfg).state
        ensemble_bins.append(int(classifier.predict(state.reshape(1, -1))[0]))

    summary = circular_summary([bin_to_heading_deg(b, heading_bins) for b in ensemble_bins])
    ensemble_bin = int(round(summary["mean_deg"] / (FULL_CIRCLE_DEG / heading_bins))) % heading_bins
    _, ensemble_err = circular_bin_error(true_bin, ensemble_bin, heading_bins)
    histogram = np.bincount(np.asarray(ensemble_bins, dtype=int), minlength=heading_bins)
    result["ensemble"] = {
        "size": len(ensemble_bins),
        "bins": ensemble_bins,
        "histogram": [int(v) for v in histogram],
        "bin": ensemble_bin,
        "heading_deg": float(summary["mean_deg"]),
        "error_deg": float(ensemble_err),
        "circular_sd_deg": float(summary["circular_sd_deg"]),
        "resultant": float(summary["resultant"]),
    }
    return result


def build_flight_plan(
    bundle: CheckpointBundle,
    points: np.ndarray | Sequence[Sequence[float]],
    *,
    models: Sequence[str] = ("A", "B"),
    ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
    band: TrainingBand | None = None,
    level_id: int = 900_001,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Turn a drawn stroke into a complete, JSON-serialisable flight plan.

    Pass ``reference`` (from ``reference_errors``) to place this single route
    inside each model's held-out error distribution — without it, one lucky
    flight reads like a result.

    Returns ``None`` when the stroke is too short to define a direction.
    """
    config = bundle.config
    heading_bins = int(config.n_input)
    band = band or training_band(heading_bins=heading_bins)
    drawn = prepare_drawn_path(
        points,
        level_id=level_id,
        heading_bins=heading_bins,
        band=band,
        velocity=config.encoding_velocity,
        dt=config.encoding_dt,
    )
    if drawn is None:
        return None

    level = drawn.level
    true_bin = home_heading_bin(level, heading_bins)
    home_distance = float(np.hypot(*level.home_xy))

    nets = {
        "A": (bundle.network_a, bundle.classifier_a),
        "B": (bundle.network_b, bundle.classifier_b),
    }
    results = [
        _evaluate_model(
            name,
            *nets[name],
            level=level,
            config=config,
            true_bin=true_bin,
            ensemble_size=ensemble_size,
            heading_bins=heading_bins,
        )
        for name in models
        if name in nets
    ]

    # Place this single route inside the model's held-out error distribution.
    ref_models = (reference or {}).get("models", {})
    for result in results:
        entry = ref_models.get(result["name"])
        if not entry:
            continue
        result["reference"] = {
            "n": int((reference or {}).get("n", 0)),
            "mean_deg": entry["mean_deg"],
            "std_deg": entry["std_deg"],
            "median_deg": entry["median_deg"],
            "histogram": entry["histogram"],
            "hist_step_deg": (reference or {}).get("hist_step_deg", REFERENCE_HIST_STEP_DEG),
            "better_than": _percentile_beaten(entry["errors"], result["error_deg"]),
        }

    input_spikes = encode_level_for_network(level, config)
    segments = _segment_timeline(drawn, config, heading_bins)
    return {
        "heading_bins": heading_bins,
        "n_steps": int(input_spikes.shape[0]),
        "path_steps": int(segments[-1]["t_end"]) if segments else 0,
        "trailing_silence": int(config.trailing_silence),
        "raw_points": drawn.raw_points.tolist(),
        "simplified_points": drawn.simplified_points.tolist(),
        "snapped_points": drawn.snapped_points.tolist(),
        "segments": segments,
        "release_xy": [float(level.end_xy[0]), float(level.end_xy[1])],
        "home_distance": home_distance,
        "true_bin": int(true_bin),
        "true_heading_deg": float(home_heading_deg(level.home_xy)),
        "input_spikes": _spike_pairs(input_spikes),
        "input_spike_count": int(input_spikes.sum()),
        "models": results,
        "preprocessing": {
            "raw_total_distance": drawn.raw_total_distance,
            "total_distance": drawn.total_distance,
            "scale_factor": drawn.scale_factor,
            "n_blocks": drawn.n_blocks,
            "in_distribution": bool(drawn.in_distribution),
            "band_min": band.min_total_distance,
            "band_max": band.max_total_distance,
            "band_max_blocks": band.max_blocks,
            "band_max_release": band.max_release_distance,
        },
        "reference": {
            "chance_error_deg": CHANCE_ERROR_DEG,
            "test_metrics": (bundle.metrics or {}).get("summary", {}),
            "n": int((reference or {}).get("n", 0)),
            "head_to_head": (reference or {}).get("head_to_head", {}),
        },
    }


def _segment_timeline(
    drawn: DrawnPath,
    config: ReservoirConfig,
    heading_bins: int,
) -> list[dict[str, Any]]:
    """Per-segment geometry plus the encoder timestep window it fires in."""
    from .encoding import heading_to_bin, segment_n_steps

    points = drawn.snapped_points
    out: list[dict[str, Any]] = []
    t = 0
    for i, seg in enumerate(drawn.segments):
        n_steps = segment_n_steps(
            seg.distance, velocity=config.encoding_velocity, dt=config.encoding_dt
        )
        out.append(
            {
                "index": i,
                "heading_deg": float(seg.heading_deg),
                "distance": float(seg.distance),
                "bin": int(heading_to_bin(seg.heading_deg, heading_bins)),
                "t_start": int(t),
                "t_end": int(t + n_steps),
                "x0": float(points[i][0]),
                "y0": float(points[i][1]),
                "x1": float(points[i + 1][0]),
                "y1": float(points[i + 1][1]),
            }
        )
        t += n_steps
    return out


def predicted_home_ray(
    plan: dict[str, Any],
    pred_bin: int,
) -> tuple[list[float], list[float]]:
    """Release point and the ray a predicted bin implies, at the *true* distance.

    The readout returns a direction only. Length is borrowed from ground truth,
    which is why the UI must label the homebound leg as a heading demo rather
    than a navigation result.
    """
    release = [float(v) for v in plan["release_xy"]]
    delta = heading_to_unit(bin_to_heading_deg(pred_bin, plan["heading_bins"]))
    length = float(plan["home_distance"])
    return release, [float(delta[0] * length), float(delta[1] * length)]
