"""Flight-plan assembly for the live playground.

Pure helpers run everywhere; the end-to-end plan needs BindsNET plus a saved
checkpoint and is skipped when either is missing.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pigeonpilot.flightplan import (
    CHANCE_ERROR_DEG,
    MAX_CIRCULAR_SD_DEG,
    _percentile_beaten,
    _spike_pairs,
    circular_summary,
)


def test_circular_summary_agrees_on_a_single_heading():
    out = circular_summary([90.0] * 5)
    assert out["mean_deg"] == pytest.approx(90.0)
    assert out["resultant"] == pytest.approx(1.0)
    assert out["circular_sd_deg"] == pytest.approx(0.0, abs=1e-6)


def test_circular_summary_wraps_around_north():
    """Naive averaging of 350° and 10° gives 180°; the circular mean gives 0°."""
    out = circular_summary([350.0, 10.0])
    assert out["mean_deg"] == pytest.approx(0.0, abs=1e-6)


def test_circular_sd_grows_with_spread():
    tight = circular_summary([80.0, 90.0, 100.0])["circular_sd_deg"]
    loose = circular_summary([10.0, 90.0, 200.0])["circular_sd_deg"]
    assert tight < loose


def test_opposite_headings_have_no_resultant():
    out = circular_summary([0.0, 180.0])
    assert out["resultant"] == pytest.approx(0.0, abs=1e-9)
    assert out["circular_sd_deg"] == MAX_CIRCULAR_SD_DEG


def test_summary_stays_json_safe():
    for headings in ([0.0, 180.0], [0.0], [], [10.0, 350.0]):
        json.dumps(circular_summary(headings))


def test_chance_error_is_the_uniform_guess_baseline():
    rng = np.random.default_rng(0)
    true_bins = rng.integers(0, 36, 20000)
    guesses = rng.integers(0, 36, 20000)
    diff = np.abs(true_bins - guesses) % 36
    mean_deg = np.minimum(diff, 36 - diff).mean() * 10.0
    assert mean_deg == pytest.approx(CHANCE_ERROR_DEG, abs=1.0)


def test_spike_pairs_is_sparse_and_ordered():
    matrix = np.zeros((5, 4))
    matrix[1, 2] = 1
    matrix[3, 0] = 1
    assert _spike_pairs(matrix) == [[1, 2], [3, 0]]


def test_spike_pairs_subsamples_when_capped():
    matrix = np.ones((50, 10))
    assert len(_spike_pairs(matrix, limit=25)) == 25


def test_percentile_beaten_counts_worse_reference_routes():
    errors = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert _percentile_beaten(errors, 20.0) == pytest.approx(0.4)
    assert _percentile_beaten(errors, 0.0) == pytest.approx(0.8)
    assert _percentile_beaten(errors, 180.0) == pytest.approx(0.0)
    assert _percentile_beaten([], 10.0) == 0.0


# ------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def bundle():
    torch = pytest.importorskip("torch")  # noqa: F841
    pytest.importorskip("bindsnet")
    from pigeonpilot.snn import load_run

    try:
        return load_run("latest")
    except FileNotFoundError:
        pytest.skip("no saved checkpoint under outputs/checkpoints")


@pytest.fixture(scope="module")
def reference(bundle):
    from pigeonpilot.flightplan import reference_errors

    return reference_errors(bundle)


@pytest.fixture(scope="module")
def stroke():
    """A hand-drawn arc, jitter included, as the canvas would deliver it."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 150)
    return np.stack([4 * np.sin(2.5 * t), 4 * t], axis=1) + rng.normal(0.0, 0.08, (150, 2))


@pytest.fixture(scope="module")
def plan(bundle, reference, stroke):
    from pigeonpilot.flightplan import build_flight_plan

    return build_flight_plan(bundle, stroke, reference=reference)


def test_reference_reproduces_the_stored_training_metrics(bundle, reference):
    """The playground must not disagree with the numbers Models_staged reported."""
    stored = bundle.metrics["summary"]
    for name, entry in reference["models"].items():
        assert entry["mean_deg"] == pytest.approx(stored[name]["mean_deg"], abs=0.05)
        assert entry["std_deg"] == pytest.approx(stored[name]["std_deg"], abs=0.05)
        assert entry["exact_acc"] == pytest.approx(stored[name]["exact_acc"], abs=1e-6)


def test_reference_head_to_head_sums_to_one(reference):
    h2h = reference["head_to_head"]
    assert h2h["A_better"] + h2h["B_better"] + h2h["tie"] == pytest.approx(1.0)


def test_single_route_spread_exceeds_the_model_gap(reference):
    """Why one drawn route cannot rank the models: the per-route spread is larger."""
    a, b = reference["models"]["A"], reference["models"]["B"]
    gap = abs(a["mean_deg"] - b["mean_deg"])
    assert max(a["std_deg"], b["std_deg"]) > gap
    assert reference["head_to_head"]["B_better"] > 0.0


def test_reference_histogram_counts_every_route(reference):
    for entry in reference["models"].values():
        assert sum(entry["histogram"]) == reference["n"]
        assert len(entry["errors"]) == reference["n"]


def test_reference_is_cached_and_stable(bundle, reference):
    from pigeonpilot.flightplan import reference_errors

    again = reference_errors(bundle)
    assert again["models"]["A"]["mean_deg"] == pytest.approx(reference["models"]["A"]["mean_deg"])


def test_plan_places_the_route_in_the_error_distribution(plan):
    for model in plan["models"]:
        ref = model["reference"]
        assert 0.0 <= ref["better_than"] <= 1.0
        assert ref["n"] == plan["reference"]["n"] > 0
        assert sum(ref["histogram"]) == ref["n"]


def test_plan_is_json_serialisable(plan):
    json.dumps(plan)


def test_plan_timeline_is_contiguous_and_matches_the_encoder(plan):
    segments = plan["segments"]
    assert segments[0]["t_start"] == 0
    for prev, nxt in zip(segments, segments[1:]):
        assert prev["t_end"] == nxt["t_start"]
        assert prev["x1"] == pytest.approx(nxt["x0"])
        assert prev["y1"] == pytest.approx(nxt["y0"])
    assert segments[-1]["t_end"] == plan["path_steps"]
    assert plan["n_steps"] == plan["path_steps"] + plan["trailing_silence"]


def test_spikes_fall_inside_the_timeline(plan):
    for t, bin_idx in plan["input_spikes"]:
        assert 0 <= t < plan["n_steps"]
        assert 0 <= bin_idx < plan["heading_bins"]
    for model in plan["models"]:
        for t, unit in model["reservoir_spikes"]:
            assert 0 <= t < plan["n_steps"]
            assert unit >= 0


def test_input_spikes_land_on_the_segment_that_fires_them(plan):
    """Bird position and raster share one clock — this is what keeps them in sync."""
    windows = {(s["t_start"], s["t_end"]): s["bin"] for s in plan["segments"]}
    for t, bin_idx in plan["input_spikes"]:
        for (start, end), expected in windows.items():
            if start <= t < end:
                assert bin_idx == expected
                break


def test_every_model_reports_a_full_score_profile(plan):
    for model in plan["models"]:
        assert len(model["scores"]) == plan["heading_bins"]
        assert 0 <= model["pred_bin"] < plan["heading_bins"]
        assert 0.0 <= model["error_deg"] <= 180.0
        assert int(np.argmax(model["scores"])) == model["pred_bin"]


def test_default_plan_carries_no_ensemble(plan):
    """The demo reports one trial against the test distribution, not a seed spread."""
    for model in plan["models"]:
        assert "ensemble" not in model


def test_ensemble_histogram_counts_every_draw(bundle, stroke):
    """Opt-in path: ensemble_size > 0 still yields a consistent seed histogram."""
    from pigeonpilot.flightplan import build_flight_plan

    opted_in = build_flight_plan(bundle, stroke, ensemble_size=4)
    for model in opted_in["models"]:
        ensemble = model["ensemble"]
        assert ensemble["size"] == 4
        assert sum(ensemble["histogram"]) == ensemble["size"]
        assert len(ensemble["histogram"]) == opted_in["heading_bins"]
        assert 0 <= ensemble["bin"] < opted_in["heading_bins"]


def test_release_point_closes_the_geometry(plan):
    release = np.asarray(plan["release_xy"])
    np.testing.assert_allclose(release, plan["snapped_points"][-1], atol=1e-9)
    assert plan["home_distance"] == pytest.approx(float(np.hypot(*release)))


def test_preprocessing_audit_trail_is_present(plan):
    audit = plan["preprocessing"]
    assert audit["scale_factor"] > 0
    assert audit["n_blocks"] <= audit["band_max_blocks"]
    assert plan["reference"]["chance_error_deg"] == CHANCE_ERROR_DEG


def test_short_stroke_returns_no_plan(bundle):
    from pigeonpilot.flightplan import build_flight_plan

    assert build_flight_plan(bundle, [[0.0, 0.0], [0.0001, 0.0]]) is None
