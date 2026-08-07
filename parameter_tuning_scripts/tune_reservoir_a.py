"""
Hyperparameter tuning for Reservoir A (fixed, non-plastic LIF reservoir) only.

Standalone script — does not import or modify Models.ipynb. Encoding/dataset
parameters (N_INPUT_NODES, TARGET_STEPS_LONGEST, INPUT_RATE_HZ, ...) are copied
verbatim from Models.ipynb's hyperparameter cell; keep them in sync by hand.

Three things are actually searched, in this order, each stage holding the
previous stage's winner fixed. Everything else (INHIB_FRACTION, reservoir
size, LIF params, ...) is a fixed, accepted default rather than a search
target — RESERVOIR_SCALE gets a confirmation pass in Stage 3, but is not
itself searched; the rest are just used as-is throughout.

  Stage 1 — FEEDFORWARD_STRENGTH and INHIB_WEIGHT_RATIO, chosen together
      A reservoir can look healthy on mean firing rate alone (good "coverage")
      while still firing in a synchronized whole-population avalanche within a
      handful of timesteps — nearly the same population vector regardless of
      input, so almost uninformative for the readout. This stage checks BOTH:
      mean-rate coverage/saturation (as before) AND per-timestep burst size,
      across every (FEEDFORWARD_STRENGTH, INHIB_WEIGHT_RATIO) pair in the
      grid, sampled across all five difficulties. Picks the smallest FF, then
      smallest ratio, that's avalanche-free and inside the coverage band.

  Stage 2 — RIDGE_ALPHA
      k-fold cross-validation on the TRAIN split only (never the test split —
      tuning against test error would invalidate it as an unbiased estimate).
      The test split is touched exactly once at the end, purely to report a
      number, after alpha is already fixed.

  Stage 3 — confirm RESERVOIR_SCALE
      RESERVOIR_SCALE is NOT searched — it's fixed at the value used
      throughout Stages 1-2. This re-runs the same avalanche/coverage check
      from Stage 1 at that fixed value, on a larger level sample, purely to
      show it's actually a sound choice rather than an unverified default.

Connection clamps: the feedforward connection clamps to (WMIN, WMAX) = (0, 10),
matching Models.ipynb's Step 4 cell. The recurrent connection carries negative
(inhibitory) weights after the Dale's-law split in init_weights() below, so it
clamps to the symmetric (RESERVOIR_WMIN, RESERVOIR_WMAX) instead — reusing
WMIN/WMAX there would zero out every inhibitory weight. Keep both pairs in
sync with the notebook by hand if it changes.

Run:
    python tune_reservoir_a.py
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import KFold

# --- bindsnet's outdated torch._six import (same shim as Models.ipynb) ---
import collections.abc

_six_shim = types.ModuleType("torch._six")
_six_shim.container_abcs = collections.abc
_six_shim.string_classes = (str,)
_six_shim.int_classes = (int,)
_six_shim.inf = float("inf")
sys.modules["torch._six"] = _six_shim

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor

from resources.paths import DEFAULT_HEADING_BINS, FULL_CIRCLE_DEG, home_heading_bin
from resources.encoding import encode_level
from resources.curriculum import generate_curriculum_dataset, split_dataset


# =============================================================================
# Config — encoding / dataset params copied from Models.ipynb's hyperparameter
# cell. Keep these in sync by hand; do not "improve" them here, this script is
# only for tuning FEEDFORWARD_STRENGTH / INHIB_WEIGHT_RATIO / RIDGE_ALPHA (and
# confirming RESERVOIR_SCALE).
# =============================================================================

DATASET_SEED = 42
SPLIT_SEED = 0

N_INPUT_NODES = DEFAULT_HEADING_BINS
N_RESERVOIR_NODES = 1000
NETWORK_DT = 1.0
ENCODING_DT = NETWORK_DT
TARGET_STEPS_LONGEST = 1000
INPUT_RATE_HZ = 40.0
ENCODING_SEED = 0
TRAILING_SILENCE = 20
WEIGHT_SEED = 0

# Feedforward connection clamp — must match Models.ipynb's Step 4 cell.
# Weights here are always >= 0 (uniform random draws), so a one-sided clamp is fine.
WMIN, WMAX = 0.0, 10.0

# Fixed, accepted without a dedicated search (standard ~80/20 cortical ratio).
INHIB_FRACTION = 0.2

# Fixed, accepted without a dedicated search — confirmed (not searched) in Stage 3.
RESERVOIR_SCALE = 0.9

# Recurrent connection carries negative (inhibitory) weights after the Dale's
# law split in init_weights(), so it needs its own symmetric clamp instead of
# reusing WMIN/WMAX (which would zero out every inhibitory weight).
RESERVOIR_WMIN, RESERVOIR_WMAX = -WMAX, WMAX

LIF_KW = dict(rest=-65.0, reset=-65.0, thresh=-52.0, refrac=5, tc_decay=250.0)

# A neuron cannot fire faster than once per (refrac + dt): this is the hard
# ceiling on per-neuron mean rate used by the coverage/saturation check.
REFRACTORY_CEILING = NETWORK_DT / (LIF_KW["refrac"] + NETWORK_DT)

# A single-timestep spike count above this fraction of the reservoir counts as
# an "avalanche" (synchronized burst) rather than graded, input-dependent
# activity. Used by both Stage 1 and Stage 3.
AVALANCHE_FRACTION = 0.3
AVALANCHE_THRESHOLD = int(AVALANCHE_FRACTION * N_RESERVOIR_NODES)

# =============================================================================
# Sweep grids and decision thresholds — edit these to widen/narrow the search.
# =============================================================================

FF_STRENGTH_GRID = [10, 20, 30, 40, 50, 60, 70, 80, 90]
INHIB_WEIGHT_RATIO_GRID = [ 2.0, 4.0, 6.0, 8.0, 10.0]
RIDGE_ALPHA_GRID = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

STAGE1_LEVELS_PER_DIFFICULTY = 3     # levels probed per difficulty, per (FF, ratio) pair
STAGE1_COVERAGE_BAND = (0.20, 0.50)  # target fraction of neurons "active" on easy levels
STAGE1_ACTIVE_EPS = 1e-6             # mean-rate threshold above which a neuron counts as "active"
STAGE1_SATURATION_MARGIN = 0.85      # fraction of REFRACTORY_CEILING that counts as "saturated"
STAGE1_SATURATION_LIMIT = 0.25       # max acceptable fraction of saturated active neurons on hard levels

STAGE2_TRAIN_SUBSAMPLE = None        # None = use the full train split; set an int to subsample for CV speed
STAGE2_KFOLDS = 5
STAGE2_CV_SEED = 0
RUN_FINAL_TEST_CHECK = True          # one-shot look at the real test split, after alpha is fixed

STAGE3_LEVELS_PER_DIFFICULTY = 5     # broader sample than Stage 1, since this is the final confirmation


# =============================================================================
# Dataset (same seeds as Models.ipynb, so results transfer back directly)
# =============================================================================

dataset = generate_curriculum_dataset(seed=DATASET_SEED)
train_set, test_set = split_dataset(dataset, train_frac=0.8, seed=SPLIT_SEED)

_max_dist = max(sum(s.distance for s in lv.segments) for lv in dataset)
ENCODING_VELOCITY = _max_dist / (TARGET_STEPS_LONGEST * ENCODING_DT)

DIFFICULTY_ORDER = list(dict.fromkeys(lv.difficulty for lv in train_set))
EASY_DIFFICULTY = DIFFICULTY_ORDER[0]
HARD_DIFFICULTY = DIFFICULTY_ORDER[-1]

print(
    f"dataset: {len(dataset)} levels | train={len(train_set)} test={len(test_set)} "
    f"| velocity={ENCODING_VELOCITY:.4f}"
)


# =============================================================================
# Shared helpers (mirroring Models.ipynb's Step 3 helpers, network-A only)
# =============================================================================

def home_bin(level, heading_bins: int = DEFAULT_HEADING_BINS) -> int:
    return home_heading_bin(level, heading_bins)


def circular_bin_error(true_bin: int, pred_bin: int, heading_bins: int = DEFAULT_HEADING_BINS) -> tuple[int, float]:
    d = abs(int(true_bin) - int(pred_bin)) % heading_bins
    d = min(d, heading_bins - d)
    return d, d * (FULL_CIRCLE_DEG / heading_bins)


def encode_level_for_network(level) -> np.ndarray:
    spikes = encode_level(
        level,
        velocity=ENCODING_VELOCITY,
        dt=ENCODING_DT,
        heading_bins=N_INPUT_NODES,
        rate_hz=INPUT_RATE_HZ,
        seed=ENCODING_SEED + int(level.level_id),
    )
    if TRAILING_SILENCE > 0:
        pad = np.zeros((TRAILING_SILENCE, N_INPUT_NODES), dtype=np.float32)
        spikes = np.vstack([spikes, pad])
    return spikes


def to_bindsnet_input(spikes: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(spikes, dtype=np.float32)).unsqueeze(1)


def pool_reservoir_state(spike_tensor: torch.Tensor) -> np.ndarray:
    return spike_tensor.float().mean(dim=0).squeeze(0).detach().cpu().numpy()


def init_weights(
    ff_strength: float,
    reservoir_scale: float,
    inhib_weight_ratio: float,
    seed: int = WEIGHT_SEED,
):
    """Same construction as Models.ipynb's init_weights, parametrized instead of
    global, plus a Dale's-law split on the recurrent matrix: INHIB_FRACTION of
    reservoir neurons get their OUTGOING weights negated and scaled up by
    inhib_weight_ratio (inhibitory), fixing the all-excitatory avalanche
    dynamics an unbalanced reservoir produces (see module docstring)."""
    generator = torch.Generator().manual_seed(seed)

    feedforward_weights = torch.rand(N_INPUT_NODES, N_RESERVOIR_NODES, generator=generator)
    w_ff = ff_strength * feedforward_weights / feedforward_weights.sum(dim=0, keepdim=True)

    reservoir_weights = torch.rand(N_RESERVOIR_NODES, N_RESERVOIR_NODES, generator=generator)
    reservoir_weights.fill_diagonal_(0.0)

    # Dale's law: pick INHIB_FRACTION of neurons, flip the sign of the ROW each
    # owns (= what that neuron sends to everyone else), and scale it up by
    # inhib_weight_ratio so the smaller inhibitory population can actually
    # restrain the larger excitatory one instead of just opposing it. They
    # still receive input/recurrent drive normally on the column side; only
    # their outgoing influence changes.
    n_inhibitory = int(round(INHIB_FRACTION * N_RESERVOIR_NODES))
    inhibitory_idx = torch.randperm(N_RESERVOIR_NODES, generator=generator)[:n_inhibitory]
    reservoir_weights[inhibitory_idx, :] *= -inhib_weight_ratio

    # Spectral radius of the actual (now mixed-sign) matrix, so reservoir_scale
    # still describes the matrix that's really being used.
    spectral_radius = torch.linalg.eigvals(reservoir_weights).abs().max().item()
    w_res = reservoir_weights * (reservoir_scale / spectral_radius)

    return w_ff, w_res


def build_reservoir_a(w_ff: torch.Tensor, w_res: torch.Tensor) -> Network:
    """Fixed (non-plastic) reservoir, network A only. Feedforward connection
    clamped to WMIN/WMAX (matching Models.ipynb's Step 4 cell); recurrent
    connection clamped to RESERVOIR_WMIN/RESERVOIR_WMAX instead, since it now
    carries negative (inhibitory) weights after the Dale's-law split."""
    net = Network(dt=NETWORK_DT)
    input_layer = Input(n=N_INPUT_NODES, traces=True)
    reservoir = LIFNodes(n=N_RESERVOIR_NODES, traces=True, **LIF_KW)
    net.add_layer(input_layer, name="Input")
    net.add_layer(reservoir, name="Reservoir")

    ff_conn = Connection(source=input_layer, target=reservoir, w=w_ff.clone(), wmin=WMIN, wmax=WMAX)
    res_conn = Connection(
        source=reservoir, target=reservoir, w=w_res.clone(), wmin=RESERVOIR_WMIN, wmax=RESERVOIR_WMAX
    )
    net.add_connection(ff_conn, source="Input", target="Reservoir")
    net.add_connection(res_conn, source="Reservoir", target="Reservoir")

    net.add_monitor(Monitor(obj=reservoir, state_vars=["s"], time=None), name="Reservoir Spikes")
    return net


def run_trial_full(net: Network, level) -> tuple[torch.Tensor, int]:
    """Encode -> run -> return the FULL (T, 1, n_res) spike tensor (not pooled).
    Resets state after."""
    spikes = encode_level_for_network(level)
    inputs = to_bindsnet_input(spikes)
    net.train(False)
    net.run(inputs={"Input": inputs}, time=spikes.shape[0])
    s = net.monitors["Reservoir Spikes"].get("s").clone()
    net.reset_state_variables()
    return s, spikes.shape[0]


def run_trial(net: Network, level) -> np.ndarray:
    """Encode -> run -> pooled per-neuron mean rate. Resets state after."""
    s, _ = run_trial_full(net, level)
    return pool_reservoir_state(s)


def levels_for_difficulty(levels: Sequence, difficulty: str, n: int) -> list:
    """First n levels matching a difficulty (deterministic, not random, so
    stage results are reproducible run-to-run)."""
    return [lv for lv in levels if lv.difficulty == difficulty][:n]


def sample_levels_all_difficulties(levels: Sequence, n_per_difficulty: int) -> list:
    """First n_per_difficulty levels from EACH difficulty, flattened."""
    return [lv for d in DIFFICULTY_ORDER for lv in levels_for_difficulty(levels, d, n_per_difficulty)]


def stratified_subsample(levels: Sequence, n: int, seed: int) -> list:
    """Roughly proportional-by-difficulty subsample, for Stage 2 CV speed."""
    rng = np.random.default_rng(seed)
    by_diff: dict[str, list] = {}
    for lv in levels:
        by_diff.setdefault(lv.difficulty, []).append(lv)
    total = len(levels)
    chosen: list = []
    for diff, group in by_diff.items():
        idx = rng.permutation(len(group))
        k = max(1, round(n * len(group) / total))
        chosen.extend(group[i] for i in idx[:k])
    rng.shuffle(chosen)
    return chosen[:n]


def _spike_health(mean_rates: np.ndarray) -> tuple[float, float, float]:
    """(coverage, mean rate among active neurons, fraction of active neurons saturated)."""
    active = mean_rates > STAGE1_ACTIVE_EPS
    coverage = float(active.mean())
    if not active.any():
        return coverage, 0.0, 0.0
    active_rates = mean_rates[active]
    mean_active_rate = float(active_rates.mean())
    saturated = active_rates >= STAGE1_SATURATION_MARGIN * REFRACTORY_CEILING
    saturation_frac = float(saturated.mean())
    return coverage, mean_active_rate, saturation_frac


@dataclass
class TrialDiagnostics:
    """Everything Stage 1 and Stage 3 need from one simulated trial, computed
    from a single run (no need to re-simulate for the mean-rate vs. burst-timing
    checks separately)."""

    difficulty: str
    T: int
    total_spikes: float
    mean_rates: np.ndarray             # per-neuron mean rate, for coverage/saturation
    max_spikes_in_one_step: int        # largest single-timestep population spike count
    nonzero_steps: int                 # how many timesteps had at least one spike


def diagnose_trial(net: Network, level) -> TrialDiagnostics:
    s, T = run_trial_full(net, level)
    per_step = s.float().squeeze(1).sum(dim=1)
    nonzero_steps = int((per_step > 0).sum())
    max_step = int(per_step.max()) if nonzero_steps else 0
    return TrialDiagnostics(
        difficulty=level.difficulty,
        T=T,
        total_spikes=float(s.sum()),
        mean_rates=pool_reservoir_state(s),
        max_spikes_in_one_step=max_step,
        nonzero_steps=nonzero_steps,
    )


# =============================================================================
# Stage 1 — FEEDFORWARD_STRENGTH and INHIB_WEIGHT_RATIO, chosen together
# =============================================================================

@dataclass
class Stage1Result:
    ff_strength: float
    inhib_ratio: float
    easy_coverage: float   # fraction of neurons active on the sparsest (easy) trials
    hard_saturation: float  # fraction of hard-trial active neurons pinned near the refractory ceiling
    max_burst: int          # largest single-timestep spike count seen across all probed levels
    n_avalanche: int        # how many probed levels exceeded AVALANCHE_THRESHOLD in one step
    n_silent: int            # how many probed levels produced zero spikes (informational only)
    ok: bool = field(init=False)

    def __post_init__(self):
        lo, hi = STAGE1_COVERAGE_BAND
        self.ok = (
            lo <= self.easy_coverage <= hi
            and self.hard_saturation <= STAGE1_SATURATION_LIMIT
            and self.n_avalanche == 0
        )


def choose_ff_and_inhib_ratio() -> tuple[float, float]:
    print("\n=== Stage 1: FEEDFORWARD_STRENGTH + INHIB_WEIGHT_RATIO ===")
    levels = sample_levels_all_difficulties(train_set, STAGE1_LEVELS_PER_DIFFICULTY)
    print(
        f"reservoir_scale fixed at {RESERVOIR_SCALE} (the value we intend to use; confirmed, "
        f"not searched, in Stage 3). For each (FF, ratio) pair, probing {len(levels)} levels "
        f"({STAGE1_LEVELS_PER_DIFFICULTY} per difficulty x {len(DIFFICULTY_ORDER)} difficulties). "
        f"A reservoir can look healthy on mean firing rate alone while still firing in a "
        f"synchronized whole-population avalanche within a handful of timesteps — nearly the "
        f"same population vector regardless of input — so this checks BOTH mean-rate coverage/"
        f"saturation AND per-timestep burst size.\n"
    )
    print(
        f"target: easy coverage in {STAGE1_COVERAGE_BAND}, hard saturation <= "
        f"{STAGE1_SATURATION_LIMIT}, no probed level exceeds {AVALANCHE_THRESHOLD} "
        f"({AVALANCHE_FRACTION:.0%} of {N_RESERVOIR_NODES}) neurons firing in a single step\n"
    )

    results: list[Stage1Result] = []
    header = (
        f"{'FF':>5} | {'ratio':>5} | {'easy_cov':>8} | {'hard_sat':>8} | "
        f"{'max_burst':>9} | {'n_avalanche':>11} | {'n_silent':>8} | ok?"
    )
    print(header)
    print("-" * len(header))

    for ff in FF_STRENGTH_GRID:
        for ratio in INHIB_WEIGHT_RATIO_GRID:
            w_ff, w_res = init_weights(ff, RESERVOIR_SCALE, inhib_weight_ratio=ratio)
            net = build_reservoir_a(w_ff, w_res)

            diags = [diagnose_trial(net, lv) for lv in levels]
            easy_cov = np.mean(
                [_spike_health(d.mean_rates)[0] for d in diags if d.difficulty == EASY_DIFFICULTY]
            )
            hard_sat = np.mean(
                [_spike_health(d.mean_rates)[2] for d in diags if d.difficulty == HARD_DIFFICULTY]
            )
            max_burst = max(d.max_spikes_in_one_step for d in diags)
            n_avalanche = sum(1 for d in diags if d.max_spikes_in_one_step > AVALANCHE_THRESHOLD)
            n_silent = sum(1 for d in diags if d.total_spikes == 0)

            res = Stage1Result(
                ff_strength=ff,
                inhib_ratio=ratio,
                easy_coverage=float(easy_cov),
                hard_saturation=float(hard_sat),
                max_burst=max_burst,
                n_avalanche=n_avalanche,
                n_silent=n_silent,
            )
            results.append(res)
            print(
                f"{ff:>5} | {ratio:>5} | {res.easy_coverage:>8.2%} | {res.hard_saturation:>8.2%} | "
                f"{res.max_burst:>9} | {res.n_avalanche:>11} | {res.n_silent:>8} | "
                f"{'yes' if res.ok else 'no'}"
            )

    candidates = [r for r in results if r.ok]
    if candidates:
        # Smallest FF, then smallest ratio: minimal input drive and minimal
        # inhibitory strength needed to reach healthy, avalanche-free activity.
        chosen = min(candidates, key=lambda r: (r.ff_strength, r.inhib_ratio))
        print(
            f"\n-> chosen FEEDFORWARD_STRENGTH = {chosen.ff_strength}, INHIB_WEIGHT_RATIO = "
            f"{chosen.inhib_ratio} (smallest pair meeting target, in that priority order)"
        )
    else:
        band_center = sum(STAGE1_COVERAGE_BAND) / 2
        chosen = min(results, key=lambda r: (r.n_avalanche, abs(r.easy_coverage - band_center)))
        print(
            f"\n! no (FF, ratio) pair met the full target — falling back to FF={chosen.ff_strength}, "
            f"ratio={chosen.inhib_ratio} (fewest avalanches, closest easy_coverage to band center). "
            f"Consider widening FF_STRENGTH_GRID / INHIB_WEIGHT_RATIO_GRID."
        )

    if chosen.n_silent:
        print(
            f"   (note: chosen combo still had {chosen.n_silent} silent probed level(s) — a "
            f"known, separate FEEDFORWARD_STRENGTH edge case, not fixed by inhibition tuning)"
        )

    return chosen.ff_strength, chosen.inhib_ratio


# =============================================================================
# Stage 2 — RIDGE_ALPHA
# =============================================================================

def collect_states(net: Network, levels: Sequence) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for lv in levels:
        X.append(run_trial(net, lv))
        y.append(home_bin(lv))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def mean_circular_error_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errs = [circular_bin_error(int(t), int(p))[1] for t, p in zip(y_true, y_pred)]
    return float(np.mean(errs))


def tune_ridge_alpha(ff_strength: float, inhib_ratio: float, reservoir_scale: float = RESERVOIR_SCALE) -> float:
    print("\n=== Stage 2: RIDGE_ALPHA (cross-validated on TRAIN split only) ===")

    if STAGE2_TRAIN_SUBSAMPLE is None:
        subsample = list(train_set)
        print(
            f"feedforward_strength={ff_strength}, inhib_weight_ratio={inhib_ratio}, "
            f"reservoir_scale={reservoir_scale} (Stage 1 winners, all fixed). Using the FULL "
            f"train split ({len(subsample)} levels) for {STAGE2_KFOLDS}-fold CV; the test "
            f"split is not touched during this search.\n"
        )
    else:
        subsample = stratified_subsample(train_set, STAGE2_TRAIN_SUBSAMPLE, seed=STAGE2_CV_SEED)
        print(
            f"feedforward_strength={ff_strength}, inhib_weight_ratio={inhib_ratio}, "
            f"reservoir_scale={reservoir_scale} (Stage 1 winners, all fixed). Using a "
            f"stratified subsample of {STAGE2_TRAIN_SUBSAMPLE} train levels for "
            f"{STAGE2_KFOLDS}-fold CV speed; the test split is not touched during this search.\n"
        )

    w_ff, w_res = init_weights(ff_strength, reservoir_scale, inhib_weight_ratio=inhib_ratio)
    net = build_reservoir_a(w_ff, w_res)

    print(f"collecting reservoir states for {len(subsample)} train levels...")
    X, y = collect_states(net, subsample)
    print(f"done. X shape={X.shape}\n")

    kfold = KFold(n_splits=STAGE2_KFOLDS, shuffle=True, random_state=STAGE2_CV_SEED)

    header = f"{'alpha':>10} | {'cv_mean_deg':>11} | {'cv_accuracy':>11}"
    print(header)
    print("-" * len(header))

    cv_scores: dict[float, float] = {}
    for alpha in RIDGE_ALPHA_GRID:
        fold_deg, fold_acc = [], []
        for train_idx, val_idx in kfold.split(X):
            clf = RidgeClassifier(alpha=alpha).fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[val_idx])
            fold_deg.append(mean_circular_error_deg(y[val_idx], pred))
            fold_acc.append(float((pred == y[val_idx]).mean()))
        cv_scores[alpha] = float(np.mean(fold_deg))
        print(f"{alpha:>10} | {cv_scores[alpha]:>11.2f} | {np.mean(fold_acc):>11.2%}")

    best_alpha = min(cv_scores, key=cv_scores.get)
    print(f"\n-> chosen RIDGE_ALPHA = {best_alpha} (lowest mean CV circular error: {cv_scores[best_alpha]:.2f} deg)")

    if RUN_FINAL_TEST_CHECK:
        print(
            "\nrunning ONE-TIME check on the real test split (informational only — "
            "alpha was already chosen above without looking at this)..."
        )
        clf = RidgeClassifier(alpha=best_alpha).fit(X, y)
        X_test, y_test = collect_states(net, test_set)
        pred_test = clf.predict(X_test)
        test_deg = mean_circular_error_deg(y_test, pred_test)
        test_acc = float((pred_test == y_test).mean())
        print(f"test split (n={len(test_set)}): mean circular error = {test_deg:.2f} deg | accuracy = {test_acc:.2%}")

    return best_alpha


# =============================================================================
# Stage 3 — confirm RESERVOIR_SCALE (not searched, just verified)
# =============================================================================

def verify_reservoir_scale(ff_strength: float, inhib_ratio: float, reservoir_scale: float = RESERVOIR_SCALE) -> bool:
    print("\n=== Stage 3: confirm RESERVOIR_SCALE ===")
    levels = sample_levels_all_difficulties(train_set, STAGE3_LEVELS_PER_DIFFICULTY)
    print(
        f"RESERVOIR_SCALE={reservoir_scale} was never searched — it's used as-is throughout "
        f"Stages 1-2 above. This just checks it's actually a sound choice: the same avalanche/"
        f"coverage diagnostic as Stage 1, but on a broader sample ({len(levels)} levels, "
        f"{STAGE3_LEVELS_PER_DIFFICULTY} per difficulty) for more confidence.\n"
    )

    w_ff, w_res = init_weights(ff_strength, reservoir_scale, inhib_weight_ratio=inhib_ratio)
    net = build_reservoir_a(w_ff, w_res)

    diags = [diagnose_trial(net, lv) for lv in levels]

    header = f"{'diff':8s} {'T':>5} {'total':>7} {'nz_steps':>9} {'max_1step':>10}"
    print(header)
    for d in diags:
        print(f"{d.difficulty:8s} {d.T:5d} {d.total_spikes:7.0f} {d.nonzero_steps:9d} {d.max_spikes_in_one_step:10d}")

    max_bursts = np.array([d.max_spikes_in_one_step for d in diags])
    n_avalanche = int((max_bursts > AVALANCHE_THRESHOLD).sum())
    n_silent = sum(1 for d in diags if d.total_spikes == 0)
    print(
        f"\nacross {len(diags)} levels: max_spikes_in_one_step mean={max_bursts.mean():.1f} "
        f"median={np.median(max_bursts):.0f} max={max_bursts.max()} (threshold={AVALANCHE_THRESHOLD})"
    )
    print(f"avalanche-like levels: {n_avalanche}/{len(diags)} | completely silent levels: {n_silent}/{len(diags)}")

    ok = n_avalanche == 0
    verdict = "PASS" if ok else "FAIL"
    print(
        f"\n-> RESERVOIR_SCALE={reservoir_scale} verdict: {verdict}"
        + ("" if ok else " — avalanches still present, consider a lower RESERVOIR_SCALE")
    )
    if n_silent:
        print(
            f"   (note: {n_silent} silent level(s) is the same known FEEDFORWARD_STRENGTH "
            f"edge case flagged in Stage 1, not a RESERVOIR_SCALE problem)"
        )
    return ok


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ff_strength, inhib_ratio = choose_ff_and_inhib_ratio()
    ridge_alpha = tune_ridge_alpha(ff_strength, inhib_ratio)
    verify_reservoir_scale(ff_strength, inhib_ratio)

    print("\n=== Recommended Reservoir A hyperparameters ===")
    print(f"FEEDFORWARD_STRENGTH = {ff_strength}  (searched, Stage 1)")
    print(f"INHIB_WEIGHT_RATIO   = {inhib_ratio}  (searched, Stage 1)")
    print(f"RIDGE_ALPHA          = {ridge_alpha}  (searched, Stage 2)")
    print(f"RESERVOIR_SCALE      = {RESERVOIR_SCALE}  (fixed, confirmed in Stage 3 — not searched)")
    print(f"INHIB_FRACTION       = {INHIB_FRACTION}  (fixed, not searched — standard 80/20 ratio)")
    print(
        f"\nReminder: these numbers assume Models.ipynb's WMIN/WMAX clamp on network A's "
        f"connections is set to ({WMIN}, {WMAX}), matching this script. If you change the "
        f"clamp in the notebook again, update WMIN/WMAX here too and rerun."
    )


if __name__ == "__main__":
    main()
