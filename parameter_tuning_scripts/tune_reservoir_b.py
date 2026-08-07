"""
Hyperparameter tuning for Reservoir B (plastic LIF reservoir with STDP) only.

Standalone script — does not import or modify Models.ipynb or tune_reservoir_a.py.
Encoding/dataset parameters are copied verbatim from Models.ipynb's hyperparameter
cell; keep them in sync by hand.

FEEDFORWARD_STRENGTH, RESERVOIR_SCALE, RIDGE_ALPHA, INHIB_FRACTION, and
INHIB_WEIGHT_RATIO are all FIXED here at the values tune_reservoir_a.py already
found for Reservoir A — they are not re-tuned by this script. Only the STDP-
specific knobs are searched, in this order, each stage holding the previous
stage's winner fixed:

  Stage 1 — STDP_NU (learning rate pair, nu_pre and nu_post)
      The biggest lever: too low and B barely differs from A (no real
      learning); too high and STDP can push the same recurrent matrix whose
      avalanche-free balance took real work to establish (see
      tune_reservoir_a.py) back out of that balance. Sweeps (nu_pre, nu_post)
      pairs directly, varying the RATIO between depression and potentiation
      strength (from balanced 1:1 up to the notebook's own 100:1 default) —
      not just overall magnitude, since the 100:1 imbalance turned out to
      make potentiation dominate regardless of magnitude (see
      apply_dale_clamp's docstring). Each candidate gets a short plastic
      exposure (1 epoch per difficulty, no mixed rehearsal — cheap compared
      to the full schedule), then two checks: did the weights actually move
      (mean |Δw|), and is the reservoir's activity still healthy afterward
      (same avalanche/coverage diagnostic as script A). Picks the most
      balanced ratio that still passes both.

  Stage 2 — STDP_TC_PRE / STDP_TC_POST (trace time constants, moved together)
      With nu fixed, this is a "does it actually help" question rather than a
      stability one, so it's judged by downstream readout performance:
      k-fold CV (RidgeClassifier at the fixed RIDGE_ALPHA) after the same
      short exposure used in Stage 1. Picks the tc with the lowest mean CV
      circular error.

  Stage 3 — confirm the training schedule (STDP_EPOCHS_PER_DIFFICULTY /
      STDP_MIXED_EPOCHS_AFTER), NOT searched
      Runs the real, full curriculum schedule from pigeonpilot.curriculum's
      defaults (matching Models.ipynb) exactly once, with the Stage 1/2
      winners fixed, and reports final train/test performance plus a
      post-training health check. This is expensive (the full schedule is
      roughly 15-16k plastic trials) — it's a confirmation pass, not a grid
      search over schedule length.

Dale's-law-preserving clamp: WeightDependentPostPre is a generic Hebbian rule
with no notion of which synapses are "supposed to" stay inhibitory — given
enough correlated (pre, post) activity (which this reservoir has a lot of), it
will happily potentiate inhibitory synapses toward positive, eroding the E/I
balance tune_reservoir_a.py established and reintroducing the avalanche
dynamics that balance was built to fix. apply_dale_clamp() re-clamps the
recurrent connection after every plastic trial so inhibitory rows stay <= 0
and excitatory rows stay >= 0 — STDP can still change *how* strong a synapse
is, but never *which side of zero* it's on.

Run:
    python tune_reservoir_b.py
"""

from __future__ import annotations

import math
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
from bindsnet.learning import WeightDependentPostPre

from resources.paths import DEFAULT_HEADING_BINS, FULL_CIRCLE_DEG, home_heading_bin
from resources.encoding import encode_level
from resources.curriculum import (
    generate_curriculum_dataset,
    split_dataset,
    iter_train_schedule,
    DEFAULT_EPOCHS_PER_DIFFICULTY,
    DEFAULT_MIXED_EPOCHS_AFTER,
)


# =============================================================================
# Config — encoding / dataset params copied from Models.ipynb's hyperparameter
# cell. Keep these in sync by hand.
# =============================================================================

DATASET_SEED = 42
SPLIT_SEED = 0

N_INPUT_NODES = DEFAULT_HEADING_BINS
N_RESERVOIR_NODES = 100
NETWORK_DT = 1.0
ENCODING_DT = NETWORK_DT
TARGET_STEPS_LONGEST = 1000
INPUT_RATE_HZ = 40.0
ENCODING_SEED = 0
TRAILING_SILENCE = 20
WEIGHT_SEED = 0

LIF_KW = dict(rest=-65.0, reset=-65.0, thresh=-52.0, refrac=5, tc_decay=250.0)

# =============================================================================
# FIXED parameters — already tuned by tune_reservoir_a.py, NOT re-tuned here.
# =============================================================================

FEEDFORWARD_STRENGTH = 50
RESERVOIR_SCALE = 0.9
RIDGE_ALPHA = 0.01
INHIB_FRACTION = 0.2
INHIB_WEIGHT_RATIO = 1.0
WMIN, WMAX = 0.0, 10.0                       # feedforward connection clamp
RESERVOIR_WMIN, RESERVOIR_WMAX = -10.0, 10.0  # recurrent connection clamp (allows inhibition)

# A neuron cannot fire faster than once per (refrac + dt): the hard ceiling on
# per-neuron mean rate used by the coverage/saturation check.
REFRACTORY_CEILING = NETWORK_DT / (LIF_KW["refrac"] + NETWORK_DT)

# A single-timestep spike count above this fraction of the reservoir counts as
# an "avalanche" (synchronized burst) — same check and threshold as script A.
AVALANCHE_FRACTION = 0.3
AVALANCHE_THRESHOLD = int(AVALANCHE_FRACTION * N_RESERVOIR_NODES)

STAGE1_COVERAGE_BAND = (0.20, 0.50)
STAGE1_ACTIVE_EPS = 1e-6
STAGE1_SATURATION_MARGIN = 0.85
STAGE1_SATURATION_LIMIT = 0.25

# =============================================================================
# STDP search grids and settings — edit these to widen/narrow the search.
# =============================================================================

# (nu_pre, nu_post) candidate pairs, tested directly rather than one overall
# multiplier. The notebook's own default (1e-5, 1e-3) has a 100:1 imbalance
# between depression (nu_pre) and potentiation (nu_post) strength — that
# turned out to make potentiation dominate regardless of overall magnitude:
# scaling the whole pair down just slowed the same one-directional drift, it
# never changed direction (inhibitory weights eroded to exactly zero even at
# 10x weaker than default).
#
# Three questions this grid answers:
#   1. Ratio, potentiation-dominant (nu_pre < nu_post) — original direction,
#      from fully balanced (1:1) up to the notebook's own imbalance (100:1).
#   2. Ratio, depression-dominant (nu_pre > nu_post) — the untested mirror
#      case. Never checked before; could be fine, could fail differently
#      (e.g. weights collapsing toward zero instead of avalanching).
#   3. Magnitude at the balanced 1:1 point — is 0.001 special, or does any
#      balanced pair work as long as it's balanced?
STDP_NU_GRID = [
    # 1) potentiation-dominant (already tested)
    (1e-5, 1e-3),   # 100:1 — Models.ipynb's original default
    (3e-5, 1e-3),   # ~33:1
    (1e-4, 1e-3),   # 10:1
    (3e-4, 1e-3),   # ~3:1
    # 3) balanced, different magnitudes
    (1e-4, 1e-4),   # 10x smaller than chosen value, still balanced
    (1e-3, 1e-3),   # 1:1, fully balanced — the value we chose
    (1e-2, 1e-2),   # 10x larger, still balanced
    (1e-1, 1e-1),   # 100x larger, still balanced
    # 2) depression-dominant (mirror direction — NEW)
    (3e-4, 1e-4),   # ~3:1 reversed
    (1e-3, 1e-4),   # 10:1 reversed
    (1e-3, 3e-5),   # ~33:1 reversed
    (1e-3, 1e-5),   # 100:1 reversed — mirror of the original default
]
BASE_STDP_TC = 20.0                                   # held fixed while nu is searched (Stage 1)

STDP_TC_GRID = [10.0, 20.0, 40.0, 80.0]

# Short plastic exposure shared by Stage 1 and Stage 2: 1 epoch per difficulty,
# no mixed rehearsal — cheap compared to the full schedule, structured the
# same way (curriculum-ordered) rather than a flat random sample.
SHORT_EXPOSURE_MIXED_AFTER = 0

STAGE1_HEALTH_LEVELS_PER_DIFFICULTY = 3
STAGE1_MIN_MEAN_DW = 1e-4   # mean |Δw| must clear this to count as "real learning happened"
STAGE1_SEED = 0

STAGE2_CV_SUBSAMPLE = 150
STAGE2_KFOLDS = 5
STAGE2_CV_SEED = 0

STAGE3_SEED = 0
STAGE3_VERBOSE_EVERY = 1000  # progress print interval during the full curriculum (it's long)

# Reservoir A's final test numbers, for reference when interpreting Stage 3's
# result (from tune_reservoir_a.py's tuning — not recomputed here).
RESERVOIR_A_TEST_DEG = 35.17
RESERVOIR_A_TEST_ACC = 0.2245


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

SHORT_EXPOSURE_EPOCHS = {d: 1 for d in DIFFICULTY_ORDER}

print(
    f"dataset: {len(dataset)} levels | train={len(train_set)} test={len(test_set)} "
    f"| velocity={ENCODING_VELOCITY:.4f}"
)
print(
    f"fixed from Reservoir A: FF={FEEDFORWARD_STRENGTH}, scale={RESERVOIR_SCALE}, "
    f"ridge_alpha={RIDGE_ALPHA}, inhib_fraction={INHIB_FRACTION}, inhib_ratio={INHIB_WEIGHT_RATIO}"
)


# =============================================================================
# Shared helpers
# =============================================================================

def home_bin(level, heading_bins: int = DEFAULT_HEADING_BINS) -> int:
    return home_heading_bin(level, heading_bins)


def circular_bin_error(true_bin: int, pred_bin: int, heading_bins: int = DEFAULT_HEADING_BINS) -> tuple[int, float]:
    d = abs(int(true_bin) - int(pred_bin)) % heading_bins
    d = min(d, heading_bins - d)
    return d, d * (FULL_CIRCLE_DEG / heading_bins)


def mean_circular_error_deg(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    errs = [circular_bin_error(int(t), int(p))[1] for t, p in zip(y_true, y_pred)]
    return float(np.mean(errs))


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


def init_weights(seed: int = WEIGHT_SEED) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same construction as Models.ipynb's init_weights (with the Dale's-law
    split from tune_reservoir_a.py), using the FIXED parameters above. Also
    returns inhibitory_idx so callers can keep enforcing the E/I split during
    plastic training (see apply_dale_clamp)."""
    generator = torch.Generator().manual_seed(seed)

    feedforward_weights = torch.rand(N_INPUT_NODES, N_RESERVOIR_NODES, generator=generator)
    w_ff = FEEDFORWARD_STRENGTH * feedforward_weights / feedforward_weights.sum(dim=0, keepdim=True)

    reservoir_weights = torch.rand(N_RESERVOIR_NODES, N_RESERVOIR_NODES, generator=generator)
    reservoir_weights.fill_diagonal_(0.0)

    n_inhibitory = int(round(INHIB_FRACTION * N_RESERVOIR_NODES))
    inhibitory_idx = torch.randperm(N_RESERVOIR_NODES, generator=generator)[:n_inhibitory]
    reservoir_weights[inhibitory_idx, :] *= -INHIB_WEIGHT_RATIO

    spectral_radius = torch.linalg.eigvals(reservoir_weights).abs().max().item()
    w_res = reservoir_weights * (RESERVOIR_SCALE / spectral_radius)

    return w_ff, w_res, inhibitory_idx


def apply_dale_clamp(net: Network, inhibitory_idx: torch.Tensor) -> None:
    """Re-clamp the recurrent connection so inhibitory rows stay <= 0 and
    excitatory rows stay >= 0. WeightDependentPostPre has no notion of Dale's
    law on its own — it potentiates whatever fires together, including
    inhibitory synapses, which erodes the E/I balance over many plastic
    trials (see module docstring). Call this after every trial trained with
    learning=True."""
    w = net.connections[("Reservoir", "Reservoir")].w
    inhibitory_mask = torch.zeros(w.shape[0], dtype=torch.bool)
    inhibitory_mask[inhibitory_idx] = True
    w[inhibitory_mask, :] = w[inhibitory_mask, :].clamp(min=RESERVOIR_WMIN, max=0.0)
    w[~inhibitory_mask, :] = w[~inhibitory_mask, :].clamp(min=0.0, max=RESERVOIR_WMAX)


def build_reservoir_b(
    w_ff: torch.Tensor,
    w_res: torch.Tensor,
    nu: tuple[float, float],
    tc_pre: float,
    tc_post: float,
) -> Network:
    """Plastic reservoir (STDP on both connections), network B only."""
    net = Network(dt=NETWORK_DT)
    input_layer = Input(n=N_INPUT_NODES, traces=True)
    reservoir = LIFNodes(n=N_RESERVOIR_NODES, traces=True, **LIF_KW)
    net.add_layer(input_layer, name="Input")
    net.add_layer(reservoir, name="Reservoir")

    stdp_kwargs = dict(update_rule=WeightDependentPostPre, nu=nu, tc_pre=tc_pre, tc_post=tc_post)
    ff_conn = Connection(source=input_layer, target=reservoir, w=w_ff.clone(), wmin=WMIN, wmax=WMAX, **stdp_kwargs)
    res_conn = Connection(
        source=reservoir, target=reservoir, w=w_res.clone(),
        wmin=RESERVOIR_WMIN, wmax=RESERVOIR_WMAX, **stdp_kwargs,
    )
    net.add_connection(ff_conn, source="Input", target="Reservoir")
    net.add_connection(res_conn, source="Reservoir", target="Reservoir")

    net.add_monitor(Monitor(obj=reservoir, state_vars=["s"], time=None), name="Reservoir Spikes")
    return net


def run_trial(net: Network, level, learning: bool) -> np.ndarray:
    """Encode -> run -> pooled per-neuron mean rate. Resets state after."""
    spikes = encode_level_for_network(level)
    inputs = to_bindsnet_input(spikes)
    net.train(learning)
    net.run(inputs={"Input": inputs}, time=spikes.shape[0])
    state = pool_reservoir_state(net.monitors["Reservoir Spikes"].get("s"))
    net.reset_state_variables()
    return state


def run_trial_full(net: Network, level) -> tuple[torch.Tensor, int]:
    """Encode -> run (learning off) -> FULL (T, 1, n_res) spike tensor. For
    diagnostics only. Resets state after."""
    spikes = encode_level_for_network(level)
    inputs = to_bindsnet_input(spikes)
    net.train(False)
    net.run(inputs={"Input": inputs}, time=spikes.shape[0])
    s = net.monitors["Reservoir Spikes"].get("s").clone()
    net.reset_state_variables()
    return s, spikes.shape[0]


@dataclass
class TrialDiagnostics:
    difficulty: str
    T: int
    total_spikes: float
    mean_rates: np.ndarray
    max_spikes_in_one_step: int
    nonzero_steps: int


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


def _spike_health(mean_rates: np.ndarray) -> tuple[float, float, float]:
    active = mean_rates > STAGE1_ACTIVE_EPS
    coverage = float(active.mean())
    if not active.any():
        return coverage, 0.0, 0.0
    active_rates = mean_rates[active]
    mean_active_rate = float(active_rates.mean())
    saturated = active_rates >= STAGE1_SATURATION_MARGIN * REFRACTORY_CEILING
    saturation_frac = float(saturated.mean())
    return coverage, mean_active_rate, saturation_frac


def levels_for_difficulty(levels: Sequence, difficulty: str, n: int) -> list:
    return [lv for lv in levels if lv.difficulty == difficulty][:n]


def sample_levels_all_difficulties(levels: Sequence, n_per_difficulty: int) -> list:
    return [lv for d in DIFFICULTY_ORDER for lv in levels_for_difficulty(levels, d, n_per_difficulty)]


def stratified_subsample(levels: Sequence, n: int, seed: int) -> list:
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


def collect_states(net: Network, levels: Sequence) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for lv in levels:
        X.append(run_trial(net, lv, learning=False))
        y.append(home_bin(lv))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def train_exposure(
    net: Network,
    epochs_per_difficulty: dict,
    mixed_epochs_after: int,
    seed: int,
    inhibitory_idx: torch.Tensor,
    verbose_every: int | None = None,
) -> int:
    """Runs iter_train_schedule's curriculum ordering through `net` with
    learning on. Returns total trial count. Leaves net in eval mode after."""
    n = 0
    for phase, phase_epoch, _, level in iter_train_schedule(
        train_set,
        mode="curriculum",
        epochs_per_difficulty=epochs_per_difficulty,
        n_epochs_mixed_after=mixed_epochs_after,
        seed=seed,
    ):
        run_trial(net, level, learning=True)
        apply_dale_clamp(net, inhibitory_idx)
        n += 1
        if verbose_every and n % verbose_every == 0:
            print(f"  ... {n} plastic trials (phase={phase}, epoch={phase_epoch})")
    net.train(False)
    return n


# =============================================================================
# Stage 1 — STDP_NU
# =============================================================================

@dataclass
class Stage1Result:
    nu: tuple[float, float]
    mean_dw_ff: float
    mean_dw_rec: float
    easy_coverage: float
    hard_saturation: float
    max_burst: int
    n_avalanche: int
    ratio: float = field(init=False)  # nu_post / nu_pre — potentiation vs. depression imbalance
    ok: bool = field(init=False)

    def __post_init__(self):
        self.ratio = self.nu[1] / self.nu[0]
        lo, hi = STAGE1_COVERAGE_BAND
        learned = self.mean_dw_ff > STAGE1_MIN_MEAN_DW or self.mean_dw_rec > STAGE1_MIN_MEAN_DW
        healthy = (
            lo <= self.easy_coverage <= hi
            and self.hard_saturation <= STAGE1_SATURATION_LIMIT
            and self.n_avalanche == 0
        )
        self.ok = learned and healthy


def choose_stdp_nu() -> tuple[float, float]:
    print("\n=== Stage 1: STDP_NU ===")
    health_levels = sample_levels_all_difficulties(train_set, STAGE1_HEALTH_LEVELS_PER_DIFFICULTY)
    print(
        f"Fixed init from Reservoir A's tuning; TC held at baseline {BASE_STDP_TC} (searched in "
        f"Stage 2). Sweeping (nu_pre, nu_post) pairs directly rather than one overall multiplier "
        f"— the notebook's 100:1 default imbalance between depression and potentiation strength "
        f"made potentiation dominate regardless of overall magnitude, eroding inhibitory weights "
        f"to zero even at 10x weaker than default. This grid instead varies the RATIO between "
        f"them, from balanced (1:1) up to the original imbalance (100:1). Each candidate gets a "
        f"short plastic exposure (1 epoch per difficulty, no mixed rehearsal), then checks: did "
        f"the weights actually move (mean |Δw|), and is the reservoir's activity still healthy "
        f"afterward (same avalanche/coverage diagnostic as tune_reservoir_a.py).\n"
    )
    print(
        f"target: mean |Δw| > {STAGE1_MIN_MEAN_DW} (real learning happened), easy coverage in "
        f"{STAGE1_COVERAGE_BAND}, hard saturation <= {STAGE1_SATURATION_LIMIT}, no probed level "
        f"exceeds {AVALANCHE_THRESHOLD} ({AVALANCHE_FRACTION:.0%}) neurons firing in one step\n"
    )

    results: list[Stage1Result] = []
    header = (
        f"{'nu_pre':>8} | {'nu_post':>8} | {'ratio':>7} | {'dw_ff':>8} | {'dw_rec':>8} | "
        f"{'easy_cov':>8} | {'hard_sat':>8} | {'max_burst':>9} | {'n_avalanche':>11} | ok?"
    )
    print(header)
    print("-" * len(header))

    for nu in STDP_NU_GRID:
        w_ff, w_res, inhibitory_idx = init_weights()
        net = build_reservoir_b(w_ff, w_res, nu=nu, tc_pre=BASE_STDP_TC, tc_post=BASE_STDP_TC)

        w_ff_before = net.connections[("Input", "Reservoir")].w.detach().clone()
        w_res_before = net.connections[("Reservoir", "Reservoir")].w.detach().clone()

        train_exposure(
            net, SHORT_EXPOSURE_EPOCHS, SHORT_EXPOSURE_MIXED_AFTER,
            seed=STAGE1_SEED, inhibitory_idx=inhibitory_idx,
        )

        dw_ff = float((net.connections[("Input", "Reservoir")].w.detach() - w_ff_before).abs().mean())
        dw_rec = float((net.connections[("Reservoir", "Reservoir")].w.detach() - w_res_before).abs().mean())

        diags = [diagnose_trial(net, lv) for lv in health_levels]
        easy_cov = float(np.mean([_spike_health(d.mean_rates)[0] for d in diags if d.difficulty == EASY_DIFFICULTY]))
        hard_sat = float(np.mean([_spike_health(d.mean_rates)[2] for d in diags if d.difficulty == HARD_DIFFICULTY]))
        max_burst = max(d.max_spikes_in_one_step for d in diags)
        n_avalanche = sum(1 for d in diags if d.max_spikes_in_one_step > AVALANCHE_THRESHOLD)

        res = Stage1Result(
            nu=nu, mean_dw_ff=dw_ff, mean_dw_rec=dw_rec,
            easy_coverage=easy_cov, hard_saturation=hard_sat,
            max_burst=max_burst, n_avalanche=n_avalanche,
        )
        results.append(res)
        print(
            f"{nu[0]:>8.1e} | {nu[1]:>8.1e} | {res.ratio:>7.1f} | {dw_ff:>8.5f} | {dw_rec:>8.5f} | "
            f"{easy_cov:>8.2%} | {hard_sat:>8.2%} | {max_burst:>9} | {n_avalanche:>11} | "
            f"{'yes' if res.ok else 'no'}"
        )

    candidates = [r for r in results if r.ok]
    if candidates:
        # Most balanced ratio first — measured as distance from 1:1 in LOG
        # space, since ratio=2 (potentiation-dominant) and ratio=0.5
        # (depression-dominant) are equally imbalanced multiplicatively; a
        # plain min(ratio) would wrongly favor small fractional ratios now
        # that the grid includes the depression-dominant (ratio<1) mirror
        # cases. Then largest magnitude among ties (most learning per unit
        # of exposure).
        chosen = min(candidates, key=lambda r: (abs(math.log(r.ratio)), -r.nu[1]))
        print(
            f"\n-> chosen STDP_NU = {chosen.nu} (ratio {chosen.ratio:.2f}:1 — most balanced "
            f"candidate meeting target, largest magnitude among ties)"
        )
    else:
        # Fewest avalanches first; among ties, LEAST weight change (the
        # safest, least aggressive option) rather than the most, then most
        # balanced ratio.
        chosen = min(
            results,
            key=lambda r: (r.n_avalanche, max(r.mean_dw_ff, r.mean_dw_rec), abs(math.log(r.ratio))),
        )
        print(
            f"\n! no (nu_pre, nu_post) pair met both the learning and health targets — falling "
            f"back to nu={chosen.nu} (fewest avalanches, then least weight change, i.e. the "
            f"safest candidate). Consider widening STDP_NU_GRID toward smaller magnitudes."
        )

    return chosen.nu


# =============================================================================
# Stage 2 — STDP_TC_PRE / STDP_TC_POST
# =============================================================================

def choose_stdp_tc(nu: tuple[float, float]) -> float:
    print("\n=== Stage 2: STDP_TC_PRE / STDP_TC_POST ===")
    print(
        f"STDP_NU fixed at {nu} (Stage 1 winner). Each candidate gets the same short exposure "
        f"as Stage 1, then we fit RidgeClassifier(alpha={RIDGE_ALPHA}, fixed) on a "
        f"{STAGE2_CV_SUBSAMPLE}-level stratified train subsample and cross-validate. This is a "
        f"downstream-performance question, not a stability one (Stage 1 already screened nu for "
        f"stability), so tc is judged by what actually helps the readout.\n"
    )

    subsample = stratified_subsample(train_set, STAGE2_CV_SUBSAMPLE, seed=STAGE2_CV_SEED)
    kfold = KFold(n_splits=STAGE2_KFOLDS, shuffle=True, random_state=STAGE2_CV_SEED)

    header = f"{'tc':>6} | {'cv_mean_deg':>11} | {'cv_accuracy':>11}"
    print(header)
    print("-" * len(header))

    scores: dict[float, float] = {}
    for tc in STDP_TC_GRID:
        w_ff, w_res, inhibitory_idx = init_weights()
        net = build_reservoir_b(w_ff, w_res, nu=nu, tc_pre=tc, tc_post=tc)
        train_exposure(
            net, SHORT_EXPOSURE_EPOCHS, SHORT_EXPOSURE_MIXED_AFTER,
            seed=STAGE1_SEED, inhibitory_idx=inhibitory_idx,
        )

        X, y = collect_states(net, subsample)
        fold_deg, fold_acc = [], []
        for train_idx, val_idx in kfold.split(X):
            clf = RidgeClassifier(alpha=RIDGE_ALPHA).fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[val_idx])
            fold_deg.append(mean_circular_error_deg(y[val_idx], pred))
            fold_acc.append(float((pred == y[val_idx]).mean()))
        scores[tc] = float(np.mean(fold_deg))
        print(f"{tc:>6} | {scores[tc]:>11.2f} | {np.mean(fold_acc):>11.2%}")

    best_tc = min(scores, key=scores.get)
    print(f"\n-> chosen STDP_TC_PRE = STDP_TC_POST = {best_tc} (lowest mean CV circular error: {scores[best_tc]:.2f} deg)")
    return best_tc


# =============================================================================
# Stage 3 — confirm the training schedule (not searched)
# =============================================================================

def confirm_training_schedule(nu: tuple[float, float], tc: float) -> tuple[float, float]:
    print("\n=== Stage 3: confirm training schedule (full curriculum) ===")
    print(
        f"STDP_NU={nu}, STDP_TC_PRE=STDP_TC_POST={tc} (Stage 1/2 winners, fixed). "
        f"STDP_EPOCHS_PER_DIFFICULTY / STDP_MIXED_EPOCHS_AFTER are NOT searched — this runs the "
        f"real, full schedule ({DEFAULT_EPOCHS_PER_DIFFICULTY}, mixed_after="
        f"{DEFAULT_MIXED_EPOCHS_AFTER}) once, to confirm it's adequate rather than optimize it. "
        f"This is the expensive part — full curriculum, not a short exposure.\n"
    )

    w_ff, w_res, inhibitory_idx = init_weights()
    net = build_reservoir_b(w_ff, w_res, nu=nu, tc_pre=tc, tc_post=tc)

    n_trials = train_exposure(
        net, DEFAULT_EPOCHS_PER_DIFFICULTY, DEFAULT_MIXED_EPOCHS_AFTER,
        seed=STAGE3_SEED, inhibitory_idx=inhibitory_idx, verbose_every=STAGE3_VERBOSE_EVERY,
    )
    print(f"trained on {n_trials} plastic trials total\n")

    health_levels = sample_levels_all_difficulties(train_set, STAGE1_HEALTH_LEVELS_PER_DIFFICULTY)
    diags = [diagnose_trial(net, lv) for lv in health_levels]
    n_avalanche = sum(1 for d in diags if d.max_spikes_in_one_step > AVALANCHE_THRESHOLD)
    print(
        f"post-training health check: {n_avalanche}/{len(diags)} probed levels avalanche-like "
        f"({'PASS' if n_avalanche == 0 else 'FAIL'})\n"
    )

    print(f"collecting reservoir states for {len(train_set)} train + {len(test_set)} test levels...")
    X_train, y_train = collect_states(net, train_set)
    X_test, y_test = collect_states(net, test_set)

    clf = RidgeClassifier(alpha=RIDGE_ALPHA).fit(X_train, y_train)
    pred = clf.predict(X_test)
    test_deg = mean_circular_error_deg(y_test, pred)
    test_acc = float((pred == y_test).mean())
    print(
        f"\nFinal Model B test performance (n={len(test_set)}): "
        f"mean circular error = {test_deg:.2f} deg | accuracy = {test_acc:.2%}"
    )
    print(
        f"For reference, Reservoir A (fixed, no STDP) scored {RESERVOIR_A_TEST_DEG:.2f} deg / "
        f"{RESERVOIR_A_TEST_ACC:.2%} on the same test split (from tune_reservoir_a.py) — compare "
        f"against that, not chance, to see whether STDP actually helped."
    )
    return test_deg, test_acc


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    nu = choose_stdp_nu()
    tc = choose_stdp_tc(nu)
    test_deg, test_acc = confirm_training_schedule(nu, tc)

    print("\n=== Recommended Reservoir B (STDP) hyperparameters ===")
    print(f"STDP_NU      = {nu}  (searched, Stage 1)")
    print(f"STDP_TC_PRE  = {tc}  (searched, Stage 2)")
    print(f"STDP_TC_POST = {tc}  (searched, Stage 2, tied to STDP_TC_PRE)")
    print(
        "STDP_EPOCHS_PER_DIFFICULTY / STDP_MIXED_EPOCHS_AFTER = resources.curriculum defaults "
        "(confirmed in Stage 3, not searched)"
    )
    print(
        f"\nFixed (from Reservoir A's tuning, not touched here): "
        f"FEEDFORWARD_STRENGTH={FEEDFORWARD_STRENGTH}, RESERVOIR_SCALE={RESERVOIR_SCALE}, "
        f"RIDGE_ALPHA={RIDGE_ALPHA}, INHIB_FRACTION={INHIB_FRACTION}, "
        f"INHIB_WEIGHT_RATIO={INHIB_WEIGHT_RATIO}"
    )
    print(f"Final test performance: {test_deg:.2f} deg / {test_acc:.2%} accuracy")


if __name__ == "__main__":
    main()
