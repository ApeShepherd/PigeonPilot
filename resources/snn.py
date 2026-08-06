"""
SNN stack helpers for PigeonPilot — Model A (fixed) and Model B (STDP).

Shared by ``Models.ipynb`` and the playground: encode → run trial → readout,
plus checkpoint save/load for both pigeons. Mirrors the hyperparameter contract
from ``Models.ipynb``; encoding knobs must match a saved checkpoint or live
inference will silently drift off the training distribution.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .encoding import encode_level
from .paths import (
    DEFAULT_HEADING_BINS,
    FULL_CIRCLE_DEG,
    Level,
    home_heading_bin,
)

# ---------------------------------------------------------------------------
# BindsNET / torch (optional until first use)
# ---------------------------------------------------------------------------

_BINDSNET_READY = False


def ensure_bindsnet() -> None:
    """Install the torch._six shim and import BindsNET once."""
    global _BINDSNET_READY
    if _BINDSNET_READY:
        return

    import collections.abc

    six_shim = types.ModuleType("torch._six")
    six_shim.container_abcs = collections.abc
    six_shim.string_classes = (str,)
    six_shim.int_classes = (int,)
    six_shim.inf = float("inf")
    sys.modules["torch._six"] = six_shim
    _BINDSNET_READY = True


@dataclass
class ReservoirConfig:
    """Hyperparameters needed to encode levels and rebuild frozen networks."""

    n_input: int = DEFAULT_HEADING_BINS
    n_reservoir: int = 10_000
    network_dt: float = 1.0
    encoding_dt: float = 1.0
    encoding_velocity: float = 1.0
    input_rate_hz: float = 40.0
    encoding_seed: int = 0
    trailing_silence: int = 20
    weight_seed: int = 0
    feedforward_strength: float = 50.0
    reservoir_scale: float = 0.9
    wmin: float = 0.0
    wmax: float = 10.0
    inhib_fraction: float = 0.2
    inhib_weight_ratio: float = 1.0
    ridge_alpha: float = 0.01
    lif: dict[str, float] = field(
        default_factory=lambda: {
            "rest": -65.0,
            "reset": -65.0,
            "thresh": -52.0,
            "refrac": 5.0,
            "tc_decay": 250.0,
        }
    )
    stdp_nu: tuple[float, float] = (1e-3, 1e-3)
    stdp_tc_pre: float = 20.0
    stdp_tc_post: float = 20.0

    @property
    def reservoir_wmin(self) -> float:
        return -self.wmax

    @property
    def reservoir_wmax(self) -> float:
        return self.wmax

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["stdp_nu"] = list(self.stdp_nu)
        return d

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> ReservoirConfig:
        payload = dict(data)
        if "stdp_nu" in payload:
            payload["stdp_nu"] = tuple(payload["stdp_nu"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class CheckpointBundle:
    """Loaded network(s), classifier(s), config, and optional jury metrics.

    Model B (STDP) is optional — a reservoir-only run (no plasticity, see
    e.g. Reservoir_only_staged.ipynb) saves/loads with network_b=None,
    classifier_b=None. Playground's model="A" path never touches B, so an
    A-only bundle works there too; only model="B"/"both" require B.
    """

    config: ReservoirConfig
    network_a: Any
    classifier_a: Any
    network_b: Any = None
    classifier_b: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)
    # Directory this run was loaded from, so derived artefacts can be cached beside it.
    source: Optional[Path] = None


def home_bin(level: Level, heading_bins: int = DEFAULT_HEADING_BINS) -> int:
    """Classification label: heading bin of ``home_xy``."""
    return home_heading_bin(level, heading_bins)


def circular_bin_error(
    true_bin: int,
    pred_bin: int,
    heading_bins: int = DEFAULT_HEADING_BINS,
) -> tuple[int, float]:
    """Return ``(bin_error, degrees)`` on the circular heading ring."""
    d = abs(int(true_bin) - int(pred_bin)) % heading_bins
    d = min(d, heading_bins - d)
    return d, d * (FULL_CIRCLE_DEG / heading_bins)


def bin_to_heading_deg(bin_idx: int, heading_bins: int = DEFAULT_HEADING_BINS) -> float:
    """Convert a heading bin index to compass degrees (bin center)."""
    step = FULL_CIRCLE_DEG / heading_bins
    return float((int(bin_idx) % heading_bins) * step)


def heading_to_unit(heading_deg: float) -> np.ndarray:
    """Compass degrees → unit ``(x, y)`` with ``0°`` = North ``(+y)``."""
    rad = np.deg2rad(heading_deg)
    return np.array([np.sin(rad), np.cos(rad)], dtype=float)


def encode_level_for_network(level: Level, config: ReservoirConfig) -> np.ndarray:
    """Poisson rate-code a level, then pad trailing silence (Models.ipynb contract)."""
    spikes = encode_level(
        level,
        velocity=config.encoding_velocity,
        dt=config.encoding_dt,
        heading_bins=config.n_input,
        rate_hz=config.input_rate_hz,
        seed=config.encoding_seed + int(level.level_id),
    )
    if config.trailing_silence > 0:
        pad = np.zeros((config.trailing_silence, config.n_input), dtype=np.float32)
        spikes = np.vstack([spikes, pad])
    return spikes


def to_bindsnet_input(spikes: np.ndarray):
    import torch

    return torch.from_numpy(np.asarray(spikes, dtype=np.float32)).unsqueeze(1)


def pool_reservoir_state(spike_tensor) -> np.ndarray:
    return spike_tensor.float().mean(dim=0).squeeze(0).detach().cpu().numpy()


def init_weights(config: ReservoirConfig, seed: Optional[int] = None):
    """Dale-law recurrent init + spectral-radius rescale (Models.ipynb)."""
    ensure_bindsnet()
    import torch

    n_input = config.n_input
    n_res = config.n_reservoir
    generator = torch.Generator().manual_seed(config.weight_seed if seed is None else seed)

    feedforward_weights = torch.rand(n_input, n_res, generator=generator)
    w_ff = (
        config.feedforward_strength
        * feedforward_weights
        / feedforward_weights.sum(dim=0, keepdim=True)
    )

    reservoir_weights = torch.rand(n_res, n_res, generator=generator)
    reservoir_weights.fill_diagonal_(0.0)

    n_inhibitory = int(round(config.inhib_fraction * n_res))
    inhibitory_idx = torch.randperm(n_res, generator=generator)[:n_inhibitory]
    reservoir_weights[inhibitory_idx, :] *= -config.inhib_weight_ratio

    spectral_radius = torch.linalg.eigvals(reservoir_weights).abs().max().item()
    w_res = reservoir_weights * (config.reservoir_scale / spectral_radius)
    return w_ff, w_res


def build_reservoir(
    *,
    config: ReservoirConfig,
    plastic: bool,
    w_ff,
    w_res,
):
    """Build a BindsNET LIF reservoir; ``plastic=True`` attaches STDP rules."""
    ensure_bindsnet()
    import torch
    from bindsnet.learning import WeightDependentPostPre
    from bindsnet.network import Network
    from bindsnet.network.monitors import Monitor
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection

    n_input, n_res = w_ff.shape
    assert w_res.shape == (n_res, n_res)

    network = Network(dt=config.network_dt)
    input_layer = Input(n=n_input, traces=True)
    lif_kw = {
        "rest": config.lif["rest"],
        "reset": config.lif["reset"],
        "thresh": config.lif["thresh"],
        "refrac": int(config.lif["refrac"]),
        "tc_decay": config.lif["tc_decay"],
    }
    reservoir = LIFNodes(n=n_res, traces=True, **lif_kw)
    network.add_layer(input_layer, name="Input")
    network.add_layer(reservoir, name="Reservoir")

    feedforward_kwargs: dict[str, Any] = dict(wmin=config.wmin, wmax=config.wmax)
    recurrent_kwargs: dict[str, Any] = dict(
        wmin=config.reservoir_wmin, wmax=config.reservoir_wmax
    )
    if plastic:
        stdp_kwargs = dict(
            update_rule=WeightDependentPostPre,
            nu=config.stdp_nu,
            tc_pre=config.stdp_tc_pre,
            tc_post=config.stdp_tc_post,
        )
        feedforward_kwargs.update(stdp_kwargs)
        recurrent_kwargs.update(stdp_kwargs)

    network.add_connection(
        Connection(source=input_layer, target=reservoir, w=w_ff.clone(), **feedforward_kwargs),
        source="Input",
        target="Reservoir",
    )
    network.add_connection(
        Connection(source=reservoir, target=reservoir, w=w_res.clone(), **recurrent_kwargs),
        source="Reservoir",
        target="Reservoir",
    )
    network.add_monitor(
        Monitor(obj=reservoir, state_vars=["s"], time=None),
        name="Reservoir Spikes",
    )
    return network


def run_trial(net, level: Level, config: ReservoirConfig, *, learning: bool = False) -> np.ndarray:
    """Encode → run → pool mean rates → reset state (weights kept)."""
    spikes = encode_level_for_network(level, config)
    inputs = to_bindsnet_input(spikes)
    net.train(learning)
    net.run(inputs={"Input": inputs}, time=spikes.shape[0])
    state = pool_reservoir_state(net.monitors["Reservoir Spikes"].get("s"))
    net.reset_state_variables()
    return state


def predict_home_bin(net, classifier, level: Level, config: ReservoirConfig) -> int:
    """Frozen reservoir + RidgeClassifier → predicted home heading bin."""
    state = run_trial(net, level, config, learning=False)
    pred = classifier.predict(state.reshape(1, -1))
    return int(pred[0])


def _connection_weights(net) -> tuple[Any, Any]:
    w_ff = net.connections[("Input", "Reservoir")].w.detach().cpu().clone()
    w_res = net.connections[("Reservoir", "Reservoir")].w.detach().cpu().clone()
    return w_ff, w_res


DEFAULT_CHECKPOINT_ROOT = Path("outputs/checkpoints")
_LATEST_POINTER = "latest.txt"


def checkpoint_root(root: str | Path | None = None) -> Path:
    """Root folder that holds named A/B runs side by side."""
    return Path(root) if root is not None else DEFAULT_CHECKPOINT_ROOT


def resolve_run_dir(name: str = "latest", *, root: str | Path | None = None) -> Path:
    """Resolve a run name (or ``\"latest\"``) to its directory.

    Layout::

        outputs/checkpoints/
          latest.txt          # pointer: name of the current default run
          n10000_jury/        # one folder per training run (A + B inside)
          n1000_smoke/
    """
    base = checkpoint_root(root)
    if name == "latest":
        pointer = base / _LATEST_POINTER
        if not pointer.exists():
            # Backward compatible: folder literally named ``latest``
            legacy = base / "latest"
            if legacy.exists():
                return legacy
            raise FileNotFoundError(
                f"No latest pointer at {pointer} and no legacy folder {legacy}. "
                "Save a run with save_run(...) first."
            )
        name = pointer.read_text(encoding="utf-8").strip()
        if not name:
            raise FileNotFoundError(f"Empty latest pointer: {pointer}")
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint run not found: {path}")
    return path


def list_runs(*, root: str | Path | None = None) -> list[dict[str, Any]]:
    """List saved runs (newest first). Each entry has name, path, n_reservoir, metrics summary."""
    base = checkpoint_root(root)
    if not base.exists():
        return []

    latest_name: Optional[str] = None
    pointer = base / _LATEST_POINTER
    if pointer.exists():
        latest_name = pointer.read_text(encoding="utf-8").strip() or None

    runs: list[dict[str, Any]] = []
    for child in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        meta_path = child / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        cfg = meta.get("config") or {}
        metrics = meta.get("metrics") or {}
        runs.append(
            {
                "name": child.name,
                "path": child,
                "is_latest": child.name == latest_name,
                "n_reservoir": cfg.get("n_reservoir"),
                "metrics": metrics,
            }
        )
    return runs


def set_latest(name: str, *, root: str | Path | None = None) -> Path:
    """Point ``latest`` at an existing named run (does not copy files)."""
    if name in {"", "latest", _LATEST_POINTER}:
        raise ValueError(f"Pass a concrete run folder name, not {name!r}")
    base = checkpoint_root(root)
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint run not found: {path}")
    base.mkdir(parents=True, exist_ok=True)
    (base / _LATEST_POINTER).write_text(name + "\n", encoding="utf-8")
    return path


def save_checkpoint(
    directory: str | Path,
    *,
    network_a,
    classifier_a,
    network_b=None,
    classifier_b=None,
    config: ReservoirConfig,
    metrics: Optional[Mapping[str, Any]] = None,
    run_name: Optional[str] = None,
) -> Path:
    """Persist reservoir weights, sklearn classifier(s), config, and optional metrics.

    Model B is optional — pass network_b=None (the default) for a reservoir-only
    (no-plasticity) run; only A gets written. Prefer ``save_run(name, ...)`` so
    multiple trainings sit side by side under ``outputs/checkpoints/<name>/``.
    """
    ensure_bindsnet()
    import joblib
    import torch

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    w_ff_a, w_res_a = _connection_weights(network_a)
    weights_blob = {"A": {"w_ff": w_ff_a, "w_res": w_res_a}}
    if network_b is not None:
        w_ff_b, w_res_b = _connection_weights(network_b)
        weights_blob["B"] = {"w_ff": w_ff_b, "w_res": w_res_b}
    torch.save(weights_blob, root / "weights.pt")

    joblib.dump(classifier_a, root / "classifier_a.joblib")
    if classifier_b is not None:
        joblib.dump(classifier_b, root / "classifier_b.joblib")

    meta = {
        "name": run_name or root.name,
        "config": config.to_jsonable(),
        "metrics": dict(metrics or {}),
        "has_model_b": network_b is not None,
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def save_run(
    name: str,
    *,
    network_a,
    classifier_a,
    network_b=None,
    classifier_b=None,
    config: ReservoirConfig,
    metrics: Optional[Mapping[str, Any]] = None,
    root: str | Path | None = None,
    set_as_latest: bool = True,
) -> Path:
    """Save one training run under ``checkpoints/<name>/``.

    Model B is optional — omit network_b/classifier_b (or pass None) for a
    reservoir-only run (no plasticity).

    Examples::

        save_run("n10000_jury", network_a=..., network_b=..., ...)  # both pigeons
        save_run("n1000_reservoir_only", network_a=..., classifier_a=..., config=...)
    """
    if name in {"", "latest", _LATEST_POINTER}:
        raise ValueError(f"Invalid run name: {name!r}")
    if "/" in name or "\\" in name:
        raise ValueError("Run name must be a single folder segment, not a path")

    base = checkpoint_root(root)
    path = save_checkpoint(
        base / name,
        network_a=network_a,
        classifier_a=classifier_a,
        network_b=network_b,
        classifier_b=classifier_b,
        config=config,
        metrics=metrics,
        run_name=name,
    )
    if set_as_latest:
        set_latest(name, root=base)
    return path


def load_checkpoint(directory: str | Path) -> CheckpointBundle:
    """Rebuild frozen network(s) and load classifier(s) from a run directory.

    Model B is loaded only if it was saved (reservoir-only runs have no "B"
    entry in weights.pt / no classifier_b.joblib) — bundle.network_b and
    bundle.classifier_b are None in that case.
    """
    ensure_bindsnet()
    import joblib
    import torch

    root = Path(directory)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    config = ReservoirConfig.from_mapping(meta["config"])
    blob = torch.load(root / "weights.pt", map_location="cpu", weights_only=False)

    # Inference only — rebuild without STDP rules; trained weights are loaded in.
    network_a = build_reservoir(
        config=config, plastic=False, w_ff=blob["A"]["w_ff"], w_res=blob["A"]["w_res"]
    )
    network_a.train(False)

    network_b = None
    classifier_b = None
    classifier_b_path = root / "classifier_b.joblib"
    if "B" in blob and classifier_b_path.exists():
        network_b = build_reservoir(
            config=config, plastic=False, w_ff=blob["B"]["w_ff"], w_res=blob["B"]["w_res"]
        )
        network_b.train(False)
        classifier_b = joblib.load(classifier_b_path)

    return CheckpointBundle(
        config=config,
        network_a=network_a,
        classifier_a=joblib.load(root / "classifier_a.joblib"),
        network_b=network_b,
        classifier_b=classifier_b,
        metrics=dict(meta.get("metrics") or {}),
        source=root,
    )


def load_run(name: str = "latest", *, root: str | Path | None = None) -> CheckpointBundle:
    """Load a named run (or the current ``latest``) — both A and B."""
    return load_checkpoint(resolve_run_dir(name, root=root))


def config_from_models_globals(
    *,
    encoding_velocity: float,
    n_reservoir: int = 10_000,
    network_dt: float = 1.0,
    encoding_dt: float = 1.0,
    input_rate_hz: float = 40.0,
    encoding_seed: int = 0,
    trailing_silence: int = 20,
    weight_seed: int = 0,
    feedforward_strength: float = 50.0,
    reservoir_scale: float = 0.9,
    inhib_fraction: float = 0.2,
    inhib_weight_ratio: float = 1.0,
    ridge_alpha: float = 0.01,
    lif_kw: Optional[Mapping[str, float]] = None,
    stdp_nu: Sequence[float] = (1e-3, 1e-3),
    stdp_tc_pre: float = 20.0,
    stdp_tc_post: float = 20.0,
) -> ReservoirConfig:
    """Build a ``ReservoirConfig`` from the usual Models.ipynb hyperparameter names."""
    return ReservoirConfig(
        n_input=DEFAULT_HEADING_BINS,
        n_reservoir=n_reservoir,
        network_dt=network_dt,
        encoding_dt=encoding_dt,
        encoding_velocity=float(encoding_velocity),
        input_rate_hz=input_rate_hz,
        encoding_seed=encoding_seed,
        trailing_silence=trailing_silence,
        weight_seed=weight_seed,
        feedforward_strength=feedforward_strength,
        reservoir_scale=reservoir_scale,
        inhib_fraction=inhib_fraction,
        inhib_weight_ratio=inhib_weight_ratio,
        ridge_alpha=ridge_alpha,
        lif=dict(lif_kw)
        if lif_kw is not None
        else {
            "rest": -65.0,
            "reset": -65.0,
            "thresh": -52.0,
            "refrac": 5.0,
            "tc_decay": 250.0,
        },
        stdp_nu=(float(stdp_nu[0]), float(stdp_nu[1])),
        stdp_tc_pre=stdp_tc_pre,
        stdp_tc_post=stdp_tc_post,
    )
