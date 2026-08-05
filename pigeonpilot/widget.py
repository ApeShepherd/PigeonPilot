"""
Live playground widget — draw a displacement route, watch both pigeons home.

Split of responsibility: the kernel owns every number (``flightplan``), the
browser owns every pixel (``static/playground.js``). One stroke triggers one
``build_flight_plan`` call of roughly a second, after which the animation runs
entirely in the frontend at display refresh rate. Nothing is recomputed per
frame, so the flight cannot stutter and the spike raster cannot drift out of
sync with the bird.

Usage::

    from pigeonpilot.snn import load_run
    from pigeonpilot.widget import PigeonPlayground

    PigeonPlayground(load_run("latest"))

Requires ``anywidget``; works in Cursor / VS Code, JupyterLab and Colab.
``playground.py`` (Matplotlib) stays available as a fallback.
"""

from __future__ import annotations

import pathlib
import traceback
from typing import Any, Sequence

import traitlets

from .drawing import training_band
from .flightplan import DEFAULT_ENSEMBLE_SIZE, build_flight_plan, reference_errors
from .snn import CheckpointBundle

try:
    import anywidget
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The live playground needs anywidget. Install it with:\n"
        "    pip install anywidget"
    ) from exc

_STATIC = pathlib.Path(__file__).parent / "static"
DEFAULT_FIELD_HALF = 7.0
TOO_SHORT_MESSAGE = "Route too short — draw further away from the home star."


class PigeonPlayground(anywidget.AnyWidget):
    """Interactive draw → displace → predict demo bound to a loaded checkpoint.

    Parameters
    ----------
    bundle :
        Loaded run (``snn.load_run``). Its ``ReservoirConfig`` defines the
        encoding contract; nothing here overrides it.
    models :
        Which pigeons to evaluate on the drawn route. Both by default, since
        the A/B comparison is the point of the demo.
    ensemble_size :
        Extra Poisson re-draws per model, off by default. The widget places
        each flight inside the held-out error distribution instead, which is
        the honest answer to "was that a fluke?" and costs nothing per click.
        Set it above zero only if you want the per-stroke spread in
        ``last_plan`` for the write-up.
    field_half :
        Half-width of the drawing field in world units. Defaults to a little
        beyond the curriculum's farthest release point, so the canvas invites
        routes the reservoir has actually seen.
    reference :
        Evaluate the held-out test split once so every flight can be shown
        against the model's error distribution. Takes a few seconds on first
        use and is then cached next to the checkpoint. Turn off only if you
        deliberately want the demo without that context.
    """

    _esm = _STATIC / "playground.js"
    _css = _STATIC / "playground.css"

    # Carries a nonce alongside the points so redrawing the same stroke still fires.
    request = traitlets.Dict(default_value={}).tag(sync=True)
    plan = traitlets.Dict(default_value={}, allow_none=True).tag(sync=True)
    band = traitlets.Dict(default_value={}).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    busy = traitlets.Bool(False).tag(sync=True)
    field_half = traitlets.Float(DEFAULT_FIELD_HALF).tag(sync=True)

    def __init__(
        self,
        bundle: CheckpointBundle,
        *,
        models: Sequence[str] = ("A", "B"),
        ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
        field_half: float | None = None,
        reference: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.bundle = bundle
        self.models = tuple(models)
        self.ensemble_size = int(ensemble_size)
        self.reference = reference_errors(bundle, models=self.models) if reference else None
        self._band = training_band(heading_bins=int(bundle.config.n_input))
        self.band = {
            "band_min": self._band.min_total_distance,
            "band_max": self._band.max_total_distance,
            "band_max_blocks": self._band.max_blocks,
            "band_max_release": self._band.max_release_distance,
        }
        self.field_half = float(
            field_half if field_half is not None else self._band.max_release_distance * 1.15
        )

    @traitlets.observe("request")
    def _on_request(self, change: dict[str, Any]) -> None:
        request = change["new"] or {}
        # Echoed back on every outcome so a repeated stroke still updates the UI.
        nonce = request.get("nonce", 0)
        points = request.get("points") or []
        if len(points) < 2:
            self.plan = {"nonce": nonce, "error": TOO_SHORT_MESSAGE}
            return
        self.busy = True
        try:
            plan = build_flight_plan(
                self.bundle,
                points,
                models=self.models,
                ensemble_size=self.ensemble_size,
                band=self._band,
                reference=self.reference,
            )
        except Exception:  # surface failures in the widget, not only the log
            traceback.print_exc()
            self.plan = {
                "nonce": nonce,
                "error": "Inference failed — see the traceback in the notebook log.",
            }
            return
        finally:
            self.busy = False

        if plan is None:
            self.plan = {"nonce": nonce, "error": TOO_SHORT_MESSAGE}
            return
        self.plan = {**plan, "nonce": nonce}

    @property
    def last_plan(self) -> dict[str, Any]:
        """The most recent flight plan, for inspection or export from the notebook."""
        return dict(self.plan or {})

    def self_test(self, verbose: bool = True) -> dict[str, Any]:
        """Run a synthetic stroke through the full pipeline and report the outcome.

        Use this when the widget sits on "Running the reservoir …": it exercises the
        same code path as a real flight without needing the browser, so it
        separates a kernel-side failure from a frontend one.
        """
        import json
        import time

        import numpy as np

        t = np.linspace(0.0, 1.0, 150)
        stroke = np.stack([4.0 * np.sin(2.5 * t), 4.0 * t], axis=1)
        report: dict[str, Any] = {
            "widget_module": __file__,
            "checkpoint": str(getattr(self.bundle, "source", "unknown")),
            "reservoir_size": self.bundle.config.n_reservoir,
            "models": list(self.models),
            "reference_loaded": self.reference is not None,
        }
        started = time.perf_counter()
        try:
            plan = build_flight_plan(
                self.bundle,
                stroke,
                models=self.models,
                ensemble_size=self.ensemble_size,
                band=self._band,
                reference=self.reference,
            )
            report["seconds"] = round(time.perf_counter() - started, 2)
            report["plan_built"] = plan is not None
            if plan is not None:
                json.dumps(plan)
                report["json_serialisable"] = True
                report["segments"] = len(plan["segments"])
                report["errors_deg"] = {m["name"]: m["error_deg"] for m in plan["models"]}
            report["ok"] = plan is not None
        except Exception as exc:
            report["ok"] = False
            report["error"] = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()

        if verbose:
            for key, value in report.items():
                print(f"{key:20} {value}")
            if report.get("ok"):
                print(
                    "\nKernel side is healthy. If the widget still hangs, the browser is "
                    "running older widget code: restart the kernel and re-run all cells."
                )
        return report


def launch(bundle: CheckpointBundle, **kwargs: Any) -> PigeonPlayground:
    """Create the live playground for a loaded checkpoint."""
    return PigeonPlayground(bundle, **kwargs)
