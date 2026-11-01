"""Explicit boundary accounting for the existing P3 live-chunk executor.

No device, model, or environment is created here. The supplied executor may
dispatch environment steps; the caller must separately authorize and meter them.
"""
from __future__ import annotations

from . import codex_p3_live_chunk_adapter_20261101 as executor
from . import p3_c3_model_boundary_20261102_exclusive as boundary


class _MeteredModel:
    """Minimal JEPA ABI used by the existing live-chunk executor."""

    def __init__(self, ledger):
        self._ledger = ledger

    def action_encoder(self, actions):
        ledger = self._ledger
        boundary._require(ledger._active, "candidate scope required")
        return ledger._call("signal", lambda: ledger._model.action_encoder(actions),
                            {"action_encoder": 1})

    def predict(self, context, embedded):
        ledger = self._ledger
        shape = getattr(context, "shape", ())
        boundary._require(ledger._active and len(shape) >= 1
                          and type(shape[0]) is int and shape[0] > 0,
                          "candidate scope and context batch required")
        return ledger._call("signal", lambda: ledger._model.predict(context, embedded),
                            {"predict": 1, "predictor_sample": shape[0]})


def run_candidate(ledger, *, root, index, **executor_arguments):
    """Run the existing executor with a metered model and frozen candidate quota.

    ``executor_arguments`` is the exact ``run_chunk`` ABI excluding ``model``.
    Exceptions propagate; the caller retains ``ledger`` to inspect partial
    attempted/returned counts. No GPU or environment access occurs until the
    caller invokes this function.
    """
    boundary._require(isinstance(ledger, boundary.BoundaryAdapter), "boundary ledger")
    boundary._require("model" not in executor_arguments, "model is bound by bridge")
    boundary._require(executor_arguments.get("arm") == "dynamic",
                      "live executor arm must be dynamic; allocation arm is separate")
    return ledger.candidate(
        root, index,
        lambda _: executor.run_chunk(model=_MeteredModel(ledger), **executor_arguments),
    )
