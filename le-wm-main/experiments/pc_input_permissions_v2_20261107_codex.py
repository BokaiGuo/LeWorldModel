"""Versioned input boundary: successful call receipts, not episode evidence.

The v1 module remains unchanged because existing experiments bind its hash.
This guard is a data-flow check, not a sandbox against arbitrary Python.
"""
from pathlib import Path

from experiments import pc_input_permissions as v1


class _SuccessfulCalls:
    def __init__(self, model, audit):
        paths = {Path(path).resolve() for path, _ in audit.dependencies}
        if not {Path(__file__).resolve(), Path(v1.__file__).resolve()}.issubset(paths):
            raise ValueError("input audit must bind both v1 and v2 boundary sources")
        super().__init__(model, audit)

    def _check_frozen(self):
        if any(module.training for module in self._model.modules()):
            raise ValueError("model must remain in evaluation mode")
        if any(parameter.requires_grad for parameter in self._model.parameters()):
            raise ValueError("model parameters must remain frozen")

    def _invoke(self, operation, inputs, fn):
        self._check_frozen()
        receipt = {"track_id": self.audit.track_id, "input_audit_hash": self.audit.fingerprint,
                   "operation": operation, "inputs": {k: v1.tensor_receipt(v) for k, v in inputs.items()}}
        value = fn()
        self._check_frozen()
        self.receipts.append(receipt)
        return value


class RGBLeWMv2(_SuccessfulCalls, v1.RGBLeWM):
    def encode(self, inputs):
        if set(inputs) != {"pixels"}:
            raise ValueError("RGB encoder accepts pixels only")
        return self._invoke("encode", inputs, lambda: self._model.encode(dict(inputs)))

    def score(self, inputs, actions):
        if set(inputs) != {"pixels", "goal", "action"}:
            raise ValueError("RGB planner requires pixels, goal, legal action history only")
        context = dict(inputs)
        cost = self._invoke("score", {**inputs, "candidate_actions": actions},
                            lambda: self._model.get_cost(context, actions))
        return cost, context


class StateAssistedPointMazev2(_SuccessfulCalls, v1.StateAssistedPointMaze):
    def predict(self, inputs, actions):
        if set(inputs) != {"visual", "proprio"}:
            raise ValueError("official visual + proprio required; no default or zero fill")
        visual, proprio = inputs["visual"], inputs["proprio"]
        if visual.ndim != 5 or proprio.shape != (*visual.shape[:2], 4):
            raise ValueError("expected B,T,C,H,W RGB and B,T,4 current-state proprio")
        if actions.ndim != 3 or actions.shape[1] != visual.shape[0] or actions.shape[2] != 10:
            raise ValueError("expected T,B,10 normalized five-action packets")
        def run():
            encoded = self._model.encode(dict(inputs))
            return self._model.unroll(encoded, act_suffix=actions)
        return self._invoke("predict", {**inputs, "candidate_actions": actions}, run)


def call_manifest(audit, *, algorithm_version, protocol_path, task_manifest_path, receipts):
    """Bind successful model calls without claiming an episode was completed."""
    record = v1.manifest(audit, algorithm_version=algorithm_version, protocol_path=protocol_path,
                         task_manifest_path=task_manifest_path, receipts=receipts)
    record["schema"] = "pc.dual_track.call_receipts.v2"
    record["execution_status"] = "model_calls_recorded"
    return record
