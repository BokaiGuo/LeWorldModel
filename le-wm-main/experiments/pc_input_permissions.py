"""Fail-closed input boundaries for new Planning Compiler runs.

This is an API/data-flow guard, not a security sandbox against arbitrary Python.
Legacy runners are deliberately not relabelled or grandfathered into either track.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

RGB = "primary_rgb"
STATE = "official_state_assisted"
SCOPES = {RGB: "visual_closed_loop", STATE: "state_assisted_mechanism"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def track(value):
    if value not in SCOPES:
        raise ValueError("explicit, recognized evaluation track required")
    return value


def tensor_receipt(value):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError("only finite tensor inputs are accepted")
    tensor = value.detach().cpu().contiguous()
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest()}


@dataclass(frozen=True)
class InputAudit:
    track_id: str
    family: str
    dependencies: tuple  # (absolute path, sha256), including weights/config/runtime
    proprio_fields: tuple
    preprocessing: str
    clock: str

    def verify(self):
        track(self.track_id)
        expected = {"lewm": (RGB, ()), "pointmaze_jepa": (STATE, ("x", "y", "vx", "vy"))}
        if self.family not in expected or (self.track_id, self.proprio_fields) != expected[self.family]:
            raise ValueError("checkpoint input requirements incompatible with track")
        if not self.preprocessing or not self.clock or not self.dependencies:
            raise ValueError("incomplete input audit")
        paths = [path for path, _ in self.dependencies]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate dependency")
        for path, expected_hash in self.dependencies:
            if not Path(path).is_absolute() or file_hash(path) != expected_hash:
                raise ValueError("input dependency drift")
        return self

    @property
    def fingerprint(self):
        return digest(asdict(self))


def freeze_audit(track_id, family, paths, *, preprocessing, clock, proprio_fields=()):
    audit = InputAudit(track_id, family,
                       tuple((str(Path(p).resolve()), file_hash(p)) for p in paths),
                       tuple(proprio_fields), preprocessing, clock)
    return audit.verify()


class Boundary:
    def __init__(self, model, audit):
        audit.verify()
        if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
            raise ValueError("model must be eval and all weights frozen")
        self._model = model
        self.audit = audit
        self.receipts = []

    def _record(self, operation, inputs):
        record = {"track_id": self.audit.track_id, "input_audit_hash": self.audit.fingerprint,
                  "operation": operation, "inputs": {k: tensor_receipt(v) for k, v in inputs.items()}}
        self.receipts.append(record)


class RGBLeWM(Boundary):
    def __init__(self, model, audit):
        if audit.family != "lewm" or audit.track_id != RGB:
            raise ValueError("RGB LeWM audit required")
        super().__init__(model, audit)

    def encode(self, inputs):
        if set(inputs) != {"pixels"}:
            raise ValueError("RGB encoder accepts pixels only")
        self._record("encode", inputs)
        return self._model.encode(dict(inputs))

    def score(self, inputs, actions):
        # Historical action packets are explicit: no automatic zero filling.
        if set(inputs) != {"pixels", "goal", "action"}:
            raise ValueError("RGB planner requires pixels, goal, legal action history only")
        self._record("score", {**inputs, "candidate_actions": actions})
        context = dict(inputs)  # upstream mutates context; never mutate caller data
        cost = self._model.get_cost(context, actions)
        return cost, context


class StateAssistedPointMaze(Boundary):
    def __init__(self, model, audit):
        if audit.family != "pointmaze_jepa" or audit.track_id != STATE:
            raise ValueError("state-assisted PointMaze audit required")
        # Inspect the loaded path, not just the caller's declaration.
        core = model.model
        if not getattr(core, "use_proprio", False) or getattr(core, "proprio_encoder", None) is None:
            raise ValueError("loaded model does not match audited proprio path")
        super().__init__(model, audit)

    def predict(self, inputs, actions):
        if set(inputs) != {"visual", "proprio"}:
            raise ValueError("official visual + proprio required; no default or zero fill")
        visual, proprio = inputs["visual"], inputs["proprio"]
        if visual.ndim != 5 or proprio.shape != (*visual.shape[:2], 4):
            raise ValueError("expected B,T,C,H,W RGB and B,T,4 current-state proprio")
        if actions.ndim != 3 or actions.shape[1] != visual.shape[0] or actions.shape[2] != 10:
            raise ValueError("expected T,B,10 normalized five-action packets")
        self._record("predict", {**inputs, "candidate_actions": actions})
        encoded = self._model.encode(dict(inputs))
        return self._model.unroll(encoded, act_suffix=actions)


def manifest(audit, *, algorithm_version, protocol_path, task_manifest_path, receipts):
    audit.verify()
    if not algorithm_version or not receipts:
        raise ValueError("algorithm binding and actual call receipts required")
    if any(r["track_id"] != audit.track_id or r["input_audit_hash"] != audit.fingerprint for r in receipts):
        raise ValueError("mixed input receipts")
    return {"schema": "pc.dual_track.v1", "track_id": audit.track_id,
            "evidence_scope": SCOPES[audit.track_id], "algorithm_version": algorithm_version,
            "protocol_hash": file_hash(protocol_path), "task_manifest_hash": file_hash(task_manifest_path),
            "input_audit": asdict(audit), "input_audit_hash": audit.fingerprint,
            "input_receipts": receipts, "execution_status": "completed",
            "scientific_status": "not_evaluated"}


def validate_evidence(record, *, target_track, algorithm_version, protocol_hash):
    """Eligibility is not scientific gate passage. Missing legacy metadata fails."""
    track(target_track)
    if (record.get("schema") != "pc.dual_track.v1"
            or record.get("track_id") != target_track
            or record.get("evidence_scope") != SCOPES[target_track]
            or record.get("algorithm_version") != algorithm_version
            or record.get("protocol_hash") != protocol_hash):
        raise ValueError("incompatible or legacy evidence; cross-track promotion forbidden")
    payload = record["input_audit"]
    audit = InputAudit(payload["track_id"], payload["family"],
                       tuple(tuple(x) for x in payload["dependencies"]),
                       tuple(payload["proprio_fields"]), payload["preprocessing"], payload["clock"])
    audit.verify()
    if audit.track_id != target_track or audit.fingerprint != record["input_audit_hash"]:
        raise ValueError("audit/track mismatch")
    receipts = record["input_receipts"]
    if not receipts or any(r["track_id"] != target_track or r["input_audit_hash"] != audit.fingerprint
                           for r in receipts):
        raise ValueError("missing or mixed runtime receipts")
    return True


def cache_key(audit, *, algorithm_version, inputs):
    audit.verify()
    return digest({"track": audit.track_id, "audit": audit.fingerprint,
                   "algorithm": algorithm_version, "inputs": inputs})
