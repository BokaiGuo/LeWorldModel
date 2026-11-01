"""CPU-only synthetic checks; no model or CUDA calls."""
import hashlib
import json

import pytest

from experiments.spectral_reachability import p1_dev303_v3_cuda_executor_20261102_exclusive as m


class Handle:
    def __init__(self, module):
        self.module = module

    def remove(self):
        self.module.hook = None


class Module:
    def __init__(self):
        self.hook = None

    def register_forward_pre_hook(self, fn):
        self.hook = fn
        return Handle(self)

    def call(self, batch):
        self.hook(self, (batch,))


class Batch:
    def __init__(self, size):
        self.shape = (size, 3, 10)


class Model:
    def __init__(self):
        self.encoder, self.projector = Module(), Module()
        self.action_encoder, self.predictor = Module(), Module()


def test_hook_counts_samples_and_rejects_visual_encode():
    model = Model()
    meter = m.Meter()
    meter.attach(model)
    model.action_encoder.call(Batch(1))
    model.predictor.call(Batch(3))
    assert meter.counts["action_encoder"] == 1
    assert meter.counts["predictor"] == 1
    assert meter.counts["predictor_samples"] == 3
    with pytest.raises(ValueError, match="unexpected image"):
        model.encoder.call(Batch(1))
    assert meter.counts["image_encoder"] == 1
    meter.close()
    assert model.predictor.hook is None


def test_binding_sha_and_symlink_rejected(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"fixed")
    digest = hashlib.sha256(b"fixed").hexdigest()
    m.verify_bindings({str(path): digest})
    path.write_bytes(b"drift")
    with pytest.raises(ValueError, match="source drift"):
        m.verify_bindings({str(path): digest})
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="source drift"):
        m.verify_bindings({str(link): hashlib.sha256(b"drift").hexdigest()})


def test_authority_rejected_before_any_io(tmp_path):
    with pytest.raises(ValueError, match="authority"):
        m.execute(tmp_path / "absent", "sha", tmp_path / "output", "not-authorized")
    assert not (tmp_path / "output").exists()


def test_verify_contract_rejects_tampering(tmp_path, monkeypatch):
    expected = {"a": 1}
    monkeypatch.setattr(m, "prepare_contract", lambda: expected)
    path = tmp_path / "prepare.json"
    raw = json.dumps(expected).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert m.verify_contract(path, digest) == expected
    path.write_text('{"a":2}')
    with pytest.raises(ValueError, match="SHA"):
        m.verify_contract(path, digest)


def test_task_uses_checkpoint_sha_not_state_dict_sha():
    slot = {"task_id": "task1"}
    envelope = {"task_id": "task1", "P1_admission": False, "task": {"model_sha256": "c" * 64}}
    bindings = {"/fixture/model/weights.pt": "c" * 64}
    m.validate_task_identity(envelope, slot, "d" * 64, bindings, "/fixture/model")
    envelope["task"]["model_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        m.validate_task_identity(envelope, slot, "d" * 64, bindings, "/fixture/model")
