import copy

import numpy as np
import pytest

from experiments.spectral_reachability import p5_rgb_forward_recursive_probe_20261030_codex_v1 as m


def fake_state(device):
    rec = {"shape": [1], "dtype": "float32", "sha256": "mock", "device": device}
    return {"modules": [{"name": "", "class": "Mock", "training": False}],
            "parameters": {"p": {**rec, "requires_grad": False}},
            "normalization": {"_state_mean": rec, "_state_std": rec},
            "precision": {"default_dtype": "torch.float32", "matmul_precision": "highest",
                          "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True,
                          "cuda_autocast_enabled": False}}


def fake_prepared():
    rows = [{} for _ in range(164)]
    for i in m.INDICES:
        rows[i] = {"index": i, "root_slot": i // 82, "condition": ("off", "on")[(i // 41) % 2],
                   "task_id": str(i), "selection_digest": str(i), "actions": [[0., 0.]] * 5,
                   "actions_sha256": "actions", "previous_rgb": {"model_input_chw_float32_sha256": "prev"},
                   "current_rgb": {"model_input_chw_float32_sha256": "current"}}
    return {"rows": rows}


class FakeBackend:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail
    def metadata(self):
        return {"device": "cuda:0", "device_name": "mock", "torch_version": "mock", "cuda_version": "mock",
                "strict_load": True, "frozen": True, "eval": True,
                "checkpoint_sha256": m.bridge.FILES["checkpoint"][1]}
    def capture(self):
        return fake_state("cuda:0")
    def current(self):
        return self.state.copy()
    def forward(self, inputs):
        self.calls.append("forward")
        return np.zeros((1, 6), dtype="<f4"), np.ones((1, 6), dtype="<f4")
    def encode(self, inputs):
        self.calls.append("encode")
        self.initial = np.zeros((1, 6), dtype="<f4")
        self.state = self.initial.copy()
        return self.state.copy()
    def reset(self, coord, sign):
        self.calls.append("reset")
        self.state = m.perturb(self.initial[0], coord, sign)[None]
        return self.state.copy()
    def step(self, inputs, index):
        self.calls.append("step")
        if self.fail:
            raise RuntimeError("injected")
        self.state = self.state + np.float32(1)
        return self.state.copy()


def generate(tmp_path, fail=False):
    p, b = fake_prepared(), {"state": fake_state("cpu")}
    backend = FakeBackend(fail)
    directory = tmp_path / "run"
    journal = m.runner.Journal(directory)
    try:
        status = m.execute(p, "contract", backend, journal, 100, b, clock=lambda: 0, loader=lambda row: None)
    finally:
        journal.close()
    return p, b, backend, directory, status


def test_full_schedule_review_and_finite_difference(tmp_path):
    p, b, backend, directory, status = generate(tmp_path)
    assert status == "complete"
    assert {k: backend.calls.count(k) for k in ("forward", "encode", "reset", "step")} == {
        "forward": 4, "encode": 4, "reset": 52, "step": 264}
    raw = (directory / "events.jsonl").read_bytes()
    report = m.review(raw, m.bridge.digest(raw), p, "contract", b)
    branch = report["branches"]["0"]
    assert branch["interface_decoded_difference"] == [0.] * 6
    assert branch["interface_predicted_difference"] == [0.] * 6
    for coord in range(6):
        assert branch["finite_difference"][0]["columns_by_input_coordinate"][coord][coord] == pytest.approx(1, abs=4e-5)
    assert len(report["branches"]) == 4


def rechain(events):
    prev = "0" * 64
    for i, event in enumerate(events):
        event["seq"], event["prev"] = i, prev
        event.pop("hash", None)
        event["hash"] = m.bridge.digest(m.bridge.canonical(event))
        prev = event["hash"]
    return b"".join(m.bridge.canonical(e) for e in events)


@pytest.mark.parametrize("mutation", ["order", "identity", "input", "state_link", "output", "snapshot", "extra", "truncate"])
def test_rehashed_tampering_rejected(tmp_path, mutation):
    p, b, _, directory, _ = generate(tmp_path)
    events = [m.bridge.read_json(line) for line in (directory / "events.jsonl").read_bytes().splitlines()]
    if mutation == "order":
        events[2], events[4] = events[4], events[2]
    elif mutation == "identity":
        events[3]["actions_sha256"] = "other"
    elif mutation == "input":
        event = next(e for e in events if e.get("kind") == "returned" and
                     e.get("op") == "reset_state" and e.get("trajectory") == "0:plus")
        event["output"] = m.runner.vector(np.zeros((1, 6), dtype="<f4"))
    elif mutation == "state_link":
        event = next(e for e in events if e.get("kind") == "attempt" and e.get("op") == "predict_next")
        event["state_in"] = m.runner.vector(np.ones((1, 6), dtype="<f4"))
    elif mutation == "output":
        events[3]["output"]["decoded"]["values"][0] = 8
    elif mutation == "snapshot":
        events[-2]["state"]["precision"]["cuda_autocast_enabled"] = True
    elif mutation == "extra":
        events[3]["future_label"] = 1
    else:
        events.pop()
    raw = rechain(events)
    with pytest.raises(ValueError):
        m.review(raw, m.bridge.digest(raw), p, "contract", b)


def test_partial_failure_is_never_complete(tmp_path):
    p, b, backend, directory, status = generate(tmp_path, fail=True)
    assert status == "partial" and backend.calls == ["forward", "encode", "step"]
    raw = (directory / "events.jsonl").read_bytes()
    assert [m.bridge.read_json(line)["kind"] for line in raw.splitlines()][-2:] == ["failed", "end"]
    with pytest.raises(ValueError):
        m.review(raw, m.bridge.digest(raw), p, "contract", b)


def test_denominator_uses_realized_float32_spacing():
    base = np.asarray([100, 0, 0, 0, 0, 0], dtype="<f4")
    minus, plus = m.perturb(base, 0, -1), m.perturb(base, 0, 1)
    assert float(plus[0]) - float(minus[0]) != 2 * float(m.EPS)
    with pytest.raises(ValueError):
        m.perturb([1e20, 0, 0, 0, 0, 0], 0, 1)
