from contextlib import contextmanager
import json

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.spectral_reachability import p1_cp5_two_candidate_rgbq_20261101_leibniz as audit


class Analytic:
    def __init__(self):
        self.actions = []
    def action_encode(self, actions, *, phase):
        self.actions.append(actions.clone())
        return actions
    def predict(self, states, actions, *, phase):
        return states + actions[..., :1]


def test_actual_packets_are_normalized_in_order():
    leaf = Analytic()
    raw = torch.tensor([[[.1, .1]] * 5, [[.2, .2]] * 5, [[.3, .3]] * 5])
    result = audit.predict(leaf, torch.zeros(1, 3, 192), torch.zeros(1, 2, 10),
                           raw, torch.zeros(2), torch.ones(2))
    assert result.shape == (192,) and len(leaf.actions) == 3
    for index in range(3):
        assert torch.allclose(leaf.actions[index][0, -1], raw[index].reshape(10))
    with pytest.raises(ValueError):
        audit.predict(leaf, torch.zeros(1, 3, 192), torch.zeros(1, 2, 10),
                      raw * 20, torch.zeros(2), torch.ones(2))


def test_separate_signed_metrics_and_roster():
    rows = [dict(arm="cem", predicted_q=2., actual_rgb_q=1., physical_joint_xy=4.),
            dict(arm="structured_own", predicted_q=1., actual_rgb_q=3., physical_joint_xy=2.)]
    assert audit.comparisons(rows) == dict(predicted_q_structured_minus_cem=-1.,
                                          actual_rgb_q_structured_minus_cem=2.,
                                          physical_joint_xy_structured_minus_cem=-2.)
    with pytest.raises(ValueError):
        audit.comparisons(rows[::-1])
    assert audit.costs()["predict_calls"] == 6 and audit.costs()["encode_calls"] == 2


@pytest.mark.parametrize("tamper", ["roster", "action", "image"])
def test_physical_receipt_rejects_mismatch(monkeypatch, tmp_path, tamper):
    import copy
    import hashlib

    monkeypatch.setattr(audit, "RUN", tmp_path)
    points = []
    for step in (0, 5, 10, 15, 20):
        pixel = np.full((224, 224, 3), step, dtype=np.uint8)
        path = tmp_path / f"{step}.png"
        Image.fromarray(pixel).save(path)
        state = {"step": step}
        points.append(dict(step=step, png=str(path), file_sha256=audit.family.sha(path),
                           pixel_sha256=hashlib.sha256(pixel.tobytes()).hexdigest(),
                           state=state, state_sha256=audit.own.spec.digest(state)))
    commands = [[[0., 0.]] * 20, [[0., 0.]] * 5 + [[0.2, 0.]] * 15,
                [[0., 0.]] * 20]
    schedule = [dict(arm=arm, commands=actions) for arm, actions in zip(audit.physical.ARMS, commands)]
    rows = [dict(arm=slot["arm"], status="completed20", steps_attempted=20,
                 steps_returned=20, executed_raw=copy.deepcopy(slot["commands"]),
                 checkpoints=copy.deepcopy(points)) for slot in schedule]
    protocol = dict(schedule=copy.deepcopy(schedule),
                    cp5={"pixel_sha256": points[1]["pixel_sha256"],
                         "state_sha256": points[1]["state_sha256"]})
    report = dict(schema=audit.physical.NAME + ".report.v1", protocol_sha256=audit.PROTOCOL_SHA,
                  sources_verified=True, neural_calls=0, P1_admission=False, rows=rows)
    assert len(audit.validate_pair(report, protocol, schedule)) == 2
    if tamper == "roster":
        report["rows"].reverse()
    elif tamper == "action":
        report["rows"][1]["executed_raw"][7][0] = 0.3
    else:
        report["rows"][1]["checkpoints"][-1]["pixel_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        audit.validate_pair(report, protocol, schedule)


def test_run_real_lock_path_without_cuda(monkeypatch, tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    source = tmp_path / "source"
    source.write_text("source")
    value = {"sources": {str(source): audit.family.sha(source)}}
    prepared = results / (audit.NAME + "_PREP.json")
    prepared.write_text(json.dumps(value))
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "build", lambda: value)
    seen = []
    @contextmanager
    def lock(path):
        seen.append(path)
        yield
    monkeypatch.setattr(audit.prev.launcher.extra.gpu.locking, "gpu_lock", lock)
    def no_cuda(_):
        raise RuntimeError("before CUDA")
    monkeypatch.setattr(audit.own.gpu_preflight, "require_cuda_allocation", no_cuda)
    output = results / (audit.NAME + "_RUN")
    result = audit.run(prepared, audit.family.sha(prepared), output, audit.AUTH)
    assert seen == [audit.prev.launcher.extra.gpu.LOCK]
    assert result["status"] == "partial"
    assert json.loads((output / "report.json").read_text())["failure"]["message"] == "before CUDA"
