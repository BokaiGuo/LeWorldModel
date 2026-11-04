import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_cem_bridge_20261102_codexexclusive as bridge

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results/reacher_pc_fd_suffix_cuda_20261102_v1/report.json"
AUDIT = ROOT / "results/reacher_pc_fd_suffix_saved_audit_20261102_v1.json"
PHYSICAL_PLAN = ROOT / "results/reacher_pc_fd_common_physical_20261102_PLAN_v1.json"


def test_real_sources_build_four_fixed_development_arms():
    manifest = bridge.prepare(MODEL, AUDIT, PHYSICAL_PLAN)
    assert [r["seed"] for r in manifest["roots"]] == list(bridge.ROOTS)
    assert manifest["planned_arms"] == 4
    assert manifest["cem"]["rollout_candidates"] == 32
    assert manifest["cem"]["predict_token_samples"] == 64
    assert manifest["physical_outcome_used"] is False
    for root in manifest["roots"]:
        assert [(arm["arm"], arm["proposal_index"]) for arm in root["arms"]] == list(bridge.ARMS)
        assert all(bridge.sha_array(arm["raw_prefix"]) == arm["raw_prefix_sha256"] for arm in root["arms"])


def test_root_context_requires_exact_warmup_and_identity():
    root = bridge.prepare(MODEL, AUDIT, PHYSICAL_PLAN)["roots"][0]
    context = {"root_rgb_sha256": root["root_rgb_sha256"], "goal_rgb_sha256": root["goal_rgb_sha256"],
               "warmup_actions_sha256": root["warmup_sha256"],
               "warmup_raw_steps": np.asarray(root["warmup_packets"], np.float32).reshape(15, 2),
               "frames": [object(), object(), object()]}
    bridge.check_context(context, root)
    context["warmup_raw_steps"][3, 0] += .1
    with pytest.raises(ValueError, match="warmup mismatch"):
        bridge.check_context(context, root)
    context["root_rgb_sha256"] = "wrong"
    with pytest.raises(ValueError, match="identity mismatch"):
        bridge.check_context(context, root)


def test_forbids_cpu_neural_inference_before_environment():
    class UntouchableEnv:
        def __getattr__(self, _name):
            raise AssertionError("environment should not be touched")
    model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    with pytest.raises(ValueError, match="CPU inference forbidden"):
        bridge.run_root({}, model=model, env=UntouchableEnv(), processor=None, mean=None, std=None,
                        lower=None, upper=None, device="cpu", parameter_hash_expected="x")


def test_rejects_source_or_selection_tampering(tmp_path, monkeypatch):
    model = json.loads(MODEL.read_text())
    saved = json.loads(AUDIT.read_text())
    physical = json.loads(PHYSICAL_PLAN.read_text())
    model["rows"][0]["compiler_receipt"]["strict_selected_index"] = 4
    fake_paths = [tmp_path / "model.json", tmp_path / "audit.json", tmp_path / "physical.json"]
    for path, value in zip(fake_paths, (model, saved, physical), strict=True):
        path.write_text(json.dumps(value))
    monkeypatch.setattr(bridge, "MODEL_SHA", bridge.sha_file(fake_paths[0]))
    monkeypatch.setattr(bridge, "AUDIT_SHA", bridge.sha_file(fake_paths[1]))
    monkeypatch.setattr(bridge, "PHYSICAL_PLAN_SHA", bridge.sha_file(fake_paths[2]))
    saved["report_sha256"] = bridge.MODEL_SHA
    physical["model_sha256"] = bridge.MODEL_SHA
    fake_paths[1].write_text(json.dumps(saved))
    monkeypatch.setattr(bridge, "AUDIT_SHA", bridge.sha_file(fake_paths[1]))
    physical["audit_sha256"] = bridge.AUDIT_SHA
    fake_paths[2].write_text(json.dumps(physical))
    monkeypatch.setattr(bridge, "PHYSICAL_PLAN_SHA", bridge.sha_file(fake_paths[2]))
    with pytest.raises(ValueError, match="selection mismatch"):
        bridge.prepare(*fake_paths)
    fake_paths[0].write_text("tampered")
    with pytest.raises(ValueError, match="source SHA"):
        bridge.prepare(*fake_paths)


def test_run_root_wires_both_fixed_arms_to_same_cem_without_inference(monkeypatch):
    from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
    from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner

    root = bridge.prepare(MODEL, AUDIT, PHYSICAL_PLAN)["roots"][0]
    context = {"root_rgb_sha256": root["root_rgb_sha256"], "goal_rgb_sha256": root["goal_rgb_sha256"],
               "warmup_actions_sha256": root["warmup_sha256"],
               "warmup_raw_steps": np.asarray(root["warmup_packets"], np.float32).reshape(15, 2),
               "frames": [object(), object(), object()]}
    monkeypatch.setattr(bridge, "require_host_cuda", lambda *_: None)
    monkeypatch.setattr(setup, "_parameter_hash", lambda _model: "frozen")
    monkeypatch.setattr(setup, "collect_executed_context", lambda *_a, **_k: context)
    original_as_tensor = torch.as_tensor
    def fake_as_tensor(value, *, dtype, device):
        return original_as_tensor(value, dtype=dtype, device="cpu")
    monkeypatch.setattr(torch, "as_tensor", fake_as_tensor)
    def decode(params, *_args):
        for arm in root["arms"]:
            if np.array_equal(params.numpy().reshape(1, 10), np.asarray(arm["params"], np.float32)):
                return np.asarray(arm["raw_prefix"], np.float32), None
        raise AssertionError("unexpected prefix")
    monkeypatch.setattr(setup.p0, "_to_raw_actions", decode)
    calls = []
    def fake_two_stage(**kwargs):
        calls.append(kwargs)
        params, receipt = kwargs["first_program"](None, None, None, root["seed"])
        raw, _ = decode(params)
        return {"first_action_witness_raw": raw.tolist(),
                "suffix_receipt": {"forward_rollout_evaluations": 32}, "first_receipt": receipt}
    monkeypatch.setattr(runner, "_two_stage", fake_two_stage)
    class ActionSpace:
        low = [-1, -1]
        high = [1, 1]
    class Env:
        action_space = ActionSpace()
    result = bridge.run_root(root, model=object(), env=Env(), processor=None, mean=None, std=None,
                             lower=None, upper=None, device="cuda", parameter_hash_expected="frozen")
    assert [row["status"] for row in result["arms"]] == ["complete", "complete"]
    assert [row["suffix_seed"] for row in result["arms"]] == [root["seed"] + 5000] * 2
    assert [(c["progress_solver"], c["progress_horizon"], c["cem_samples"], c["cem_steps"],
             c["cem_elite"], c["seed"]) for c in calls] == [
                 ("cem", 2, 16, 2, 4, root["seed"])] * 2
    assert all(c["context"] is context for c in calls)
