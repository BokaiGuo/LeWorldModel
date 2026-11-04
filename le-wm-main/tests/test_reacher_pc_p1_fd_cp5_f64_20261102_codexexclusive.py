"""No neural inference, simulator, or GPU."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import preflight_reacher_d2p_p1_reconfiguration as setup
try:
    from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_f64_20261102_codexexclusive as fixed
except ImportError:
    from experiments.spectral_reachability import reacher_pc_p1_fd_cp5_cem_bridge_20261102_codexexclusive as fixed

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results/reacher_pc_fd_suffix_cuda_20261102_v1/report.json"
AUDIT = ROOT / "results/reacher_pc_fd_suffix_saved_audit_20261102_v1.json"
PHYSICAL = ROOT / "results/reacher_pc_fd_common_physical_20261102_PLAN_v1.json"
PREFLIGHT = ROOT / "results/reacher_pc_p1_common_tail_preflight_20261102_codexexclusive_02.json"


def test_preparation_preserves_original_float64_bank_and_raw():
    source = json.loads(MODEL.read_text())
    prepared = fixed.prepare(MODEL, AUDIT, PHYSICAL)
    for row, root in zip(source["rows"], prepared["roots"], strict=True):
        for arm in root["arms"]:
            index = arm["proposal_index"]
            assert np.array_equal(np.asarray(arm["params"], np.float64),
                                  np.asarray(row["bank_params"][index], np.float64))
            assert np.array_equal(np.asarray(arm["raw_prefix"], np.float32),
                                  np.asarray(row["prefix_raw"][index], np.float32))


def test_float64_real_source_decodes_to_strict_saved_actions():
    p = json.loads(PREFLIGHT.read_text())
    mean, std = setup._load_action_stats(Path(p["action_stats_path"]), "swm/ReacherDMControl-v0")
    low = np.array([-1., -1.], dtype=np.float32)
    high = -low
    lo, hi = setup._normalized_action_bounds(mean, std, low, high)
    plan = fixed.prepare(MODEL, AUDIT, PHYSICAL)
    for root in plan["roots"]:
        for arm in root["arms"]:
            params, raw = fixed.decode_arm(arm, mean=mean, std=std, lower=torch.tensor(lo),
                                           upper=torch.tensor(hi), raw_low=low, raw_high=high, device="cpu")
            assert params.dtype == torch.float64
            assert np.array_equal(raw, np.asarray(arm["raw_prefix"], np.float32))
            assert fixed.sha_array(raw) == arm["raw_prefix_sha256"]
    broken = torch.tensor(plan["roots"][0]["arms"][0]["params"], dtype=torch.float32).unsqueeze(0)
    raw, _ = setup.p0._to_raw_actions(broken, mean, std, torch.tensor(lo), torch.tensor(hi), low, high)
    assert fixed.sha_array(raw) != plan["roots"][0]["arms"][0]["raw_prefix_sha256"]
    altered = dict(plan["roots"][0]["arms"][0], raw_prefix_sha256="0" * 64)
    with pytest.raises(ValueError, match="prefix action mismatch"):
        fixed.decode_arm(altered, mean=mean, std=std, lower=torch.tensor(lo), upper=torch.tensor(hi),
                         raw_low=low, raw_high=high, device="cpu")
    arm = plan["roots"][0]["arms"][0]
    truncated = dict(arm, params=np.asarray(arm["params"], dtype=np.float32).tolist())
    with pytest.raises(ValueError, match="prefix action mismatch"):
        fixed.decode_arm(truncated, mean=mean, std=std, lower=torch.tensor(lo), upper=torch.tensor(hi),
                         raw_low=low, raw_high=high, device="cpu")


def test_source_hash_mismatch_rejected(tmp_path):
    other = tmp_path / "model.json"
    other.write_text("{}")
    with pytest.raises(ValueError, match="source SHA"):
        fixed.prepare(other, AUDIT, PHYSICAL)


def test_two_stage_float64_first_does_not_promote_successor_cem_history(monkeypatch):
    from experiments.spectral_reachability import run_reacher_d2p_p1_reconfiguration as runner
    p0 = runner.p0
    frame = np.zeros((2, 2, 3), np.uint8)
    context = {"frames": [frame] * 3, "root_frame": frame, "goal": frame,
               "history_action_blocks": np.zeros((3, 10), np.float32), "execution_snapshot": {}}
    successor = dict(context, context_mode="test", root_rgb_sha256="r", goal_rgb_sha256="g",
                     advanced_action_sha256="a")
    env = SimpleNamespace(action_space=SimpleNamespace(low=-np.ones(2), high=np.ones(2)))
    env.unwrapped = SimpleNamespace(env=SimpleNamespace(
        physics=SimpleNamespace(data=SimpleNamespace(qpos=np.zeros(2))),
        task=SimpleNamespace(target_qpos=np.zeros(2))))
    monkeypatch.setattr(p0, "_prepare", lambda *a: (torch.zeros(1, 3, 1), torch.zeros(1, 1),
                                                    torch.zeros(1, 3, 10, dtype=torch.float32)))
    monkeypatch.setattr(runner, "_base_distance", lambda *a, **k: (0., {}))
    monkeypatch.setattr(runner, "advance_executed_context", lambda *a, **k: successor)
    monkeypatch.setattr(runner, "restore_execution_snapshot", lambda *a, **k: (frame, frame))
    monkeypatch.setattr(p0, "_verify_execution_identity", lambda *a: {})
    monkeypatch.setattr(p0, "_qpos_distance", lambda *a: 0.)
    monkeypatch.setattr(p0, "_goal_latent", lambda *a: torch.zeros(1))
    monkeypatch.setattr(p0, "_latent_context", lambda *a: torch.zeros(1, 3, 1))
    monkeypatch.setattr(runner, "_observed_goal_cost", lambda *a: (0., "test"))
    observed = {}
    def cem(model, pixels, goal, history, **kwargs):
        observed.update(dtype=history.dtype, kwargs=kwargs)
        return torch.zeros(1, 2, 10, dtype=history.dtype), {"forward_rollout_evaluations": 32}
    monkeypatch.setattr(p0, "_cem", cem)
    monkeypatch.setattr(p0, "_execute", lambda *a, **k: (
        {"initial_qpos_distance": 0., "final_qpos_distance": 0., "min_qpos_distance": 0.}, {}))
    result = runner._two_stage(
        name="test", first_program=lambda *args: (torch.zeros(1, 1, 10, dtype=torch.float64), {}),
        env=env, seed=1, context=context, preprocessor=None, model=None, mean=np.zeros(10, np.float32),
        std=np.ones(10, np.float32), lower=-torch.ones(10), upper=torch.ones(10), action_block=5,
        progress_horizon=2, damping=0., device=torch.device("cpu"), progress_solver="cem",
        cem_samples=16, cem_steps=2, cem_elite=4)
    assert observed["dtype"] == torch.float32
    assert observed["kwargs"]["seed"] == 5001
    assert result["suffix_receipt"]["forward_rollout_evaluations"] == 32
