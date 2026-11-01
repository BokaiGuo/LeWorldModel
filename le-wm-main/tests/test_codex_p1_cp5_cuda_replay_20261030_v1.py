import json
from pathlib import Path

import pytest
import torch

from experiments.spectral_reachability import codex_p1_cp5_cuda_replay_20261030_v1 as m
from experiments.spectral_reachability import codex_p1_suffix_budget_outcome_20261030_v1 as budget
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod


def row():
    path = Path(__file__).resolve().parents[1] / "results/codex_p1_three_prefix_blind_inputs_20261030_v1.json"
    return json.loads(path.read_text())["rows"][0]


def test_encoder_actual_leaf_calls_failure_and_block():
    events = []
    caps = dict.fromkeys(budget.AXES, 0)
    caps.update(encoder=1, projector=1)
    meter = budget.Budget(caps, events.append, wall_seconds=10)
    class Model:
        encoder = torch.nn.Identity()
        projector = torch.nn.Identity()
        predictor = torch.nn.Identity()
        action_encoder = torch.nn.Identity()
        pred_proj = torch.nn.Identity()
        def predict(self, *args): return None
    model = Model()
    with m.encode_modules(model, meter, lambda: None) as counts:
        model.encoder(torch.ones(1))
        model.projector(torch.ones(1))
        with pytest.raises(ValueError, match="resource cap"):
            model.encoder(torch.ones(1))
        with pytest.raises(ValueError, match="failed"):
            model.projector(torch.ones(1))
    assert counts == {"encoder": 2, "projector": 1}
    assert meter.counts["encoder"]["attempted"] == 1
    assert [e["kind"] for e in events] == ["attempt", "return", "attempt", "return"]
    assert torch.equal(model.encoder(torch.ones(1)), torch.ones(1))


def test_encoder_actual_exception_is_preserved():
    events = []
    meter = budget.Budget(dict.fromkeys(budget.AXES, 1), events.append, wall_seconds=10)
    class Broken(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("encoder failure")
    class Model:
        encoder, projector = Broken(), torch.nn.Identity()
        predictor, action_encoder, pred_proj = torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()
        def predict(self, *args): return None
    with m.encode_modules(Model(), meter, lambda: None):
        with pytest.raises(RuntimeError, match="encoder failure"):
            Model.encoder(torch.ones(1))
    assert meter.counts["encoder"] == {"attempted": 1, "returned": 0, "failed": 1}
    assert events[-1]["kind"] == "failure"


def test_production_encoding_rejects_cpu():
    meter = budget.Budget(dict.fromkeys(budget.AXES, 1), lambda e: None, wall_seconds=10)
    with pytest.raises(ValueError, match="host CUDA"):
        m.encode_cp5(row(), torch.nn.Identity(), meter, None, "cpu")


def test_replay_charges_post_update_actions(tmp_path):
    class Model:
        def encode(self, x): return x
        def action_encoder(self, x): return x
        def predict(self, z, a): return z + a.mean(dim=-1, keepdim=True)
    cfg = {"steps": 2, "learning_rate": 0.05, "history_size": 3}
    r = row()
    policy = {"schema": m.replay.POLICY_SCHEMA, "candidate_ids": ["optimized"],
              "horizon": 2, "history_size": 3, "metric": "terminal_latent_mean_squared_error",
              "tie": "first_declared_candidate", "replay_batch": 1,
              "action_chart": "raw_packet_5x2_box_minus1_to1",
              "mean": torch.tensor(r["action_mean"]).tolist(), "std": torch.tensor(r["action_std"]).tolist()}
    identity = {k: "a"*64 for k in spec.IDENTITY}
    identity["optimizer_config"] = spec.digest(cfg)
    arms = {a: {"initializer_sha256": "b"*64, "optimizer_sha256": identity["optimizer"],
                "optimizer_config_sha256": identity["optimizer_config"]} for a in spec.ARMS}
    caps = dict.fromkeys(spec.AXES, 100)
    contract = {"schema": spec.NAME+".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id": "a", "caps": caps},
                           {"id": "b", "caps": {k: v+1 for k, v in caps.items()}}]}
    sha = spec.digest(contract)
    driver = durable.Driver(tmp_path/"ledger", contract, sha, "structured", "a", "cp5")
    leaf = leafmod.LeafAdapter(Model(), driver, contract, sha, analytic_fixture=True)
    result = m.plan_and_replay(r, torch.zeros(192), leaf, cfg, policy, contract)
    assert result["actions"].shape == (1, 2, 5, 2)
    assert driver.charged["action_encode_calls"] == driver.charged["predict_calls"] == 6
    assert driver.charged["backward_calls"] == 2
    assert driver.charged["replay_calls"] == driver.charged["selection_evaluations"] == 1
    driver.finish()
