"""Four-node wrapper wiring with arithmetic fixtures, no model or GPU."""
import numpy as np
import pytest
import torch

from experiments.spectral_reachability import p1_h4_parfit_20261101 as h4
from experiments.spectral_reachability import p1_reverse_h4_parfit_20261101 as wrapper


class TensorLeaf:
    def __init__(self, caps):
        self.driver = type("FixtureCounter", (), {})()
        self.driver.caps = caps.copy()
        self.driver.charged = dict.fromkeys(h4.AXES, 0)
        self.driver.failed = self.driver.closed = False

    def _dispatch(self, phase, kind, cost, fn):
        for key, value in cost.items():
            self.driver.charged[key] += value
            assert self.driver.charged[key] <= self.driver.caps[key]
        return fn()

    def action_encode(self, actions, *, phase):
        return self._dispatch(phase, "action_encode", {"action_encode_calls": 1}, lambda: actions)

    def predict(self, states, actions, *, phase):
        return self._dispatch(phase, "predict", {"predict_calls": 1, "predict_samples": 1},
                              lambda: states + actions[:, -1:, :].mean(dim=-1, keepdim=True))

    def gradient(self, loss, actions, *, phase):
        return self._dispatch(phase, "backward", {"backward_calls": 1, "vjp_rows": 1},
                              lambda: torch.autograd.grad(loss, actions)[0])

    def _abort(self):
        self.driver.failed = True


def test_episode_carries_actual_packets_across_all_boundaries(monkeypatch):
    runner = wrapper.extra.runner
    results = []

    def physical(env, ref, data, layer, base, row, nodes, visual, runtime, meter, store, plan_node):
        control = wrapper.reverse.Q.carry.control
        for boundary in (0, 5, 10, 15):
            raw10 = np.concatenate((ref["raw_history"],
                                     np.asarray(row["executed_raw"], np.float32).reshape(-1, 2)))[-10:]
            task = {"goal": wrapper.reverse.Q.kernels.GOAL,
                    "root_context": {"latent_history": np.zeros((3, 192)).tolist(),
                                     "action_history": runner.rgb.moving.normalized(base, raw10)},
                    "action_mean": base["action_mean"], "action_std": base["action_std"],
                    "projection": np.eye(192).tolist()}
            if boundary == 0:
                invalid = dict(task, projection=np.zeros((192, 192)).tolist())
                with pytest.raises(ValueError, match="identity projection"):
                    plan_node(invalid, "random", boundary, store, runtime)
            result = plan_node(task, "random", boundary, store, runtime)
            results.append(result)
            row["executed_raw"].extend(result["commands"])
            row["last_observation"] = {"pixel_sha256": str(boundary // 5 + 1) * 64}

    monkeypatch.setattr(runner.rgb, "episode", physical)
    exp = wrapper.setup("random")
    control = wrapper.reverse.Q.carry.control

    def bridge(task, arm, boundary, store, runtime):
        leaf = TensorLeaf(h4.node_cost(arm, boundary))
        config = {"steps": 6, "learning_rate": .05, **wrapper.boundary_config(boundary)}
        policy = {"metric": "goal_encoder_cometric_v1", "horizon": h4.dimensions(boundary)[0]}
        contract = {"phase_policy_sha256": control.spec.digest(policy)}
        leaf.config_sha = control.spec.digest(config)
        leaf.contract_sha = control.spec.digest(contract)
        result = control.plan(leaf, arm, task, config, policy, contract,
                              device=torch.device("cpu"), emit=lambda *_: None)
        assert leaf.driver.charged == control.cost(arm)
        packet = control.base.first_packet(result)
        return {"commands": packet.tolist()}

    base = {"action_mean": [0., 0.], "action_std": [1., 1.]}
    ref = {"raw_history": np.zeros((10, 2), dtype=np.float32)}
    row = {"episode_id": "s0_random", "executed_raw": [],
           "last_observation": {"pixel_sha256": "a" * 64}}
    exp.runner.rgb.episode(None, ref, {}, None, base, row, [], None, {}, None, None, bridge)
    assert [len(x["selected_full_plan"]) for x in results] == [4, 3, 2, 1]
    assert results[0]["carry_feedback"] is None
    for i in range(1, 4):
        assert results[i]["carry_feedback"]["previous_plan_sha256"] == results[i-1]["selected_full_plan_sha256"]
