"""Four initialization variants with a common metered raw-coordinate CEM.

New engineering variant, not the installed native CEM or a fairness claim.
Calls are charged only at the existing cooperative dispatch boundaries.
"""
import json
import math

import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay

CONFIG_SCHEMA = "codex_four_arm_raw_candidates_20261030.config.v1"


def _normal(leaf, phase, states, history, raw, mean, std):
    h = states.shape[1]
    z, past = states, history
    for step in range(raw.shape[1]):
        current = ((raw[:, step] - mean) / std).reshape(1, 1, 10)
        context = torch.cat((past, current), dim=1)
        encoded = leaf.action_encode(context, phase=phase)
        out = leaf.predict(z, encoded, phase=phase)
        spec.require(isinstance(out, torch.Tensor) and out.shape == z.shape
                     and leafmod._finite(out), "invalid prediction")
        z = torch.cat((z, out[:, -1:]), dim=1)[:, -h:]
        past = context[:, -(h-1):] if h > 1 else context[:, :0]
    return z[:, -1]


def run_arm(leaf, arm, states, history, goal, mean, std, config, policy, contract):
    """One declared arm; CEM samples all candidates and replays final generation."""
    spec.require(type(leaf) is leafmod.LeafAdapter, "leaf adapter required")
    try:
        spec.require(spec.digest(contract) == leaf.contract_sha and spec.digest(config) == leaf.config_sha,
                     "contract/config identity")
        with leaf.driver.path.open("rb") as stream:
            header = json.loads(stream.readline(1024 * 1024))["payload"]
        spec.require(arm in spec.ARMS and header["arm"] == arm, "driver arm mismatch")
        spec.require(type(config) is dict and set(config) ==
                     {"schema", "iterations", "elite", "seed", "initial_std", "std_floor", "ridge"}
                     and config["schema"] == CONFIG_SCHEMA, "exact CEM config")
        spec.require(type(config["iterations"]) is int and config["iterations"] > 0
                     and type(config["seed"]) is int and 0 <= config["seed"] < 2**63
                     and type(config["elite"]) is int and config["elite"] > 0, "CEM integer config")
        for name in ("initial_std", "std_floor", "ridge"):
            value = config[name]
            spec.require(type(value) in (int, float) and math.isfinite(value) and value > 0, "CEM positive config")
        spec.require(config["std_floor"] <= config["initial_std"], "CEM std floor")
        spec.require(type(policy) is dict and set(policy) ==
                     {"schema", "candidate_ids", "horizon", "history_size", "metric", "tie", "replay_batch",
                      "action_chart", "mean", "std"}
                     and spec.digest(policy) == contract["phase_policy_sha256"]
                     and policy.get("schema") == replay.POLICY_SCHEMA, "replay policy identity")
        ids = policy["candidate_ids"]
        count = len(ids)
        spec.require(count >= 2 and config["elite"] <= count, "population/elite")
        # Validate contract and normalizer using the same rules as final replay
        # before any model operation; avoid charging an artificial zero replay.
        spec.require(policy.get("action_chart") == "raw_packet_5x2_box_minus1_to1"
                     and policy.get("metric") == "terminal_latent_mean_squared_error"
                     and policy.get("tie") == "first_declared_candidate"
                     and type(policy.get("replay_batch")) is int and policy["replay_batch"] == 1
                     and type(ids) is list and all(type(x) is str and x and x.strip() == x for x in ids)
                     and len(set(ids)) == count, "fixed replay policy")
        horizon, h = policy["horizon"], policy["history_size"]
        spec.require(type(horizon) is int and horizon > 0 and type(h) is int and h > 0, "horizon/history")
        for t in (states, history, goal, mean, std):
            leaf._tensor(t)
            spec.require(t.is_floating_point() and leafmod._finite(t) and
                         t.device == states.device and t.dtype == states.dtype, "finite aligned tensor")
        spec.require(states.ndim == 3 and states.shape[:2] == (1, h) and history.shape == (1, h-1, 10)
                     and goal.shape == (1, states.shape[2]) and mean.shape == std.shape == (2,)
                     and bool((std > 0).all().item()), "task/action shape")
        spec.require(torch.equal(mean, torch.tensor(policy["mean"], dtype=mean.dtype, device=mean.device))
                     and torch.equal(std, torch.tensor(policy["std"], dtype=std.dtype, device=std.device)),
                     "normalizer mismatch")
        generator = torch.Generator(device="cpu").manual_seed(config["seed"])
        shape = (1, horizon, 5, 2)
        center = torch.zeros(shape, dtype=states.dtype, device=states.device)
        scale = torch.full_like(center, config["initial_std"])

        def sample(n):
            noise = torch.randn((n, horizon, 5, 2), generator=generator, dtype=states.dtype, device="cpu")
            return (center + scale * noise.to(states.device)).clamp(-1, 1)

        def evaluate(raw):
            return replay.replay_and_select(leaf, states, history, raw, goal, mean, std, policy, contract)

        if arm == "random":
            center = sample(1).detach()
        elif arm == "immediate_progress":
            trials = sample(count)
            scores = []
            with torch.no_grad():
                for index in range(count):
                    leaf._dispatch("screening", "replay", {"replay_calls": 1}, lambda: None)
                    predicted = _normal(leaf, "screening", states, history, trials[index:index+1, :1], mean, std)
                    def score():
                        val = float((predicted - goal).square().mean().item())
                        spec.require(math.isfinite(val), "nonfinite immediate score")
                        return val
                    scores.append(leaf._dispatch("screening", "selection", {"selection_evaluations": 1}, score))
            winner = min(range(count), key=lambda i: (scores[i], i))
            center = trials[winner:winner+1]
        elif arm == "structured":
            leaf._dispatch("jacobian", "jacobian", {"jacobian_calls": 1}, lambda: None)
            raw = center.detach().requires_grad_(True)
            predicted = _normal(leaf, "jacobian", states.detach(), history.detach(), raw, mean, std)
            dim = predicted.shape[1]
            rows = []
            for i in range(dim):
                row = leaf._dispatch("jacobian", "backward", {"backward_calls": 1, "vjp_rows": 1},
                    lambda i=i: torch.autograd.grad(predicted[0, i], raw, retain_graph=i < dim-1)[0])
                spec.require(leafmod._finite(row), "nonfinite Jacobian")
                rows.append(row.reshape(-1))
            jac = torch.stack(rows)
            target = (goal.detach() - predicted.detach()).reshape(-1)
            def solve():
                eye = torch.eye(jac.shape[1], dtype=jac.dtype, device=jac.device)
                return torch.linalg.solve(jac.T @ jac + config["ridge"] * eye, jac.T @ target)
            correction = leaf._dispatch("initialization", "solve", {"solve_calls": 1}, solve)
            spec.require(leafmod._finite(correction), "nonfinite structured correction")
            center = correction.reshape(shape).clamp(-1, 1).detach()
        # native uses the zero-centered native-style CEM distribution only;
        # this is not the separately implemented native CEM baseline.
        for _ in range(config["iterations"]):
            samples = sample(count)
            measured = evaluate(samples)
            elite = sorted(range(count), key=lambda i: (measured["scores"][i], i))[:config["elite"]]
            chosen = samples[elite]
            center = chosen.mean(dim=0, keepdim=True)
            scale = chosen.std(dim=0, unbiased=False, keepdim=True).clamp_min(config["std_floor"])
        final = evaluate(samples)
        index = final["selected_index"]
        return {"selected_raw": final["actions"][index:index+1].detach().clone(),
                "final_candidate_ids": final["candidate_ids"], "final_scores": final["scores"],
                "selected_index": index, "arm": arm,
                "optimizer": "raw_clipped_gaussian_cem_variant",
                "fairness_proven": False, "physical_execution_authorized": False}
    except BaseException:
        leaf._abort()
        raise
