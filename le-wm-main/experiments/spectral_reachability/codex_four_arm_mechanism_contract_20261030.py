"""Four-arm shared-suffix mechanism preparation; NOT a performance launcher.

CPU analytic fixtures are allowed; real models remain subject to LeafAdapter's
frozen CUDA checks. This module never loads a model or executes physics.
"""
import json
import math

import torch

from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_durable_20261030 as durable
from experiments.spectral_reachability import codex_four_arm_leaf_suffix_20261030 as leafmod
from experiments.spectral_reachability import codex_four_arm_raw_candidates_20261030 as raw
from experiments.spectral_reachability import codex_four_arm_raw_suffix_20261030_main as suffix
from experiments.spectral_reachability import codex_four_arm_final_replay_20261030 as replay
from experiments.spectral_reachability import codex_four_arm_spectral8_suffix_20261030 as spectral

SEEDS = {"native": 20261030, "random": 20261031, "immediate_progress": 20261101}
PERF_SCHEMA = "codex_four_arm_own_optimizer_draft.v1"
PERF_PHASES = ("setup", "initialization", "search", "final_replay", "physical")
LABELS = {"structured": spectral.LABEL,
          "native": "native_style_initialization_variant_not_installed_native_cem",
          "random": "clipped_gaussian_random_initializer",
          "immediate_progress": "one_packet_screened_gaussian_initializer"}


def expected_cost(arm, config):
    spec.require(arm in spec.ARMS, "unknown arm")
    result = spectral.required_caps(config)
    if arm != "structured":
        result["action_encode_calls"] -= 3
        result["predict_calls"] -= 3
        result["predict_samples"] -= 3
        result["backward_calls"] -= 192
        result["vjp_rows"] -= 192
        result["jacobian_calls"] = result["svd_calls"] = 0
    if arm == "immediate_progress":
        for axis in ("action_encode_calls", "predict_calls", "predict_samples",
                     "replay_calls", "selection_evaluations"):
            result[axis] += 16
    return result


def make_contract(task, config, model_sha, hardware_sha):
    """Build same-feedback mechanism contract, not production authentication."""
    spec.require(type(task) is dict and set(task) == {"binding", "goal", "projection", "root_context",
                 "action_mean", "action_std"} and spec.is_hash(model_sha) and spec.is_hash(hardware_sha),
                 "task and identity")
    spec.require(config["history_size"] == 3, "three-state shared feedback required")
    costs = {a: expected_cost(a, config) for a in spec.ARMS}
    policy = spectral.policy(task["action_mean"], task["action_std"], 3)
    identity = {"root": task["binding"]["root_sha256"], "task": spec.digest(task),
                "model": model_sha, "action_domain": task["binding"]["action_domain_sha256"],
                "history": spec.digest(task["root_context"]), "goal": spec.digest(task["goal"]),
                "optimizer": spectral.file_sha(suffix.__file__), "optimizer_config": spec.digest(config),
                "meter": spectral.file_sha(durable.__file__), "hardware": hardware_sha}
    arms = {a: {"initializer_sha256": spectral.file_sha(spectral.__file__) if a == "structured"
                else spectral.file_sha(__file__),
                "optimizer_sha256": identity["optimizer"],
                "optimizer_config_sha256": identity["optimizer_config"]} for a in spec.ARMS}
    caps = {axis: max(costs[a][axis] for a in spec.ARMS) for axis in spec.AXES}
    contract = {"schema": spec.NAME+".contract.v1", "identity": identity, "arms": arms,
                "scope": "standalone_per_root_per_arm_including_attributable_setup",
                "phase_policy_sha256": spec.digest(policy),
                "levels": [{"id":"common", "caps":caps},
                           {"id":"unused_larger", "caps":{k:v+1 for k,v in caps.items()}}]}
    spec.preflight(contract, spec.digest(contract))
    return policy, contract


def initial_samples(arm):
    """Fixed CPU draws in raw space; native is a declared variant, not CEM."""
    spec.require(arm in SEEDS, "no direct sample for spectral arm")
    count = 16 if arm == "immediate_progress" else 8
    generator = torch.Generator(device="cpu").manual_seed(SEEDS[arm])
    result = (0.4 * torch.randn((count,3,5,2), generator=generator, dtype=torch.float32)).clamp(-1,1)
    if arm == "native":
        result[0].zero_()
    return result


def performance_requirements():
    return {"implementation_ready": False, "shared_suffix_is_performance_evidence": False,
            "required_own_optimizers": {a: "not_yet_bound_or_authenticated" for a in spec.ARMS},
            "same_feedback_required": True, "total_cost_equality_proven": False,
            "independent_confirmation": False, "physical_execution_authorized": False}


def check_performance_draft(draft):
    """Validate declared vectors only; hashes/estimates are not actual evidence."""
    spec.require(type(draft) is dict and set(draft) == {"schema", "feedback_sha256", "arms", "caps"}
                 and draft["schema"] == PERF_SCHEMA, "performance draft fields")
    spec.require(spec.is_hash(draft["feedback_sha256"]) and type(draft["arms"]) is dict
                 and set(draft["arms"]) == set(spec.ARMS), "feedback/arm fields")
    caps = spec.vector(draft["caps"])
    totals, remaining = {}, {}
    for arm in spec.ARMS:
        item = draft["arms"][arm]
        spec.require(type(item) is dict and set(item) == {"feedback_sha256", "optimizer_source_sha256",
                     "initializer_source_sha256", "phase_costs"}, "arm draft fields")
        spec.require(item["feedback_sha256"] == draft["feedback_sha256"], "same feedback required")
        spec.require(spec.is_hash(item["optimizer_source_sha256"])
                     and spec.is_hash(item["initializer_source_sha256"]), "source identity")
        phases = item["phase_costs"]
        spec.require(type(phases) is dict and set(phases) == set(PERF_PHASES), "phase fields")
        for vector in phases.values():
            spec.vector(vector)
        total = {axis:sum(phases[phase][axis] for phase in PERF_PHASES) for axis in spec.AXES}
        spec.require(all(total[axis] <= caps[axis] for axis in spec.AXES), "declared cap exceeded")
        totals[arm] = total
        remaining[arm] = {axis:caps[axis]-total[axis] for axis in spec.AXES}
    return {"schema":PERF_SCHEMA+".checked", "declared_totals":totals, "remaining_caps":remaining,
            "implementation_ready":False, "fairness_proven":False, "actual_usage_verified":False,
            "source_hashes_verified":False, "physical_execution_authorized":False}


def run_mechanism(leaf, arm, task, config, policy, contract, *, device, emit):
    """Evaluate fixed initialization effects using one shared gradient suffix.

    The supplied event sink must persist before returning; this function cannot
    attest to sink durability or certify external root/model identities.
    """
    spec.require(type(leaf) is leafmod.LeafAdapter, "LeafAdapter required")
    try:
        spec.require(arm in spec.ARMS and callable(emit) and spec.digest(contract) == leaf.contract_sha
                     and spec.digest(config) == leaf.config_sha, "contract binding")
        expected_policy, expected_contract = make_contract(
            task, config, contract["identity"]["model"], contract["identity"]["hardware"])
        spec.require(contract == expected_contract and policy == expected_policy, "task/source/policy binding")
        with leaf.driver.path.open("rb") as stream:
            header = json.loads(stream.readline(1024*1024))["payload"]
        spec.require(header["arm"] == arm, "ledger arm binding")
        cost = expected_cost(arm, config)
        spec.require(all(leaf.driver.charged[k]+cost[k] <= leaf.driver.caps[k] for k in spec.AXES),
                     "full path reserve unavailable")
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        states = tensor(task["root_context"]["latent_history"])[None]
        history = tensor(task["root_context"]["action_history"])[None]
        goal = tensor(task["goal"])[None]
        mean, std, projection = tensor(task["action_mean"]), tensor(task["action_std"]), tensor(task["projection"])
        for value in (states, history, goal, mean, std, projection):
            leaf._tensor(value)
            spec.require(leafmod._finite(value), "finite task tensors")
        spec.require(states.shape == (1,3,192) and history.shape == (1,2,10) and goal.shape == (1,192)
                     and mean.shape == std.shape == (2,) and bool((std>0).all().item())
                     and projection.shape == (192,192)
                     and torch.equal(projection, torch.eye(192, device=states.device)), "shared feedback ABI")
        if arm == "structured":
            return spectral.run(leaf, states, history, goal, mean, std, projection, config, policy, contract, emit=emit)
        samples = initial_samples(arm).to(device=states.device)
        emit("initial_candidates", {"arm": arm, "raw": samples.detach().cpu().tolist(),
                                    "seed": SEEDS[arm], "label": LABELS[arm]})
        if arm == "immediate_progress":
            scores = []
            with torch.no_grad():
                for i in range(16):
                    leaf._dispatch("screening", "replay", {"replay_calls":1}, lambda: None)
                    predicted = raw._normal(leaf, "screening", states, history, samples[i:i+1,:1], mean, std)
                    def score():
                        value = float((predicted-goal).square().mean().item())
                        spec.require(math.isfinite(value), "nonfinite screening score")
                        return value
                    value = leaf._dispatch("screening", "selection", {"selection_evaluations":1}, score)
                    scores.append(value)
                    emit("screening_score", {"index":i, "score":value})
            order = sorted(range(16), key=lambda i:(scores[i],i))[:8]
            emit("screening_selection", {"indices":order})
            samples = samples[order]
        emit("suffix_initial_candidates", {"arm":arm, "raw":samples.detach().cpu().tolist()})
        optimized = torch.cat([suffix.optimize(leaf, states, history, samples[i:i+1], goal, mean, std, config)
                               for i in range(8)], dim=0)
        emit("optimized_candidates", {"arm":arm, "raw":optimized.detach().cpu().tolist()})
        final = replay.replay_and_select(leaf, states, history, optimized, goal, mean, std, policy, contract)
        emit("final_selection", {"scores":final["scores"], "index":final["selected_index"],
                                 "id":final["selected_id"]})
        return {"initial_raw":samples.detach(), "optimized_raw":optimized.detach(),
                "scores":final["scores"], "selected_index":final["selected_index"],
                "selected_id":final["selected_id"], "label":LABELS[arm],
                "fairness_proven":False, "physical_execution_authorized":False}
    except BaseException:
        leaf._abort()
        raise
