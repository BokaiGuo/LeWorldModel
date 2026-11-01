"""Structured model-only planner on the prospective common-budget node.

No environment dispatch, fairness, or independent-confirmation authorization.
"""
import json
from pathlib import Path

import torch

from . import p1_common_budget_node_20261101_leibniz as common
from . import p1_common_budget_cem_events_20261101_leibniz as common_v2
from . import codex_four_arm_common_suffix_control_20261030 as original

spec, leafmod = original.spec, original.leafmod
DEPENDENCIES = (original, original.spectral, original.suffix, original.raw, original.replay)


def plan(leaf, task, protocol, protocol_sha, boundary, *, device, emit):
    """Plan with a caller-owned, already-opened common structured node.

    Caller persists emitted records and closes/consumes the durable ledger.
    Feedback authentication, physical execution and source transitive closure
    remain external responsibilities.
    """
    spec.require(type(leaf) is leafmod.LeafAdapter, "leaf")
    try:
        spec.require(callable(emit), "emit")
        is_v2 = protocol.get("schema") == "p1_common_budget_cem_events_20261101_leibniz.v1"
        sources = protocol["base_protocol"]["sources"] if is_v2 else protocol["sources"]
        required = {str(Path(module.__file__).resolve()) for module in DEPENDENCIES}
        required.add(str(Path(__file__).resolve()))
        spec.require(required <= set(sources), "structured planner sources missing")
        # Recover the old task-dependent contract, then permit only the exact
        # caps substitution produced by the common-budget binder.
        with leaf.driver.path.open("rb") as stream:
            header = json.loads(stream.readline(1024 * 1024))["payload"]
        current = header["contract"]
        policy, template, config = original.contract_for(
            task, current["identity"]["model"], current["identity"]["hardware"])
        binder = common_v2.bind if is_v2 else common.bind
        bound = binder(protocol, protocol_sha, "structured", boundary, template)
        spec.require(current == bound and spec.digest(bound) == leaf.contract_sha
                     and leaf.config_sha == spec.digest(config)
                     and header["arm"] == "structured" and header["level"] == "node"
                     and header["run_id"] == f"structured_b{boundary:02}",
                     "common structured contract binding")
        expected = original.cost("structured")
        spec.require(all(leaf.driver.charged[k] + expected[k] <= leaf.driver.caps[k]
                         for k in spec.AXES), "full node reserve unavailable")
        tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        states = tensor(task["root_context"]["latent_history"])[None]
        history = tensor(task["root_context"]["action_history"])[None]
        goal = tensor(task["goal"])[None]
        mean, std = tensor(task["action_mean"]), tensor(task["action_std"])
        projection = tensor(task["projection"])
        for value in (states, history, goal, mean, std, projection):
            leaf._tensor(value)
            spec.require(leafmod._finite(value), "nonfinite input")
        spec.require(states.shape == (1, 3, 192) and history.shape == (1, 2, 10)
                     and goal.shape == (1, 192) and mean.shape == std.shape == (2,)
                     and bool((std > 0).all().item()) and projection.shape == (192, 192)
                     and torch.equal(projection, torch.eye(192, device=states.device)), "feedback ABI")
        emit("node_header", {"arm": "structured", "task_sha256": spec.digest(task),
                             "label": original.LABELS["structured"], "config": config})
        leaf._dispatch("jacobian", "jacobian", {"jacobian_calls": 1}, lambda: None)
        nominal = torch.zeros((1, 3, 5, 2), device=states.device, requires_grad=True)
        pred = original.raw._normal(leaf, "jacobian", states.detach(), history.detach(), nominal, mean, std)
        rows = []
        for i in range(192):
            grad = leaf._dispatch("jacobian", "backward", {"backward_calls": 1, "vjp_rows": 1},
                                  lambda i=i: torch.autograd.grad(pred[0, i], nominal, retain_graph=i < 191)[0])
            spec.require(grad.shape == nominal.shape and leafmod._finite(grad), "invalid Jacobian")
            rows.append(grad.detach().reshape(-1))
        jac = torch.stack(rows).cpu().numpy()
        emit("spectral_signal", {"nominal": pred.detach().cpu().tolist(), "jacobian": jac.tolist(),
                                 "projection": projection.cpu().tolist()})
        bank, singular = leaf._dispatch("initialization", "svd", {"svd_calls": 1},
                                        lambda: original.spectral.spectral_bank(jac, projection.cpu().numpy()))
        initial = torch.from_numpy(bank).to(device=states.device, dtype=states.dtype)
        emit("spectral_values", {"singular_values": singular.tolist()})
        emit("initial_candidates", {"raw": initial.detach().cpu().tolist()})
        optimized = torch.cat([original.suffix.optimize(leaf, states, history, initial[i:i + 1],
                                                        goal, mean, std, config) for i in range(8)])
        emit("optimized_candidates", {"raw": optimized.detach().cpu().tolist()})
        final = original.replay.replay_and_select(leaf, states, history, optimized, goal, mean, std, policy, bound)
        emit("final_selection", {"scores": final["scores"], "index": final["selected_index"], "id": final["selected_id"]})
        return {"initial_raw": initial.detach(), "optimized_raw": optimized.detach(),
                "selected_raw": optimized[final["selected_index"]].detach(), "scores": final["scores"],
                "selected_index": final["selected_index"], "label": original.LABELS["structured"],
                "fairness_proven": False, "physical_execution_authorized": False,
                "independent_confirmation": False}
    except BaseException:
        leaf._abort()
        raise
