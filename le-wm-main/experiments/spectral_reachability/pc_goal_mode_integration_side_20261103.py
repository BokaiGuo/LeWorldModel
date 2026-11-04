"""Opt-in task-mode compiler and bounded exact-backend integration check."""
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import d2p_atlas_search_v2 as atlas
from . import d2p_goal_mode_rank_20261102 as modes
from . import d2p_local_compiler as local
from . import pc_atlas_unicycle_smoke_20261103 as unicycle
from . import pc_response_program_joint_refinement_20261103_codex as joint

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_GOAL_MODE_INTEGRATION_SIDE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_goal_mode_integration_side_20261103.json"
INITIAL = {
    "layout": "states_with_between_state_actions_v1", "context_length": 2,
    "latent_dim": 3, "action_dim": 2,
    "latent_history": [[0., 0., 0.], [0., 0., 0.]], "action_history": [[0., 0.]],
}


def geometry(jacobian, residual, sqrt_q, inverse_sqrt_r, damping, count, rule):
    """One SVD; ridge update and bridge basis share the same selected modes."""
    if rule not in ("task", "gain") or type(count) is not int or count < 1:
        raise ValueError("explicit mode rule and positive integer count required")
    g, r, q, ri = map(local._array, (jacobian, residual, sqrt_q, inverse_sqrt_r))
    if (g.ndim != 2 or min(g.shape) < 1 or r.shape != (g.shape[0],)
            or q.shape != (g.shape[0], g.shape[0]) or ri.shape != (g.shape[1], g.shape[1])):
        raise ValueError("aligned response, residual and metric factors required")
    matrix, weighted = q @ g @ ri, q @ r
    ranked = modes.rank_goal_modes(matrix, weighted, damping)
    singular = ranked["singular"]
    tolerance = singular[0] * max(matrix.shape) * np.finfo(float).eps
    valid = singular > tolerance
    order = ranked["order"] if rule == "task" else np.arange(len(singular))
    selected = [int(i) for i in order if valid[i]][:count]
    w = ranked["right_t"][selected].T @ ranked["coefficients"][selected]
    remainder = weighted - matrix @ w
    basis = ranked["left"][:, selected]
    perpendicular = weighted - basis @ (basis.T @ weighted)
    return {
        "rank": int(valid.sum()), "retained_rank": len(selected),
        "mode_rule": rule, "selected_mode_indices": selected,
        "singular_values": singular.tolist(),
        "mode_contributions": ranked["contributions"].tolist(),
        "raw_delta": (ri @ w).tolist(),
        "raw_spectral_directions": (ri @ ranked["right_t"][selected].T).T.tolist(),
        "unreachable_fraction": float(perpendicular @ perpendicular / (weighted @ weighted + 1e-12)),
        "unreachable_residual": perpendicular.tolist(),
        "linear_reach_value": float((remainder @ remainder + damping * (w @ w)) / len(r)),
        "linear_reach_scope": "selected_modes_unconstrained_ridge_not_physical_capability",
    }


class TaskModeCompiler(local.LocalCompiler):
    """Explicit opt-in: defaults and prior experiment semantics are unchanged."""

    def __init__(self, *args, mode_rule, atlas_config, **kwargs):
        if mode_rule not in ("task", "gain") or not isinstance(atlas_config, atlas.AtlasConfig):
            raise ValueError("mode rule and frozen Atlas configuration required")
        super().__init__(*args, **kwargs)
        self.mode_rule, self.atlas_config = mode_rule, atlas_config
        self.mode_trace = []
        self.svd_calls = 0

    def geometry(self, context, actions):
        nominal = self._call("rollout", context, actions)
        jacobian = self._call("jacobian", context, actions)
        ri = np.diag(np.tile(self.scale, len(actions)))
        result = geometry(jacobian, self.goal - np.asarray(nominal["latent_history"][-1]),
                          self.q, ri, self.config.damping, self.config.spectral_modes, self.mode_rule)
        self.svd_calls += 1
        self.mode_trace.append(deepcopy(result))
        return nominal, result

    def compile(self, context):
        before = len(self.mode_trace)
        result = atlas.AtlasSearch(self, self.atlas_config).search(context)
        # Adapt metadata without changing selection or the action witness.
        for program in result["programs"]:
            program["candidate"] = program["root_name"]
        result["schema"] = "pc_goal_mode_compiler.side.v1"
        result["mode_rule"] = self.mode_rule
        result["mode_trace"] = deepcopy(self.mode_trace[before:])
        result["svd_calls"] = len(self.mode_trace) - before
        result["selected_is_model_qualified"] = (
            result["selected_index"] == result["qualified_reconfiguration_index"])
        return result


class MeteredBackend(unicycle.UnicycleBackend):
    def __init__(self):
        self.counters = {"rollout_action_steps": 0, "jacobian_action_steps": 0}

    def rollout(self, context, actions):
        self.counters["rollout_action_steps"] += len(actions)
        return super().rollout(context, actions)

    def jacobian(self, context, actions):
        self.counters["jacobian_action_steps"] += len(actions)
        return super().jacobian(context, actions)


def make_compiler(goal, index, rule):
    return TaskModeCompiler(
        MeteredBackend(), goal=[*goal, 0.], sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[1., 1.], low=[-1., -unicycle.TURN], high=[1., unicycle.TURN],
        config=local.CompilerConfig(packet_steps=1, spectral_modes=1,
                                    max_rollout_operations=1000, max_jacobian_operations=100),
        mode_rule=rule, atlas_config=atlas.AtlasConfig(depth=2, beam_width=2,
                                                     random_bridges=2, seed=2026111200 + index))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_joint():
    report = json.loads(joint.OUTPUT.read_text())
    for key, path in (("protocol_sha256", joint.PROTOCOL), ("runner_sha256", Path(joint.__file__)),
                      ("dynamics_sha256", Path(joint.dynamics.__file__)),
                      ("gradient_sha256", Path(joint.strong.__file__))):
        if sha(path) != report[key]:
            raise ValueError(f"joint source binding failed: {key}")
    if len(report["rows"]) != 16:
        raise ValueError("joint panel incomplete")
    replays, max_error = 0, 0.
    for row in report["rows"]:
        goal = np.asarray(row["goal"])
        for cap in (64, 128, 256):
            items = [x for x in row["results"] if x["cap"] == cap]
            if len(items) != 4 or {x["method"] for x in items} != set(joint.METHODS):
                raise ValueError("missing joint arm")
            arms = {x["method"]: x for x in items}
            for arm in items:
                replay, _ = joint.dynamics.evaluate(joint.actions(arm["point"]), goal)
                difference = abs(replay - arm["terminal_distance"])
                max_error = max(max_error, difference)
                if difference > 1e-12 or not (
                    arm["total_calls"] == arm["generation_calls"] + arm["optimizer_objectives"] <= cap):
                    raise ValueError("joint action or budget mismatch")
                replays += 1
            observed = joint.intervention(goal, arms["response"]["point"], arms["same_bank_greedy"]["point"])
            if observed != arms["response"]["intervention"]:
                raise ValueError("joint final-chain mechanism mismatch")
    for cap in (64, 128, 256):
        entries = [x for row in report["rows"] for x in row["results"] if x["cap"] == cap]
        for method in joint.METHODS:
            mean = float(np.mean([x["terminal_distance"] for x in entries if x["method"] == method]))
            if not np.isclose(mean, report["summary"][str(cap)][method]["mean_error"], rtol=0, atol=1e-15):
                raise ValueError("joint summary mismatch")
        witnesses = sum(x["intervention"]["limited_witness"] for x in entries if x["method"] == "response")
        if witnesses != report["summary"][str(cap)]["limited_witnesses"]:
            raise ValueError("joint witness count mismatch")
    return {"source_sha256": sha(joint.OUTPUT), "replayed_selected_chains": replays,
            "max_distance_discrepancy": max_error, "summary": report["summary"]}


def run():
    rows = []
    for index in range(16):
        angle = 2 * np.pi * (index + .37) / 16
        goal = .9 * np.array([np.cos(angle), np.sin(angle)])
        arms = {}
        for rule in ("gain", "task"):
            started = time.monotonic()
            compiler = make_compiler(goal, index, rule)
            result = compiler.compile(deepcopy(INITIAL))
            elapsed = time.monotonic() - started
            chosen = result["programs"][result["selected_index"]]
            error, _ = unicycle.evaluate(chosen["actions"], goal)
            result["terminal_distance"] = error
            result["planning_seconds"] = elapsed
            result["backend_action_steps"] = dict(compiler.backend.counters)
            arms[rule] = result
        rows.append({"index": index, "goal": goal.tolist(), "arms": arms,
                     "identical_selected_actions": arms["gain"]["programs"][arms["gain"]["selected_index"]]["actions"]
                     == arms["task"]["programs"][arms["task"]["selected_index"]]["actions"]})
    return rows


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    audit = audit_joint()
    print("Joint result audited; running exact-backend task/gain integration only.", flush=True)
    rows = run()
    summary = {
        "identical_selected_actions": sum(x["identical_selected_actions"] for x in rows),
        "mean_terminal_distance": {rule: float(np.mean([x["arms"][rule]["terminal_distance"] for x in rows]))
                                   for rule in ("gain", "task")},
        "observed_ranks": sorted({g["rank"] for x in rows for arm in x["arms"].values()
                                 for g in arm["mode_trace"]}),
        "geometry_calls": sum(len(arm["mode_trace"]) for x in rows for arm in x["arms"].values()),
    }
    dependencies = (PROTOCOL, Path(__file__), Path(atlas.__file__), Path(local.__file__),
                    Path(modes.__file__), Path(unicycle.__file__), Path(joint.__file__),
                    Path(joint.dynamics.__file__), Path(joint.strong.__file__))
    report = {"scope": "exact backend integration development, no visual model or gate admission",
              "dependencies": {str(path): sha(path) for path in dependencies},
              "joint_saved_audit": audit, "rows": rows, "summary": summary,
              "full_cost_certified": False, "P1_admission": False}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
