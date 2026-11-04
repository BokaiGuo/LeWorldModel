"""Non-oracle depth-3 Atlas pruning alternatives and matched-call experiment."""
import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend
from .pc_exact_unicycle_development import TURN


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parents[2] / "docs/PC_ATLAS_DEPTH3_PRUNING_ALTERNATIVE_PROTOCOL_20261103_CODEX.md"
ROOT = {"layout": "states_with_between_state_actions_v1", "context_length": 2,
        "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
        "action_history": [[0., 0.]]}
MODES = ("v2", "lookahead", "lookahead_diverse", "random")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose_intermediate(candidates, mode, rng, action_range, tolerance):
    """Return selected indices, in beam order, using only ordinary candidates."""
    if mode not in MODES or len(candidates) < 2:
        raise ValueError("known policy and at least two candidates required")
    action_range = np.asarray(action_range, dtype=float)
    if (action_range.ndim != 1 or not np.isfinite(action_range).all()
            or np.any(action_range <= 0) or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("positive finite action ranges and finite nonnegative tolerance required")
    for candidate in candidates:
        if (not np.isfinite(candidate["cost"])
                or np.asarray(candidate["last_action"]).shape != action_range.shape
                or not np.isfinite(candidate["last_action"]).all()):
            raise ValueError("finite candidate score and aligned action required")
    if mode == "v2":
        return sorted(range(len(candidates)), key=lambda i: candidates[i]["cost"])[:2]
    if mode == "random":
        return sorted(int(i) for i in rng.choice(len(candidates), 2, replace=False))
    if any(c["lookahead_cost"] is None or not np.isfinite(c["lookahead_cost"]) for c in candidates):
        raise ValueError("finite lookahead score required")
    ordered = sorted(range(len(candidates)), key=lambda i: candidates[i]["lookahead_cost"])
    if mode == "lookahead":
        return ordered[:2]
    first = ordered[0]
    eligible = [i for i in ordered[1:]
                if candidates[i]["lookahead_cost"] <= candidates[first]["lookahead_cost"] + tolerance]
    if not eligible:
        return ordered[:2]
    anchor = np.asarray(candidates[first]["last_action"])
    second = min(eligible, key=lambda i: (
        -float(np.linalg.norm((np.asarray(candidates[i]["last_action"]) - anchor) / action_range)), i))
    return [first, second]


class IntermediatePolicyAtlas(AtlasSearch):
    """Drop-in independent search: only depth-2 beam policy differs from v2."""

    def __init__(self, compiler, config, *, mode, tolerance=0.01, pad_lookahead=False):
        super().__init__(compiler, config)
        if config.depth != 3 or config.beam_width != 2 or mode not in MODES:
            raise ValueError("depth 3, beam width 2, and known mode required")
        if not np.isfinite(tolerance) or tolerance < 0 or type(pad_lookahead) is not bool:
            raise ValueError("valid policy options required")
        self.mode, self.tolerance, self.pad_lookahead = mode, float(tolerance), pad_lookahead

    def search(self, context):
        c, cfg = self.compiler, self.config
        rng = np.random.default_rng(cfg.seed)
        policy_rng = np.random.default_rng(cfg.seed + 1000003)
        initial = dict(c.counts)
        root_cost = c._cost(context)
        roots = []
        for name, actions in self._bank(context, rng):
            successor = c._call("rollout", context, actions)
            roots.append({"root_name": name, "root_cost": c._cost(successor),
                          "context": successor, "actions": actions.tolist(),
                          "edges": [{"name": name, "actions": actions.tolist(),
                                     "output_sha256": digest(successor)}]})
        greedy = min(range(len(roots)), key=lambda i: (roots[i]["root_cost"], i))
        complete, traces = [], []
        for root_index, root in enumerate(roots):
            beam = [root]
            for depth in (2, 3):
                expanded = []
                for node in beam:
                    for name, actions in self._bank(node["context"], rng):
                        successor = c._call("rollout", node["context"], actions)
                        expanded.append({"root_name": node["root_name"], "root_cost": node["root_cost"],
                                         "context": successor, "actions": node["actions"] + actions.tolist(),
                                         "edges": node["edges"] + [
                                             {"name": name, "actions": actions.tolist(),
                                              "output_sha256": digest(successor)}]})
                if depth == 2:
                    scores = []
                    for node in expanded:
                        predicted = None
                        if self.pad_lookahead or self.mode in ("lookahead", "lookahead_diverse"):
                            predicted = c.progress(node["context"], packets=1)["nonlinear_cost"]
                        scores.append({"cost": c._cost(node["context"]), "lookahead_cost": predicted,
                                       "last_action": node["actions"][-1]})
                    selected = choose_intermediate(scores, self.mode, policy_rng, c.high - c.low,
                                                   self.tolerance)
                    traces.append({"root_index": root_index, "root_name": root["root_name"],
                                   "candidates": [
                                       {"index": i, "names": [e["name"] for e in n["edges"]],
                                        "actions": n["actions"], "context_sha256": digest(n["context"]),
                                        **scores[i], "retained": i in selected}
                                       for i, n in enumerate(expanded)],
                                   "selected_indices_in_beam_order": selected})
                    beam = [expanded[i] for i in selected]
                else:
                    beam = sorted(expanded, key=lambda node: c._cost(node["context"]))[:cfg.beam_width]
            best = beam[0]
            direct = c._call("rollout", context, np.asarray(best["actions"]))
            if digest(direct) != digest(best["context"]):
                raise ValueError("full action witness differs from split composition")
            complete.append({"root_index": root_index, "root_name": best["root_name"],
                             "prefix_cost": best["root_cost"], "terminal_cost": c._cost(direct),
                             "actions": best["actions"], "edges": best["edges"],
                             "output_context": direct, "witness_replayed": True})
        selected = min(range(len(complete)), key=lambda i: (complete[i]["terminal_cost"], i))
        control_cost = min(complete[0]["terminal_cost"], complete[greedy]["terminal_cost"])
        eligible = [i for i, item in enumerate(complete)
                    if item["root_name"] not in ("zero", "progress")
                    and item["prefix_cost"] >= root_cost - cfg.nonprogress_tolerance
                    and item["terminal_cost"] + cfg.improvement_margin < control_cost]
        qualified = min(eligible, key=lambda i: (complete[i]["terminal_cost"], i)) if eligible else None
        return {"schema": "pc_atlas_depth3_intermediate_policy.v1",
                "configuration": {**vars(cfg), "mode": self.mode, "tolerance": self.tolerance,
                                  "pad_lookahead": self.pad_lookahead},
                "root_cost": root_cost, "programs": complete, "selected_index": selected,
                "zero_index": 0, "same_bank_greedy_index": greedy,
                "qualified_reconfiguration_index": qualified,
                "operation_counts": {k: c.counts[k] - initial[k] for k in c.counts},
                "intermediate_trace": traces,
                "scope": "model_only_no_privileged_actions_no_physical_claim"}


class MeteredUnicycle(UnicycleBackend):
    def __init__(self):
        self.steps = {"rollout_attempted": 0, "rollout_returned": 0,
                      "jacobian_attempted": 0, "jacobian_returned": 0}

    def rollout(self, context, actions):
        self.steps["rollout_attempted"] += len(actions)
        result = super().rollout(context, actions)
        self.steps["rollout_returned"] += len(actions)
        return result

    def jacobian(self, context, actions):
        self.steps["jacobian_attempted"] += len(actions)
        result = super().jacobian(context, actions)
        self.steps["jacobian_returned"] += len(actions)
        return result


def new_compiler(goal):
    backend = MeteredUnicycle()
    compiler = LocalCompiler(
        backend, goal=[*goal, 0.], sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
        config=CompilerConfig(packet_steps=1, spectral_modes=2,
                              max_rollout_operations=10000, max_jacobian_operations=2000))
    return compiler, backend


def signature(result):
    return {"selected_index": result["selected_index"],
            "same_bank_greedy_index": result["same_bank_greedy_index"],
            "qualified_reconfiguration_index": result["qualified_reconfiguration_index"],
            "programs": [{"root_name": p["root_name"], "actions": p["actions"],
                          "prefix_cost": p["prefix_cost"], "terminal_cost": p["terminal_cost"],
                          "output_sha256": digest(p["output_context"])}
                         for p in result["programs"]]}


def run_pair(index, seed):
    angle = 2 * np.pi * index / 64 + .031
    goal = [float(np.cos(angle)), float(np.sin(angle))]
    cfg = AtlasConfig(depth=3, beam_width=2, seed=seed)
    compiler, backend = new_compiler(goal)
    original = AtlasSearch(compiler, cfg).search(ROOT)
    original_counts, original_steps = original["operation_counts"], deepcopy(backend.steps)
    arms = {}
    for mode in MODES:
        compiler, backend = new_compiler(goal)
        result = IntermediatePolicyAtlas(compiler, cfg, mode=mode, pad_lookahead=True).search(ROOT)
        result["backend_action_steps"] = deepcopy(backend.steps)
        selected = result["programs"][result["selected_index"]]
        position = np.asarray(selected["output_context"]["latent_history"][-1][:2])
        result["position_error"] = float(np.linalg.norm(position - goal))
        arms[mode] = result
    if signature(original) != signature(arms["v2"]):
        raise RuntimeError("padded v2 does not reproduce original search")
    accounting = {(tuple(r["operation_counts"].values()), tuple(r["backend_action_steps"].values()))
                  for r in arms.values()}
    if len(accounting) != 1:
        raise RuntimeError(f"call/action-step budget mismatch at goal {index}, seed {seed}")
    return {"index": index, "seed": seed, "goal": goal, "arms": arms,
            "original_v2_parity": True, "original_v2_counts": original_counts,
            "original_v2_backend_action_steps": original_steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Run non-oracle depth-3 pruning arms with matched trial calls and independent v2 parity checks.")
    pairs = [run_pair(i, 42 + i + offset) for offset in (0, 100, 200) for i in range(8)]
    summary = {}
    for mode in MODES:
        errors = [p["arms"][mode]["position_error"] for p in pairs]
        summary[mode] = {"mean_position_error": float(np.mean(errors)),
                         "mean_model_cost": float(np.mean([
                             p["arms"][mode]["programs"][p["arms"][mode]["selected_index"]]["terminal_cost"]
                             for p in pairs]))}
    dependencies = [Path(__file__), PROTOCOL, HERE / "d2p_atlas_search_v2.py",
                    HERE / "d2p_local_compiler.py", HERE / "d2p_composition_witness_v2.py",
                    HERE / "pc_atlas_unicycle_smoke_20261103.py",
                    HERE / "pc_exact_unicycle_development.py"]
    report = {"schema": "pc_atlas_depth3_pruning_alternative.v1",
              "scope": "exposed analytical development; no oracle actions; no physical or independent claim",
              "dependency_sha256": {p.name: sha(p) for p in dependencies},
              "numpy_version": np.__version__, "python_version": sys.version,
              "summary": summary, "pairs": pairs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
