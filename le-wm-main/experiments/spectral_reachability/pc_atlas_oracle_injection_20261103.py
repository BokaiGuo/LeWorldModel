"""Diagnostic insertion of a strong planner's action witness into v2 atlas."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend
from .pc_exact_unicycle_development import TURN, evaluate
from .pc_exact_unicycle_strong_baseline import multistart_gradient


class InjectedAtlas(AtlasSearch):
    """Append the prescribed prefix and its suffix only at matching contexts."""

    def __init__(self, compiler, config, *, root, actions):
        super().__init__(compiler, config)
        self.actions = np.asarray(actions, dtype=np.float64)
        self.root_digest = digest(root)
        self.prefix_context = compiler.backend.rollout(root, self.actions[:1])
        self.successor_digest = digest(self.prefix_context)
        self.injection_visits = 0

    def _bank(self, context, rng):
        bank = super()._bank(context, rng)
        context_hash = digest(context)
        if context_hash == self.root_digest:
            bank.append(("injected_prefix", self.actions[:1].copy()))
        if context_hash == self.successor_digest:
            self.injection_visits += 1
            bank.append(("injected_suffix", self.actions[1:].copy()))
        return bank


def diagnose(index):
    angle = 2 * np.pi * index / 64 + .031
    goal = np.array([np.cos(angle), np.sin(angle)])
    root = {"layout": "states_with_between_state_actions_v1", "context_length": 2,
            "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
            "action_history": [[0., 0.]]}
    actions, gradient_ledger = multistart_gradient(goal)
    backend = UnicycleBackend()
    compiler = LocalCompiler(
        backend, goal=[*goal, 0.], sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
        config=CompilerConfig(packet_steps=1, spectral_modes=2,
                              max_rollout_operations=1000, max_jacobian_operations=100))
    search = InjectedAtlas(compiler, AtlasConfig(seed=42+index), root=root, actions=actions)
    direct = backend.rollout(root, actions)
    split = backend.rollout(search.prefix_context, actions[1:])
    direct_hash, split_hash = digest(direct), digest(split)
    if direct_hash != split_hash:
        raise ValueError("injected full and split replay diverged")
    result = search.search(root)
    injected = [p for p in result["programs"] if p["root_name"] == "injected_prefix"]
    if len(injected) != 1 or not search.injection_visits:
        raise ValueError("injected action witness was not offered at matching successor")
    branch = injected[0]
    candidate_cost = compiler._cost(direct)
    branch_cost = branch["terminal_cost"]
    if branch_cost > candidate_cost + 1e-12:
        raise ValueError("branch rejected an offered lower-cost suffix")
    selected = result["programs"][result["selected_index"]]
    return {
        "index": index, "goal": goal.tolist(), "gradient_actions": actions.tolist(),
        "gradient_ledger": gradient_ledger,
        "diagnostic_external_rollouts": 3,
        "prefix_cost": compiler._cost(search.prefix_context),
        "candidate_terminal_cost": candidate_cost,
        "candidate_physical_error": evaluate(actions, goal)[0],
        "direct_sha256": direct_hash, "split_sha256": split_hash,
        "candidate_offered_at_matching_successor": search.injection_visits,
        "branch_retained_exact_injected_suffix": branch["edges"][-1]["name"] == "injected_suffix",
        "branch_terminal_cost": branch_cost,
        "branch_strictly_better_than_injected": branch_cost < candidate_cost - 1e-12,
        "branches_strictly_better_than_candidate": sum(
            p["terminal_cost"] < candidate_cost - 1e-12 for p in result["programs"]),
        "selected_root_name": selected["root_name"],
        "selected_terminal_cost": selected["terminal_cost"],
        "selected_exact_injected_chain": (
            selected["root_name"] == "injected_prefix"
            and selected["edges"][-1]["name"] == "injected_suffix"),
        "selected_strictly_worse_than_candidate": selected["terminal_cost"] > candidate_cost + 1e-12,
        "atlas_counts": result["operation_counts"],
        "injected_branch": branch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Diagnostic: can Atlas replay, retain and score a supplied gradient action chain?")
    rows = [diagnose(i) for i in range(8)]
    report = {"schema": "pc_atlas_oracle_injection.v1",
              "scope": "development diagnostic with privileged injected actions; not Atlas performance",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "atlas_sha256": hashlib.sha256(Path(__file__).with_name("d2p_atlas_search_v2.py").read_bytes()).hexdigest(),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps({"roots": len(rows),
                      "branch_retained": sum(r["branch_retained_exact_injected_suffix"] for r in rows),
                      "selected": sum(r["selected_exact_injected_chain"] for r in rows),
                      "selected_worse": sum(r["selected_strictly_worse_than_candidate"] for r in rows)}))


if __name__ == "__main__":
    main()
