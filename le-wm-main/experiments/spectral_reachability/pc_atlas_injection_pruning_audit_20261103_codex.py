"""Instrument v2 injection, plus counterfactual cross-root pruning; model only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig
from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_oracle_injection_20261103 import InjectedAtlas
from .pc_atlas_unicycle_smoke_20261103 import UnicycleBackend
from .pc_exact_unicycle_development import TURN


def ranks(costs, index):
    """Strict and stable ranks, with actual v2 ordering for exact ties."""
    if not costs or not 0 <= index < len(costs) or not np.isfinite(costs).all():
        raise ValueError("finite costs and valid candidate required")
    target = costs[index]
    return {"strict_rank": 1 + sum(x < target for x in costs),
            "stable_rank": 1 + sum(x < target or (x == target and i < index)
                                   for i, x in enumerate(costs))}


class InstrumentedInjection(InjectedAtlas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.banks = []

    def _bank(self, context, rng):
        bank = super()._bank(context, rng)
        self.banks.append((context, [(name, action.copy()) for name, action in bank]))
        return bank


def audit_row(saved):
    index = saved["index"]
    goal = saved["goal"]
    actions = np.asarray(saved["gradient_actions"], dtype=np.float64)
    root = {"layout": "states_with_between_state_actions_v1", "context_length": 2,
            "latent_dim": 3, "action_dim": 2, "latent_history": [[0., 0., 0.]] * 2,
            "action_history": [[0., 0.]]}
    backend = UnicycleBackend()
    compiler = LocalCompiler(
        backend, goal=[*goal, 0.], sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
        config=CompilerConfig(packet_steps=1, spectral_modes=2,
                              max_rollout_operations=1000, max_jacobian_operations=100))
    search = InstrumentedInjection(compiler, AtlasConfig(seed=42 + index),
                                  root=root, actions=actions)
    result = search.search(root)
    root_context, root_bank = search.banks[0]
    if digest(root_context) != digest(root):
        raise ValueError("root bank context mismatch")
    root_names = [name for name, _ in root_bank]
    root_index = root_names.index("injected_prefix")
    root_scores = [compiler._cost(backend.rollout(root, a)) for _, a in root_bank]
    if len(search.banks) != 1 + len(root_bank):
        raise ValueError("diagnostic assumes v2 depth two")
    successor, suffix_bank = search.banks[1 + root_index]
    if digest(successor) != search.successor_digest:
        raise ValueError("injected successor mismatch")
    suffix_names = [name for name, _ in suffix_bank]
    suffix_index = suffix_names.index("injected_suffix")
    suffix_scores = [compiler._cost(backend.rollout(successor, a)) for _, a in suffix_bank]
    direct = backend.rollout(root, actions)
    split = backend.rollout(successor, actions[1:])
    branch = result["programs"][root_index]
    selected = result["programs"][result["selected_index"]]
    if (digest(direct) != digest(split) or digest(direct) != saved["direct_sha256"]
            or branch["terminal_cost"] != saved["branch_terminal_cost"]
            or selected["root_name"] != saved["selected_root_name"]):
        raise ValueError("historical injection replay mismatch")
    root_rank, suffix_rank = ranks(root_scores, root_index), ranks(suffix_scores, suffix_index)
    return {"index": index, "direct_sha256": digest(direct),
            "prefix_candidates": [{"name": name, "cost": score}
                                  for name, score in zip(root_names, root_scores)],
            "suffix_candidates": [{"name": name, "cost": score}
                                  for name, score in zip(suffix_names, suffix_scores)],
            "prefix_rank": root_rank, "suffix_rank": suffix_rank,
            "hypothetical_global_prefix_retained": {
                str(width): root_rank["stable_rank"] <= width for width in (1, 2, 3, 4)},
            "actual_v2_prefix_pruned": False,
            "actual_v2_suffix_survived_beam": suffix_rank["stable_rank"] <= search.config.beam_width,
            "actual_v2_suffix_branch_winner": branch["edges"][-1]["name"] == "injected_suffix",
            "actual_v2_selected": selected["root_name"] == "injected_prefix"
                                  and selected["edges"][-1]["name"] == "injected_suffix",
            "compiler_counts": result["operation_counts"],
            "external_diagnostic_rollouts": len(root_bank) + len(suffix_bank) + 3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.input.read_bytes()
    saved = json.loads(raw)
    if saved["schema"] != "pc_atlas_oracle_injection.v1":
        raise ValueError("unexpected source schema")
    print("Replay saved privileged chains; score root and successor banks, then simulate prefix-rank pruning.")
    rows = [audit_row(row) for row in saved["rows"]]
    report = {"schema": "pc_atlas_injection_pruning_audit.v1",
              "scope": "exposed exact-dynamics development diagnostic; privileged injection is not performance",
              "counterfactual": "stable prefix-cost ranking of the observed root bank; not actual v2 pruning or a rerun of a global beam",
              "source_sha256": hashlib.sha256(raw).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps({"rows": len(rows), "prefix_stable_ranks": [r["prefix_rank"]["stable_rank"] for r in rows],
                      "suffix_stable_ranks": [r["suffix_rank"]["stable_rank"] for r in rows],
                      "actual_selected": sum(r["actual_v2_selected"] for r in rows)}))


if __name__ == "__main__":
    main()
