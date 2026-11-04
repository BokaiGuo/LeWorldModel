"""Privileged successful-chain injection into wall-unicycle Atlas."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import pc_wall_unicycle_atlas_cem_20261103 as wall
from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/pc_wall_unicycle_waypoint_control_20261103_DEV01.json"
EXPECTED = "ea1462b748b17663711ceda0da77606f9ee98e209c92dd24d06c1f51af897570"
PROTOCOL = ROOT.parent / "docs/PC_WALL_UNICYCLE_INJECTION_PROTOCOL_20261103.md"


class Injection(AtlasSearch):
    def __init__(self, compiler, config, chain):
        super().__init__(compiler, config)
        self.chain = np.asarray(chain, dtype=float)
        if self.chain.shape != (16, 2):
            raise ValueError("invalid chain")
        self.bindings, self.offered = {}, [0] * 4
        context = wall.ROOT
        # Independent binding construction; these transitions are not compiler calls.
        builder = wall.Backend()
        for i in range(4):
            key = digest(context)
            self.bindings.setdefault(key, []).append(i)
            context = builder.rollout(context, self.chain[4*i:4*i+4])
        self.final_digest = digest(context)
        self.construction_transitions = builder.transitions

    def _bank(self, context, rng):
        bank = super()._bank(context, rng)
        for i in self.bindings.get(digest(context), []):
            self.offered[i] += 1
            bank.append((f"injected_{i}", self.chain[4*i:4*i+4].copy()))
        return bank


def run(seed, actions):
    backend = wall.Backend()
    compiler = LocalCompiler(backend, goal=wall.GOAL, sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[.3, np.pi / 2], low=wall.LOW, high=wall.HIGH,
                             config=CompilerConfig(packet_steps=4, spectral_modes=2, prefix_radius=.3,
                                                   max_rollout_operations=10000, max_jacobian_operations=2000))
    search = Injection(compiler, AtlasConfig(depth=4, beam_width=2, random_bridges=2, seed=seed), actions)
    result = search.search(wall.ROOT)
    selected = result["programs"][result["selected_index"]]
    matching = [i for i, p in enumerate(result["programs"]) if digest(p["output_context"]) == search.final_digest]
    same_actions = [i for i, p in enumerate(result["programs"]) if p["actions"] == np.asarray(actions).tolist()]
    return {"seed": seed, "offered": search.offered, "matching_context_branches": matching,
            "matching_action_branches": same_actions, "selected_index": result["selected_index"],
            "selected_actions": selected["actions"],
            "selected_error": wall.replay(selected["actions"])["terminal_error"],
            "selected_name": selected["root_name"], "programs": result["programs"],
            "compiler_counts": result["operation_counts"], "backend": vars(backend).copy(),
            "binding_construction_transitions": search.construction_transitions,
            "selected_replay_transitions": wall.HORIZON}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(SOURCE) != EXPECTED:
        raise ValueError("source receipt changed")
    source = json.loads(SOURCE.read_text())
    print("Privileged diagnostic: inject successful waypoint chains into exact Atlas.", flush=True)
    rows = []
    for prior in source["rows"]:
        if prior["selected"]["terminal_error"] > 1e-8:
            raise ValueError("source is not successful")
        row = run(prior["seed"], prior["selected"]["actions"])
        rows.append(row)
        print(f"seed={row['seed']} offers={row['offered']} selected error={row['selected_error']:.6g}", flush=True)
    report = {"schema": "wall_unicycle_chain_injection.v1", "source_sha256": sha(SOURCE),
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "backend_sha256": sha(Path(wall.__file__)), "rows": rows,
              "scope": "privileged injection into single exact task; not planner performance or admission"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
