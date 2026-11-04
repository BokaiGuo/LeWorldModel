"""Non-oracle model-state diverse beam for the exposed wall task."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import pc_wall_unicycle_atlas_cem_20261103 as wall
from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/pc_wall_unicycle_atlas_cem_20261103_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_WALL_UNICYCLE_DIVERSE_BEAM_PROTOCOL_20261103.md"
SCALE = np.array([1., 1., .3])


def select_beam(expanded, cost, policy):
    if len(expanded) < 2 or policy not in ("cost", "diverse"):
        raise ValueError("width-two beam requires valid nonempty candidates")
    order = sorted(range(len(expanded)), key=lambda i: (cost(expanded[i]["context"]), i))
    if policy == "cost":
        return [expanded[i] for i in order[:2]], 0
    best = order[0]
    base = np.asarray(expanded[best]["context"]["latent_history"][-1], dtype=float)
    distances = {}
    for i in range(len(expanded)):
        if i != best:
            state = np.asarray(expanded[i]["context"]["latent_history"][-1], dtype=float)
            if state.shape != SCALE.shape or not np.isfinite(state).all():
                raise ValueError("invalid state")
            distances[i] = float(np.linalg.norm((state - base) * SCALE))
    second = min(distances, key=lambda i: (-distances[i], i))
    return [expanded[best], expanded[second]], len(distances)


class BeamVariant(AtlasSearch):
    def __init__(self, compiler, config, policy):
        super().__init__(compiler, config)
        self.policy = policy

    def search(self, context):
        c, cfg = self.compiler, self.config
        if cfg.beam_width != 2:
            raise ValueError("only width two is implemented")
        rng = np.random.default_rng(cfg.seed)
        start = dict(c.counts)
        roots = []
        distance_evaluations = 0
        for name, actions in self._bank(context, rng):
            successor = c._call("rollout", context, actions)
            roots.append({"root_name": name, "root_cost": c._cost(successor),
                          "context": successor, "actions": actions.tolist(),
                          "edges": [{"name": name, "actions": actions.tolist(),
                                     "output_sha256": digest(successor)}]})
        complete = []
        for root_index, root in enumerate(roots):
            beam = [root]
            for _ in range(1, cfg.depth):
                expanded = []
                for node in beam:
                    for name, actions in self._bank(node["context"], rng):
                        successor = c._call("rollout", node["context"], actions)
                        expanded.append({"root_name": node["root_name"], "root_cost": node["root_cost"],
                                         "context": successor, "actions": node["actions"] + actions.tolist(),
                                         "edges": node["edges"] + [{"name": name, "actions": actions.tolist(),
                                                                    "output_sha256": digest(successor)}]})
                beam, n = select_beam(expanded, c._cost, self.policy)
                distance_evaluations += n
            # AtlasSearch returns the first node of its beam, which is always
            # the lowest-cost member for both policies here.
            best = beam[0]
            direct = c._call("rollout", context, np.asarray(best["actions"]))
            if digest(direct) != digest(best["context"]):
                raise ValueError("witness replay mismatch")
            complete.append({"root_index": root_index, "root_name": best["root_name"],
                             "prefix_cost": best["root_cost"], "terminal_cost": c._cost(direct),
                             "actions": best["actions"], "edges": best["edges"],
                             "output_context": direct, "witness_replayed": True})
        selected = min(range(len(complete)), key=lambda i: (complete[i]["terminal_cost"], i))
        return {"programs": complete, "selected_index": selected,
                "operation_counts": {k: c.counts[k] - start[k] for k in c.counts},
                "distance_evaluations": distance_evaluations}


def make(seed, mode):
    backend = wall.Backend()
    compiler = LocalCompiler(backend, goal=wall.GOAL, sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[.3, np.pi / 2], low=wall.LOW, high=wall.HIGH,
                             config=CompilerConfig(packet_steps=4, spectral_modes=2, prefix_radius=.3,
                                                   max_rollout_operations=10000, max_jacobian_operations=2000))
    config = AtlasConfig(depth=4, beam_width=2, random_bridges=2, seed=seed)
    search = AtlasSearch(compiler, config) if mode == "original" else BeamVariant(compiler, config, mode)
    start = perf_counter()
    result = search.search(wall.ROOT)
    elapsed = perf_counter() - start
    selected = result["programs"][result["selected_index"]]
    return {"programs": result["programs"], "selected_index": result["selected_index"],
            "selected": wall.replay(selected["actions"]), "counts": result["operation_counts"],
            "backend": deepcopy(vars(backend)), "distance_evaluations": result.get("distance_evaluations", 0),
            "wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prior = json.loads(SOURCE.read_text())
    rows = []
    print("Compare ordinary Atlas beam with non-oracle model-state diversity.", flush=True)
    for ref in prior["rows"]:
        seed = ref["seed"]
        original, cost, diverse = (make(seed, mode) for mode in ("original", "cost", "diverse"))
        for key in ("programs", "selected_index", "selected", "counts", "backend"):
            if original[key] != cost[key]:
                raise ValueError(f"cost-mode parity failed: {key}")
        if (original["selected"] != ref["atlas"]["selected"]
                or original["counts"] != ref["atlas"]["compiler_counts"]
                or original["backend"] != ref["atlas"]["backend"]):
            raise ValueError("original saved result mismatch")
        rows.append({"seed": seed, "original": original, "cost_parity": True, "diverse": diverse})
        print(f"seed={seed}: old={original['selected']['terminal_error']:.6g}, "
              f"diverse={diverse['selected']['terminal_error']:.6g}", flush=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"schema": "wall_unicycle_diverse_beam.v1", "source_sha256": sha(SOURCE),
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "backend_sha256": sha(Path(wall.__file__)), "rows": rows,
              "scope": "exposed single exact task; non-oracle development, not independent or admission"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
