"""Development-only Atlas depth-3 root-source ablation on exact unicycle."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_matched_exact_dev_20261103_exclusive import MeteredBackend, ROOT
from .pc_exact_unicycle_development import TURN

ARMS = ("random", "spectral", "ordinary")


class RootSourceAtlas(AtlasSearch):
    """Change only the first bank; use one common bank thereafter."""

    def __init__(self, compiler, config, arm):
        super().__init__(compiler, config)
        if arm not in ARMS:
            raise ValueError("unknown root candidate source")
        self.arm = arm
        self.bank_count = 0
        self.root_bank = None

    def _bank(self, context, rng):
        c = self.compiler
        progress = c.progress(context, packets=1)
        p = np.asarray(progress["actions"])
        root = self.bank_count == 0
        self.bank_count += 1
        if root and self.arm == "random":
            extras = [rng.uniform(c.low, c.high, size=(1, 2)) for _ in range(2)]
        elif root and self.arm == "spectral":
            directions = progress["geometry"]["raw_spectral_directions"]
            if not directions:
                raise ValueError("missing spectral direction")
            d = np.asarray(directions[0], dtype=float).reshape(1, 2)
            d = c.config.prefix_radius * d / np.max(np.abs(d))
            extras = [-d, d]
        else:
            extras = [np.array([[p[0, 0], sign * TURN]]) for sign in (-1, 1)]
        bank = [("zero", np.zeros((1, 2))), ("progress", p),
                ("extra_minus", extras[0]), ("extra_plus", extras[1])]
        for _, action in bank:
            if not np.isfinite(action).all() or np.any(action < c.low) or np.any(action > c.high):
                raise ValueError("candidate outside action domain")
        if root:
            self.root_bank = [a.tolist() for _, a in bank]
        return bank


def run_arm(index, seed, arm):
    if type(index) is not int or not 0 <= index < 64 or type(seed) is not int or seed < 0:
        raise ValueError("invalid panel index or seed")
    angle = 2 * np.pi * index / 64 + .031
    goal = np.array([np.cos(angle), np.sin(angle), 0.])
    backend = MeteredBackend()
    compiler = LocalCompiler(backend, goal=goal, sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[1., 1.], low=[-1., -TURN], high=[1., TURN],
                             config=CompilerConfig(packet_steps=1, spectral_modes=2,
                                                   max_rollout_operations=512,
                                                   max_jacobian_operations=64))
    atlas = RootSourceAtlas(compiler, AtlasConfig(depth=3, beam_width=2, seed=seed), arm)
    start = perf_counter()
    result = atlas.search(ROOT)
    duration = perf_counter() - start
    selected = result["programs"][result["selected_index"]]
    terminal = np.asarray(selected["output_context"]["latent_history"][-1])
    return {"index": index, "seed": seed, "arm": arm, "goal": goal.tolist(),
            "root_bank": atlas.root_bank, "selected_index": result["selected_index"],
            "selected_actions": selected["actions"], "selected_cost": selected["terminal_cost"],
            "terminal_error": float(np.linalg.norm(terminal[:2] - goal[:2])),
            "programs": result["programs"], "counts": result["operation_counts"],
            "backend_action_steps": backend.steps, "bank_count": atlas.bank_count,
            "wall_seconds": duration}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print("Running exposed exact-unicycle Atlas depth-3 root-source ablation.", flush=True)
    rows = [run_arm(i, seed, arm) for i in range(8) for seed in (42, 43, 44) for arm in ARMS]
    for offset in range(0, len(rows), len(ARMS)):
        group = rows[offset:offset + len(ARMS)]
        if any((r["counts"], r["backend_action_steps"], r["bank_count"]) !=
               (group[0]["counts"], group[0]["backend_action_steps"], group[0]["bank_count"])
               for r in group):
            raise RuntimeError("matched backend-call budget violated")
        if any(r["root_bank"][:2] != group[0]["root_bank"][:2] for r in group):
            raise RuntimeError("shared root controls differ")
    summary = {arm: {"mean_terminal_error": float(np.mean([r["terminal_error"] for r in rows if r["arm"] == arm])),
                     "mean_wall_seconds": float(np.mean([r["wall_seconds"] for r in rows if r["arm"] == arm]))}
               for arm in ARMS}
    files = [Path(__file__), Path(__file__).with_name("d2p_atlas_search_v2.py"),
             Path(__file__).with_name("d2p_local_compiler.py"),
             Path(__file__).with_name("pc_atlas_matched_exact_dev_20261103_exclusive.py"),
             Path(__file__).with_name("pc_atlas_unicycle_smoke_20261103.py"),
             Path(__file__).with_name("pc_exact_unicycle_development.py"),
             Path(__file__).with_name("d2p_composition_witness_v2.py")]
    protocol = Path(__file__).resolve().parents[3] / "docs/PC_ATLAS_DEPTH3_MATCHED_ROOT_SOURCE_PROTOCOL_20261103.md"
    report = {"schema": "pc_atlas_depth3_matched_root_source.v1",
              "scope": "exposed exact-dynamics development; not independent, neural, physical or admission",
              "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              "numpy_version": np.__version__, "summary": summary, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
