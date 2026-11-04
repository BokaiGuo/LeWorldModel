"""Matched root-source development ablation on the exposed wall task."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_wall_unicycle_atlas_cem_20261103 import (
    Backend, GOAL, HIGH, LOW, ROOT, replay, witness,
)

ARMS = ("spectral", "random", "ordinary")
SEEDS = (20261103, 20261104, 20261105)


class RootSourceAtlas(AtlasSearch):
    def __init__(self, compiler, config, arm):
        super().__init__(compiler, config)
        if arm not in ARMS:
            raise ValueError("unknown source")
        self.arm = arm
        self.banks = 0
        self.root_bank = None

    def _bank(self, context, rng):
        compiler = self.compiler
        progress = compiler.progress(context, packets=1)
        shape = (compiler.config.packet_steps, len(compiler.scale))
        zero = np.zeros(shape)
        turns = []
        for sign in (-1, 1):
            action = zero.copy()
            action[0, 1] = sign * HIGH[1]
            turns.append(action)
        is_root = self.banks == 0
        self.banks += 1
        if is_root and self.arm == "spectral":
            directions = progress["geometry"]["raw_spectral_directions"]
            if not directions:
                raise ValueError("missing spectral direction")
            direction = np.asarray(directions[0], dtype=float).reshape(shape)
            magnitude = float(np.max(np.abs(direction)))
            if not np.isfinite(magnitude) or magnitude <= 1e-12:
                raise ValueError("invalid spectral direction")
            direction = compiler.config.prefix_radius * direction / magnitude
            extras = [-direction, direction]
        elif is_root and self.arm == "random":
            extras = [rng.uniform(LOW, HIGH, size=shape) for _ in range(2)]
        else:
            extras = turns
        bank = [("zero", zero), ("progress", np.asarray(progress["actions"])),
                ("extra_minus", extras[0]), ("extra_plus", extras[1])]
        for _, action in bank:
            if (action.shape != shape or not np.isfinite(action).all()
                    or np.any(action < LOW) or np.any(action > HIGH)):
                raise ValueError("invalid candidate action")
        if is_root:
            self.root_bank = [{"name": name, "actions": action.tolist()}
                              for name, action in bank]
        return bank


def run_arm(seed, arm):
    backend = Backend()
    compiler = LocalCompiler(
        backend, goal=GOAL, sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[.3, np.pi / 2], low=LOW, high=HIGH,
        config=CompilerConfig(packet_steps=4, spectral_modes=2, prefix_radius=.3,
                              max_rollout_operations=10000, max_jacobian_operations=2000),
    )
    atlas = RootSourceAtlas(compiler, AtlasConfig(depth=4, beam_width=2, seed=seed), arm)
    start = perf_counter()
    result = atlas.search(ROOT)
    elapsed = perf_counter() - start
    selected = result["programs"][result["selected_index"]]
    checked = replay(selected["actions"])
    if not np.allclose(checked["trajectory"][-1],
                       selected["output_context"]["latent_history"][-1],
                       atol=1e-12, rtol=0):
        raise ValueError("selected program replay mismatch")
    return {"arm": arm, "seed": seed, "root_bank": atlas.root_bank,
            "bank_count": atlas.banks, "selected_index": result["selected_index"],
            "selected_root": selected["root_name"], "selected_replay": checked,
            "programs": result["programs"], "compiler_counts": result["operation_counts"],
            "backend": vars(backend).copy(), "wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    known = replay(witness())
    if known["terminal_error"] > 1e-12 or any(known["blocked"]):
        raise ValueError("feasibility witness failed")
    print("Running exposed wall task: root candidate source with common continuation.",
          flush=True)
    rows = []
    for seed in SEEDS:
        group = [run_arm(seed, arm) for arm in ARMS]
        if any((row["compiler_counts"], row["backend"], row["bank_count"]) !=
               (group[0]["compiler_counts"], group[0]["backend"], group[0]["bank_count"])
               for row in group[1:]):
            raise ValueError("matched padded backend counts violated")
        if any(row["root_bank"][:2] != group[0]["root_bank"][:2] for row in group[1:]):
            raise ValueError("shared root controls differ")
        rows.extend(group)
        print(f"seed {seed}: " + ", ".join(
            f"{row['arm']}={row['selected_replay']['terminal_error']:.6g}" for row in group),
            flush=True)
    root = Path(__file__).resolve().parents[3]
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("d2p_atlas_search_v2.py"),
        Path(__file__).with_name("d2p_local_compiler.py"),
        Path(__file__).with_name("d2p_composition_witness_v2.py"),
        Path(__file__).with_name("pc_wall_unicycle_atlas_cem_20261103.py"),
        Path(__file__).with_name("pc_exact_wall_kill.py"),
        root / "docs/PC_WALL_ATLAS_ROOT_SOURCE_PROTOCOL_20261103_CODEX.md",
    ]
    report = {
        "schema": "pc_wall_atlas_root_source.v1",
        "scope": "exposed exact single task; not independent physical or admission",
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        "numpy_version": np.__version__, "witness": known, "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
