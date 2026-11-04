"""Post-outcome bound-normalization correction for the wall root ablation."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_wall_atlas_root_source_20261103_codex import ARMS, SEEDS, RootSourceAtlas
from .pc_wall_unicycle_atlas_cem_20261103 import Backend, GOAL, HIGH, LOW, ROOT, replay, witness


class BoundNormalizedAtlas(RootSourceAtlas):
    def _bank(self, context, rng):
        root = self.banks == 0
        bank = super()._bank(context, rng)
        if root and self.arm == "spectral":
            direction = bank[3][1]
            norm = float(np.max(np.abs(direction) / HIGH))
            if not np.isfinite(norm) or norm <= 1e-12:
                raise ValueError("invalid bound-normalized direction")
            direction = .3 * direction / norm
            if np.any(direction < LOW) or np.any(direction > HIGH):
                raise ValueError("normalized action out of bounds")
            bank[2] = ("extra_minus", -direction)
            bank[3] = ("extra_plus", direction)
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
    atlas = BoundNormalizedAtlas(compiler, AtlasConfig(depth=4, beam_width=2, seed=seed), arm)
    start = perf_counter()
    result = atlas.search(ROOT)
    elapsed = perf_counter() - start
    selected = result["programs"][result["selected_index"]]
    checked = replay(selected["actions"])
    if not np.allclose(checked["trajectory"][-1],
                       selected["output_context"]["latent_history"][-1], atol=1e-12, rtol=0):
        raise ValueError("selected replay mismatch")
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
    print("Running post-outcome bound-normalized wall root-source correction.", flush=True)
    rows = []
    for seed in SEEDS:
        group = [run_arm(seed, arm) for arm in ARMS]
        if any((r["compiler_counts"], r["backend"], r["bank_count"]) !=
               (group[0]["compiler_counts"], group[0]["backend"], group[0]["bank_count"])
               for r in group[1:]):
            raise ValueError("padded backend counts differ")
        if any(r["root_bank"][:2] != group[0]["root_bank"][:2] for r in group[1:]):
            raise ValueError("shared root controls differ")
        rows.extend(group)
        print(f"seed {seed}: " + ", ".join(
            f"{r['arm']}={r['selected_replay']['terminal_error']:.6g}" for r in group),
            flush=True)
    root = Path(__file__).resolve().parents[3]
    files = [
        Path(__file__),
        Path(__file__).with_name("pc_wall_atlas_root_source_20261103_codex.py"),
        Path(__file__).with_name("pc_wall_unicycle_atlas_cem_20261103.py"),
        Path(__file__).with_name("pc_exact_wall_kill.py"),
        Path(__file__).with_name("d2p_atlas_search_v2.py"),
        Path(__file__).with_name("d2p_local_compiler.py"),
        Path(__file__).with_name("d2p_composition_witness_v2.py"),
        root / "docs/PC_WALL_ATLAS_BOUND_NORMALIZED_PROTOCOL_20261103_CODEX.md",
    ]
    report = {"schema": "pc_wall_atlas_bound_normalized.v1",
              "scope": "post-outcome development; one exposed exact task",
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in files},
              "numpy_version": np.__version__, "witness": known, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
