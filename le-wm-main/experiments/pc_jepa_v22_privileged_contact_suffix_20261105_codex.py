"""Evaluation-only contact coverage using fixed, explicitly privileged witness actions."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments import pc_jepa_v22_unqueried_suffix_dev_20261105_codex as run
from experiments import audit_pc_jepa_v22_unqueried_suffix_dev_20261105_codex as audit

ROOT, SOURCE = run.ROOT, run.SOURCE
OUT = ROOT / "results/pc_jepa_v22_privileged_contact_suffix_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_V22_PRIVILEGED_CONTACT_SUFFIX_PROTOCOL_20261105_CODEX.md"
WITNESS = SOURCE / "witness_actions_evaluation_only.npy"


def make_bank(response, cem):
    witness = np.load(WITNESS, allow_pickle=False)
    if witness.shape != (15, 2) or witness.dtype != np.float32 or not np.isfinite(witness).all() or np.max(np.abs(witness)) > 1:
        raise ValueError("bad evaluator witness")
    prefixes = np.stack((np.zeros((5, 2), np.float32), response[:5], cem[:5], witness[:5]))
    scaled = np.stack([(witness[5:] * x).astype(np.float32) for x in (0.25, 0.5, 0.75, 1.)])
    random = np.random.default_rng(2026110504).uniform(-1, 1, (3, 10, 2)).astype(np.float32)
    noise = np.random.default_rng(2026110505).uniform(-0.2, 0.2, (8, 10, 2))
    validation = np.clip(witness[None, 5:].astype(np.float64) + noise, -1, 1).astype(np.float32)
    suffixes = np.concatenate((np.zeros((1, 10, 2), np.float32), scaled, random, validation))
    if len({p.tobytes() for p in prefixes}) != 4 or len({s.tobytes() for s in suffixes}) != 16:
        raise ValueError("duplicate prefixes or suffixes")
    return prefixes, suffixes, np.stack([np.concatenate((p, s)) for p in prefixes for s in suffixes])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "physical", "neural", "audit"))
    phase = parser.parse_args().phase
    run.OUT, run.PROTOCOL, run.make_bank = OUT, PROTOCOL, make_bank
    pinned = [Path(__file__), WITNESS, PROTOCOL]
    if phase == "prepare":
        run.prepare()
        with (OUT / "contact_receipt.json").open("x") as handle:
            json.dump({"bindings": {str(p): run.sha(p) for p in pinned},
                       "manifest_sha256": run.sha(OUT / "manifest.json")},
                      handle, indent=2, allow_nan=False)
            handle.write("\n")
    else:
        receipt = json.loads((OUT / "contact_receipt.json").read_text())
        if (receipt["manifest_sha256"] != run.sha(OUT / "manifest.json")
                or any(run.sha(Path(p)) != d for p, d in receipt["bindings"].items())):
            raise ValueError("privileged source or implementation changed")
        print(f"Privileged contact coverage diagnostic phase={phase}; no planner feedback.", flush=True)
        {"physical": run.physical, "neural": run.neural_score, "audit": audit.main}[phase]()


if __name__ == "__main__":
    main()
