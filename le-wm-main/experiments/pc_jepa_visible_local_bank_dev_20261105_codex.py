"""Explicit evaluation-only locally conditioned action bank around a held-out goal."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments import pc_jepa_reachable_random_goal_dev_20261105_codex as core

OUT = core.ROOT / "results/pc_jepa_visible_local_bank_dev_20261105_codex"
PROTOCOL = core.ROOT.parent / "docs/PC_JEPA_VISIBLE_LOCAL_BANK_DEV_PROTOCOL_20261105_CODEX.md"
SEEDS = (2026110621, 2026110622, 2026110623, 2026110624)


def bank():
    rng = np.random.default_rng(2026110620)
    reference = rng.uniform(-.12, .12, (3, 2))
    near = np.clip(reference + rng.uniform(-.01, .01, (32, 3, 2)), -1, 1)
    far = rng.uniform(-.2, .2, (32, 3, 2))
    result = np.repeat(np.concatenate((near, far, reference[None])).astype(np.float32), 5, axis=1)
    if len({a.tobytes() for a in result}) != 65:
        raise ValueError("duplicate action")
    return result


def configure():
    # Explicitly scoped to this fresh process; no upstream on-disk edits.
    core.OUT, core.PROTOCOL, core.SEEDS, core.bank = OUT, PROTOCOL, SEEDS, bank


def physical():
    configure()
    core.physical()
    manifest = {"status": "completed", "runner_sha256": core.sha(Path(__file__)),
                "physical_sha256": core.sha(OUT / "physical.json"), "protocol_sha256": core.sha(PROTOCOL)}
    with (OUT / "local_manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")


def verify():
    configure()
    core.verify()
    manifest = json.loads((OUT / "local_manifest.json").read_text())
    if (manifest["status"] != "completed" or manifest["runner_sha256"] != core.sha(Path(__file__))
            or manifest["physical_sha256"] != core.sha(OUT / "physical.json")
            or manifest["protocol_sha256"] != core.sha(PROTOCOL)):
        raise ValueError("local manifest changed")


def neural():
    verify()
    core.neural()


def summarize():
    verify()
    rows = []
    for i in range(4):
        path = OUT / f"root_{i}"
        receipt = json.loads((path / "output/receipt.json").read_text())
        if (receipt["status"] != "completed" or receipt["worker_sha256"] != core.sha(core.old.WORKER)
                or receipt["latents_sha256"] != core.sha(path / "output/latents.npz")
                or any(receipt[k] != core.sha(path / "input" / n) for k, n in (
                    ("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                    ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")))):
            raise ValueError("worker receipt mismatch")
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        with np.load(path / "physical.npz", allow_pickle=False) as f:
            states, target, history = (f[k].copy() for k in ("states", "goal_state", "history_states"))
        pos = np.linalg.norm(states[:, :4] - target[:4], axis=1)
        angle = np.abs((states[:, 4] - target[4] + np.pi) % (2 * np.pi) - np.pi)
        success = (pos < 20) & (angle < np.pi / 9)
        ps = ((pred-goal)**2).mean(axis=(1, 2))
        actual_score = ((actual-goal)**2).mean(axis=(1, 2))
        groups = {}
        for name, indices in (("all", np.arange(64)), ("near", np.arange(32)), ("far", np.arange(32, 64))):
            scores = {}
            for label, values in (("predicted", ps), ("actual_encoded", actual_score)):
                order = indices[np.argsort(values[indices], kind="stable")]
                hits = np.flatnonzero(success[order])
                best = int(order[0])
                scores[label] = {"index": best, "success": bool(success[best]),
                                 "first_success_rank": int(hits[0]+1) if len(hits) else None,
                                 "score": float(values[best])}
            groups[name] = {"count": len(indices), "success_count": int(success[indices].sum()), **scores}
        center_ok = bool(np.all((target[:4] >= 40) & (target[:4] <= 472)))
        rows.append({"root": i, "goal_centers_qualified": center_ok, "goal_state": target.tolist(),
                     "goal_object_displacement": float(np.linalg.norm(target[2:4]-history[-1, 2:4])),
                     "candidate_center_outside": int(np.sum(np.any((states[:, :4] < 40) | (states[:, :4] > 472), axis=1))),
                     "mean_prediction_mse": float(np.mean((pred-actual)**2)),
                     "predicted_vs_actual": core.old.independent_order(ps, actual_score),
                     "success_by_index": success.tolist(), "groups": groups,
                     "latents_sha256": core.sha(path / "output/latents.npz")})
    report = {"status": "completed", "manifest_sha256": core.sha(OUT / "local_manifest.json"),
              "rows": rows, "scope": "privileged conditioned candidate bank; not planning"}
    with (OUT / "report.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("physical", "neural", "summarize"))
    phase = p.parse_args().phase
    print(f"Visible local-bank evaluation: {phase}", flush=True)
    {"physical": physical, "neural": neural, "summarize": summarize}[phase]()
