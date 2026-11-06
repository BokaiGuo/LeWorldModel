"""Fixed-opportunity CEM evaluation of ranked action-effect retained chains."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import pc_action_effect_ranked_end_to_end_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_shared_multistart_cem as optimizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pc_action_effect_fixed_refinement_dev_20261105_codex"
PROTOCOL = ROOT.parent / "docs/PC_ACTION_EFFECT_FIXED_REFINEMENT_DEV_PROTOCOL_20261105_CODEX.md"
ARMS = ("random_min", "random_effect", "spectral_min", "spectral_effect")
SAMPLES = 628


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def starts_from_result(result: dict) -> tuple[np.ndarray, list[int]]:
    global_bank = np.asarray(result["global_proposals"]["actions"], dtype=np.float32)
    scores = np.asarray(result["global_proposals"]["scores"], dtype=np.float64)
    if global_bank.ndim != 3 or global_bank.shape[1:] != (15, 2) or scores.shape != (len(global_bank),):
        raise ValueError("invalid global bank")
    indices = []
    for index in np.argsort(scores, kind="stable"):
        if all(not np.array_equal(global_bank[index], global_bank[old]) for old in indices):
            indices.append(int(index))
        if len(indices) == 2:
            break
    if len(indices) != 2:
        raise ValueError("insufficient unique global starts")
    layer = result["child"]["layers"][-1]
    retained = layer["retained"]
    if len(retained) != 2 or len(set(retained)) != 2 or layer["depth"] != 2:
        raise ValueError("invalid final retained slots")
    chains = [np.asarray(layer["candidates"][i]["chain"], dtype=np.float32) for i in retained]
    starts = np.stack([global_bank[i] for i in indices] + chains)
    if starts.shape != (4, 15, 2) or not np.isfinite(starts).all() or np.abs(starts).max() > 1:
        raise ValueError("invalid starting actions")
    return starts, indices


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_path = prior.OUT / "manifest.json"
    source_manifest = json.loads(manifest_path.read_text())
    prior.prior.prior.seal.verify(source_manifest["dependencies"])
    audit_path = prior.OUT / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["run_count"] != 80:
        raise ValueError("source audit missing")
    if source_manifest["tasks"] != prior.tasks():
        raise ValueError("task list mismatch")
    OUT.mkdir()
    manifest = {
        "schema": "pc_action_effect_fixed_refinement_dev.v1",
        "scope": "post-hoc same-family development fixed opportunities",
        "protocol_sha256": digest(PROTOCOL), "code_sha256": digest(Path(__file__)),
        "optimizer_sha256": digest(Path(optimizer.__file__)),
        "source_manifest_sha256": digest(manifest_path), "source_audit_sha256": digest(audit_path),
        "source_rows": {}, "tasks": prior.tasks(), "arms": ARMS, "samples": SAMPLES,
    }
    rows = []
    for root, task in enumerate(prior.tasks()):
        shared = None
        shared_scores = None
        for arm in ARMS:
            source_path = prior.OUT / f"plan_{root:02d}_{arm}.json"
            source = json.loads(source_path.read_text())
            if (source["root"], source["arm"], source["task"], source["status"]) != (root, arm, task, "completed"):
                raise ValueError("source identity mismatch")
            manifest["source_rows"][source_path.name] = digest(source_path)
            starts, indices = starts_from_result(source["result"])
            selected_scores = np.asarray(source["result"]["global_proposals"]["scores"])[indices]
            if shared is None:
                shared, shared_scores = starts[:2].copy(), selected_scores.copy()
            else:
                np.testing.assert_array_equal(starts[:2], shared)
                np.testing.assert_array_equal(selected_scores, shared_scores)
            recorder = prior.prior.prior.journal.PredictionJournal(
                OUT / f"root{root:02d}_{arm}_calls",
                prior.prior.prior.old.previous.fixed.predictor(task),
                action_dim=2, packet_length=5, token_cap=SAMPLES * 3,
            )
            result = optimizer.optimize(
                lambda actions: recorder(actions)[1],
                np.empty((0, 2), np.float32), starts, SAMPLES,
                seed=2046000000 + root, batch_size=12, elites=4,
            )
            if recorder.tokens != 1884 or any(s["samples"] != 157 for s in result["starts"]):
                raise ValueError("fixed optimization budget mismatch")
            np.testing.assert_allclose(
                [result["starts"][i]["seed_score"] for i in (0, 1)], shared_scores,
                atol=1e-12, rtol=0,
            )
            rows.append({
                "root": root, "arm": arm, "seed": 2046000000 + root,
                "global_indices": indices, "initial_actions": starts.tolist(),
                "score": float(result["score"]),
                "retained_best_score": float(min(r["best_score"] for r in result["starts"][2:])),
                "actions": result["actions"].tolist(), "starts": result["starts"],
                "tokens": recorder.tokens, "calls": recorder.calls,
            })
            print(f"fixed refinement root={root} arm={arm} score={result['score']:.6f}", flush=True)
    prior.prior.prior.seal.verify(source_manifest["dependencies"])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summaries = {}
    for direction in ("random", "spectral"):
        subsets = {
            policy: [r for r in rows if r["arm"] == f"{direction}_{policy}"]
            for policy in ("min", "effect")
        }
        summaries[direction] = {}
        for metric in ("score", "retained_best_score"):
            diff = np.asarray([b[metric] - a[metric] for a, b in zip(subsets["min"], subsets["effect"], strict=True)])
            summaries[direction][metric] = {
                "min_mean": float(np.mean([r[metric] for r in subsets["min"]])),
                "effect_mean": float(np.mean([r[metric] for r in subsets["effect"]])),
                "effect_minus_min": float(diff.mean()),
                "effect_better_roots": int((diff < -1e-12).sum()),
                "effect_worse_roots": int((diff > 1e-12).sum()),
                "ties": int((np.abs(diff) <= 1e-12).sum()),
                "min_success": sum(r[metric] < 0.05 for r in subsets["min"]),
                "effect_success": sum(r[metric] < 0.05 for r in subsets["effect"]),
            }
    output = {"status": "completed", "rows": rows, "summaries": summaries,
              "diagnostic_prediction_tokens": sum(r["tokens"] for r in rows)}
    (OUT / "result.json").write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    print("Comparing fixed shared-global and retained-chain refinement opportunities.", flush=True)
    main()
