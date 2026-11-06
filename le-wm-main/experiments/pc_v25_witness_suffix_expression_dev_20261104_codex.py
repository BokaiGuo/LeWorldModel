"""Paired equal-sample suffix expression diagnostic on privileged prefixes."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from experiments import pc_v25_scale_witness_prefix_suffix_posthoc_20261104_codex as prior
from experiments.spectral_reachability import pc_response_temporal_search_v1 as temporal

source, common, witness = prior.source, prior.common, prior.witness
cem = common.suffix.pc_shared_multistart_cem
OUT = source.ROOT / "results/pc_v25_witness_suffix_expression_dev_20261104_codex"
PROTOCOL = source.ROOT.parent / "docs/PC_V25_WITNESS_SUFFIX_EXPRESSION_DEV_PROTOCOL_20261104_CODEX.md"
ARMS = ("step_zero", "step_random", "knot_zero")


def optimize(predict, prefixes, root, arm):
    if arm not in ARMS:
        raise ValueError("unknown optimizer arm")
    records = []
    for index, prefix in enumerate(np.asarray(prefixes, np.float32)):
        runs = []
        for restart in range(common.RESTARTS):
            seed = 2030902000 + root + restart
            if arm == "knot_zero":
                zero = np.zeros((10, 2), np.float32)
                _, values = predict(np.concatenate((prefix, zero))[None])
                chain, score, generations = temporal.cem(zero, prefix, common.SAMPLES - 1,
                    np.random.default_rng(seed), predict, float(values[0]), 4)
                run = {"score": float(score), "actions": np.asarray(chain, np.float32).tolist(),
                       "samples": common.SAMPLES, "generations": generations}
            else:
                if arm == "step_zero":
                    initial = np.zeros((1, 10, 2), np.float32)
                else:
                    initial = np.random.default_rng(np.random.SeedSequence(
                        [2030910000, root, restart, index])).uniform(-1, 1, (1, 10, 2)).astype(np.float32)
                def score(chains):
                    return predict(chains)[1]
                result = cem.optimize(score, prefix, initial, common.SAMPLES, seed=seed,
                                      batch_size=12, elites=4)
                run = {"score": result["score"], "actions": result["actions"].tolist(),
                       "samples": result["samples"], "starts": result["starts"]}
            runs.append(run)
        best = min(range(len(runs)), key=lambda i: (runs[i]["score"], i))
        records.append({"index": index, "runs": runs, "best_restart": best,
                        "best_score": runs[best]["score"]})
    return records


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    witnesses = json.loads(witness.OUT.read_text())
    old_manifest = json.loads((prior.OUT / "manifest.json").read_text())
    old_audit = json.loads((prior.OUT / "audit.json").read_text())
    source.seal.verify(witnesses["dependencies"])
    source.seal.verify(old_manifest["dependencies"])
    if old_audit["status"] != "passed" or len(old_audit["rows"]) != len(source.tasks()):
        raise ValueError("prior prefix diagnostic is not audited")
    deps = source.seal.capture([Path(__file__), PROTOCOL, Path(temporal.__file__), Path(cem.__file__),
        Path(prior.__file__), witness.OUT, prior.OUT / "manifest.json", prior.OUT / "audit.json"])
    OUT.mkdir()
    source.journal.write_json(OUT / "manifest.json", {"dependencies": deps, "arms": ARMS,
        "samples_per_restart": common.SAMPLES, "restarts": common.RESTARTS,
        "scope": "post-hoc privileged prefix, development diagnostic only"})
    print("Comparing stepwise zero/random and four-knot zero CEM at equal prefix budgets.", flush=True)
    for root, task in enumerate(source.tasks()):
        prefixes = prior.prefixes_for(root, task, witnesses)
        for arm in ARMS:
            rec = source.journal.PredictionJournal(OUT / f"root{root:02d}_{arm}_calls",
                source.prior.predictor(task), action_dim=2, packet_length=5,
                token_cap=2 * common.RESTARTS * common.SAMPLES * 3)
            try:
                result = optimize(rec, prefixes, root, arm)
                row = {"root": root, "arm": arm, "task": task, "status": "completed", "prefixes": result}
            except Exception:
                row = {"root": root, "arm": arm, "task": task, "status": "failed",
                       "error": traceback.format_exc()}
            row["tokens"] = rec.tokens
            source.journal.write_json(OUT / f"root{root:02d}_{arm}.json", row)
            print(f"root={root} arm={arm} status={row['status']} tokens={rec.tokens}", flush=True)
    source.seal.verify(deps)


def audit():
    path = OUT / "audit.json"
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads((OUT / "manifest.json").read_text())
    source.seal.verify(manifest["dependencies"])
    witnesses = json.loads(witness.OUT.read_text())
    old = json.loads((prior.OUT / "audit.json").read_text())
    summaries = []
    for root, task in enumerate(source.tasks()):
        prefixes = prior.prefixes_for(root, task, witnesses)
        for arm in ARMS:
            row = json.loads((OUT / f"root{root:02d}_{arm}.json").read_text())
            if (row["root"], row["task"], row["arm"]) != (root, task, arm):
                raise ValueError("row identity mismatch")
            directory = OUT / f"root{root:02d}_{arm}_calls"
            info = source.prospective.journal_audit.audit_journal(directory)
            if info["tokens"] != row["tokens"]:
                raise ValueError("journal token mismatch")
            if row["status"] != "completed":
                summaries.append(row)
                continue
            calls = list(source.prior.read_calls(directory))
            for actions, positions, scores in calls:
                path_geometry, _ = source.OLD.previous.independent(task["start"], actions)
                np.testing.assert_allclose(path_geometry[:, -1, :2], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(scores, np.linalg.norm(positions - task["goal"], axis=1),
                                           atol=1e-12, rtol=0)
            replay = iter(calls)
            def predict(actions):
                recorded, features, scores = next(replay)
                np.testing.assert_array_equal(np.asarray(actions, np.float32), recorded)
                return features, scores
            result = optimize(predict, prefixes, root, arm)
            if next(replay, None) is not None or result != row["prefixes"]:
                raise ValueError("optimizer proposal replay mismatch")
            if row["tokens"] != 2 * common.RESTARTS * common.SAMPLES * 3:
                raise ValueError("unequal or incomplete budget")
            scores = [p["best_score"] for p in result]
            if arm == "step_zero" and scores != old["rows"][root]["prefix_scores"]:
                raise ValueError("prior step-zero diagnostic mismatch")
            summaries.append({"root": root, "arm": arm, "status": "completed",
                              "prefix_scores": scores, "tokens": info["tokens"]})
            print(f"audited root={root} arm={arm}", flush=True)
    source.seal.verify(manifest["dependencies"])
    source.journal.write_json(path, {"status": "passed", "rows": summaries,
        "scope": "alternate same-environment geometry and proposal replay, not independent replication"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
