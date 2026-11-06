"""Replay each metered E0 call and reconstruct parent/selected chain scores."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_good_chain_response_edit_v2_dev_20261105_codex as run

OUT = run.OUT


def main():
    destination = OUT / "audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((OUT / "manifest.json").read_text())
    summary = json.loads((OUT / "summary.json").read_text())
    if any(run.digest(Path(p)) != value for p, value in manifest["bindings"].items()):
        raise ValueError("dependency mismatch")
    run.fixed.prior.prior.prior.seal.verify(manifest["source_dependencies"])
    if len(summary["rows"]) != 192:
        raise ValueError("missing rows")
    verified = 0
    for row in summary["rows"]:
        root, repeat, arm = row["root"], row["repeat"], row["arm"]
        path = OUT / f"root{root:02d}_rep{repeat}_{arm}.json"
        if row != json.loads(path.read_text()) or row["status"] != "completed":
            raise ValueError("failed or mismatched row")
        record = row["result"]
        source = json.loads((run.fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = run.fixed.starts_from_result(source["result"])[0][0]
        target = np.asarray(manifest["tasks"][root]["goal"])
        predict = run.fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        directory = OUT / f"root{root:02d}_rep{repeat}_{arm}_calls"
        all_actions, all_scores, tokens = [], [], 0
        for j in range(row["calls"]):
            stem = f"{j:06d}"
            attempt = json.loads((directory / f"{stem}_attempt.json").read_text())
            returned = json.loads((directory / f"{stem}_returned.json").read_text())
            apath, rpath = directory / attempt["actions"], directory / returned["result"]
            if run.digest(apath) != attempt["sha256"] or run.digest(rpath) != returned["sha256"]:
                raise ValueError("journal hash mismatch")
            with np.load(apath, allow_pickle=False) as data:
                actions = data["actions"].copy()
            with np.load(rpath, allow_pickle=False) as data:
                features, scores = data["features"].copy(), data["scores"].copy()
            observed_features, observed_scores = predict(actions)
            np.testing.assert_allclose(features, observed_features, atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, observed_scores, atol=1e-12, rtol=0)
            tokens += len(actions)*3
            all_actions.extend(actions)
            all_scores.extend(scores)
        if tokens != row["tokens"] or not np.array_equal(all_actions[0], parent):
            raise ValueError("token or parent mismatch")
        parent_score = float(np.asarray(all_scores)[0])
        if abs(parent_score - record["parent_score"]) > 1e-12:
            raise ValueError("parent score mismatch")
        chosen = np.asarray(record["final_actions"], np.float32)
        matches = [float(score) for action, score in zip(all_actions, all_scores, strict=True)
                   if np.array_equal(action, chosen)]
        if not matches or abs(min(matches) - record["final_score"]) > 1e-12:
            raise ValueError("returned chain not in journal")
        if abs(min(all_scores) - record["final_score"]) > 1e-12:
            raise ValueError("final is not best scored incumbent")
        if arm.startswith("response"):
            # Journal order: parent, three batches of 20 clipped finite differences.
            first_feature, _ = predict(parent[None])
            for segment in range(3):
                attempt = json.loads((directory / f"{segment+1:06d}_attempt.json").read_text())
                with np.load(directory / attempt["actions"], allow_pickle=False) as data:
                    probes = data["actions"].copy()
                features, _ = predict(probes)
                denom = np.array([float(probes[2*k].reshape(-1)[segment*10+k])
                                  - float(probes[2*k+1].reshape(-1)[segment*10+k]) for k in range(10)])
                matrix = ((features[::2] - features[1::2]) / denom[:, None]).T
                np.testing.assert_allclose(matrix, record["matrices"][segment], atol=1e-12, rtol=0)
                np.testing.assert_allclose(denom, record["denominators"][segment], atol=0, rtol=0)
                if not np.isfinite(matrix.T @ (target - first_feature[0])).all():
                    raise ValueError("invalid direction")
        verified += 1
    summaries = {}
    for arm in run.ARMS:
        items = [r for r in summary["rows"] if r["arm"] == arm]
        finals = np.asarray([r["result"]["final_score"] for r in items]).reshape(16, 2)
        starts = np.asarray([r["result"]["refine_start_score"] for r in items]).reshape(16, 2)
        parents = np.asarray([r["result"]["parent_score"] for r in items]).reshape(16, 2)
        summaries[arm] = {"mean_final": float(finals.mean()), "mean_refine_start": float(starts.mean()),
                          "mean_parent": float(parents.mean()), "success_over_32": int((finals < .05).sum()),
                          "roots_success_both_repeats": int(np.all(finals < .05, axis=1).sum()),
                          "edit_selected": sum(r["result"]["selected_source"] != "parent" for r in items),
                          "mean_wall_seconds": float(np.mean([r["elapsed_seconds"] for r in items]))}
    paired = {}
    for suffix in ("fixed", "total"):
        for comparator in ("random", "cem"):
            lhs = np.asarray([r["result"]["final_score"] for r in summary["rows"] if r["arm"] == f"response_{suffix}"]).reshape(16, 2).mean(axis=1)
            rhs = np.asarray([r["result"]["final_score"] for r in summary["rows"] if r["arm"] == f"{comparator}_{suffix}"]).reshape(16, 2).mean(axis=1)
            delta = lhs-rhs
            paired[f"response_minus_{comparator}_{suffix}"] = {"mean": float(delta.mean()),
                "response_better_roots": int((delta < -1e-12).sum()), "response_worse_roots": int((delta > 1e-12).sum()),
                "ties": int((np.abs(delta) <= 1e-12).sum()), "per_root": delta.tolist()}
    result = {"status": "passed", "rows": verified, "summary_sha256": run.digest(OUT / "summary.json"),
              "auditor_sha256": run.digest(Path(__file__)), "summaries": summaries, "paired": paired,
              "scope": "same predictor implementation replay, not independent replication"}
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
