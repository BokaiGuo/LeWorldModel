"""Independently reconstruct labelled selection and saved latent statistics."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_pusht_scored_success_failure_visual_audit_dev_20261105_codex as run


def main():
    run.verify()
    out, census, sha = run.OUT, run.census, run.sha
    report_path = out / "report.json"
    report = json.loads(report_path.read_text())
    assert report["physical_sha256"] == sha(out / "physical.json")
    if (out / "audit.json").exists():
        raise FileExistsError(out / "audit.json")
    rows = []
    for row, arm in zip(report["rows"], ("response", "cem"), strict=True):
        assert row["arm"] == arm
        path = out / f"root_11_{arm}"
        source = census.OUT / f"root_11_{arm}"
        with np.load(source / "bank.npz", allow_pickle=False) as f:
            all_actions, all_scores = f["actions"].copy(), f["score_min"].copy()
            selected = int(f["selected_index"])
        with np.load(source / "physical.npz", allow_pickle=False) as f:
            states, target, valid = f["states"].copy(), f["target"].copy(), f["valid"].copy()
        assert valid.all()
        dist = np.linalg.norm(states[:, :4] - target[:4], axis=1)
        angle = np.abs((states[:, 4] - target[4] + np.pi) % (2*np.pi) - np.pi)
        success = (dist < 20) & (angle < np.pi/9)
        order = np.argsort(all_scores, kind="stable")
        successful = [int(i) for i in order if success[i]]
        failed = [int(i) for i in order if not success[i]]
        expected_ids = np.asarray(successful + failed[:64-len(successful)])
        with np.load(path / "selection.npz", allow_pickle=False) as f:
            ids, labels, original = f["source_indices"].copy(), f["success"].copy(), f["original_scores"].copy()
            assert int(f["selected_source_index"]) == selected
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_array_equal(labels, success[ids])
        np.testing.assert_array_equal(original, all_scores[ids])
        np.testing.assert_array_equal(np.load(path / "input/bank.npy", allow_pickle=False), all_actions[ids])
        receipt = json.loads((path / "output/receipt.json").read_text())
        assert receipt["status"] == "completed"
        assert receipt["worker_sha256"] == sha(census.old.src.WORKER)
        assert receipt["latents_sha256"] == sha(path / "output/latents.npz")
        for key, filename in (("context_sha256", "context.npz"), ("bank_sha256", "bank.npy"),
                              ("images_sha256", "images.npz"), ("metadata_sha256", "model_metadata.json")):
            assert receipt[key] == sha(path / "input" / filename)
        with np.load(path / "output/latents.npz", allow_pickle=False) as f:
            pred, actual, goal = (f[k].astype(np.float64) for k in ("predicted", "actual", "goal"))
        predicted = ((pred - goal)**2).mean(axis=(1, 2))
        encoded = ((actual - goal)**2).mean(axis=(1, 2))
        mse = ((pred - actual)**2).mean(axis=(1, 2))
        assert row["count"] == len(ids) == 64 and row["success_count"] == int(labels.sum())
        assert np.isclose(row["max_abs_prediction_rescore_diff"], np.max(np.abs(predicted-original)))
        assert np.isclose(row["mean_prediction_latent_mse"], np.mean(mse))
        for label, score in (("original", original), ("predicted_rescore", predicted), ("actual_encoded", encoded)):
            ranked = np.argsort(score, kind="stable")
            hits = np.flatnonzero(labels[ranked])
            good, bad = score[labels], score[~labels]
            r = row["metrics"][label]
            assert r["top_source_index"] == int(ids[ranked[0]])
            assert r["top_success"] == bool(labels[ranked[0]])
            assert r["first_success_rank"] == (int(hits[0])+1 if len(hits) else None)
            assert r["success_failure_wins"] == int(np.sum(good[:, None] < bad[None, :]))
            assert r["success_failure_ties"] == int(np.sum(good[:, None] == bad[None, :]))
            assert r["success_failure_pairs"] == len(good)*len(bad)
        idx = int(np.flatnonzero(ids == selected)[0])
        assert row["selected"]["source_index"] == selected
        assert row["selected"]["success"] == bool(labels[idx])
        for key, values in (("original", original), ("predicted_rescore", predicted),
                            ("actual_encoded", encoded), ("latent_mse", mse)):
            assert np.isclose(row["selected"][key], values[idx])
        assert [x["source_index"] for x in row["successful"]] == ids[labels].tolist()
        rows.append({"arm": arm, "selected_candidates": len(ids), "successes": int(labels.sum())})
        print(f"{arm}: labelled sample, receipts and saved scores checked", flush=True)
    with (out / "audit.json").open("x") as f:
        json.dump({"status": "passed", "auditor_sha256": sha(Path(__file__)),
                   "report_sha256": sha(report_path), "rows": rows,
                   "scope": "saved physical labels and neural output arithmetic; no independent neural or image replay"}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
