"""Audit saved v17 Reacher decisions, actions, contexts and token ledgers."""
import json

import numpy as np

from experiments import pc_v17_reacher_score_response_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    output = source.OUT / "audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    summary = json.loads((source.OUT / "summary.json").read_text())
    seal.verify(manifest["dependencies"])
    if (manifest["arms"] != list(source.ARMS) or manifest["task_seed"] != source.TASK_SEED
            or summary["manifest_sha256"] != seal.digest(source.OUT / "manifest.json")):
        raise ValueError("manifest binding mismatch")
    initial = None
    rows = []
    for arm in source.ARMS:
        execution_path = source.OUT / f"{arm}_execution.json"
        execution = json.loads(execution_path.read_text())
        if execution["status"] != "completed":
            rows.append({"arm": arm, "status": "failed", "error": execution["error"]})
            continue
        if len(execution["events"]) != 3:
            raise ValueError("invalid events")
        with np.load(source.OUT / f"{arm}_context0.npz", allow_pickle=False) as data:
            binding = {name: data[name].copy() for name in ("frames", "root", "goal", "history")}
        if initial is None:
            initial = binding
        else:
            for name in initial:
                np.testing.assert_array_equal(initial[name], binding[name])
        prior_chain = None
        arm_rows = []
        for step, event in enumerate(execution["events"]):
            decision_path = source.OUT / f"{arm}_decision{step}.json"
            context_path = source.OUT / f"{arm}_context{step}.npz"
            decision = json.loads(decision_path.read_text())
            if (event["step"] != step or event["decision_sha256"] != seal.digest(decision_path)
                    or event["context_sha256"] != seal.digest(context_path)
                    or decision["context_sha256"] != seal.digest(context_path)
                    or decision["manifest_sha256"] != seal.digest(source.OUT / "manifest.json")
                    or event["sealed_ns"] >= event["execution_return_ns"]):
                raise ValueError("decision binding or chronology mismatch")
            carry = decision["carry_in"]
            if step == 0:
                if carry is not None:
                    raise ValueError("unexpected first carry")
            else:
                np.testing.assert_array_equal(np.asarray(carry, np.float32), prior_chain[5:])
            selected = np.asarray(decision["decision"]["selected"], np.float32)
            if selected.shape != (15 - 5 * step, 2):
                raise ValueError("selected horizon mismatch")
            np.testing.assert_array_equal(event["dispatch"], selected[:5])
            np.testing.assert_array_equal(decision["dispatch"], selected[:5])
            count, full_scores, selected_scores = 0, [], []
            for batch in decision["trace"]:
                requested, raw = np.asarray(batch["requested"], np.float64), np.asarray(batch["raw"], np.float32)
                positions, scores = np.asarray(batch["positions"]), np.asarray(batch["scores"], np.float64)
                if (raw.ndim != 3 or raw.shape[1] % 5 or raw.shape[2] != 2
                        or positions.shape != (len(raw), 2) or scores.shape != (len(raw),)
                        or not np.isfinite(raw).all() or np.max(np.abs(raw)) > 1):
                    raise ValueError("invalid trace")
                np.testing.assert_array_equal(requested.astype(np.float32), raw)
                count += len(raw) * raw.shape[1] // 5
                if raw.shape[1] == len(selected):
                    full_scores.extend(scores.tolist())
                    selected_scores.extend(scores[np.all(raw == selected, axis=(1, 2))].tolist())
            score = float(decision["decision"]["selected_score"])
            if (count != decision["counts"]["prediction_token_samples"] or count > source.CAP
                    or not selected_scores or min(abs(x - score) for x in selected_scores) > 1e-6
                    or score > min(full_scores) + 1e-5):
                raise ValueError("meter or full-chain selection mismatch")
            with np.load(source.OUT / f"{arm}_context{step + 1}.npz", allow_pickle=False) as successor:
                if source.old.context_lib.rgb_digest(successor["root"]) != event["successor_root_sha256"]:
                    raise ValueError("successor image mismatch")
            prior_chain = selected
            arm_rows.append({"step": step, "tokens": count, "selected_score": score,
                             "best_scored": min(full_scores), "full_samples": len(full_scores)})
        rows.append({"arm": arm, "status": "completed", "decisions": arm_rows})
    seal.verify(manifest["dependencies"])
    source.old.host.write_new(output, {"status": "passed", "summary_sha256": seal.digest(source.OUT / "summary.json"),
        "arms": rows, "scope": "saved-data audit; not physical replay or source independence"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
