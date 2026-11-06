"""Replay packet-bank candidate and CEM predictions."""
import json
from pathlib import Path

import numpy as np
from experiments import pc_e0_packet_parent_coverage_diag_20261105_codex as run


def main():
    out, fixed, prior = run.OUT, run.fixed, run.prior
    manifest = json.loads((out / "manifest.json").read_text())
    rows = json.loads((out / "summary.json").read_text())["rows"]
    seal = fixed.prior.prior.prior.seal
    seal.verify(manifest["dependencies"])
    seal.verify(manifest["source_dependencies"])
    baseline = json.loads((prior.OUT / "summary.json").read_text())["rows"]
    baseline = {(x["root"], x["repeat"]): x for x in baseline if x["arm"] == "cem"}
    digest = prior.prior.old.prior.digest
    if len(rows) != 32 or len(baseline) != 32:
        raise ValueError("panel incomplete")
    seen, packet_scores, raw_scores, packet_start, raw_start = set(), [], [], [], []
    for row in sorted(rows, key=lambda x: (x["root"], x["repeat"])):
        root, rep = row["root"], row["repeat"]
        if (root, rep) in seen or row["status"] != "completed":
            raise ValueError("failure or duplicate")
        seen.add((root, rep))
        name = f"root{root:02d}_rep{rep}"
        if json.loads((out / f"{name}.json").read_text()) != row:
            raise ValueError("row mismatch")
        log = out / f"{name}_calls"
        predictor = fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        best, tokens, parent, first = float("inf"), 0, None, None
        for j in range(row["calls"]):
            stem = f"{j:06d}"
            req = json.loads((log / f"{stem}_attempt.json").read_text())
            resp = json.loads((log / f"{stem}_returned.json").read_text())
            ap, rp = log / req["actions"], log / resp["result"]
            if digest(ap) != req["sha256"] or digest(rp) != resp["sha256"]:
                raise ValueError("hash mismatch")
            with np.load(ap, allow_pickle=False) as f:
                actions = f["actions"].copy()
            with np.load(rp, allow_pickle=False) as f:
                feats, scores = f["features"].copy(), f["scores"].copy()
            re_f, re_s = predictor(actions)
            np.testing.assert_allclose(feats, re_f, rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, re_s, rtol=0, atol=1e-12)
            if j == 0:
                np.testing.assert_array_equal(actions, run.packet_bank(row["seed"]))
                index = int(np.argmin(scores))
                if index != row["parent_index"] or abs(float(scores[index])-row["parent_score"]) > 1e-12:
                    raise ValueError("bad parent")
                parent = actions[index]
            if j == 1:
                first = actions[0]
            best = min(best, float(scores.min()))
            tokens += len(actions)*3
        if (not np.array_equal(first, parent) or tokens != 663 or row["tokens"] != 663
                or abs(best-row["result"]["final_score"]) > 1e-12):
            raise ValueError("budget or incumbent")
        _, terminal = predictor(np.asarray(row["result"]["final_actions"], np.float32)[None])
        if abs(float(terminal[0])-best) > 1e-12:
            raise ValueError("selected action")
        base = baseline[(root, rep)]
        if base["status"] != "completed" or base["total_allocated_tokens"] != 663:
            raise ValueError("invalid baseline")
        baseline_parent = json.loads((prior.OUT / f"{name}_parent.json").read_text())
        packet_start.append(row["parent_score"])
        raw_start.append(baseline_parent["selected_score"])
        packet_scores.append(row["result"]["final_score"])
        raw_scores.append(base["result"]["final_score"])
    if seen != {(r, p) for r in range(16) for p in range(2)}:
        raise ValueError("missing row")
    packet, raw = np.asarray(packet_scores), np.asarray(raw_scores)
    delta = packet.reshape(16, 2).mean(axis=1) - raw.reshape(16, 2).mean(axis=1)
    result = {"status": "passed", "rows": len(rows),
              "packet_mean_parent": float(np.mean(packet_start)), "step_mean_parent": float(np.mean(raw_start)),
              "packet_mean_final": float(packet.mean()), "step_mean_final": float(raw.mean()),
              "packet_success": int((packet < .05).sum()), "step_success": int((raw < .05).sum()),
              "packet_better_roots": int((delta < -1e-12).sum()), "packet_worse_roots": int((delta > 1e-12).sum()),
              "ties": int((np.abs(delta) <= 1e-12).sum()), "per_root_delta": delta.tolist(),
              "summary_sha256": digest(out / "summary.json"), "auditor_sha256": digest(Path(__file__)),
              "scope": "same dynamics implementation and exposed task panel; not independent feasibility certificate"}
    with (out / "audit.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
