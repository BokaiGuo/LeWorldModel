"""Audit probe-only ablation against identical recorded cheap probes."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_cheap_probe_ablation_dev_20261105_codex as run


def main():
    out, fixed, prior = run.OUT, run.fixed, run.prior
    manifest = json.loads((out / "manifest.json").read_text())
    rows = json.loads((out / "summary.json").read_text())["rows"]
    seal = fixed.prior.prior.prior.seal
    seal.verify(manifest["dependencies"])
    seal.verify(manifest["source_dependencies"])
    old = json.loads((prior.OUT / "summary.json").read_text())["rows"]
    indexed = {(r["root"], r["repeat"], r["arm"]): r for r in old}
    seen = set()
    for row in rows:
        root, repeat, arm = row["root"], row["repeat"], row["arm"]
        key = root, repeat, arm
        if key in seen or row["status"] != "completed":
            raise ValueError("failure or duplicate")
        seen.add(key)
        name = f"root{root:02d}_rep{repeat}_{arm}"
        if row != json.loads((out / f"{name}.json").read_text()):
            raise ValueError("row differs")
        parent_source = json.loads((fixed.prior.OUT / f"plan_{root:02d}_random_min.json").read_text())
        parent = fixed.starts_from_result(parent_source["result"])[0][0]
        predictor = fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        log, best, tokens = out / f"{name}_calls", float("inf"), 0
        cheap_log = prior.OUT / f"root{root:02d}_rep{repeat}_cheap_{arm.split('_')[1]}_calls"
        for j in range(row["calls"]):
            stem = f"{j:06d}"
            request = json.loads((log / f"{stem}_attempt.json").read_text())
            response = json.loads((log / f"{stem}_returned.json").read_text())
            a_path, r_path = log / request["actions"], log / response["result"]
            if run.prior.prior.digest(a_path) != request["sha256"] or run.prior.prior.digest(r_path) != response["sha256"]:
                raise ValueError("hash mismatch")
            with np.load(a_path, allow_pickle=False) as f:
                actions = f["actions"].copy()
            with np.load(r_path, allow_pickle=False) as f:
                features, scores = f["features"].copy(), f["scores"].copy()
            if j < 2:
                with np.load(cheap_log / f"{stem}_actions.npz", allow_pickle=False) as f:
                    np.testing.assert_array_equal(actions, f["actions"])
            if j == 0:
                np.testing.assert_array_equal(actions[0], parent)
            x, y = predictor(actions)
            np.testing.assert_allclose(x, features, rtol=0, atol=1e-12)
            np.testing.assert_allclose(y, scores, rtol=0, atol=1e-12)
            best = min(best, float(scores.min()))
            tokens += len(actions)*3
        expected = 273 if arm == "probe_fixed" else 471
        if tokens != expected or row["tokens"] != expected or abs(best-row["result"]["final_score"]) > 1e-12:
            raise ValueError("budget or incumbent mismatch")
        _, value = predictor(np.asarray(row["result"]["final_actions"], np.float32)[None])
        if abs(float(value[0]) - best) > 1e-12:
            raise ValueError("incorrect returned action")
    if seen != {(r, p, arm) for r in range(16) for p in range(2) for arm in ("probe_fixed", "probe_total")}:
        raise ValueError("missing data")
    summary = {}
    for mode in ("fixed", "total"):
        delta, probe_values, cheap_values = [], [], []
        for root in range(16):
            p, c = [], []
            for repeat in range(2):
                row = next(r for r in rows if (r["root"], r["repeat"], r["arm"]) == (root, repeat, f"probe_{mode}"))
                p.append(row["result"]["final_score"])
                c.append(indexed[(root, repeat, f"cheap_{mode}")]["result"]["final_score"])
            probe_values.extend(p)
            cheap_values.extend(c)
            delta.append(float(np.mean(c) - np.mean(p)))
        d = np.asarray(delta)
        summary[mode] = {"probe_mean": float(np.mean(probe_values)), "cheap_mean": float(np.mean(cheap_values)),
                         "probe_success": int(np.sum(np.asarray(probe_values) < .05)),
                         "cheap_success": int(np.sum(np.asarray(cheap_values) < .05)),
                         "cheap_minus_probe": float(d.mean()), "cheap_better_roots": int((d < -1e-12).sum()),
                         "cheap_worse_roots": int((d > 1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
                         "per_root_delta": d.tolist()}
    result = {"status": "passed", "rows": len(seen), "summary": summary,
              "summary_sha256": prior.prior.digest(out / "summary.json"),
              "auditor_sha256": prior.prior.digest(Path(__file__)),
              "scope": "same predictor implementation; outcomes and tasks previously exposed"}
    with (out / "audit.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
