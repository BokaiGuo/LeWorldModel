"""Replay and audit shared-parent and three-arm new-parameter E0 experiments."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_paired_new_task_costed_dev_20261105_codex as run


def main():
    out, fixed, prior = run.OUT, run.fixed, run.prior
    manifest = json.loads((out / "manifest.json").read_text())
    rows = json.loads((out / "summary.json").read_text())["rows"]
    seal = fixed.prior.prior.prior.seal
    seal.verify(manifest["dependencies"])
    seal.verify(manifest["source_dependencies"])
    if manifest["tasks"] != run.tasks() or len(rows) != 96:
        raise ValueError("task or row mismatch")
    digest = prior.old.prior.digest
    parent_chains, seen, witnesses, selected, final_witness = {}, set(), [], 0, 0

    def read_call(log, index, predictor):
        stem = f"{index:06d}"
        req = json.loads((log / f"{stem}_attempt.json").read_text())
        ret = json.loads((log / f"{stem}_returned.json").read_text())
        a_path, r_path = log / req["actions"], log / ret["result"]
        if digest(a_path) != req["sha256"] or digest(r_path) != ret["sha256"]:
            raise ValueError("hash mismatch")
        with np.load(a_path, allow_pickle=False) as f:
            actions = f["actions"].copy()
        with np.load(r_path, allow_pickle=False) as f:
            features, scores = f["features"].copy(), f["scores"].copy()
        a, b = predictor(actions)
        np.testing.assert_allclose(features, a, rtol=0, atol=1e-12)
        np.testing.assert_allclose(scores, b, rtol=0, atol=1e-12)
        return actions, scores

    for root, task in enumerate(manifest["tasks"]):
        predictor = fixed.prior.prior.prior.old.previous.fixed.predictor(task)
        for rep in range(2):
            name = f"root{root:02d}_rep{rep}"
            record = json.loads((out / f"{name}_parent.json").read_text())
            log = out / f"{name}_parent_calls"
            a, s = read_call(log, 0, predictor)
            np.testing.assert_array_equal(a, run.parent_bank(2051700000 + 2*root + rep))
            index = int(np.argmin(s))
            if (record["calls"] != 1 or record["tokens"] != 192 or record["selected_index"] != index
                    or abs(record["selected_score"] - s[index]) > 1e-12):
                raise ValueError("parent selection mismatch")
            parent_chains[(root, rep)] = a[index].copy()
    for row in rows:
        root, rep, arm = row["root"], row["repeat"], row["arm"]
        key = root, rep, arm
        if key in seen or row["status"] != "completed":
            raise ValueError("duplicate or failed row")
        seen.add(key)
        name = f"root{root:02d}_rep{rep}_{arm}"
        if json.loads((out / f"{name}.json").read_text()) != row:
            raise ValueError("row mismatch")
        parent = parent_chains[(root, rep)]
        predictor = fixed.prior.prior.prior.old.previous.fixed.predictor(manifest["tasks"][root])
        singles, pairs, extras, segments = prior.make_candidates(parent, 2051800000+2*root+rep)
        best, tokens, recorded = float("inf"), 0, []
        for j in range(row["calls"]):
            a, s = read_call(out / f"{name}_calls", j, predictor)
            if j == 0:
                np.testing.assert_array_equal(a, parent[None])
            if arm != "cem" and j in (1, 2):
                np.testing.assert_array_equal(a, singles if j == 1 else pairs if arm == "pair" else extras)
            if j < 3 and arm != "cem":
                recorded.append(s)
            best = min(best, float(s.min()))
            tokens += 3*len(a)
        if tokens != 471 or row["total_allocated_tokens"] != 663 or abs(best-row["result"]["final_score"]) > 1e-12:
            raise ValueError("budget or incumbent mismatch")
        _, score = predictor(np.asarray(row["result"]["final_actions"], np.float32)[None])
        if abs(float(score[0])-best) > 1e-12:
            raise ValueError("final action mismatch")
        if arm == "pair":
            p, s, t = recorded
            flags = []
            for i, (a, b) in enumerate(segments):
                changed = np.any(singles[2*i, 5*a:5*a+5] != parent[5*a:5*a+5]) and \
                          np.any(singles[2*i+1, 5*b:5*b+5] != parent[5*b:5*b+5])
                flags.append(bool(changed and t[i] <= p[0]-.01 and p[0]-s[2*i] < .01 and p[0]-s[2*i+1] < .01))
            if flags != row["result"]["witnesses"]:
                raise ValueError("witness mismatch")
            if any(flags):
                witnesses.append([root, rep, sum(flags)])
            idx = row["result"]["start_index"]
            selected += int(idx >= 25 and flags[idx-25])
            final = np.asarray(row["result"]["final_actions"], np.float32)
            final_witness += int(any(flag and np.array_equal(final, pairs[i]) for i, flag in enumerate(flags)))
    if seen != {(r, p, a) for r in range(16) for p in range(2) for a in ("pair", "single", "cem")}:
        raise ValueError("incomplete cells")
    groups, values, paired = {}, {}, {}
    for arm in ("pair", "single", "cem"):
        chosen = sorted((r for r in rows if r["arm"] == arm), key=lambda r: (r["root"], r["repeat"]))
        v = np.asarray([r["result"]["final_score"] for r in chosen]).reshape(16, 2)
        values[arm] = v.mean(axis=1)
        groups[arm] = {"mean": float(v.mean()), "success_over_32": int((v < .05).sum())}
    for arm in ("single", "cem"):
        d = values["pair"] - values[arm]
        paired[arm] = {"mean_delta": float(d.mean()), "pair_better_roots": int((d < -1e-12).sum()),
                       "pair_worse_roots": int((d > 1e-12).sum()),
                       "ties": int((np.abs(d) <= 1e-12).sum()), "per_root": d.tolist()}
    result = {"status": "passed", "rows": len(seen), "parents": len(parent_chains), "groups": groups,
              "paired": paired, "witness_runs": witnesses, "witness_root_count": len({x[0] for x in witnesses}),
              "witness_selected_start_count": selected, "witness_exact_final_count": final_witness,
              "summary_sha256": digest(out / "summary.json"), "auditor_sha256": digest(Path(__file__)),
              "scope": "same dynamics implementation, new parameters do not certify independent provenance"}
    with (out / "audit.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
