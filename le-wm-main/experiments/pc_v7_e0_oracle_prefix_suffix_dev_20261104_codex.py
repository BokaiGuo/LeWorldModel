"""Post-hoc oracle-prefix / common-budget suffix-optimization diagnostic."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v5_e0_prospective_interventions_dev_20261104_codex as old
from experiments import pc_v5_e0_prospective_interventions_v2_dev_20261104_codex as old_v2
from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as source
from experiments import pc_v7_e0_exposed_intervention_dev_20261104_codex as v7
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_E0_ORACLE_PREFIX_SUFFIX_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_e0_oracle_prefix_suffix_dev_20261104_codex"
MANIFEST, RESULT, AUDIT = [PREFIX.with_name(PREFIX.name + suffix + ".json")
                           for suffix in ("_manifest", "_result", "_audit")]
WITNESSES = source.PREFIX.with_name(source.PREFIX.name + "_privileged_feasibility_diag.json")
V7_AUDIT = v7.OUTPUT.with_name(v7.OUTPUT.stem + "_binding_audit.json")
sha, exact, write = source.sha, source.exact, source.write


def inputs():
    witnesses = json.loads(WITNESSES.read_text())
    records = json.loads(v7.OUTPUT.read_text())
    audit = json.loads(V7_AUDIT.read_text())
    if audit["intervention_sha256"] != sha(v7.OUTPUT) or len(records["rows"]) != 18:
        raise ValueError("v7 intervention binding mismatch")
    seal.verify(records["dependencies"])
    output = []
    for w, binding in zip(witnesses["rows"], records["rows"], strict=True):
        root = w["root"]
        meta = old_v2.PREFIX.with_name(old_v2.PREFIX.name + f"_root{root:02d}.json")
        arr = old_v2.PREFIX.with_name(old_v2.PREFIX.name + f"_root{root:02d}.npz")
        if (root != binding["root"] or sha(meta) != binding["reference_metadata_sha256"]
                or sha(arr) != binding["reference_arrays_sha256"]):
            raise ValueError("historic intervention binding mismatch")
        data = json.loads(meta.read_text())
        if data["task"] != w["task"] or data["selected_chain"] != binding["selected_chain"]:
            raise ValueError("task or selected chain differs")
        witness = next(x for x in w["witnesses"] if x["eligible"] and x["successful"])
        full = np.asarray(witness["float32_actions"], dtype=np.float32)
        if full.shape != (15, 2) or not np.isfinite(full).all() or np.abs(full).max() > 1:
            raise ValueError("invalid witness actions")
        replay = float(np.linalg.norm(exact.endpoints(w["task"], full[None])[0] - np.asarray(w["task"]["goal"])))
        if replay >= .05 or abs(replay - witness["terminal_distance"]) > 1e-12:
            raise ValueError("witness replay failure")
        output.append((w["task"], full, data, meta, arr, replay))
    return output


def summary(rows):
    result = {}
    for name in ("oracle_prefix", "zero", "greedy"):
        scores = np.asarray([row["scores"][name] for row in rows])
        result[name] = {"mean_distance": float(scores.mean()),
                        "successful_streams": int((scores < .05).sum()),
                        "roots_any_success": int((scores < .05).any(axis=1).sum())}
    target = np.asarray([np.mean(row["scores"]["oracle_prefix"]) for row in rows])
    for name in ("zero", "greedy"):
        delta = target - np.asarray([np.mean(row["scores"][name]) for row in rows])
        result["oracle_vs_" + name] = {"wins": int((delta < -1e-10).sum()),
                                       "ties": int((np.abs(delta) <= 1e-10).sum()),
                                       "losses": int((delta > 1e-10).sum())}
    return result


def run():
    if any(p.exists() for p in (MANIFEST, RESULT, AUDIT)):
        raise FileExistsError("existing or partial diagnostic")
    data = inputs()
    deps = seal.capture([Path(__file__), PROTOCOL, WITNESSES, v7.OUTPUT, V7_AUDIT,
                         Path(old.__file__), Path(old_v2.__file__), Path(core.__file__),
                         Path(exact.__file__), Path(exact.wall.__file__),
                         *[p for _, _, _, meta, arr, _ in data for p in (meta, arr)]])
    write(MANIFEST, {"seal": deps, "roots": len(data), "scope": "post-hoc privileged prefix diagnostic"})
    print("Testing fixed oracle prefixes with the original common-budget suffix optimizer.", flush=True)
    rows = []
    for root, (task, full, original, _, _, oracle_distance) in enumerate(data):
        goal, prefix = np.asarray(task["goal"]), full[:5]
        runs = []
        for repeat in range(3):
            actions_seen, scores_seen = [], []

            def predict(actions):
                raw = np.asarray(actions, dtype=np.float32)
                pos = exact.endpoints(task, raw)
                scores = np.linalg.norm(pos-goal, axis=1)
                actions_seen.extend(raw.tolist())
                scores_seen.extend(scores.tolist())
                return pos, scores

            zero = np.concatenate((prefix, np.zeros((10, 2), dtype=np.float32)))
            initial = float(predict(zero[None])[1][0])
            seed = 2027112400 + root * 100 + repeat
            chain, score, generations = core.cem(np.zeros((10, 2)), prefix, 480,
                np.random.default_rng(seed), predict, initial, 4)
            runs.append({"seed": seed, "chain": chain.tolist(), "score": score,
                         "generations": generations, "actions": actions_seen, "scores": scores_seen})
        controls = {name: [r["score"] for r in original["controls"][name]["own"]]
                    for name in ("zero", "greedy")}
        rows.append({"root": root, "task": task, "prefix": prefix.tolist(),
                     "oracle_full_distance": oracle_distance, "runs": runs,
                     "scores": {"oracle_prefix": [r["score"] for r in runs], **controls}})
        print(f"root={root} done", flush=True)
    seal.verify(deps)
    write(RESULT, {"manifest_sha256": sha(MANIFEST), "rows": rows, "summary": summary(rows)})


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest, result = json.loads(MANIFEST.read_text()), json.loads(RESULT.read_text())
    seal.verify(manifest["seal"])
    if result["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("manifest hash mismatch")
    data = inputs()
    if len(result["rows"]) != len(data):
        raise ValueError("root count mismatch")
    for root, (row, (task, full, original, _, _, oracle_distance)) in enumerate(zip(result["rows"], data, strict=True)):
        if row["root"] != root or row["task"] != task or row["oracle_full_distance"] != oracle_distance:
            raise ValueError("root binding mismatch")
        np.testing.assert_array_equal(np.asarray(row["prefix"], np.float32), full[:5])
        for name in ("zero", "greedy"):
            if row["scores"][name] != [x["score"] for x in original["controls"][name]["own"]]:
                raise ValueError("control binding mismatch")
        for repeat, run in enumerate(row["runs"]):
            raw = np.asarray(run["actions"], np.float32)
            if raw.shape != (481, 15, 2) or run["seed"] != 2027112400 + root*100 + repeat:
                raise ValueError("sample count or seed mismatch")
            np.testing.assert_array_equal(raw[:, :5], np.broadcast_to(full[:5], (481, 5, 2)))
            scores = np.linalg.norm(exact.endpoints(task, raw) - np.asarray(task["goal"]), axis=1)
            np.testing.assert_allclose(scores, run["scores"], atol=1e-12, rtol=0)
            if run["score"] != min(run["scores"]) or run["score"] != row["scores"]["oracle_prefix"][repeat]:
                raise ValueError("incumbent mismatch")
            np.testing.assert_array_equal(np.asarray(run["chain"], np.float32), raw[int(np.argmin(scores))])
    if result["summary"] != summary(result["rows"]):
        raise ValueError("summary mismatch")
    write(AUDIT, {"result_sha256": sha(RESULT), "verified_roots": len(data), "trajectories": 18*3*481,
                  "scope": "same-backend action/score and binding recomputation; not independent replication"})
    print("Audited 18 roots and 25,974 suffix trajectories.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
