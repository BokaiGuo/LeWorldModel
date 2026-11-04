"""Compare per-step and four-knot CEM under a fixed privileged prefix."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v7_e0_oracle_prefix_suffix_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_E0_ORACLE_SUFFIX_ACTION_CEM_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_e0_oracle_suffix_action_cem_dev_20261104_codex"
MANIFEST, RESULT, AUDIT = [PREFIX.with_name(PREFIX.name + suffix + ".json")
                          for suffix in ("_manifest", "_result", "_audit")]


def load_previous():
    prior_manifest = json.loads(prior.MANIFEST.read_text())
    seal.verify(prior_manifest["seal"])
    prev, audit = json.loads(prior.RESULT.read_text()), json.loads(prior.AUDIT.read_text())
    if (prev["manifest_sha256"] != prior.sha(prior.MANIFEST)
            or audit["result_sha256"] != prior.sha(prior.RESULT) or len(prev["rows"]) != 18):
        raise ValueError("previous diagnostic binding mismatch")
    return prev


def summary(rows):
    result = {}
    for name in ("four_knot", "per_step"):
        scores = np.asarray([row[name] for row in rows])
        result[name] = {"mean_distance": float(scores.mean()), "successful_streams": int((scores < .05).sum()),
                        "roots_any_success": int((scores < .05).any(axis=1).sum())}
    difference = np.asarray([np.mean(row["per_step"]) - np.mean(row["four_knot"]) for row in rows])
    result["per_step_vs_four_knot"] = {"wins": int((difference < -1e-10).sum()),
                                       "ties": int((np.abs(difference) <= 1e-10).sum()),
                                       "losses": int((difference > 1e-10).sum())}
    return result


def calculate(root, task, prefix, repeat):
    goal = np.asarray(task["goal"])
    actions, values = [], []

    def predict(batch):
        raw = np.asarray(batch, dtype=np.float32)
        pos = prior.exact.endpoints(task, raw)
        scores = np.linalg.norm(pos-goal, axis=1)
        actions.extend(raw.tolist())
        values.extend(scores.tolist())
        return pos, scores

    zero = np.concatenate((prefix, np.zeros((10, 2), dtype=np.float32)))
    initial = float(predict(zero[None])[1][0])
    seed = 2027112400 + root*100 + repeat
    chain, score, generations = core.cem(np.zeros((10, 2)), prefix, 480,
                                          np.random.default_rng(seed), predict, initial, 10)
    return {"seed": seed, "chain": chain.tolist(), "score": score, "generations": generations,
            "actions": actions, "scores": values}


def run():
    if any(path.exists() for path in (MANIFEST, RESULT, AUDIT)):
        raise FileExistsError("existing or partial diagnostic")
    prev = load_previous()
    inputs = prior.inputs()
    deps = seal.capture([Path(__file__), PROTOCOL, prior.MANIFEST, prior.RESULT, prior.AUDIT,
                         Path(prior.__file__), Path(core.__file__), Path(prior.exact.__file__),
                         Path(prior.exact.wall.__file__),
                         *[path for _, _, _, meta, arr, _ in inputs for path in (meta, arr)]])
    prior.write(MANIFEST, {"seal": deps, "previous_result_sha256": prior.sha(prior.RESULT),
                           "scope": "post-hoc privileged fixed-prefix optimizer diagnostic"})
    print("Comparing per-step and four-knot suffix CEM with equal samples and fixed prefixes.", flush=True)
    rows = []
    for root, (old, (task, full, _, _, _, witness)) in enumerate(zip(prev["rows"], inputs, strict=True)):
        prefix = full[:5]
        if old["root"] != root or old["task"] != task:
            raise ValueError("root mismatch")
        np.testing.assert_array_equal(np.asarray(old["prefix"], np.float32), prefix)
        runs = [calculate(root, task, prefix, repeat) for repeat in range(3)]
        rows.append({"root": root, "task": task, "prefix": prefix.tolist(), "witness_distance": witness,
                     "four_knot": old["scores"]["oracle_prefix"],
                     "per_step": [run["score"] for run in runs], "runs": runs})
        print(f"root={root} done", flush=True)
    seal.verify(deps)
    prior.write(RESULT, {"manifest_sha256": prior.sha(MANIFEST), "rows": rows, "summary": summary(rows)})


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest, result = json.loads(MANIFEST.read_text()), json.loads(RESULT.read_text())
    seal.verify(manifest["seal"])
    if result["manifest_sha256"] != prior.sha(MANIFEST):
        raise ValueError("manifest mismatch")
    prev, inputs = load_previous(), prior.inputs()
    if len(result["rows"]) != len(prev["rows"]) or len(inputs) != 18:
        raise ValueError("panel size mismatch")
    for root, (row, old, (task, full, _, _, _, witness)) in enumerate(
            zip(result["rows"], prev["rows"], inputs, strict=True)):
        if (row["root"] != root or row["task"] != task or row["four_knot"] != old["scores"]["oracle_prefix"]
                or row["witness_distance"] != witness):
            raise ValueError("input/control mismatch")
        np.testing.assert_array_equal(np.asarray(row["prefix"], np.float32), full[:5])
        for repeat, run in enumerate(row["runs"]):
            raw = np.asarray(run["actions"], np.float32)
            if raw.shape != (481, 15, 2) or run["seed"] != 2027112400 + root*100 + repeat:
                raise ValueError("sample count or seed mismatch")
            np.testing.assert_array_equal(raw[:, :5], np.broadcast_to(full[:5], (481, 5, 2)))
            np.testing.assert_array_equal(raw[0, 5:], np.zeros((10, 2)))
            values = np.linalg.norm(prior.exact.endpoints(task, raw) - np.asarray(task["goal"]), axis=1)
            np.testing.assert_allclose(values, run["scores"], atol=1e-12, rtol=0)
            if run["score"] != min(run["scores"]) or run["score"] != row["per_step"][repeat]:
                raise ValueError("best score mismatch")
            np.testing.assert_array_equal(np.asarray(run["chain"], np.float32), raw[int(np.argmin(values))])
            if sum(g["size"] for g in run["generations"]) != 480:
                raise ValueError("generation budget mismatch")
            rebuilt = calculate(root, task, full[:5], repeat)
            if rebuilt != run:
                raise ValueError("deterministic CEM reconstruction mismatch")
    if result["summary"] != summary(result["rows"]):
        raise ValueError("summary mismatch")
    prior.write(AUDIT, {"result_sha256": prior.sha(RESULT), "roots": 18, "trajectories": 18*3*481,
                        "scope": "same-code deterministic CEM and same-backend score reconstruction"})
    print("Audited 18 roots, 54 streams and 25,974 trajectories.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
