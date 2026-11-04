"""Post-hoc paired suffix CEM budget curves on V5 selected/control prefixes."""
import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v5_canonical_response_local_mechanism_20261104_codex as local
from experiments import pc_v5_canonical_random_bridge_20261104_codex as random
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = local.bank.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_SUFFIX_BUDGET_CURVE_DIAG_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_suffix_budget_curve_diag_20261104_codex"
MANIFEST, RESULT, AUDIT = [PREFIX.with_name(PREFIX.name + s + ".json") for s in ("_manifest", "_result", "_audit")]
NAMES = ("selected", "zero", "greedy", "lookahead", "random")
sha, exact = local.sha, local.exact


def save_json(path, obj):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def root_path(root, suffix):
    return PREFIX.with_name(PREFIX.name + f"_root{root:02d}{suffix}")


def inputs():
    for data, audit in ((local.BANK, local.BANK_AUDIT), (local.DATA, local.AUDIT), (random.DATA, random.AUDIT)):
        record = json.loads(audit.read_text())
        key = "plans_sha256" if data == local.BANK else "data_sha256"
        if record[key] != sha(data):
            raise ValueError(f"audit mismatch: {data}")
    bank = json.loads(local.BANK.read_text())["rows"]
    bridges = json.loads(random.DATA.read_text())["rows"]
    out = []
    for index, (row, bridge) in enumerate(zip(bank, bridges, strict=True)):
        if row["root"] != index or bridge["root"] != index:
            raise ValueError("root mismatch")
        candidates = row["candidates"]
        zero = next(item for item in candidates[:-1] if item["name"] == "zero")
        prefixes = [candidates[-1]["prefix"], zero["prefix"], candidates[row["greedy_index"]]["prefix"],
                    candidates[row["lookahead_index"]]["prefix"], bridge["prefix"]]
        out.append((row["task"], prefixes))
    if len(out) != 24:
        raise ValueError("incomplete panel")
    return out


def compute(root, task, prefixes):
    goal = np.asarray(task["goal"])
    outcomes, action_arrays, position_arrays, score_arrays = [], [], [], []
    for name, prefix in zip(NAMES, prefixes, strict=True):
        prefix = np.asarray(prefix, dtype=float)
        arm, actions_reps, positions_reps, scores_reps = [], [], [], []
        for repeat in range(3):
            batches, positions_list, scores_list = [], [], []

            def predict(actions):
                raw = np.asarray(actions, dtype=np.float32)
                positions = exact.endpoints(task, raw)
                scores = np.linalg.norm(positions-goal, axis=1)
                batches.append(raw)
                positions_list.append(positions)
                scores_list.append(scores)
                return positions, scores

            zero = np.concatenate((prefix, np.zeros((10, 2))))[None]
            initial = float(predict(zero)[1][0])
            seed = 2027112400 + root*100 + repeat
            chain, score, generations = core.cem(np.zeros((10, 2)), prefix, 480,
                np.random.default_rng(seed), predict, initial, 4)
            if len(generations) != 40 or any(g["size"] != 12 for g in generations):
                raise ValueError("CEM generation count mismatch")
            arm.append({"seed": seed, "initial_score": initial, "selected_chain": chain.tolist(), "score": score,
                "checkpoint_scores": [generations[g-1]["best_score"] for g in (10, 20, 40)],
                "generations": generations})
            actions_reps.append(np.concatenate(batches))
            positions_reps.append(np.concatenate(positions_list))
            scores_reps.append(np.concatenate(scores_list))
        outcomes.append({"name": name, "prefix": prefix.tolist(), "repeats": arm})
        action_arrays.append(actions_reps)
        position_arrays.append(positions_reps)
        score_arrays.append(scores_reps)
    return outcomes, np.asarray(action_arrays), np.asarray(position_arrays), np.asarray(score_arrays)


def main(mode):
    if mode == "plan":
        targets = [MANIFEST, RESULT, AUDIT] + [root_path(i, s) for i in range(24) for s in (".npz", ".json")]
        if any(p.exists() for p in targets):
            raise FileExistsError("existing or partial budget diagnostic")
        cohort = inputs()
        deps = seal.capture([Path(__file__), PROTOCOL, local.BANK, local.BANK_AUDIT, local.DATA, local.AUDIT,
            random.DATA, random.AUDIT, Path(core.__file__), Path(exact.__file__), Path(exact.wall.__file__)])
        save_json(MANIFEST, {"seal": deps, "names": NAMES, "checkpoints": [120, 240, 480],
            "scope": "post-hoc exposed roots; fixed prefixes, equal per-prefix optimization"})
        print("V5 suffix budget curves: 24 roots, five prefixes, three matched CEM streams.", flush=True)
        rows = []
        for root, (task, prefixes) in enumerate(cohort):
            outcomes, actions, positions, scores = compute(root, task, prefixes)
            file = root_path(root, ".npz")
            with file.open("xb") as stream:
                np.savez_compressed(stream, actions=actions, positions=positions, scores=scores)
                stream.flush()
                os.fsync(stream.fileno())
            metadata = root_path(root, ".json")
            save_json(metadata, {"root": root, "task": task, "arrays_sha256": sha(file), "outcomes": outcomes})
            rows.append({"root": root, "metadata_sha256": sha(metadata), "arrays_sha256": sha(file)})
            print(f"root={root} complete", flush=True)
        seal.verify(deps)
        save_json(RESULT, {"manifest_sha256": sha(MANIFEST), "rows": rows})
    else:
        if AUDIT.exists():
            raise FileExistsError(AUDIT)
        cohort = inputs()
        manifest, result = json.loads(MANIFEST.read_text()), json.loads(RESULT.read_text())
        seal.verify(manifest["seal"])
        if result["manifest_sha256"] != sha(MANIFEST) or len(result["rows"]) != 24:
            raise ValueError("manifest mismatch")
        for root, (task, prefixes) in enumerate(cohort):
            meta_path, data_path = root_path(root, ".json"), root_path(root, ".npz")
            meta, ref = json.loads(meta_path.read_text()), result["rows"][root]
            if (meta["root"] != root or meta["task"] != task or ref["root"] != root
                    or ref["metadata_sha256"] != sha(meta_path) or ref["arrays_sha256"] != sha(data_path)
                    or meta["arrays_sha256"] != sha(data_path)):
                raise ValueError("result identity mismatch")
            outcomes, actions, positions, scores = compute(root, task, prefixes)
            if outcomes != meta["outcomes"]:
                raise ValueError("optimizer deterministic replay mismatch")
            with np.load(data_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["actions"], actions)
                np.testing.assert_allclose(data["positions"], positions, atol=1e-12, rtol=0)
                np.testing.assert_allclose(data["scores"], scores, atol=1e-12, rtol=0)
            print(f"root={root} audited", flush=True)
        seal.verify(manifest["seal"])
        save_json(AUDIT, {"result_sha256": sha(RESULT), "verified_roots": 24, "prefixes_per_root": 5,
            "restarts_per_prefix": 3, "chains": 24*5*3*481,
            "scope": "same-code exact dynamics deterministic optimizer reconstruction"})
        print("All paired prefix optimizations independently reconstructed.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    main(sys.argv[1])
