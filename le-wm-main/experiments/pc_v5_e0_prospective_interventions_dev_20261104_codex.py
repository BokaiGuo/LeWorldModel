"""Predeclared same-chain, common-tail and replanning interventions for V5."""
import json
import os
import sys
from pathlib import Path

import numpy as np

from experiments import pc_target_reach_v4_local_response_diag_20261104_codex as local
from experiments import pc_v5_e0_prospective_mechanism_dev_20261104_codex as source
from experiments.pc_v5_suffix_budget_curve_diag_20261104_codex import save_json
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PREFIX = source.PREFIX.with_name(source.PREFIX.name + "_intervention")
MANIFEST, RESULT, AUDIT = [PREFIX.with_name(PREFIX.name + part + ".json")
                          for part in ("_manifest", "_result", "_audit")]
NAMES = ("selected", "zero", "greedy", "lookahead", "random")
sha, exact = source.sha, source.exact


def root_path(index, suffix):
    return PREFIX.with_name(PREFIX.name + f"_root{index:02d}{suffix}")


def calculate(row):
    root, task, result = row["root"], row["task"], row["result"]
    goal = np.asarray(task["goal"])
    selected = np.asarray(result["selected"]["chain"], np.float32)
    bank = [{"name": item["name"], "prefix": item["prefix"]} for item in result["first_bank"]]
    selection_runs, selection_actions = [], []
    for item in bank:
        actions_seen = []

        def predict(actions):
            raw = np.asarray(actions, np.float32)
            pos = exact.endpoints(task, raw)
            actions_seen.append(raw)
            return pos, np.linalg.norm(pos-goal, axis=1)

        prefix = np.asarray(item["prefix"])
        initial = float(predict(np.concatenate((prefix, np.zeros((10, 2))))[None])[1][0])
        chain, score, generations = core.cem(np.zeros((10, 2)), prefix, 119,
            np.random.default_rng(2027111400 + root), predict, initial, 4)
        selection_runs.append({"chain": chain.tolist(), "score": score, "generations": generations})
        selection_actions.append(np.concatenate(actions_seen))
    lookahead = int(np.argmin([x["score"] for x in selection_runs]))
    greedy = int(np.argmin(result["prefix_scores"]))
    zero = next(i for i, item in enumerate(bank) if item["name"] == "zero")
    bridge = np.clip(core.interpolate(np.random.default_rng(2027112300+root).normal(size=(1, 4, 2)), 5)[0], -1, 1)
    prefixes = [selected[:5], np.asarray(bank[zero]["prefix"]), np.asarray(bank[greedy]["prefix"]),
                np.asarray(bank[lookahead]["prefix"]), bridge]
    tails = [np.tile(value, (10, 1)) for value in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
    tails.append(selected[5:])
    controls, all_actions, all_positions, all_scores, common_actions = {}, [], [], [], []
    for name, prefix in zip(NAMES, prefixes, strict=True):
        own, reps_a, reps_p, reps_s = [], [], [], []
        for repeat in range(3):
            aa, pp, ss = [], [], []

            def predict(actions):
                raw = np.asarray(actions, np.float32)
                pos = exact.endpoints(task, raw)
                scores = np.linalg.norm(pos-goal, axis=1)
                aa.append(raw)
                pp.append(pos)
                ss.append(scores)
                return pos, scores

            initial = float(predict(np.concatenate((prefix, np.zeros((10, 2))))[None])[1][0])
            seed = 2027112400 + root*100 + repeat
            chain, score, generations = core.cem(np.zeros((10, 2)), prefix, 480,
                np.random.default_rng(seed), predict, initial, 4)
            own.append({"seed": seed, "chain": chain.tolist(), "score": score, "generations": generations})
            reps_a.append(np.concatenate(aa))
            reps_p.append(np.concatenate(pp))
            reps_s.append(np.concatenate(ss))
        common = np.asarray([np.concatenate((prefix, tail)) for tail in tails], np.float32)
        pos = exact.endpoints(task, common)
        values = np.linalg.norm(pos-goal, axis=1)
        controls[name] = {"prefix": prefix.tolist(), "own": own, "mean_own": float(np.mean([x["score"] for x in own])),
                          "common_distances": values.tolist(), "relative_zero": (values-values[0]).tolist()}
        all_actions.append(reps_a)
        all_positions.append(reps_p)
        all_scores.append(reps_s)
        common_actions.append(common)
    response_row = {"root": root, "task": task, "selected_tail": selected[5:].tolist(),
        "candidates": bank + [{"name": "selected_final", "prefix": selected[:5].tolist()}],
        "greedy_index": greedy, "lookahead_index": lookahead}
    response = local.build(response_row)
    # Reuse the audited local-response implementation with the fixed random
    # prefix in its lookahead slot; selected computations must agree exactly.
    random_row = dict(response_row)
    random_row["candidates"] = bank + [{"name": "random_bridge", "prefix": bridge.tolist()},
                                       response_row["candidates"][-1]]
    random_row["lookahead_index"] = len(bank)
    random_response = local.build(random_row)
    if response["controls"]["selected"] != random_response["controls"]["selected"]:
        raise ValueError("selected local response changed between controls")
    response["controls"]["random"] = random_response["controls"]["lookahead"]
    response["gaps"]["random"] = random_response["gaps"]["lookahead"]

    def qualifies(others):
        if not (response["nonprogress"] and response["selected_prefix_nonzero"] and response["selected_tail_nonzero"]):
            return False
        for name in others:
            if not (controls["selected"]["relative_zero"][-1] < controls[name]["relative_zero"][-1]-1e-10
                    and controls["selected"]["mean_own"] < controls[name]["mean_own"]-1e-10):
                return False
            for eps in ("0.01", "0.05"):
                gap = response["gaps"][name][eps]
                if gap["goal_gradient"] is None or not (gap["matrix"] > 1e-6 and gap["goal_gradient"] > 1e-6):
                    return False
        return all(response["controls"][name]["scales"][eps]["branch_stable"]
                   for name in ("selected", *others) for eps in ("0.01", "0.05"))

    data = {"root": root, "task": task, "selected_chain": selected.tolist(), "bank": bank,
        "selection_runs": selection_runs, "greedy_index": greedy, "lookahead_index": lookahead,
        "random_seed": 2027112300+root, "controls": controls, "response": response,
        "base_intersection": qualifies(("zero", "greedy")),
        "all_intersection": qualifies(("zero", "greedy", "lookahead", "random"))}
    arrays = {"selection_actions": np.asarray(selection_actions), "own_actions": np.asarray(all_actions),
        "own_positions": np.asarray(all_positions), "own_scores": np.asarray(all_scores),
        "common_actions": np.asarray(common_actions)}
    return data, arrays


def main(mode):
    manifest, plans_audit = json.loads(source.MANIFEST.read_text()), json.loads(source.AUDIT.read_text())
    seal.verify(manifest["seal"])
    if plans_audit["plans_sha256"] != sha(source.PLANS):
        raise ValueError("sealed plans audit mismatch")
    rows = json.loads(source.PLANS.read_text())["rows"]
    if len(rows) != 18:
        raise ValueError("plan panel incomplete")
    if mode == "intervene":
        outputs = [MANIFEST, RESULT, AUDIT] + [root_path(i, s) for i in range(18) for s in (".json", ".npz")]
        if any(p.exists() for p in outputs):
            raise FileExistsError("existing or partial intervention")
        deps = seal.capture([Path(__file__), source.PROTOCOL, source.MANIFEST, source.PLANS, source.AUDIT,
            Path(local.__file__), Path(core.__file__), Path(exact.__file__), Path(exact.wall.__file__)])
        save_json(MANIFEST, {"seal": deps, "source_sha256": sha(source.PLANS),
            "scope": "prospective fixed-rule development mechanism; provenance not certified"})
        print("Executing predeclared V5 same-chain interventions.", flush=True)
        index_rows = []
        for row in rows:
            data, arrays = calculate(row)
            index = row["root"]
            array_path, meta_path = root_path(index, ".npz"), root_path(index, ".json")
            with array_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            data["arrays_sha256"] = sha(array_path)
            save_json(meta_path, data)
            index_rows.append({"root": index, "metadata_sha256": sha(meta_path), "arrays_sha256": sha(array_path)})
            print(f"root={index} intervened", flush=True)
        seal.verify(deps)
        save_json(RESULT, {"manifest_sha256": sha(MANIFEST), "rows": index_rows})
    else:
        if AUDIT.exists():
            raise FileExistsError(AUDIT)
        im, result = json.loads(MANIFEST.read_text()), json.loads(RESULT.read_text())
        seal.verify(im["seal"])
        if result["manifest_sha256"] != sha(MANIFEST) or len(result["rows"]) != 18:
            raise ValueError("intervention manifest mismatch")
        for row, binding in zip(rows, result["rows"], strict=True):
            root = row["root"]
            meta_path, array_path = root_path(root, ".json"), root_path(root, ".npz")
            data = json.loads(meta_path.read_text())
            if (binding["root"] != root or binding["metadata_sha256"] != sha(meta_path)
                    or binding["arrays_sha256"] != sha(array_path) or data["arrays_sha256"] != sha(array_path)):
                raise ValueError("output binding mismatch")
            expected, arrays = calculate(row)
            expected["arrays_sha256"] = sha(array_path)
            if expected != data:
                raise ValueError("intervention deterministic rebuild mismatch")
            with np.load(array_path, allow_pickle=False) as saved:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(saved[key], value)
            print(f"root={root} audited", flush=True)
        seal.verify(im["seal"])
        save_json(AUDIT, {"result_sha256": sha(RESULT), "verified_roots": 18,
            "scope": "same-code exact dynamics deterministic reconstruction"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("intervene", "audit"):
        raise SystemExit("choose intervene or audit")
    main(sys.argv[1])
