"""Post-hoc equal-opportunity suffix search for every sealed v10 prefix."""
import json
from pathlib import Path
import sys

import numpy as np

from experiments.pc_recursive_response_v10_e0_dev_20261104_codex import OUT, ROOT, previous, core, seal

PROTOCOL = ROOT.parent / "docs/PC_RECURSIVE_RESPONSE_V10_E0_FIXED_BANK_SUFFIX_PROTOCOL_20261104_CODEX.md"
PREFIX = OUT / "fixed_bank_suffix"
MANIFEST = PREFIX / "manifest.json"
RESULT = PREFIX / "result.json"
AUDIT = PREFIX / "audit.json"


def run():
    if PREFIX.exists():
        raise FileExistsError(PREFIX)
    plans = json.loads((OUT / "result.json").read_text())
    audit = json.loads((OUT / "structure_audit.json").read_text())
    if audit["result_sha256"] != previous.sha(OUT / "result.json"):
        raise ValueError("unverified source")
    deps = seal.capture([Path(__file__), PROTOCOL, OUT / "result.json", OUT / "structure_audit.json",
                         Path(core.__file__), Path(previous.exact.__file__), Path(previous.exact.wall.__file__)])
    PREFIX.mkdir()
    previous.write(MANIFEST, {"dependencies": deps, "scope": "post-hoc fixed generated bank"})
    rows = []
    for root in range(8):
        for arm in ("v10_mode1", "v10_mode2"):
            plan = next(x for x in plans["rows"] if x["root"] == root and x["arm"] == arm)
            layer = plan["result"]["layers"][0]
            task, goal = plan["task"], np.asarray(plan["task"]["goal"])
            bank = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
            all_actions, all_scores, candidates = [], [], []
            for index, prefix in enumerate(bank):
                repeats = []
                for repeat in range(2):
                    seen_actions, seen_scores = [], []

                    def predict(actions):
                        raw = np.asarray(actions, np.float32)
                        pos = previous.exact.endpoints(task, raw)
                        values = np.linalg.norm(pos-goal, axis=1)
                        seen_actions.extend(raw)
                        seen_scores.extend(values)
                        return pos, values

                    zero_chain = np.concatenate((prefix, np.zeros((10, 2), np.float32)))
                    initial = float(predict(zero_chain[None])[1][0])
                    seed = 2027120100 + root*100 + repeat
                    chain, value, generations = core.cem(np.zeros((10, 2)), prefix, 480,
                        np.random.default_rng(seed), predict, initial, 4)
                    if len(seen_actions) != 481:
                        raise ValueError("incorrect budget")
                    all_actions.append(np.stack(seen_actions))
                    all_scores.append(np.asarray(seen_scores))
                    repeats.append({"seed": seed, "score": value, "chain": np.asarray(chain, np.float32).tolist(),
                                    "generations": generations})
                candidates.append({"index": index, "prefix": prefix.tolist(), "repeats": repeats})
            path = PREFIX / f"root{root:02d}_{arm}.npz"
            with path.open("xb") as f:
                np.savez_compressed(f, actions=np.stack(all_actions).reshape(len(bank), 2, 481, 15, 2),
                                    scores=np.stack(all_scores).reshape(len(bank), 2, 481))
            rows.append({"root": root, "arm": arm, "task": task, "retained": layer["retained"],
                         "greedy": int(np.argmin([x["score"] for x in layer["candidates"]])),
                         "candidates": candidates, "arrays": path.name, "arrays_sha256": previous.sha(path)})
            print(f"root={root} {arm}: {len(bank)} prefixes completed", flush=True)
    seal.verify(deps)
    previous.write(RESULT, {"manifest_sha256": previous.sha(MANIFEST), "rows": rows})


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    manifest, result = json.loads(MANIFEST.read_text()), json.loads(RESULT.read_text())
    seal.verify(manifest["dependencies"])
    if result["manifest_sha256"] != previous.sha(MANIFEST) or len(result["rows"]) != 16:
        raise ValueError("manifest mismatch")
    source = json.loads((OUT / "result.json").read_text())
    checked = 0
    for row in result["rows"]:
        plan = next(x for x in source["rows"] if x["root"] == row["root"] and x["arm"] == row["arm"])
        layer = plan["result"]["layers"][0]
        if (row["task"] != plan["task"] or row["retained"] != layer["retained"]
                or row["greedy"] != int(np.argmin([x["score"] for x in layer["candidates"]]))):
            raise ValueError("source mismatch")
        path = PREFIX / row["arrays"]
        if previous.sha(path) != row["arrays_sha256"]:
            raise ValueError("array hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            actions, scores = data["actions"], data["scores"]
            if actions.shape != (len(layer["candidates"]), 2, 481, 15, 2):
                raise ValueError("wrong shape")
            for index, candidate in enumerate(row["candidates"]):
                prefix = np.asarray(layer["candidates"][index]["chain"], np.float32)
                np.testing.assert_array_equal(candidate["prefix"], prefix)
                for rep, run_result in enumerate(candidate["repeats"]):
                    raw = actions[index, rep]
                    np.testing.assert_array_equal(raw[:, :5], np.broadcast_to(prefix, (481, 5, 2)))
                    np.testing.assert_array_equal(raw[0, 5:], np.zeros((10, 2)))
                    reconstructed = np.linalg.norm(previous.exact.endpoints(row["task"], raw) -
                                                   np.asarray(row["task"]["goal"]), axis=1)
                    np.testing.assert_allclose(reconstructed, scores[index, rep], atol=1e-12, rtol=0)
                    if (run_result["seed"] != 2027120100 + row["root"]*100 + rep
                            or sum(g["size"] for g in run_result["generations"]) != 480
                            or run_result["score"] != min(reconstructed)):
                        raise ValueError("budget, seed, or incumbent mismatch")
                    np.testing.assert_array_equal(np.asarray(run_result["chain"], np.float32),
                                                  raw[int(np.argmin(reconstructed))])
                    checked += len(raw)
    seal.verify(manifest["dependencies"])
    previous.write(AUDIT, {"result_sha256": previous.sha(RESULT), "chains": checked,
                           "scope": "same exact backend, stored optimizer trace arithmetic"})
    print(f"audited {checked} full chains")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
