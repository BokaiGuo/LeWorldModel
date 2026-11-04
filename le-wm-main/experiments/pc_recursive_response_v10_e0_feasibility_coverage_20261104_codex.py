"""Post-hoc privileged common-suffix coverage of the sealed v10 first banks."""
import json
from pathlib import Path
import sys

import numpy as np

from experiments.pc_recursive_response_v10_e0_dev_20261104_codex import OUT, previous, seal, tasks

ROOT = previous.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RECURSIVE_RESPONSE_V10_E0_FEASIBILITY_COVERAGE_PROTOCOL_20261104_CODEX.md"
RESULT = OUT / "feasibility_coverage.json"
AUDIT = OUT / "feasibility_coverage_audit.json"


def run():
    if RESULT.exists() or AUDIT.exists():
        raise FileExistsError("existing feasibility output")
    manifest = json.loads((OUT / "manifest.json").read_text())
    prior = json.loads((OUT / "result.json").read_text())
    audit = json.loads((OUT / "structure_audit.json").read_text())
    seal.verify(manifest["dependencies"])
    if audit["result_sha256"] != previous.sha(OUT / "result.json"):
        raise ValueError("unverified v10 run")
    deps = seal.capture([Path(__file__), PROTOCOL, OUT / "manifest.json", OUT / "result.json",
        OUT / "structure_audit.json", Path(previous.exact.__file__), Path(previous.exact.wall.__file__),
        Path(previous.exact.wall.physics.__file__)])
    rows = []
    for root, task in enumerate(tasks()):
        goal = np.asarray(task["goal"])
        method_rows = {x["arm"]: x for x in prior["rows"] if x["root"] == root and x["arm"].startswith("v10")}
        if set(method_rows) != {"v10_mode1", "v10_mode2"}:
            raise ValueError("missing v10 arms")
        witnesses = []
        for w in previous.exact.wall.feasibility(task):
            item = {"height": w["height"], "steps_needed": w["steps_needed"], "eligible": False}
            if w["within_horizon"] and w["steps_needed"] <= 15:
                raw = (np.asarray(w["actions"][:15]) / previous.exact.wall.HIGH).astype(np.float32)
                if raw.shape != (15, 2) or not np.isfinite(raw).all() or np.abs(raw).max() > 1:
                    raise ValueError("bad witness")
                terminal = previous.exact.endpoints(task, raw[None])[0]
                distance = float(np.linalg.norm(terminal-goal))
                item.update(eligible=True, actions=raw.tolist(), terminal_distance=distance,
                            success=distance < .05)
                if item["success"]:
                    item["coverage"] = {}
                    for arm, row in method_rows.items():
                        first = row["result"]["layers"][0]
                        bank = np.asarray([x["chain"] for x in first["candidates"]], np.float32)
                        full = np.concatenate((bank, np.broadcast_to(raw[5:], (len(bank), 10, 2))), axis=1)
                        endpoints = previous.exact.endpoints(task, full)
                        scores = np.linalg.norm(endpoints-goal, axis=1)
                        successful = np.flatnonzero(scores < .05).tolist()
                        kept = first["retained"]
                        item["coverage"][arm] = {"scores": scores.tolist(),
                            "nearest_prefix_action_l2": float(np.linalg.norm(
                                (bank - raw[:5]).reshape(len(bank), -1), axis=1).min()),
                            "successful_indices": successful, "retained": kept,
                            "retained_successful_indices": [i for i in kept if i in successful]}
            witnesses.append(item)
        rows.append({"root": root, "task": task, "witnesses": witnesses})
    seal.verify(deps)
    previous.write(RESULT, {"scope": "post-hoc privileged fixed-suffix diagnostic, not planning",
                            "dependencies": deps, "rows": rows})
    print(f"saved post-hoc witness and coverage results for {len(rows)} roots")


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(RESULT.read_text())
    prior = json.loads((OUT / "result.json").read_text())
    seal.verify(data["dependencies"])
    count = 0
    for root, row in enumerate(data["rows"]):
        if row["root"] != root or row["task"] != tasks()[root]:
            raise ValueError("task mismatch")
        task, goal = row["task"], np.asarray(row["task"]["goal"])
        for item in row["witnesses"]:
            if not item["eligible"]:
                continue
            action = np.asarray(item["actions"], np.float32)
            distance = float(np.linalg.norm(previous.exact.endpoints(task, action[None])[0]-goal))
            if abs(distance-item["terminal_distance"]) > 1e-12 or item["success"] != (distance < .05):
                raise ValueError("witness score mismatch")
            for arm, coverage in item.get("coverage", {}).items():
                plan = next(x for x in prior["rows"] if x["root"] == root and x["arm"] == arm)
                layer = plan["result"]["layers"][0]
                bank = np.asarray([x["chain"] for x in layer["candidates"]], np.float32)
                full = np.concatenate((bank, np.broadcast_to(action[5:], (len(bank), 10, 2))), axis=1)
                scores = np.linalg.norm(previous.exact.endpoints(task, full)-goal, axis=1)
                np.testing.assert_allclose(scores, coverage["scores"], rtol=0, atol=1e-12)
                valid = np.flatnonzero(scores < .05).tolist()
                if (valid != coverage["successful_indices"] or coverage["retained"] != layer["retained"]
                        or coverage["retained_successful_indices"] != [i for i in layer["retained"] if i in valid]):
                    raise ValueError("coverage index mismatch")
                nearest = float(np.linalg.norm((bank-action[:5]).reshape(len(bank), -1), axis=1).min())
                if abs(nearest-coverage["nearest_prefix_action_l2"]) > 1e-12:
                    raise ValueError("distance mismatch")
                count += len(full)
    seal.verify(data["dependencies"])
    previous.write(AUDIT, {"result_sha256": previous.sha(RESULT), "replayed_candidate_chains": count,
                           "scope": "same exact backend and fixed saved banks"})
    print(f"audited {count} privileged suffix candidate chains")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
