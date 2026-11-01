"""Verify saved paired cp10 physical episodes without new simulation or inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp10_fixed_counterfactual_20261101_exclusive as source
from . import p1_moving_live_feedback_analysis_20261027_v2 as analysis


def require(value, message):
    if not value:
        raise ValueError(message)


def check_order(events, n):
    dispatched = [e for e in events if e["kind"] == "branch_dispatch"]
    returned = [e for e in events if e["kind"] == "branch_returned"]
    seals = [e for e in events if e["kind"] == "cp10_identity_sealed"]
    require(len(dispatched) == len(returned) == n and len(seals) == int(n > 10), "dispatch/seal count")
    for i, (d, r) in enumerate(zip(dispatched, returned, strict=True), 1):
        require(d["step"] == r["step"] == i and d["command"] == r["command"]
                and d["index"] < r["index"], "dispatch/return")
        if i < n:
            require(not r["terminated"] and not r["truncated"], "action after native stop")
    if n > 10:
        require(returned[9]["index"] < seals[0]["index"] < dispatched[10]["index"], "cp10 seal order")


def verify(directory, report_sha):
    directory = Path(directory).resolve(strict=True)
    require(source.sha(directory / "report.json") == report_sha, "report SHA")
    report = json.loads((directory / "report.json").read_bytes())
    binding = json.loads((directory / "execution_binding.json").read_bytes())
    prep = Path(binding["prepared"])
    require(source.sha(prep) == binding["sha256"] == report["contract_sha256"], "contract SHA")
    contract = json.loads(prep.read_bytes())
    require(contract == source.build(), "source and contract closure")
    require(report["status"] == "recorded_development" and not report["errors"]
            and not report["pending"] and not report["secondary_errors"] and report["neural_calls"] == 0,
            "run completion")
    journal = directory / "journal.jsonl"
    require(source.sha(journal) == report["journal_sha256"], "journal SHA")
    events = analysis.journal(journal.read_bytes(), report["journal_chain_tail"])
    require(events[0]["kind"] == "run_started" and events[-1]["kind"] == "run_finished"
            and events[-1]["counts"] == report["counts"], "journal boundary")
    pending, counts = {}, Counter()
    for event in events:
        kind = event["kind"]
        if kind == "operation_attempted":
            ident = event["operation_id"]
            require(ident not in pending, "duplicate physical attempt")
            pending[ident] = event
            counts[(event["resource"], "attempted")] += event["units"]
        elif kind == "operation_completed":
            ident = event["operation_id"]
            require(ident in pending and all(event[k] == pending[ident][k]
                    for k in ("resource", "units", "region", "episode_id")), "physical pair")
            counts[(event["resource"], "completed")] += event["units"]
            del pending[ident]
    require(not pending and all(counts[(k, phase)] == n and n <= contract["physical_caps_total"][k]
            for k, phases in report["counts"].items() for phase, n in phases.items()), "resource totals/caps")
    require(not any(e["kind"] in ("neural_attempt", "neural_return") for e in events), "neural calls")
    require(len(report["rows"]) == len(contract["schedule"]) == 6, "roster")
    summaries, checkpoints = [], {}
    for row, slot in zip(report["rows"], contract["schedule"], strict=True):
        i, n = slot["index"], row["control_returned"]
        ident = f"branch_{i:02}"
        require(row["episode_id"] == ident and row["role"] == slot["role"] and row["tail"] == slot["tail"]
                and json.loads((directory / (ident + "_row.json")).read_bytes()) == row, "row identity")
        relevant = [e for e in events if e.get("branch") == i and e["kind"] in
                    ("branch_dispatch", "branch_returned", "cp10_identity_sealed")]
        check_order(relevant, n)
        commands = np.asarray(slot["commands"], dtype="<f4")
        require(hashlib.sha256(commands.tobytes()).hexdigest() == slot["commands_float32_sha256"]
                and np.array_equal(np.asarray(row["executed_raw"], dtype="<f4").reshape(-1, 2), commands[:n]),
                "action bytes")
        returned = [e for e in relevant if e["kind"] == "branch_returned"]
        require(len(row["transitions"]) == n, "transition count")
        goal = np.asarray(contract["root"]["goal"])
        for event, transition in zip(returned, row["transitions"], strict=True):
            require(all(event[k] == transition[k] for k in
                        ("state", "command", "reward", "terminated", "truncated", "n_contacts", "metrics")),
                    "transition/event")
            m = analysis.native_metrics(transition["state"], goal)
            require(all(m[k] == transition["metrics"][k] for k in
                        ("native_success", "native_distance", "native_angle_term", "negative_native_angle",
                         "wrapped_allxy_success", "wrapped_object_angle_error", "object_position_error")),
                    "physical metric")
        obs = row["observations"]
        expected_count = n if row["status"] == "censored_out_of_domain" else n + 1
        if row["status"] == "native_stopped_warmup":
            expected_count = 0
        require(len(obs) == expected_count and 0 <= n <= 15, "observed/censored frames")
        for j, point in enumerate(obs):
            path = Path(point["path"])
            require(point["step"] == j and path.parent == directory and not path.is_symlink()
                    and source.sha(path) == point["file_sha256"], "frame file")
            with Image.open(path) as image:
                pixels = np.asarray(image)
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"]
                    and (j == 0 or point["state"] == returned[j-1]["state"]), "frame pixels/state")
        if len(obs) > 10:
            source.check_cp10(slot["cp10"], obs[10], slot["commands"][:10], row["executed_raw"][:10])
            cp = {k: obs[10][k] for k in source.IDENTITY}
            if slot["role"] in checkpoints:
                require(checkpoints[slot["role"]] == cp, "within-pair cp10 identity")
            checkpoints[slot["role"]] = cp
        final = row["final_metrics"]
        last = obs[-1] if obs else None
        summaries.append({"role": row["role"], "tail": row["tail"], "status": row["status"],
                          "returned_steps": n, "unexecuted_steps": 15 - n,
                          "native_success": final["native_success"] if final else None,
                          "strict_success": final["wrapped_allxy_success"] if final else None,
                          "joint_xy": float(np.linalg.norm(np.asarray(last["state"][:4]) - goal[:4]))
                                      if last and len(obs) == n + 1 else None,
                          "native_angle_term": final["native_angle_term"] if final else None,
                          "strict_angle": final["wrapped_object_angle_error"] if final else None})
    return {"status": "verified_saved_physical_development", "report_sha256": report_sha,
            "contract_sha256": report["contract_sha256"], "rows": summaries,
            "physical_counts": report["counts"], "neural_calls_new": 0,
            "new_physical_calls_during_verification": 0,
            "independent_confirmation": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.directory, args.report_sha256)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": source.sha(args.output)}))


if __name__ == "__main__":
    main()
