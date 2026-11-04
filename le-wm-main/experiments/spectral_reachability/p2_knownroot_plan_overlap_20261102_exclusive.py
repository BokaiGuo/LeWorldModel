"""Read-only overlap check for the fixed eight-plan known-root physical join."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOIN = "results/codex_p2_knownroot_physical_join_20261030_v1_CPU01.json"
PHYSICAL = "results/p2_composition_paired_physical_20261027_v2_main01/physical_report.json"
HASHES = {
    JOIN: "68fadebe7cce9fca1b3af33dec1813de11534c447b24d6f35044e07746e34a1e",
    PHYSICAL: "e696af5758c88172dafccb30cf62901fe91f9640ebf4df7ad57d4f2f19da8a4d",
}
OUTPUT = ROOT / "results/p2_knownroot_plan_overlap_20261102_exclusive_run02.json"
ARMS = ("joint_h2_no_composition", "conditional_suffix_composition")
ROOTS = (
    "9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333",
    "23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050",
    "4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca",
    "cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def file_digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load():
    result = {}
    for path, expected in HASHES.items():
        raw = (ROOT / path).read_bytes()
        need(file_digest(raw) == expected, "source hash mismatch: " + path)
        result[path] = json.loads(raw)
    return result[JOIN], result[PHYSICAL]


def keyed(rows, arm_key):
    result = {}
    for row in rows:
        key = (row["root_sha256"], row[arm_key])
        need(key[0] in ROOTS and key[1] in ARMS and key not in result, "unexpected or duplicate branch")
        need(type(row["root_index"]) is int and 0 <= row["root_index"] < len(ROOTS)
             and ROOTS[row["root_index"]] == key[0],
             "root index mismatch")
        result[key] = row
    need(set(result) == {(r, a) for r in ROOTS for a in ARMS}, "incomplete branch census")
    return result


def audit(join, physical):
    need(join["status"] == "joined_not_physical_ready" and
         join["physical_factory_bound"] is False and join["physical_authorized"] is False and
         join["P1_admission"] is False and join["P2_admission"] is False,
         "join authorization or status drift")
    plans = keyed(join["plans"], "arm")
    rows = keyed(physical["rows"], "method")
    result = []
    for root in ROOTS:
        for arm in ARMS:
            p, r = plans[root, arm], rows[root, arm]
            need(r["status"] == "completed" and r["full_horizon_completed"] is True and
                 r["failure"] is None and
                 all(r[k] == 10 for k in ("planned_commands", "completed_commands",
                                          "physical_env_step_attempts", "physical_env_step_returns")),
                 "old physical branch incomplete")
            trace = r["trace"]
            need(len(trace) == 10 and [x["primitive_step"] for x in trace] == list(range(1, 11)) and
                 all(x["kind"] == "observation" for x in trace), "old physical trace mismatch")
            need(len(p["commands"]) == 10, "planned action length")
            ctx = r["context_identity"]
            need(p["history_rgb_sha256"] == ctx["history_rgb_sha256"] and
                 p["goal_rgb_sha256"] == ctx["goal_rgb_sha256"], "recorded RGB identity mismatch")
            same = p["commands"] == [x["command"] for x in trace]
            result.append({"root_index": r["root_index"], "root_sha256": root, "arm": arm,
                           "same_commands_as_old_physical": same,
                           "same_recorded_history_and_goal_rgb": True})
    return {
        "schema": "p2_knownroot_plan_overlap_20261102_exclusive_v1",
        "input_sha256": HASHES,
        "fixed_eight_branch_census": result,
        "identical_commands": sum(x["same_commands_as_old_physical"] for x in result),
        "different_commands": sum(not x["same_commands_as_old_physical"] for x in result),
        "physical_authorized": False,
        "P1_admission": False, "P2_admission": False,
        "limits": [
            "Fixed known-root retrospective plan, not a prospective or independent confirmation.",
            "Report-level action and recorded RGB comparison only; does not re-authenticate runtime, source closure or journal semantics.",
            "Differences do not authorize selection of a subset or native physical execution.",
            "The input join predates a later native physical run; this check does not validate that run. No new model, physical or GPU execution.",
        ],
    }


def main():
    raw = (json.dumps(audit(*load()), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"created {OUTPUT} sha256={file_digest(raw)}")


if __name__ == "__main__":
    main()
