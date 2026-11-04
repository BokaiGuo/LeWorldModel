"""Verify saved four-winner CPU execution, without replaying or using a GPU."""
import argparse
import json
from pathlib import Path

from . import p3_venv_winner_cpu_physical_20261102_codexexclusive as physical


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(prepared, report, events, directory):
    require(events[0]["kind"] == "run_started" and events[0]["prepare"] == prepared,
            "embedded prepare")
    require(report["status"] == "moving_history_physical_recorded" and not report["errors"],
            "execution status")
    require(len(report["rows"]) == len(prepared["plans"]) == 4, "fixed roster size")
    closures = [e for e in events if e["kind"] == "branch_closed"]
    require(len(closures) == 4, "branch closure count")
    for i, (row, plan) in enumerate(zip(report["rows"], prepared["plans"])):
        branch = [e for e in events if e.get("branch") == i]
        started = [e for e in branch if e["kind"] == "branch_started"]
        closed = closures[i]
        require(len(started) == 1 and started[0]["plan"] == plan
                and closed["receipt"] == row and started[0]["index"] < closed["index"]
                and all(e["index"] < closed["index"] for e in branch)
                and (i == 0 or closures[i-1]["index"] < started[0]["index"]), "branch binding")
        require((row["arm"], row["root_slot"], row["selection_digest"]) ==
                (plan["arm"], plan["root_slot"], plan["selection_digest"]), "selection")
        require(row["initial_exact"] and row["full_horizon"] and
                row["status"] == "horizon_completed" and row["own_returned"] == 10
                and row["warmup_returned"] == 10 and not row["terminated"]
                and not row["truncated"], "completion")
        warm = [e for e in branch if e["kind"] == "warmup_return"]
        require([e["step"] for e in warm] == list(range(1, 11)) and
                [e["raw_action"] for e in warm] == started[0]["raw_warmup"], "warmup")
        transitions = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        require([e["step"] for e in transitions] == list(range(1, 11)) and
                [e["command"] for e in transitions] == plan["commands"] and
                transitions[-1]["state"] == row["final_state"], "plan transitions")
        require(all(not e["terminated"] and not e["truncated"] and
                    e["returned_goal_state"] == row["goal"] for e in transitions), "goal and termination")
        require(row["initial_metrics"] == physical.moving.metric(row["goal"], row["root_state"])
                and row["final_metrics"] == physical.moving.metric(row["goal"], row["final_state"]),
                "metric reconstruction")
        guards = [e for e in branch if e["kind"] == "guard_receipt"]
        require(len(guards) == 22 and all(e["accepted"] and e["render_preserved_dynamics"] for e in guards),
                "render guards")
        observations = [e for e in branch if e["kind"] == "fresh_observation"]
        require(len(observations) == 22, "observation count")
        for receipt in row["checkpoints"]:
            path = directory / receipt["png"]
            require(path.parent == directory and physical.moving.sha(path) == receipt["file_sha256"],
                    "checkpoint PNG")
        require([(x["phase"], x["step"]) for x in row["checkpoints"]] ==
                [("plan", 0), ("plan", 5), ("plan", 10)], "checkpoints")
    require(report["native_successes"] == sum(r["native_success"] for r in report["rows"]),
            "success count")
    require(events[-1]["counts"] == report["counts"] and events[-1]["errors"] == report["errors"],
            "terminal counts")
    return {"branches": 4, "native_successes": report["native_successes"],
            "checks": "saved chain, contract, commands, guards, checkpoints, endpoint metrics",
            "independent_physics_attestation": False}


def consume(prepared_path, prepare_sha, directory):
    moving = physical.moving
    prepared_path, directory = Path(prepared_path).resolve(), Path(directory).resolve()
    require(moving.sha(prepared_path) == prepare_sha, "prepare SHA")
    prepared = moving.read(prepared_path)
    require(moving.read(directory / "execution_binding.json") ==
            {"prepared": str(prepared_path), "prepare_sha256": prepare_sha}, "execution binding")
    report = moving.read(directory / "report.json")
    journal_path = directory / "journal.jsonl"
    require(moving.sha(journal_path) == report["journal_sha256"], "journal SHA")
    events = moving.journal(journal_path, report["journal_chain_tail"])
    result = validate(prepared, report, events, directory)
    result.update(report_sha256=moving.sha(directory / "report.json"),
                  journal_sha256=report["journal_sha256"], prepare_sha256=prepare_sha)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = consume(args.prepared, args.sha256, args.run)
    output = Path(args.output)
    require(output.is_absolute() and output.parent == physical.ROOT / "results"
            and output.name.startswith("p3_venv_winner_cpu_physical_20261102_codexexclusive_CONSUME_"),
            "exclusive output path")
    with output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
