"""Identify duplicate saved SRA replays; no inference or physical execution."""
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
SOURCES = {
    "composition": ("results/p2_composition_paired_physical_20261027_v2_main01/physical_report.json",
                    "e696af5758c88172dafccb30cf62901fe91f9640ebf4df7ad57d4f2f19da8a4d"),
    "sra": ("results/p2_no_composition_sra_physical_20261027_v2_cuda_main_v1/physical_report.json",
            "fe105e05b73a9bdfd1b602eab740753b2a773b3ac42c27ca5c95156fe524bb03"),
}
ROOTS = (
    "9173218f71898716eac8fe0f3946b8e2cf20527ac71d84555c20c4e1b7fbf333",
    "23b8de7837c39fb4e0fcffe7678732a9d840c57de7a0c8376e22076770ef6050",
    "4afefa3be02a327aa69c22f38c17376d82340ba352aaf4aa29f95b070f5503ca",
    "cf1b9bb913bf7815c9184d5de649b77bf09cc268dd80905a0feca9bd2a2edbf2",
)
OUTPUT = PROJECT / "results/p2_sra_duplicate_replay_check_20261102_exclusive_run01.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_pinned(project=PROJECT):
    data = {}
    for label, (path, expected) in SOURCES.items():
        raw = (project / path).read_bytes()
        require(digest(raw) == expected, "source hash mismatch: " + label)
        data[label] = json.loads(raw)
    return data


def index_rows(rows, method):
    selected = [r for r in rows if r["method"] == method]
    require(len(selected) == 4, "expected four rows: " + method)
    result = {}
    for row in selected:
        root = row["root_sha256"]
        require(root in ROOTS and root not in result, "unknown or duplicate root")
        require(row["root_index"] == ROOTS.index(root), "root index mismatch")
        result[root] = row
    require(set(result) == set(ROOTS), "root census mismatch")
    return result


def check_branch(row):
    require(row["status"] == "completed" and row["full_horizon_completed"] is True,
            "incomplete execution")
    require(row["failure"] is None and row["planned_commands"] == 10 and
            all(row[k] == 10 for k in ("completed_commands", "attempted_commands",
                                       "physical_env_step_attempts", "physical_env_step_returns")),
            "execution count or failure mismatch")
    trace = row["trace"]
    require(len(trace) == 10 and
            [t["primitive_step"] for t in trace] == list(range(1, 11)) and
            all(t["kind"] == "observation" for t in trace) and
            trace[-1]["state"] == row["final_state"], "trace mismatch")
    for kind in ("journal", "terminal_rgb"):
        path = Path(row[kind + "_path"])
        require(digest(path.read_bytes()) == row[kind + "_sha256"],
                kind + " hash mismatch")


def compare(data):
    a = index_rows(data["composition"]["rows"], "joint_h2_no_composition")
    b = index_rows(data["sra"]["rows"], "joint_h2_no_composition_sra")
    pairs = []
    for root in ROOTS:
        left, right = a[root], b[root]
        check_branch(left)
        check_branch(right)
        x, y = left["context_identity"], right["context_identity"]
        fields = ("state", "goal", "history_rgb_sha256", "goal_rgb_sha256")
        identity = all(x[k] == y[k] for k in fields) and x["raw_history"] == y["history"]
        require(identity, "reset/goal/history mismatch")
        pairs.append({
            "root_index": left["root_index"], "root_sha256": root,
            "shared_recorded_context": identity,
            "identical_trace_including_commands_rewards_states_termination": left["trace"] == right["trace"],
            "identical_final_state": left["final_state"] == right["final_state"],
            "identical_terminal_rgb_sha256": left["terminal_rgb_sha256"] == right["terminal_rgb_sha256"],
        })
    return {
        "schema": "p2_sra_duplicate_replay_check_20261102_exclusive_v1",
        "source_sha256": {k: v[1] for k, v in SOURCES.items()},
        "pairs": pairs,
        "limits": [
            "Previously used calibration roots; matching replay is not independent confirmation.",
            "Pinned reports and their recorded journal/RGB hashes checked; source provenance and journal semantics not re-certified.",
            "No new model inference or physical execution; no admission or equal-total-budget claim.",
        ],
    }


def main():
    result = compare(load_pinned())
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(f"created {OUTPUT} sha256={digest(raw)}")


if __name__ == "__main__":
    main()
