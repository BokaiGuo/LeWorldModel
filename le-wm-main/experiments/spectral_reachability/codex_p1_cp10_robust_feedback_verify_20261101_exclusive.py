"""Independent saved-artifact verifier; never replays physics or inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_cp10_robust_feedback_20261101_exclusive as spec
from . import p1_moving_live_feedback_analysis_20261027_v2 as analysis

CAPS = {"actual_rgb_encode": 2, "action_encoder": 14, "predict": 14}


def require(value, message):
    if not value:
        raise ValueError(message)


def unobserved(planned, returned):
    require(type(returned) is int and 0 <= returned <= planned, "invalid returned count")
    return planned - returned


def expected_neural(n):
    unobserved(15, n)
    return {"actual_rgb_encode": int(n > 5) + int(n > 10),
            "action_encoder": 4 * int(n > 5) + 10 * int(n > 10),
            "predict": 4 * int(n > 5) + 10 * int(n > 10)}


def check_frame_count(n, count, status, censor_step):
    if status == "native_stopped_warmup":
        require(n == count == 0 and censor_step is None, "warmup stop")
    elif status == "censored_out_of_domain":
        require(n > 0 and count == n and censor_step == n, "capture censor")
    else:
        require(count == n + 1 and censor_step is None, "frame count")


def cp10_pair(row):
    return {"cp10": {k: row["observations"][10][k] for k in spec.IDENTITY},
            "first10": row["executed_raw"][:10]}


def check_matched_cp10(a, b):
    require(len(a["executed_raw"]) >= 10 and len(b["executed_raw"]) >= 10
            and cp10_pair(a) == cp10_pair(b), "paired cp10 identity")


def check_order(events, role, n):
    subset = [e for e in events if e.get("role") == role and
              e["kind"] in ("branch_dispatch", "branch_returned", "decision_sealed")]
    dispatched = [e for e in subset if e["kind"] == "branch_dispatch"]
    returned = [e for e in subset if e["kind"] == "branch_returned"]
    require(len(dispatched) == len(returned) == n, "dispatch/return count")
    for step, (sent, ret) in enumerate(zip(dispatched, returned, strict=True), 1):
        require(sent["step"] == ret["step"] == step and sent["command"] == ret["command"]
                and sent["index"] < ret["index"], "dispatch/return ordering")
        if step < n:
            require(not ret["terminated"] and not ret["truncated"], "action after native stop")
    decisions = [e for e in subset if e["kind"] == "decision_sealed"]
    require([e["boundary"] for e in decisions] == [b for b in (5, 10) if n > b], "decision roster")
    for d in decisions:
        b = d["boundary"]
        require(returned[b-1]["index"] < d["index"] < dispatched[b]["index"], "decision seal ordering")


def check_neural(events, role, attempts, returns):
    rows = [e for e in events if e.get("role") == role and e["kind"] in ("neural_attempt", "neural_return")]
    require(len(rows) % 2 == 0, "incomplete neural pair")
    counts = dict.fromkeys(CAPS, 0)
    for a, b in zip(rows[::2], rows[1::2], strict=True):
        require(a["kind"] == "neural_attempt" and b["kind"] == "neural_return"
                and a["resource"] == b["resource"] and a["context"] == b["context"]
                and a["resource"] in CAPS, "neural pair identity")
        counts[a["resource"]] += 1
    require(counts == attempts == returns and all(counts[k] <= CAPS[k] for k in CAPS), "neural accounting")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory, report_sha):
    directory = Path(directory).resolve(strict=True)
    report_path = directory / "report.json"
    require(sha(report_path) == report_sha, "report hash")
    report = json.loads(report_path.read_bytes())
    binding = json.loads((directory / "activation_binding.json").read_bytes())
    activation_path = Path(binding["path"])
    require(sha(activation_path) == binding["sha256"] == report["activation_sha256"], "activation hash")
    activation = json.loads(activation_path.read_bytes())
    require(binding["sha256"] == report["plan_sha256"], "plan hash")
    require(all(sha(Path(path)) == digest for path, digest in activation["bindings"].items()),
            "verification source hash")
    require(activation == spec.contract(), "contract/source closure")
    plan = activation
    require(report["status"] == "recorded_development" and not report["errors"]
            and report["model_before"] == report["model_after"] and not report["pending"]
            and not report["secondary_errors"], "run completion")
    journal = directory / "journal.jsonl"
    require(sha(journal) == report["journal_sha256"], "journal hash")
    events = analysis.journal(journal.read_bytes(), report["journal_chain_tail"])
    require(events[0]["kind"] == "run_started" and events[-1]["kind"] == "run_finished"
            and events[-1]["counts"] == report["counts"], "journal boundary")
    physical_counts = Counter()
    pending = {}
    for event in events:
        kind = event["kind"]
        if kind == "operation_attempted":
            ident = event["operation_id"]
            require(ident not in pending, "duplicate physical attempt")
            pending[ident] = event
            physical_counts[(event["resource"], "attempted")] += event["units"]
        elif kind == "operation_completed":
            ident = event["operation_id"]
            require(ident in pending and all(event[k] == pending[ident][k]
                    for k in ("resource", "units", "region", "episode_id")), "physical pair")
            physical_counts[(event["resource"], "completed")] += event["units"]
            del pending[ident]
    require(not pending and all(physical_counts[(key, phase)] == val
            for key, count in report["counts"].items() for phase, val in count.items()), "physical count")
    expected = [(r, p) for r in spec.ROLES for p in spec.POLICIES]
    require([(r["role"], r["policy"]) for r in report["rows"]] == expected, "episode roster")
    summaries = []
    for row in report["rows"]:
        role, policy = row["role"], row["policy"]
        ident = role + "_" + policy
        require(row["episode_id"] == ident and json.loads((directory / (ident + "_row.json")).read_bytes()) == row,
                "row identity")
        n = row["control_returned"]
        check_order(events, ident, n)
        check_neural(events, ident, row["neural"]["attempts"], row["neural"]["returns"])
        require(row["neural"]["attempts"] == expected_neural(n)
                and all(row["neural"]["module_hooks"][k] == row["neural"]["attempts"][k]
                        for k in ("predict", "action_encoder")), "boundary/hook neural counts")
        sent = [e for e in events if e["kind"] == "branch_dispatch" and e.get("role") == ident]
        returned = [e for e in events if e["kind"] == "branch_returned" and e.get("role") == ident]
        require(len(row["executed_raw"]) == len(row["transitions"]) == n, "returned row counts")
        for i, (d, ret, transition) in enumerate(zip(sent, returned, row["transitions"], strict=True)):
            require(d["command"] == row["executed_raw"][i] == transition["command"]
                    and all(ret[k] == transition[k] for k in
                            ("state", "reward", "terminated", "truncated", "n_contacts", "metrics")), "row transition")
            metrics = analysis.native_metrics(transition["state"], row["observations"][0]["goal"])
            require(all(metrics[k] == transition["metrics"][k] for k in
                        ("native_success", "native_distance", "native_angle_term", "negative_native_angle",
                         "wrapped_allxy_success", "wrapped_object_angle_error", "object_position_error")), "metrics")
        observations = row["observations"]
        check_frame_count(n, len(observations), row["status"], row.get("censor_step"))
        for i, point in enumerate(observations):
            path = Path(point["path"])
            require(point["step"] == i and path.parent == directory and not path.is_symlink()
                    and sha(path) == point["file_sha256"], "frame seal")
            with Image.open(path) as image:
                pixels = np.asarray(image)
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == point["pixel_sha256"], "RGB hash")
            if i:
                require(point["state"] == row["transitions"][i-1]["state"], "frame transition state")
        candidates = plan["candidates"][role]
        prefix = candidates["prefix"]
        require(row["executed_raw"][:min(n, 5)] == prefix[:min(n, 5)], "prefix bytes")
        if len(observations) > 5:
            require(all(observations[5][k] == candidates["required_cp5"][k] for k in spec.IDENTITY), "cp5 identity")
        decisions = row["decisions"]
        require([d["boundary"] for d in decisions] == [b for b in (5, 10) if n > b], "decision list")
        choice5 = None
        for decision in decisions:
            boundary = decision["boundary"]
            require(json.loads((directory / (ident + f"_cp{boundary}_decision.json")).read_bytes()) == decision,
                    "decision file")
            journal_decision = [e for e in events if e["kind"] == "decision_sealed"
                                and e.get("role") == ident and e["boundary"] == boundary]
            require(len(journal_decision) == 1 and all(journal_decision[0][k] == v for k, v in decision.items())
                    and decision["pixel_sha256"] == observations[boundary]["pixel_sha256"], "decision journal")
            if boundary == 5:
                choice5 = spec.select_cp5(policy, decision["scores"]["q1"], decision["scores"]["q2"])
                choice, packet = choice5, candidates[choice5][0]
            else:
                choice = spec.select_cp10(policy, decision["scores"])
                packet = candidates[choice][1]
            require(decision["choice"] == choice and decision["packet_sha256"] == spec.digest_actions(packet)
                    and row["executed_raw"][boundary:min(n, boundary+5)] ==
                    packet[:min(n-boundary, 5)], "sealed choice/action")
        summaries.append({"role": role, "policy": policy, "status": row["status"],
                          "returned_steps": n, "unobserved_steps": unobserved(15, n),
                          "decisions": decisions, "final_metrics": row["final_metrics"]})
    for role in spec.ROLES:
        pair = [r for r in report["rows"] if r["role"] == role]
        if all(len(r["observations"]) > 10 for r in pair):
            check_matched_cp10(*pair)
    return {"status": "verified_saved_no_neural_or_physics_replay", "report_sha256": report_sha,
            "rows": summaries, "physical_counts": report["counts"], "new_neural_calls": 0,
            "new_physical_calls": 0, "independent_confirmation": False, "P1_admission": False}


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
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
