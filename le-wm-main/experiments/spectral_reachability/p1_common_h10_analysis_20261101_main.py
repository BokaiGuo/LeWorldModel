"""Saved paired physical summaries, without filling stopped endpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("selected", "zero", "greedy")


def metrics(state, goal):
    return dict(pusher_xy=math.dist(state[:2], goal[:2]),
                object_xy=math.dist(state[2:4], goal[2:4]),
                object_angle=abs((state[4] - goal[4] + math.pi) % (2 * math.pi) - math.pi))


def delta(after, before):
    return {key: after[key] - before[key] for key in before}


def analyze():
    output, bindings, prefix_identity = [], {}, []

    def read(path):
        blob = path.read_bytes()
        bindings[str(path)] = hashlib.sha256(blob).hexdigest()
        return json.loads(blob)

    for root in (0, 1):
        reference_bank = None
        reference_scores = None
        selected_prefix = None
        directory = ROOT / f"results/p1_common_h10_20261101_main_run_root{root}_MAIN01"
        report, plan = read(directory / "report.json"), read(directory / "prepared.json")
        if report["status"] != "recorded_not_independently_consumed" or len(report["rows"]) != 27:
            raise ValueError("incomplete intervention")
        for role in ROLES:
            own = read(ROOT / f"results/p1_three_prefix_own_host_20261101_main_root{root}_{role}_RUN_MAIN01/report.json")
            row = own["rows"][0]
            node = next(n for n in own["nodes"][row["episode_id"]] if n["boundary"] == 0)
            trace_path = Path(node["receipt"]["trace"])
            blob = trace_path.read_bytes()
            trace_sha = hashlib.sha256(blob).hexdigest()
            if trace_sha != node["receipt"]["trace_sha256"]:
                raise ValueError("source trace")
            bindings[str(trace_path)] = trace_sha
            payload = next(json.loads(line)["payload"] for line in blob.splitlines()
                           if json.loads(line)["kind"] == "prefix_intervention")
            bank, scores = payload["bank"], payload["stage_scores"]
            if reference_bank is None:
                reference_bank, reference_scores = bank, scores
            if bank != reference_bank or scores != reference_scores or payload["role"] != role:
                raise ValueError("same prefix candidate bank and scores")
            if role == "zero":
                expected = [[0.0, 0.0]] * 5
            elif role == "greedy":
                expected = bank[min(range(8), key=lambda i: (scores[i], i))][0]
            else:
                expected = payload["full_selected"][0]
            if row["executed_raw"][:5] != expected:
                raise ValueError("prefix selection rule")
            if role == "selected":
                selected_prefix = row["executed_raw"][:5]
            if role == "greedy":
                prefix_identity.append(dict(root=root, selected_equals_greedy=
                                            selected_prefix == row["executed_raw"][:5]))
            cp = {c["step"]: c for c in row["checkpoints"]}
            goal = cp[0]["goal"]
            initial, prefix = metrics(cp[0]["state"], goal), metrics(cp[5]["state"], goal)
            branches = [r for r in report["rows"] if r["prefix_role"] == role]
            zero = next(r for r in branches if r["suffix_role"] == "zero")
            zero_end = metrics(zero["endpoint"]["state"], goal) if zero["endpoint"] else None
            for branch in branches:
                specification = plan["schedule"][branch["branch"]]
                if not branch["prefix_verified"] or branch["executed_raw"] != specification["commands"][:branch["control_returned"]]:
                    raise ValueError("prefix/action binding")
                endpoint = metrics(branch["endpoint"]["state"], goal) if branch["endpoint"] else None
                last = branch["transitions"][-1]
                output.append(dict(root=root, prefix=role, suffix=branch["suffix_role"],
                                   immediate_delta=delta(prefix, initial), status=branch["status"],
                                   steps=branch["control_returned"], terminal_observation=metrics(last["state"], goal),
                                   native_terminated=last["terminated"], native_truncated=last["truncated"],
                                   endpoint15=endpoint,
                                   endpoint_minus_own_zero=delta(endpoint, zero_end) if endpoint and zero_end else None,
                                   own_planner_cp15=metrics(cp[15]["state"], goal) if 15 in cp else None,
                                   own_planner_success=row["native_success"],
                                   own_planner_status=row["status"]))
    return dict(rows=output, bindings=bindings, prefix_identity=prefix_identity, P1_admission=False,
                interpretation="descriptive known-root effects; no imputation or gate modification")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(output=str(args.output), rows=len(result["rows"]), P1_admission=False)))
