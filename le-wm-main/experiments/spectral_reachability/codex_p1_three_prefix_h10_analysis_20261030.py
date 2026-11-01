"""Analyze verified saved H10 three-prefix probes, without selecting actions."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from . import codex_p1_three_prefix_h10_runner_20261030 as runner
from . import codex_p1_three_prefix_h10_saved_20261030 as saved

NAME = "codex_p1_three_prefix_h10_analysis_20261030"
ROLES = ("selected", "zero", "greedy")
COMPARISONS = (("selected", "zero"), ("selected", "greedy"), ("greedy", "zero"))
STATUSES = ("probe_completed", "native_stopped", "censored_out_of_domain")
need = runner.require


def vector(value):
    need(isinstance(value, list) and len(value) == 7 and
         all(type(x) in (float, int) and math.isfinite(x) for x in value), "finite native7 state")
    return value


def wrapped(x):
    return math.atan2(math.sin(x), math.cos(x))


def measure(state, start, goal):
    if state is None:
        return None
    state, start, goal = vector(state), vector(start), vector(goal)
    dx, dy = state[2]-start[2], state[3]-start[3]
    return dict(object_displacement_xy=[dx, dy],
                object_displacement_norm=math.hypot(dx, dy),
                object_angle_displacement=wrapped(state[4]-start[4]),
                object_position_error=math.hypot(state[2]-goal[2], state[3]-goal[3]),
                object_angle_error=abs(wrapped(state[4]-goal[4])))


def difference(a, b):
    if a is None or b is None:
        return None
    return dict(object_displacement_xy=[a["object_displacement_xy"][i]-b["object_displacement_xy"][i]
                                        for i in range(2)],
                object_angle_displacement=wrapped(a["object_angle_displacement"]-
                                                  b["object_angle_displacement"]),
                object_position_error=a["object_position_error"]-b["object_position_error"],
                object_angle_error=a["object_angle_error"]-b["object_angle_error"])


def relative_zero(actual, zero):
    if actual is None or zero is None:
        return None
    return dict(object_response_xy=[actual["object_displacement_xy"][i]-zero["object_displacement_xy"][i]
                                    for i in range(2)],
                object_response_angle=wrapped(actual["object_angle_displacement"]-
                                              zero["object_angle_displacement"]),
                position_error_improvement=zero["object_position_error"]-actual["object_position_error"],
                angle_error_improvement=zero["object_angle_error"]-actual["object_angle_error"])


def relative_difference(a, b):
    if a is None or b is None:
        return None
    return dict(object_response_xy=[a["object_response_xy"][i]-b["object_response_xy"][i] for i in range(2)],
                object_response_angle=wrapped(a["object_response_angle"]-b["object_response_angle"]),
                position_error_improvement=a["position_error_improvement"]-b["position_error_improvement"],
                angle_error_improvement=a["angle_error_improvement"]-b["angle_error_improvement"])


def analyze(original, verified, details=None):
    anchors, banks, rows = original["anchors"], original["probe_banks"], verified["rows"]
    need(original["fixed_pairs"] == 8 and original["fixed_branches"] == verified["fixed_branches"] == 1008
         and verified["saved_checks_passed"] is True and verified["P1_admission"] is False
         and verified["independent_confirmation"] is False and len(anchors) == len(rows) == 24
         and len(banks) == 8, "verified fixed denominator")
    counts = {s: 0 for s in STATUSES}
    if details is not None:
        need(len(details) == 24 and all(len(shard) == 42 for shard in details), "detail denominator")
    groups, aliases = [], {}
    for i in range(8):
        triple, checked, bank = anchors[3*i:3*i+3], rows[3*i:3*i+3], banks[i]
        need(len(bank) == 42 and [p["index"] for p in bank] == list(range(42)), "probe roster")
        need([a["index"] for a in triple] == [3*i, 3*i+1, 3*i+2]
             and [a["role"] for a in triple] == list(ROLES)
             and all(a["logical_pair"] == i for a in triple)
             and [r["anchor"] for r in checked] == [3*i, 3*i+1, 3*i+2], "anchor roster")
        a0 = triple[0]
        for a in triple:
            need(all(a[k] == a0[k] for k in ("task_id", "root_slot", "condition", "arm", "goal",
                                                "physical_root_sha256")), "same task")
            vector(a["goal"])
            vector(a["checkpoint"]["state"])
        for row in checked:
            need(len(row["statuses"]) == len(row["endpoints"]) == 42, "endpoint roster")
            for status, endpoint in zip(row["statuses"], row["endpoints"], strict=True):
                need(status in STATUSES and ((status == "probe_completed") == (endpoint is not None)),
                     "endpoint/status")
                if endpoint is not None:
                    vector(endpoint)
                counts[status] += 1
        need(bytes.fromhex(bank[0]["raw10_float32_hex"]) == bytes(80), "zero suffix")
        for probe in bank:
            raw = bytes.fromhex(probe["raw10_float32_hex"])
            need(len(raw) == 80 and hashlib.sha256(raw).hexdigest() == probe["raw10_sha256"],
                 "probe bytes")
        measurements = {
            role: [measure(endpoint, a["checkpoint"]["state"], a["goal"]) for endpoint in row["endpoints"]]
            for role, a, row in zip(ROLES, triple, checked, strict=True)
        }
        records = []
        for j, probe in enumerate(bank):
            metrics = {role: measurements[role][j] for role in ROLES}
            responses = {role: relative_zero(metrics[role], measurements[role][0]) for role in ROLES}
            role_records = {}
            for k, role in enumerate(ROLES):
                item = dict(status=checked[k]["statuses"][j], metrics=metrics[role],
                            response_vs_own_zero=responses[role])
                if details is not None:
                    detail = details[3*i+k][j]
                    need(detail["status"] == item["status"], "detail status")
                    item["observation"] = detail
                role_records[role] = item
            records.append(dict(index=j, probe=probe["name"], raw10_sha256=probe["raw10_sha256"],
                                roles=role_records,
                                comparisons={f"{a}_minus_{b}": dict(
                                    endpoint_metrics=difference(metrics[a], metrics[b]),
                                    response_vs_own_zero=relative_difference(responses[a], responses[b]))
                                    for a, b in COMPARISONS}))
        material = [a0["task_id"], a0["physical_root_sha256"], a0["condition"], a0["goal"],
                    [a["prefix5_float32_hex"] for a in triple],
                    [p["raw10_float32_hex"] for p in bank]]
        key = hashlib.sha256(json.dumps(material, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        groups.append(dict(logical_pair=i, root_slot=a0["root_slot"], condition=a0["condition"],
                           arm=a0["arm"], task_id=a0["task_id"], physical_root_sha256=a0["physical_root_sha256"],
                           duplicate_group=aliases.setdefault(key, len(aliases)), duplicate_key_sha256=key,
                           prefix_duplicate_groups={r: a["duplicate_prefix_task_group"]
                                                    for r, a in zip(ROLES, triple, strict=True)},
                           anchors={r: a["index"] for r, a in zip(ROLES, triple, strict=True)},
                           probes=records))
    need(counts == verified["counts"] and sum(counts.values()) == 1008, "status census")
    return dict(schema=NAME+".v1", fixed_groups=8, probes_per_group=42, fixed_branches=1008,
                status_counts=counts, groups=groups, P1_admission=False, freshness=False,
                independent_confirmation=False, equal_budget_certified=False,
                endpoints_policy="native stop and out-of-domain censor remain missing even at step 10",
                limitations="Known roots; post-outcome exploration; duplicate groups are not independent samples.")


def consume(directory, report_sha256, prepared, prepare_sha256):
    code = runner.runner.physical.sha(Path(__file__))
    verified = saved.consume(directory, report_sha256, prepared, prepare_sha256)
    prepared_value = runner.verify_prepare(prepared, prepare_sha256)
    root = Path(directory)
    run_report = runner.native.read(root / "run_report.json", cap=4*1024**2)
    need(len(run_report["shards"]) == 24, "shard detail denominator")
    details = []
    for i, shard in enumerate(run_report["shards"]):
        path = root / f"anchor_{i:02d}" / "report.json"
        need(shard["anchor"] == i and runner.runner.physical.sha(path) == shard["report_sha256"],
             "detail report digest")
        report = runner.native.read(path, cap=64*1024**2)
        need(len(report["rows"]) == 42, "detail row denominator")
        shard_details = []
        for j, row in enumerate(report["rows"]):
            need(row["index"] == j and row["status"] == verified["rows"][i]["statuses"][j],
                 "detail row identity")
            responses = row["responses"]
            last = responses[-1] if responses else None
            shard_details.append(dict(status=row["status"], accepted_probe_steps=len(responses),
                                      native_stop=None if row["status"] != "native_stopped" else
                                      dict(step=last["step"], terminated=last["terminated"],
                                           truncated=last["truncated"]),
                                      censor=row.get("censor")))
        need(runner.runner.physical.sha(path) == shard["report_sha256"], "detail report drift")
        details.append(shard_details)
    result = analyze(prepared_value["original"], verified, details)
    need(runner.runner.physical.sha(Path(__file__)) == code and
         runner.runner.physical.sha(Path(directory) / "run_report.json") == report_sha256,
         "post-analysis drift")
    result.update(run_report_sha256=report_sha256, prepare_sha256=prepare_sha256, analysis_code_sha256=code)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", required=True)
    ap.add_argument("--report-sha256", required=True)
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--prepare-sha256", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    output = Path(a.output)
    need(output.is_absolute() and output.parent == runner.ROOT / "results" and
         output.name.startswith(NAME+"_") and not output.is_symlink(), "exclusive output namespace")
    result = consume(a.directory, a.report_sha256, a.prepared, a.prepare_sha256)
    raw = runner.native.raw(result)
    with output.open("xb") as stream:
        need(stream.write(raw) == len(raw), "output write")
        stream.flush()
        os.fsync(stream.fileno())
    runner.base.paired.fsync_dir(output.parent)
    print(json.dumps(dict(status="saved_analysis_exploratory_only", sha256=runner.runner.physical.sha(output),
                          fixed_branches=1008, P1_admission=False)))


if __name__ == "__main__":
    main()
