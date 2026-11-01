"""Read-only verification of a completed 40-to-50 fixed-H3 run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_fixed_h3_50_20261101_codex as ctrl
from . import p1_six_feedback_saved_20261101_main as shared

require, sha, spec = ctrl.require, ctrl.sha, ctrl.spec


def verify_inherited(row, original):
    require(not original["eligible"] and row["status"] ==
            ("inherited_censored" if original["old_status"] == "canvas_censored" else "inherited_native_stopped")
            and row["old_status"] == original["old_status"] and row["steps"] == original["old_steps"]
            and row["stop_reason"] == original["old_stop_reason"]
            and row["old_final_checkpoint"] == original["old_final_checkpoint"]
            and row["replayed"] is False and not row["executed_raw"] and not row["checkpoints"]
            and not row["nodes"] and not row["node_ledgers"] and not row["encodings"]
            and row["physical_replay_steps"] == row["physical_continuation_steps"] == row["warmup_steps"] == 0
            and row["observed_cp40"] is None and row["observed_cp50"] is None,
            "inherited terminal branch")


def check_points(row, directory, goal):
    points = {p["step"]: p for p in row["checkpoints"]}
    require(len(points) == len(row["checkpoints"]) and row["steps"] in points, "checkpoint uniqueness")
    for frame in row["checkpoints"]:
        image_path = Path(frame["png"])
        require(image_path.resolve().is_relative_to(directory) and not image_path.is_symlink()
                and sha(image_path) == frame["file_sha256"], "actual PNG file identity")
        with Image.open(image_path) as image:
            pixels = np.asarray(image).copy()
        require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
                and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"], "actual RGB identity")
        native = ctrl.fb.common.metrics.qualification.native(goal, frame["state"])
        metrics = dict(**ctrl.fb.common.metrics.metrics(frame["state"], goal),
                       joint_xy=native["joint_xy_error"], native_success=native["initial_success"])
        require(metrics == frame["metrics"], "checkpoint physical metrics")
    require(row["observed_cp40"] == points.get(40) and row["observed_cp50"] == points.get(50),
            "separate endpoint identity")
    return points


def verify(directory, expected_sha):
    directory = Path(directory).resolve()
    report_path, binding_path = directory / "report.json", directory / "binding.json"
    require(spec.is_hash(expected_sha) and sha(report_path) == expected_sha, "report SHA")
    report = json.loads(report_path.read_bytes())
    require(sha(binding_path) == report["binding_sha256"]
            and sha(report["prepared"]) == report["prepare_sha256"], "binding and prepare SHA")
    binding = json.loads(binding_path.read_bytes())
    require(binding == json.loads(Path(report["prepared"]).read_bytes())
            and binding == ctrl.build(), "source closure and exact prepared input")
    require(report["schema"] == ctrl.NAME + ".report.v1" and report["error"] is None
            and report["model_before"] is not None
            and report["model_before"] == report["model_after"] == binding["expected_model_hash"]
            and report["P1_admission"] is False and report["independent_confirmation"] is False,
            "complete frozen-model run")
    require(tuple(r["prefix"] for r in report["rows"]) == ctrl.fb.PREFIXES, "six fixed rows")
    require(ctrl.study_summary(report["rows"], report["error"]) ==
            {k: report[k] for k in ("status", "all_survivors_full_horizon", "branch_status_counts")}
            and report["status"] == "complete", "valid terminal report status")
    require(report["expected_new_nodes"] == report["expected_new_encodes"] == 10
            and report["node_cost"] == binding["node_cost"]
            and report["encode_cost"] == binding["encode_cost"]
            and report["historical_dispatch_charges"] == binding["historical_dispatch_charges"],
            "expected and historical cost binding")
    _, runtime = ctrl.own.configure()
    goal, _, _ = ctrl.fb.common.target.collect()
    aggregate_node, aggregate_enc = dict.fromkeys(spec.AXES, 0), dict.fromkeys(spec.AXES, 0)
    summary = []
    for row, original in zip(report["rows"], binding["rows"], strict=True):
        require(row["prefix"] == original["prefix"] and row.get("error") is None, "branch identity/status")
        if not original["eligible"]:
            verify_inherited(row, original)
            summary.append(dict(prefix=row["prefix"], status=row["status"], steps=row["steps"],
                                new_nodes=0, cp40=None, cp50=None))
            continue
        require(row["status"] in ("completed50", "native_stopped", "canvas_censored"),
                "terminal continued branch")
        raw = np.asarray(row["executed_raw"], np.float32)
        require(raw.shape == (row["steps"], 2) and 41 <= row["steps"] <= 50
                and np.isfinite(raw).all() and (np.abs(raw) <= 1).all()
                and np.array_equal(raw[:40], np.asarray(original["old_actions"], np.float32))
                and row["physical_replay_steps"] == 40
                and row["physical_continuation_steps"] == row["steps"] - 40
                and row["warmup_steps"] == 10, "exact replay and returned actions")
        points = check_points(row, directory, goal["goal_state"])
        require(all(step in points for step in range(0, 41, 5)), "all replay checkpoints")
        for old in original["old_checkpoints"]:
            ctrl.fb.check_checkpoint(points[old["step"]], old)
        require([n["physical_boundary"] for n in row["nodes"]] ==
                list(ctrl.BOUNDARIES[:len(row["nodes"])])
                and len(row["nodes"]) == 1 + int(row["steps"] > 45)
                and len(row["nodes"]) == len(row["node_ledgers"]) == len(row["encodings"]),
                "node and ledger chronology")
        previous = ctrl.carry.BoundPlan.seal(row["prefix"], 35, np.asarray(original["previous_plan"], np.float32))
        require(previous.plan_sha256 == original["previous_plan_sha256"], "prior selected plan")
        latents = original["prior_latents"]
        for index, node in enumerate(row["nodes"]):
            boundary = node["physical_boundary"]
            require(node["logical_optimizer_boundary"] == 5 and boundary in points
                    and node["current_pixel_sha256"] == points[boundary]["pixel_sha256"]
                    and node["previous_original_plan"] == previous.plan.tolist(), "physical lineage")
            extended, linked = ctrl.extended_carry(previous, episode=row["prefix"], boundary=boundary,
                                                    issued=raw[boundary-5:boundary],
                                                    pixel_sha=points[boundary]["pixel_sha256"])
            require(node["hold_extended_previous"] == extended.tolist() and node["feedback"] == linked
                    and node["carry_kind"] == binding["carry"], "hold-extended carry")
            task = ctrl.make_task(original["cp5_task"], latents[-2:] + [node["latent"]],
                                  raw[boundary-10:boundary], points[boundary]["pixel_sha256"],
                                  extended, boundary, runtime)
            require(task == node["task"], "rolling actual context")
            policy, contract, config, _ = ctrl.prior.contracts(task, binding["model_sha256"])
            require(node["policy"] == policy and node["config"] == config, "H3 policy/config")
            placeholder = ctrl.make_task(original["cp5_task"], latents[-2:] + [[0.] * 192],
                                         raw[boundary-10:boundary], points[boundary]["pixel_sha256"],
                                         extended, boundary, runtime)
            _, _, _, encoding = ctrl.prior.contracts(placeholder, binding["model_sha256"])
            for kind, con, suffix, aggregate, entries in (
                ("node", contract, "node", aggregate_node, row["node_ledgers"]),
                ("encoding", encoding, "encode", aggregate_enc, row["encodings"])):
                entry = entries[index]
                ledger, declared = node[kind + "_ledger"], node[kind + "_account"]
                require(entry["physical_boundary"] == boundary and entry["ledger"] == ledger
                        and entry["account"] == declared, "node/ledger receipt binding")
                journal = Path(ledger["journal"])
                require(journal.resolve().is_relative_to(directory), "durable ledger path")
                account = ctrl.own.consumer.consume(journal, ledger["sha256"], spec.digest(con),
                              "structured", "node", f"{row['prefix']}_b{boundary}_{suffix}")
                expected = binding["node_cost"] if kind == "node" else dict.fromkeys(spec.AXES, 0)
                if kind == "encoding":
                    expected.update(encode_calls=1, encode_samples=1)
                require(account == declared and account["status"] == "closed_unverified"
                        and account["charged_upper_bounds"] == expected, "measured dispatch")
                for key in spec.AXES:
                    aggregate[key] += account["charged_upper_bounds"][key]
            chosen = shared.selected_from_trace(dict(node, boundary=5, previous_plan=extended.tolist()), directory)
            previous = ctrl.carry.BoundPlan.seal(row["prefix"], boundary, chosen)
            require(previous.plan_sha256 == node["selected_plan_sha256"]
                    and np.array_equal(raw[boundary:boundary+5], chosen[0, :len(raw[boundary:boundary+5])]),
                    "actual selected returned packet")
            latents = latents[-2:] + [node["latent"]]
        if row["status"] == "completed50":
            require(row["steps"] == 50 and len(row["nodes"]) == 2
                    and sorted(points) == list(range(0, 51, 5)), "completed50 chronology")
        else:
            stop = row["stop_reason"]
            require(stop["step"] == row["steps"] and stop["status"] == row["status"]
                    and stop["native_success"] == points[row["steps"]]["metrics"]["native_success"]
                    and (stop["terminated"] or stop["truncated"] or stop["outside"]), "observed stop")
        summary.append(dict(prefix=row["prefix"], status=row["status"], steps=row["steps"],
                            new_nodes=len(row["nodes"]),
                            cp40=points[40]["metrics"], cp50=points[50]["metrics"] if 50 in points else None))
    costs = dict(planner=aggregate_node, rgb_encoding=aggregate_enc)
    require(costs == report["actual_new_dispatch_charges"]
            and report["measured_new_nodes"] == sum(len(r["nodes"]) for r in report["rows"])
            and report["recorded_node_ledgers"] == sum(len(r["node_ledgers"]) for r in report["rows"])
            and report["recorded_encoding_ledgers"] == sum(len(r["encodings"]) for r in report["rows"])
            and report["measured_new_encodes"] == aggregate_enc["encode_calls"]
            and all(report[k] == sum(r[k.removeprefix("physical_")] if k == "physical_warmup_steps"
                                    else r[k] for r in report["rows"])
                    for k in ("physical_replay_steps", "physical_continuation_steps", "physical_warmup_steps")),
            "separate actual totals")
    ctrl.fb.check_sources(binding["sources"])
    return dict(verified=True, report_sha256=expected_sha, rows=summary,
                actual_new_dispatch_charges=costs, neural_calls=0, physical_execution=False,
                P1_admission=False, independent_confirmation=False,
                scope="saved accounting and lineage; not independent model replay or kernel instrumentation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory, args.report_sha256), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
