"""Saved verification of actual physical lineage and fixed-H3 selections."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_fixed_h3_40_20261101_codex as ctrl
from . import p1_six_feedback_saved_20261101_main as shared

require, sha, spec = ctrl.require, ctrl.sha, ctrl.spec


def run(directory, report_sha, output):
    directory, output = Path(directory).resolve(), Path(output)
    rp, bp = directory / "report.json", directory / "binding.json"
    require(sha(rp) == report_sha, "report SHA")
    report, binding = json.loads(rp.read_bytes()), json.loads(bp.read_bytes())
    require(sha(bp) == report["binding_sha256"] and sha(report["prepared"]) == report["prepare_sha256"]
            and binding == json.loads(Path(report["prepared"]).read_bytes()), "bound preparation")
    require(report["error"] is None and report["model_before"] is not None
            and report["model_before"] == report["model_after"] == binding["expected_model_hash"],
            "frozen model and no global error")
    require([r["prefix"] for r in report["rows"]] == list(ctrl.fb.PREFIXES)
            and [r["prefix"] for r in binding["rows"]] == list(ctrl.fb.PREFIXES), "six branches")
    sources = binding["sources"].copy()
    ctrl.fb.check_sources(sources)
    _, runtime = ctrl.own.configure()
    goal, _, _ = ctrl.fb.common.target.collect()
    aggregate = dict.fromkeys(spec.AXES, 0)
    summary = []
    for row, original in zip(report["rows"], binding["rows"], strict=True):
        require(row.get("error") is None and row["status"] in
                ("completed40", "native_stopped", "canvas_censored"), "valid terminal branch")
        raw = np.asarray(row["executed_raw"], np.float32)
        require(raw.shape == (row["steps"], 2) and np.isfinite(raw).all() and (np.abs(raw) <= 1).all(),
                "actual bounded actions")
        require(np.array_equal(raw[:10], np.asarray(original["first10"], np.float32)[:len(raw[:10])]),
                "sealed original first ten")
        points = {p["step"]: p for p in row["checkpoints"]}
        require(len(points) == len(row["checkpoints"]) and row["steps"] in points, "unique checkpoints")
        for frame in row["checkpoints"]:
            require(sha(frame["png"]) == frame["file_sha256"], "checkpoint file")
            with Image.open(frame["png"]) as image:
                pixels = np.asarray(image)
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"], "checkpoint RGB")
            native = ctrl.fb.common.metrics.qualification.native(goal["goal_state"], frame["state"])
            metrics = dict(**ctrl.fb.common.metrics.metrics(frame["state"], goal["goal_state"]),
                           joint_xy=native["joint_xy_error"], native_success=native["initial_success"])
            require(metrics == frame["metrics"], "physical metrics")
            sources[frame["png"]] = frame["file_sha256"]
        for old in original["checkpoints"]:
            if old["step"] in points:
                ctrl.fb.check_checkpoint(points[old["step"]], old)
        require([n["physical_boundary"] for n in row["nodes"]] ==
                list(ctrl.BOUNDARIES[:len(row["nodes"])]), "actual node chronology")
        previous = ctrl.carry.BoundPlan.seal(row["prefix"], 5, np.asarray(original["previous_cp5_plan"], np.float32))
        history = original["cp5_task"]["root_context"]["latent_history"]
        for node in row["nodes"]:
            b = node["physical_boundary"]
            require(node["logical_optimizer_boundary"] == 5 and b in points
                    and node["current_pixel_sha256"] == points[b]["pixel_sha256"]
                    and node["previous_original_plan"] == previous.plan.tolist(), "actual previous plan/current RGB")
            extended, linked = ctrl.extended_carry(previous, episode=row["prefix"], boundary=b,
                issued=raw[b-5:b], pixel_sha=points[b]["pixel_sha256"])
            require(node["hold_extended_previous"] == extended.tolist() and node["feedback"] == linked
                    and node["carry_kind"] == binding["carry"], "hold extension provenance")
            task = ctrl.task_for(original["cp5_task"], history[-2:] + [node["latent"]],
                                 raw[b-10:b], points[b]["pixel_sha256"], extended, b, runtime)
            require(task == node["task"], "actual history")
            policy, contract, config, _ = ctrl.contracts(task, binding["model_sha256"])
            require(node["policy"] == policy and node["config"] == config, "fixed H3 configuration")
            placeholder = ctrl.task_for(original["cp5_task"], history[-2:] + [[0.]*192],
                                raw[b-10:b], points[b]["pixel_sha256"], extended, b, runtime)
            _, _, _, encoding = ctrl.contracts(placeholder, binding["model_sha256"])
            for kind, c, suffix in (("node", contract, "node"), ("encoding", encoding, "encode")):
                ledger = node[kind + "_ledger"]
                path = Path(ledger["journal"])
                require(path.resolve().is_relative_to(directory), "ledger location")
                account = ctrl.own.consumer.consume(path, ledger["sha256"], spec.digest(c),
                            "structured", "node", f"{row['prefix']}_b{b}_{suffix}")
                cost = ctrl.h4.h4.node_cost("structured", 5) if kind == "node" else dict.fromkeys(spec.AXES, 0)
                if kind == "encoding":
                    cost.update(encode_calls=1, encode_samples=1)
                require(account == node[kind + "_account"] and account["status"] == "closed_unverified"
                        and account["charged_upper_bounds"] == cost, "measured dispatch")
                for key in aggregate:
                    aggregate[key] += cost[key]
                sources[str(path)] = ledger["sha256"]
            # The shared optimizer uses logical boundary5/H3; physical chronology
            # was independently checked above and is never replaced by this adapter.
            chosen = shared.selected_from_trace(dict(node, boundary=5, previous_plan=extended.tolist()), directory)
            previous = ctrl.carry.BoundPlan.seal(row["prefix"], b, chosen)
            require(previous.plan_sha256 == node["selected_plan_sha256"], "selected bytes")
            issued = raw[b:b+5]
            require(np.array_equal(issued, chosen[0, :len(issued)]), "executed chosen packet")
            history = history[-2:] + [node["latent"]]
            sources[node["trace"]] = node["trace_sha256"]
        if row["status"] == "completed40":
            require(row["steps"] == 40 and len(row["nodes"]) == 6 and sorted(points) == list(range(0, 41, 5)),
                    "complete40 chronology")
        else:
            stop = row["stop_reason"]
            require(stop["step"] == row["steps"]
                    and stop["native_success"] == points[row["steps"]]["metrics"]["native_success"],
                    "observed stop")
        summary.append(dict(prefix=row["prefix"], status=row["status"], steps=row["steps"],
                            cp20=points[20]["metrics"] if 20 in points else None,
                            cp40=points[40]["metrics"] if 40 in points else None,
                            last_observed=points[row["steps"]]["metrics"],
                            selected_indices=[n["selected_index"] for n in row["nodes"]]))
    for path in (rp, bp, Path(shared.__file__).resolve(), Path(__file__).resolve()):
        sources[str(path)] = sha(path)
    ctrl.fb.check_sources(sources)
    require(output.is_absolute() and output.parent == ctrl.ROOT / "results" and not output.exists(),
            "exclusive output")
    ctrl.own.write(output, dict(rows=summary, sources=sources, actual_dispatch_charges=aggregate,
        verified=True, neural_calls=0, P1_admission=False, independent_confirmation=False,
        limitations="known-root new40-step controller, not matched-cost20 comparison or independent kernel verification"))
    print(json.dumps(dict(rows=summary, actual_dispatch_charges=aggregate, sha256=sha(output))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.directory, args.sha256, args.output)
