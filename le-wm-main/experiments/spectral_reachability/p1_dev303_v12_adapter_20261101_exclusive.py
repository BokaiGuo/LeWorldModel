"""CPU-only preparation and score-selection contract for the dev303 V12 bank.

This module never executes a model, simulator, or physical controller.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_blind_adapter_20261101_codex as source
from . import p1_dev303_blind_planners_20261101_codex as planner
from . import p1_dev302_mixed_prefix_bank_v9_20261101_codex as bank

ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_dev303_v12_adapter_20261101_exclusive"
TASK = ROOT / "results/task_p1_dev303_goal_blind_20261101_codex.json.gz"
TASK_SHA = "3787aba5dfe49c97b9c738e347e228c8089449af0df84a5f91788ceb769b2b12"
BLIND_SHA = "092af72253c9b0463e5c7a5814052599843dc4b6a979bd040d7f318c93dcddd9"
RECEIPT_SHA = "4d7c23525d745278ba29f46676c88dddd1a95aae44c4e46241fb71cdd60af78d"
REPORT_SHA = "cb72e8f00b439bf4e28db6d21445993f63e54d978096ccb7bdcdcec2865b6b71"
PLAN_SHA = {
    "full10_conditional_D2P": "271561e10098fe0b14a72511862279c874f320168c42e86ba4a5644d95684c98",
    "line_search_GN": "a87740ca69e7587d3e4cb20dd92da2510c2695c28f2ba1795a2a8a8b1c14517e",
}
ARMS = ("full10_conditional_D2P", "line_search_GN")
LABELS = ("zero", "spectral", "gn", "rgb_nearest_contact",
          "rgb_goal_support_direct", "rgb_goal_support_top_route",
          "rgb_goal_support_bottom_route", "random_control",
          "rgb_delayed_task_support", "rgb_delayed_support_x12",
          "rgb_delayed_support_y12", "rgb_delayed_support_x23",
          "rgb_delayed_support_y23", "rgb_delayed_x12_pos06",
          "rgb_delayed_x12_neg06", "rgb_delayed_x23_pos06",
          "rgb_delayed_x23_neg04")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_fixed(path, expected):
    path = Path(path).absolute()
    require("outcome" not in str(path).lower() and "physical" not in str(path).lower(),
            "outcome/physical input forbidden")
    require(path.is_file() and not path.is_symlink(), f"regular input: {path}")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, f"SHA mismatch: {path}")
    return raw


def decode(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: require(False, "nonfinite JSON"))


def consume_planner(report_path, report_sha):
    with planner.patched():
        return planner.base.consume(Path(report_path).parent, report_sha)


def prepare(report_path, report_sha):
    """Prepare only after the separate dev303 planner has sealed both plans."""
    report_path = Path(report_path).absolute()
    require(report_sha == REPORT_SHA, "fixed dev303 planner report SHA")
    report = decode(read_fixed(report_path, report_sha))
    consumed = consume_planner(report_path, report_sha)
    require(consumed["status"] == "model_commands_verified_for_separate_MAIN_executor"
            and consumed["new_physics"] == 0 and consumed["P1_admission"] is False,
            "full planner consume")
    require(report["schema"] == "p1_constructive_blind_planners_cuda_20261027_v1.report"
            and report["status"] == "model_plans_sealed_not_admitted"
            and report["commands_released"] is True and report["errors"] == []
            and report["source_verified"] is True and report["model_unchanged"] is True
            and report["all_plans_before_physics"] is True
            and report["new_physics"] == 0 and report["P1_admission"] is False,
            "planner release boundary")
    prep_path = Path(report["prepared"]).absolute()
    prep = decode(read_fixed(prep_path, report["prepare_sha256"]))
    require(prep["schema"] == "p1_dev303_blind_planners_20261101_codex.prepare"
            and len(prep["slots"]) == 1 and prep["slots"][0]["order"] == 0
            and Path(prep["slots"][0]["task_export"]).absolute() == TASK
            and prep["slots"][0]["task_sha256"] == TASK_SHA
            and prep["expected_model_hash"] == report["model_hash_before"]
            == report["model_hash_after"]
            and prep["contract"] == report["contract"], "planner task/model binding")
    export = decode(gzip.decompress(read_fixed(TASK, TASK_SHA)))
    require(export["development_seed"] == 301000003
            and export["blind_source_sha256"] == BLIND_SHA
            and export["P1_admission"] is False
            and export["new_independent_confirmation"] is False, "task identity")
    arrays, receipt, audit = source.source()
    require(receipt == RECEIPT_SHA and audit["seed"] == 301000003
            and audit["P1_admission"] is False, "source receipt")
    plans = {}
    bindings = {str(TASK): TASK_SHA, str(report_path): report_sha,
                str(prep_path): report["prepare_sha256"]}
    rows = report["rows"]
    require(len(rows) == 2 and {(r["order"], r["arm"]) for r in rows}
            == {(0, arm) for arm in ARMS}, "planner roster")
    consumed_plans = {p["arm"]: p for p in consumed["plans"]}
    require(set(consumed_plans) == set(ARMS), "consumed planner roster")
    for arm in ARMS:
        row = next(r for r in rows if r["arm"] == arm)
        name = f"root_0_{arm}.plan.json"
        require(row["status"] == "model_plan_sealed" and row["plan_file"] == name,
                "plan row")
        path = report_path.parent / name
        expected = report["artifact_sha256"][name]
        require(expected == PLAN_SHA[arm], "fixed dev303 plan SHA")
        plan = decode(read_fixed(path, expected))
        require(plan["task_sha256"] == TASK_SHA and plan["arm"] == arm
                and plan["selection_sha256"] == row["selection_sha256"]
                and consumed_plans[arm]["commands"] == plan["commands"]
                and consumed_plans[arm]["selection_sha256"] == plan["selection_sha256"],
                "plan identity")
        commands = np.asarray(plan["commands"], dtype=np.float32)
        require(commands.shape == (10, 2) and np.isfinite(commands).all()
                and (np.abs(commands) <= 1).all(), "bounded 10-step plan")
        plans[arm] = commands
        bindings[str(path)] = expected
    result = bank.build(arrays["rgb_history"][-1], arrays["goal_rgb"],
                        plans[ARMS[0]][:5], plans[ARMS[1]][:5])
    candidates = result["candidates"]
    require(len(candidates) == 17 and [r["label"] for r in candidates] == list(LABELS)
            and [r["index"] for r in candidates] == list(range(17)), "fixed candidate roster")
    for row in candidates:
        actions = np.asarray(row["actions"], dtype="<f4")
        require(actions.shape == (5, 2) and np.isfinite(actions).all()
                and (np.abs(actions) <= 1).all()
                and hashlib.sha256(actions.tobytes()).hexdigest() == row["actions_sha256"],
                "candidate integrity")
    feasible_count = sum(r["contact_feasible_prior"] is True for r in candidates)
    # Bind every bank generation module, and the independent receipt-verification source.
    for module in (bank, bank.v8, bank.v8.v7, bank.v8.v7.v6,
                   bank.v8.v7.v6.v5, bank.v8.v7.v6.geometry,
                   bank.v8.v7.v6.geometry.rgb_source, source, source.core,
                   source.core.saved, planner, planner.base, Path(__file__)):
        path = Path(module if isinstance(module, Path) else module.__file__).resolve()
        bindings[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": NAME + ".prepare.v1", "task": export["task"],
        "source_receipt_sha256": receipt, "blind_source_sha256": BLIND_SHA,
        "candidate_status": "ready" if feasible_count else "abstain_no_feasible_candidate",
        "feasible_candidate_count": feasible_count,
        "pixel_sha256": [hashlib.sha256(x.tobytes()).hexdigest()
                         for x in (arrays["rgb_history"][-1], arrays["goal_rgb"])],
        "prefix_bank": result,
        "tail_seeds": [np.concatenate((plans[a][5:][None], np.zeros((1, 5, 2), np.float32))).tolist()
                       for a in ARMS] + [np.zeros((2, 5, 2), np.float32).tolist()],
        "tail_seed_labels": ["old_d2p_plus_zero", "old_gn_plus_zero", "zero"],
        "algorithm": {"seed": 30220261102, "common_random_numbers": True,
                      "iterations": 5, "samples": 64, "elites": 8,
                      "variants": 5, "tail_packets": 2,
                      "selected": "minimum robust score among feasible; stable index",
                      "greedy": "minimum cp5 latent MSE; stable index"},
        "budget": {"action_encoder_calls": 187, "predict_calls": 187,
                   "predict_samples": 54417, "candidate_evaluations": 17,
                   "wall_seconds": 1800},
        "bindings_sha256": bindings, "policy": prep["policy"],
        "model_sha256": prep["expected_model_hash"],
        "selector_execution_authorized": False, "physical_execution_authorized": False,
        "independent_confirmation": False, "P1_admission": False,
    }


def select(candidates, scores):
    """Pure selection over already available scores; does not generate scores."""
    require(len(candidates) == len(scores) == 17, "complete selector scores")
    require([r["index"] for r in candidates] == list(range(17)), "selector order")
    require(all(np.isfinite(s["robust_score"]) and np.isfinite(s["cp5_q"])
                for s in scores), "finite selector scores")
    feasible = [i for i, r in enumerate(candidates) if r["contact_feasible_prior"] is True]
    require(feasible, "abstain: no feasible candidate")
    return {"selected_index": min(feasible, key=lambda i: (scores[i]["robust_score"], i)),
            "latent_greedy_index": min(range(17), key=lambda i: (scores[i]["cp5_q"], i)),
            "task_greedy_direct_index": 4, "zero_index": 0, "random_index": 7,
            "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-report", required=True)
    parser.add_argument("--planner-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).absolute()
    require(not os.path.lexists(output), "exclusive output")
    value = prepare(args.planner_report, args.planner_report_sha256)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
