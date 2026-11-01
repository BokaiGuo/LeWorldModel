"""Saved-only verifier for dev303 feedback ledger; does not execute model or physics."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from . import p1_dev301_blind_plan_physical_saved_20261101_codex as saved
from . import p1_dev303_feedback_history_20261102_codex_exclusive as history
from . import p1_dev303_rgb_feedback_scored_tail_20261102_codex_exclusive as feedback
from . import p1_dev303_rgb_chunk_diagnostic_20261102_exclusive as prior

base, require = prior.base, prior.require


def verify_branch(directory, role, row, events, source_history):
    arm = role["arm"]
    require(events[0]["kind"] == "branch_started" and events[-1]["kind"] == "branch_closed"
            and events[-1]["status"] == row["status"], "branch boundaries")
    require(row["status"] in ("failed", "rgb_abstained", "native_early_stop",
                               "feedback_limit_completed"), "branch status")
    timeline = history.History()
    warm_refs = {p["step"]: p for p in source_history["frames"]}
    transitions, observations, decisions = [], [], []
    cp5 = False
    dispatched, observed, returned, last_decision = None, None, None, None
    plan_batches = []
    planning = None
    stopped = False
    expected_timeline = 0
    for event in events[1:-1]:
        kind = event["kind"]
        require(event.get("arm") == arm, "branch isolation")
        if kind == "reset_verified":
            require(not observations and expected_timeline == 0, "reset ordering")
        elif kind == "action_dispatch":
            require(not stopped and dispatched is None and planning is None, "dispatch state")
            expected_timeline += 1
            phase = "warmup" if expected_timeline <= 10 else "prefix" if expected_timeline <= 15 else "feedback"
            step = expected_timeline if phase == "warmup" else expected_timeline - (10 if phase == "prefix" else 15)
            require(event["timeline_step"] == expected_timeline and event["phase"] == phase
                    and event["step"] == step and step <= (5 if phase == "feedback" else 10),
                    "fixed action schedule")
            if phase == "warmup":
                expected = [0., 0.]
            elif phase == "prefix":
                expected = role["prefix"][step-1]
            else:
                require(cp5 and last_decision is not None
                        and last_decision["status"] == "proposal_only_wait_for_real_rgb", "feedback gate")
                expected = last_decision["action"]
                last_decision = None
            require(event["action"] == expected, "frozen or scored action")
            timeline.dispatched(expected, step=expected_timeline)
            dispatched, observed, returned = event, None, None
        elif kind == "action_observed":
            require(dispatched is not None and observed is None
                    and all(event[k] == dispatched[k] for k in ("phase", "step", "timeline_step")),
                    "observed order")
            observed = event
        elif kind == "action_return":
            require(dispatched is not None and observed is not None and returned is None
                    and all(event[k] == dispatched[k] for k in ("phase", "step", "timeline_step", "action"))
                    and all(event[k] == observed[k] for k in ("action", "terminated", "truncated")),
                    "actual action receipt")
            returned = {k: event[k] for k in ("phase", "step", "timeline_step", "action",
                                               "state", "reward", "terminated", "truncated")}
            transitions.append(returned)
            stopped = bool(event["terminated"] or event["truncated"])
        elif kind == "observation":
            point = {k: v for k, v in event.items()
                     if k not in ("index", "previous", "kind", "arm", "sha256")}
            saved.frame(directory, point)
            phase, step = point["phase"], point["step"]
            if not observations:
                require(dispatched is None and (phase, step) == ("warmup", 0), "initial frame")
                ref = warm_refs[0]
                pixels = base.image(directory / point["file"])
                timeline.reset(pixels, 0)
            else:
                require(returned is not None and phase == returned["phase"] and step == returned["step"]
                        and point["state"] == returned["state"], "observation receipt")
                ref = warm_refs.get(step) if phase == "warmup" else (
                    role["prefix_checkpoints"][step-1] if phase == "prefix" else None)
                pixels = base.image(directory / point["file"])
                timeline.returned(returned["action"], pixels, step=expected_timeline)
                dispatched, observed, returned = None, None, None
            if ref is not None:
                require(all(point[k] == ref[k] for k in ("state", "snapshot", "pixel_sha256")),
                        "source replay mismatch")
            observations.append(point)
        elif kind == "cp5_gate_passed":
            require(expected_timeline == 15 and not stopped and dispatched is None and not cp5,
                    "cp5 ordering")
            context = timeline.context()
            require(event["pixel_sha256"] == role["cp5_pixel_sha256"] == observations[-1]["pixel_sha256"]
                    and event["frame_ids"] == context["frame_ids"]
                    and event["pixel_sha256s"] == context["pixel_sha256"]
                    and event["actual_history_sha256"] == context["actual_history_sha256"],
                    "cp5 actual history")
            cp5 = True
        elif kind == "planning_started":
            require(cp5 and not stopped and dispatched is None and planning is None
                    and last_decision is None and event["replan"] == expected_timeline - 14,
                    "planning order")
            context = timeline.context()
            require(event["frame_id"] == context["frame_ids"][-1]
                    and event["rgb_sha256"] == context["pixel_sha256"][-1]
                    and event["actual_history_sha256"] == context["actual_history_sha256"]
                    and event["history_frame_ids"] == context["frame_ids"]
                    and event["history_pixel_sha256s"] == context["pixel_sha256"]
                    and event["history_actual_actions"] == context["actions"].tolist(),
                    "planning inputs")
            planning, plan_batches = event, []
        elif kind == "score_batch":
            require(planning is not None and event["replan"] == planning["replan"]
                    and event["iteration"] == len(plan_batches) < 5, "score batch ordering")
            name = event["file"]
            require(Path(name).name == name and base.sha(directory / name) == event["file_sha256"],
                    "score batch SHA")
            with np.load(io.BytesIO((directory / name).read_bytes()), allow_pickle=False) as data:
                require(set(data.files) == {"variants", "scores"}, "score batch content")
                variants, scores = data["variants"].copy(), data["scores"].copy()
            require(variants.shape == (64, 5, 2, 5, 2) and scores.shape == (64, 5)
                    and np.isfinite(variants).all() and np.isfinite(scores).all(), "score batch shapes")
            iteration = len(plan_batches) + 1
            require(event["counts"] == {"rgb_encode_calls": 3, "action_encoder_calls": 2*iteration,
                                        "predict_calls": 2*iteration, "predict_samples": 640*iteration}
                    and event["rgb_encoder_hook"] == 3
                    and event["hooks"] == {"action_encoder": 2*iteration, "predict": 2*iteration},
                    "score batch CUDA counters")
            plan_batches.append((variants, scores))
        elif kind == "planning_finished":
            require(planning is not None and event["replan"] == planning["replan"], "planning finish")
            context = timeline.context()
            features = feedback.geometry(context["frames"][-1], context["actions"][-1])
            decision = event["decision"]
            if features is None:
                require(not plan_batches and decision == {"status": "near_object_abstain_no_contact_claim"}
                        and all(v == 0 for v in event["counts"].values())
                        and event["rgb_encoder_hook"] == 0, "abstention")
            else:
                require(len(plan_batches) == 5, "complete model score ledger")
                index = 0
                def replay_score(variants):
                    nonlocal index
                    expected, scores = plan_batches[index]
                    require(np.array_equal(variants, expected), "candidate bank replay")
                    index += 1
                    return scores
                candidate, safe, gap = features
                action, selection = feedback.search(candidate, safe, replay_score)
                require(decision["status"] == "proposal_only_wait_for_real_rgb"
                        and decision["action"] == action.tolist()
                        and decision["action_sha256"] == feedback.sha(action.tobytes())
                        and decision["frame_id"] == context["frame_ids"][-1]
                        and decision["rgb_sha256"] == context["pixel_sha256"][-1]
                        and decision["actual_history_sha256"] == context["actual_history_sha256"]
                        and decision["gap_pixels"] == gap and decision["approach_candidate"] == candidate.tolist()
                        and decision["model_scoring"] == selection
                        and decision["budget"] == event["counts"]
                        and event["counts"] == {"rgb_encode_calls": 3, "action_encoder_calls": 10,
                                                "predict_calls": 10, "predict_samples": 3200}
                        and event["rgb_encoder_hook"] == 3
                        and event["hooks"] == {"action_encoder": 10, "predict": 10},
                        "selection or budget replay")
            decisions.append(decision)
            last_decision = decision
            planning = None
        elif kind == "planning_partial":
            require(planning is not None and event["replan"] == planning["replan"], "partial plan")
            counts, hooks = event["counts"], event["hooks"]
            require(set(counts) == {"rgb_encode_calls", "action_encoder_calls", "predict_calls",
                                    "predict_samples"}
                    and set(hooks) == {"action_encoder", "predict"}
                    and 0 <= counts["rgb_encode_calls"] <= 3
                    and 2*len(plan_batches) <= counts["action_encoder_calls"] <= 10
                    and 2*len(plan_batches) <= counts["predict_calls"] <= 10
                    and 640*len(plan_batches) <= counts["predict_samples"] <= 3200
                    and 0 <= event["rgb_encoder_hook"] <= 3
                    and 0 <= hooks["action_encoder"] <= 10 and 0 <= hooks["predict"] <= 10,
                    "partial accounting bounds")
            planning = None
        elif kind == "branch_error":
            require(row["status"] == "failed" and event["error"] == row["error"], "branch error")
        else:
            raise ValueError("unknown branch event " + kind)
    require(transitions == row["transitions"] and observations == row["observations"]
            and decisions == row["decisions"], "row and journal")
    if row["status"] == "feedback_limit_completed":
        require(cp5 and expected_timeline == 20 and not stopped and len(decisions) == 5
                and dispatched is None, "feedback completion")
    if row["status"] == "native_early_stop":
        require(stopped and transitions[-1]["phase"] == "feedback", "early stop")
    if row["status"] == "rgb_abstained":
        require(last_decision is not None and last_decision["status"] == "near_object_abstain_no_contact_claim",
                "abstention status")
    if row["status"] == "failed":
        require(row["error"] is not None and any(e["kind"] == "branch_error" for e in events), "failed row")
    return {"arm": arm, "status": row["status"], "decisions": len(decisions)}


def verify(prepared, prepared_sha, directory, report_sha):
    from . import p1_dev303_feedback_live_20261102_codex_exclusive as run
    prepared, directory = Path(prepared).resolve(strict=True), Path(directory).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == run.build(), "contract closure")
    require(base.sha(directory / "report.json") == report_sha, "report SHA")
    report = base.read(directory / "report.json")
    require(report["schema"] == run.NAME + ".report.v1" and report["prepare_sha256"] == prepared_sha
            and report["P1_admission"] is False and report["independent_confirmation"] is False
            and report["full_cost_certified"] is False
            and report["classification"] == "development_only"
            and report["status"] == ("partial_or_failed" if report["errors"] else "recorded"), "report")
    require(base.read(directory / "execution_binding.json") ==
            {"prepared": str(prepared), "prepare_sha256": prepared_sha}, "execution binding")
    journal = directory / "journal.jsonl"
    require(base.sha(journal) == report["journal_sha256"], "journal SHA")
    events = saved.journal(journal, report["journal_tail"])
    require(events[0]["kind"] == "run_started" and events[0]["prepare_sha256"] == prepared_sha
            and events[-1]["kind"] == "run_finished" and events[-1]["errors"] == report["errors"],
            "journal endpoints")
    roles = base.read(prepared)["replay"]["source"]["roles"]
    require([r["arm"] for r in report["rows"]] == [r["arm"] for r in roles][:len(report["rows"])],
            "arm order")
    require(report["errors"] or len(report["rows"]) == len(roles), "both arms required")
    _, receipt, _ = prior.source_data()
    results = []
    loaded = [e for e in events if e["kind"] == "model_loaded"]
    closed = [e for e in events if e["kind"] == "model_closed"]
    require(len(loaded) <= 1 and len(closed) <= 1
            and (not report["rows"] or len(loaded) == 1), "model event roster")
    if loaded:
        expected_hash = base.read(run.FEEDBACK_PREP)["model_hash"]
        require(events.index(loaded[0]) == 1 and loaded[0]["host"]["device"] == "cuda:0"
                and loaded[0]["model_hash"] == expected_hash, "model load binding")
    if closed:
        require(loaded and len(report["rows"]) == len(roles)
                and closed[0]["model_hash"] == loaded[0]["model_hash"], "model closure hash")
    if not report["errors"]:
        require(len(loaded) == len(closed) == 1, "model closure required")
    cursor = events.index(loaded[0]) + 1 if loaded else 1
    for role, row in zip(roles, report["rows"]):
        require(base.read(directory / (role["arm"] + ".row.json")) == row, "row file")
        end = cursor
        while end < len(events)-1 and events[end]["kind"] != "branch_closed":
            end += 1
        require(end < len(events)-1, "branch closure")
        scoped = events[cursor:end+1]
        require(all(e.get("arm") == role["arm"] for e in scoped), "contiguous fixed arms")
        results.append(verify_branch(directory, role, row, scoped, receipt["history"]))
        cursor = end + 1
    remaining = events[cursor:-1]
    require(all(e["kind"] in ("model_closed", "run_error") for e in remaining),
            "unscoped journal events")
    branch_errors = [{"arm": row["arm"], **row["error"]}
                     for row in report["rows"] if row["error"] is not None]
    require(all(error in report["errors"] for error in branch_errors), "branch errors reported")
    return {"status": "saved_ledger_verified", "run_status": report["status"], "branches": results,
            "limitations": "model outputs and instrumented counters are recorded claims, not independently rerun",
            "full_cost_certified": False, "P1_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prepared", "prepared-sha", "directory", "report-sha"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared, args.prepared_sha, args.directory, args.report_sha)))


if __name__ == "__main__":
    main()
