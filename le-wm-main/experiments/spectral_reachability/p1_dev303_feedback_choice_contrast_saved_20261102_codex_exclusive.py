"""Saved-only contrast verification: fixed methods, actual histories and score replay."""
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


def verify_branch(directory, role, spec, row, events, source_history):
    arm, method = spec["id"], spec["method"]
    require(row["arm"] == arm and row["prefix"] == spec["prefix"] and row["method"] == method
            and events[0]["kind"] == "branch_started" and events[0]["prefix"] == spec["prefix"]
            and events[0]["method"] == method and events[-1]["kind"] == "branch_closed"
            and events[-1]["status"] == row["status"], "fixed branch")
    require(row["status"] in ("failed", "rgb_abstained", "native_early_stop", "feedback_limit_completed"),
            "status")
    warm = {p["step"]: p for p in source_history["frames"]}
    timeline = history.History()
    actions, frames, decisions, batches = [], [], [], []
    step_id, pending, observed, returned, cp5, planned, stopped = 0, None, None, None, False, None, False
    plan = None
    for event in events[1:-1]:
        require(event.get("arm") == arm, "branch isolation")
        kind = event["kind"]
        if kind == "reset_verified":
            require(not frames and step_id == 0, "reset order")
        elif kind == "action_dispatch":
            require(not stopped and pending is None and plan is None, "dispatch state")
            step_id += 1
            phase = "warmup" if step_id <= 10 else "prefix" if step_id <= 15 else "feedback"
            n = step_id if phase == "warmup" else step_id - (10 if phase == "prefix" else 15)
            require(event["timeline_step"] == step_id and event["phase"] == phase
                    and event["step"] == n and (phase != "feedback" or n <= 5), "step order")
            if phase == "warmup":
                action = [0., 0.]
            elif phase == "prefix":
                action = role["prefix"][n-1]
            else:
                require(cp5 and planned is not None and planned["status"] == "proposal_only_wait_for_real_rgb",
                        "choice gate")
                action, planned = planned["action"], None
            require(event["action"] == action, "dispatched action")
            timeline.dispatched(action, step=step_id)
            pending, observed, returned = event, None, None
        elif kind == "action_observed":
            require(pending is not None and observed is None
                    and all(event[k] == pending[k] for k in ("phase", "step", "timeline_step")),
                    "action observed ordering")
            observed = event
        elif kind == "action_return":
            require(pending is not None and observed is not None and returned is None
                    and all(event[k] == pending[k] for k in ("phase", "step", "timeline_step", "action"))
                    and all(event[k] == observed[k] for k in ("action", "terminated", "truncated")),
                    "action receipt")
            returned = {k: event[k] for k in ("phase", "step", "timeline_step", "action",
                                               "state", "reward", "terminated", "truncated")}
            actions.append(returned)
            stopped = bool(event["terminated"] or event["truncated"])
        elif kind == "observation":
            point = {k: v for k, v in event.items()
                     if k not in ("index", "previous", "kind", "arm", "sha256")}
            saved.frame(directory, point)
            phase, n = point["phase"], point["step"]
            if not frames:
                require((phase, n) == ("warmup", 0), "initial frame")
                ref = warm[0]
                timeline.reset(base.image(directory / point["file"]), 0)
            else:
                require(returned is not None and phase == returned["phase"] and n == returned["step"]
                        and point["state"] == returned["state"], "frame receipt")
                ref = warm.get(n) if phase == "warmup" else (
                    role["prefix_checkpoints"][n-1] if phase == "prefix" else None)
                timeline.returned(returned["action"], base.image(directory / point["file"]), step=step_id)
                pending, observed, returned = None, None, None
            if ref is not None:
                require(all(point[k] == ref[k] for k in ("state", "snapshot", "pixel_sha256")),
                        "replay checkpoint")
            frames.append(point)
        elif kind == "cp5_gate_passed":
            require(step_id == 15 and not stopped and pending is None and not cp5, "cp5 order")
            context = timeline.context()
            require(event["pixel_sha256"] == role["cp5_pixel_sha256"] == frames[-1]["pixel_sha256"]
                    and event["frame_ids"] == context["frame_ids"]
                    and event["pixel_sha256s"] == context["pixel_sha256"]
                    and event["actual_history_sha256"] == context["actual_history_sha256"], "cp5 history")
            cp5 = True
        elif kind == "planning_started":
            require(cp5 and not stopped and pending is None and plan is None and planned is None
                    and event["method"] == method and event["replan"] == step_id-14, "planning order")
            ctx = timeline.context()
            require(event["frame_id"] == ctx["frame_ids"][-1] and event["rgb_sha256"] == ctx["pixel_sha256"][-1]
                    and event["actual_history_sha256"] == ctx["actual_history_sha256"]
                    and event["history_frame_ids"] == ctx["frame_ids"]
                    and event["history_pixel_sha256s"] == ctx["pixel_sha256"]
                    and event["history_actual_actions"] == ctx["actions"].tolist(), "planning inputs")
            plan, batches = event, []
        elif kind == "score_batch":
            require(plan is not None and event["replan"] == plan["replan"]
                    and event["iteration"] == len(batches) < 5, "score order")
            name = event["file"]
            require(Path(name).name == name and base.sha(directory / name) == event["file_sha256"], "score SHA")
            with np.load(io.BytesIO((directory / name).read_bytes()), allow_pickle=False) as data:
                require(set(data.files) == {"variants", "scores"}, "score files")
                variants, scores = data["variants"].copy(), data["scores"].copy()
            require(variants.shape == (64, 5, 2, 5, 2) and scores.shape == (64, 5)
                    and np.isfinite(variants).all() and np.isfinite(scores).all(), "score shape")
            i = len(batches) + 1
            require(event["rgb_encoder_hook"] == 3
                    and event["counts"] == {"rgb_encode_calls": 3, "action_encoder_calls": i*2,
                                            "predict_calls": i*2, "predict_samples": i*640}
                    and event["hooks"] == {"action_encoder": i*2, "predict": i*2}, "batch metering")
            batches.append((variants, scores))
        elif kind == "planning_finished":
            require(plan is not None and event["replan"] == plan["replan"], "plan completion")
            ctx = timeline.context()
            features = feedback.geometry(ctx["frames"][-1], ctx["actions"][-1])
            choice = event["decision"]
            require(choice["method"] == method, "choice method")
            if features is None:
                require(not batches and choice["status"] == "near_object_abstain_no_contact_claim"
                        and event["rgb_encoder_hook"] == 0
                        and all(v == 0 for v in event["counts"].values()), "near abstention")
            else:
                require(len(batches) == 5 and event["rgb_encoder_hook"] == 3
                        and event["counts"] == {"rgb_encode_calls": 3, "action_encoder_calls": 10,
                                                "predict_calls": 10, "predict_samples": 3200}
                        and event["hooks"] == {"action_encoder": 10, "predict": 10}, "full scored budget")
                index = 0
                def score(variants):
                    nonlocal index
                    expected, values = batches[index]
                    require(np.array_equal(variants, expected), "candidate bank")
                    index += 1
                    return values
                seed, safe, gap = features
                winner, selection = feedback.search(seed, safe, score)
                model = choice["model_decision"]
                require(model["action"] == winner.tolist() and model["model_scoring"] == selection
                        and model["action_sha256"] == feedback.sha(winner.tobytes())
                        and model["approach_candidate"] == seed.tolist()
                        and model["gap_pixels"] == gap and model["budget"] == event["counts"]
                        and model["frame_id"] == ctx["frame_ids"][-1]
                        and model["rgb_sha256"] == ctx["pixel_sha256"][-1]
                        and model["actual_history_sha256"] == ctx["actual_history_sha256"], "model selection")
                selected = winner if method == "model_choice" else seed
                require(choice["status"] == "proposal_only_wait_for_real_rgb"
                        and choice["action"] == selected.tolist()
                        and choice["action_sha256"] == feedback.sha(selected.tobytes())
                        and choice["seed_passed_existing_geometry_guard"] is True
                        and choice["same_model_budget_for_methods"] is True, "fixed intervention")
            decisions.append(choice)
            planned, plan = choice, None
        elif kind == "planning_partial":
            require(plan is not None and event["replan"] == plan["replan"], "partial model log")
            plan = None
        elif kind == "branch_error":
            require(row["status"] == "failed" and event["error"] == row["error"], "error binding")
        else:
            raise ValueError("unexpected event " + kind)
    require(actions == row["transitions"] and frames == row["observations"]
            and decisions == row["decisions"], "row consistency")
    if row["status"] == "feedback_limit_completed":
        require(cp5 and step_id == 20 and len(decisions) == 5 and pending is None and not stopped, "completion")
    elif row["status"] == "native_early_stop":
        require(stopped and actions[-1]["phase"] == "feedback", "early stop")
    elif row["status"] == "rgb_abstained":
        require(planned is not None and planned["status"] == "near_object_abstain_no_contact_claim", "abstain")
    else:
        require(row["error"] is not None and any(e["kind"] == "branch_error" for e in events), "failure")
    return {"arm": arm, "method": method, "status": row["status"], "decisions": len(decisions)}


def verify(prepared, prepared_sha, directory, report_sha):
    from . import p1_dev303_feedback_choice_contrast_20261102_codex_exclusive as run
    prepared, directory = Path(prepared).resolve(strict=True), Path(directory).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == run.build(), "contract closure")
    require(base.sha(directory / "report.json") == report_sha, "external report SHA")
    report = base.read(directory / "report.json")
    require(report["schema"] == run.NAME + ".report.v1" and report["prepare_sha256"] == prepared_sha
            and report["P1_admission"] is False and report["independent_confirmation"] is False
            and report["full_cost_certified"] is False
            and report["status"] == ("partial_or_failed" if report["errors"] else "recorded"), "report")
    require(base.read(directory / "execution_binding.json") ==
            {"prepared": str(prepared), "prepare_sha256": prepared_sha}, "execution binding")
    path = directory / "journal.jsonl"
    require(base.sha(path) == report["journal_sha256"], "journal SHA")
    events = saved.journal(path, report["journal_tail"])
    require(events[0]["kind"] == "run_started" and events[0]["prepare_sha256"] == prepared_sha
            and events[-1]["kind"] == "run_finished" and events[-1]["errors"] == report["errors"],
            "journal endpoints")
    contract = base.read(prepared)
    specs = contract["arms_in_fixed_order"]
    require([r["arm"] for r in report["rows"]] == [s["id"] for s in specs][:len(report["rows"])]
            and (bool(report["errors"]) or len(report["rows"]) == len(specs)), "four-arm order")
    loaded = [e for e in events if e["kind"] == "model_loaded"]
    closed = [e for e in events if e["kind"] == "model_closed"]
    require(len(loaded) <= 1 and len(closed) <= 1 and (not report["rows"] or loaded), "model events")
    if loaded:
        require(events.index(loaded[0]) == 1 and loaded[0]["host"]["device"] == "cuda:0"
                and loaded[0]["model_hash"] == base.read(run.live.FEEDBACK_PREP)["model_hash"], "model binding")
    if closed:
        require(loaded and len(report["rows"]) == len(specs)
                and closed[0]["model_hash"] == loaded[0]["model_hash"], "model closure")
    if not report["errors"]:
        require(len(loaded) == len(closed) == 1, "model completion")
    roles = {r["arm"]: r for r in contract["source_replay"]["source"]["roles"]}
    _, source, _ = prior.source_data()
    cursor = events.index(loaded[0])+1 if loaded else 1
    results = []
    for spec, row in zip(specs, report["rows"]):
        require(base.read(directory / (spec["id"] + ".row.json")) == row, "row file")
        end = cursor
        while end < len(events)-1 and events[end]["kind"] != "branch_closed":
            end += 1
        require(end < len(events)-1, "missing branch closure")
        results.append(verify_branch(directory, roles[spec["prefix"]], spec, row,
                                     events[cursor:end+1], source["history"]))
        cursor = end+1
    require(all(e["kind"] in ("model_closed", "run_error") for e in events[cursor:-1]),
            "unexpected global event")
    require(all({"arm": row["arm"], **row["error"]} in report["errors"]
                for row in report["rows"] if row["error"] is not None), "missing reported error")
    return {"status": "saved_contrast_chain_verified", "run_status": report["status"], "branches": results,
            "model_score_limitation": "recorded GPU results not independently rerun",
            "full_cost_certified": False, "P1_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("prepared", "prepared-sha", "directory", "report-sha"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.prepared, args.prepared_sha, args.directory, args.report_sha)))


if __name__ == "__main__":
    main()
