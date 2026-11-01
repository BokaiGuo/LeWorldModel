"""Prospective post-outcome model-choice versus RGB-approach contrast."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_feedback_live_20261102_codex_exclusive as live
from . import p1_dev303_feedback_live_saved_20261102_codex_exclusive as old_saved
from . import p1_dev303_feedback_choice_contrast_saved_20261102_codex_exclusive as saved

NAME = "p1_dev303_feedback_choice_contrast_20261102_codex_exclusive"
ROOT, base, prior = live.ROOT, live.base, live.prior
feedback, history, require = live.feedback, live.history, live.require
SOURCE_DIR = ROOT / "results/p1_dev303_feedback_live_20261102_codex_exclusive_RUN_DEV01"
SOURCE_PREP = ROOT / "results/p1_dev303_feedback_live_20261102_codex_exclusive_PREP_DEV01.json"
SOURCE_PREP_SHA = "74b31b22db7a4bfb803264aa45889876fe935d361567949f8aa41e93a054d686"
SOURCE_REPORT_SHA = "23e66529af689ed6a13e19c9b3d6f5ecc85813c5b1b05a3ec22ac31c0f5f78ee"
METHODS = ("model_choice", "direct_approach")


def build():
    source_audit = old_saved.verify(SOURCE_PREP, SOURCE_PREP_SHA, SOURCE_DIR, SOURCE_REPORT_SHA)
    require(source_audit["run_status"] == "recorded"
            and [r["arm"] for r in source_audit["branches"]] == list(feedback.frozen.ARMS)
            and all(r["decisions"] == 5 for r in source_audit["branches"]), "source ledger")
    source = live.build()
    paths = [Path(__file__), Path(saved.__file__), Path(live.__file__), Path(old_saved.__file__)]
    return {
        "schema": NAME + ".contract.v1", "classification": "post_outcome_development_contrast",
        "design_source_report_sha256": SOURCE_REPORT_SHA,
        "source_feedback_preparation_sha256": SOURCE_PREP_SHA,
        "source_replay": source["replay"], "feedback_protocol_sha256": live.FEEDBACK_SHA,
        "arms_in_fixed_order": [{"prefix": arm, "method": method, "id": arm + "_" + method}
                                for arm in feedback.frozen.ARMS for method in METHODS],
        "method_rules": {
            "model_choice": "execute the V13 robust-score winner",
            "direct_approach": "execute the predeclared RGB approach seed after the same full scoring",
        },
        "controller_lifecycle": "new proposal-only controller each opportunity; actual action and RGB verified by independent chronological history",
        "feasibility": "both abstain if the existing RGB seed geometry or one-step guard fails",
        "selection": "method fixed in advance; never select an action using physical outcome",
        "opportunities": "up to five one-action decisions and fresh real RGB per arm; no padding after stop",
        "cost": "same five-iteration V13 scoring per available decision in both methods",
        "cost_fairness_boundary": "opportunity and per-decision compute matched; realized totals can differ after abstention or failure",
        "full_cost_certified": False, "physical_execution_authorized_by_contract": False,
        "cuda_execution_authorized_by_contract": False, "P1_admission": False,
        "independent_confirmation": False,
        "implementation_sha256": {str(p.resolve()): base.sha(p) for p in paths},
    }


def decision(controller, scorer, method):
    require(method in METHODS, "fixed contrast method")
    model_decision = controller.plan(scorer)
    if model_decision["status"] != "proposal_only_wait_for_real_rgb":
        return {"method": method, "status": model_decision["status"], "model_decision": model_decision}
    original = feedback.raw_actions([model_decision["action"]], (1, 2))[0]
    direct = feedback.raw_actions([model_decision["approach_candidate"]], (1, 2))[0]
    require(np.array_equal(controller.pending, original), "model pending receipt")
    # This is an explicitly preregistered intervention after identical scoring.
    # Do not mutate the sealed proposal controller or misreport its pending action.
    action = original if method == "model_choice" else direct
    return {"method": method, "status": "proposal_only_wait_for_real_rgb",
            "action": action.tolist(), "action_sha256": feedback.sha(action.tobytes()),
            "model_decision": model_decision, "seed_passed_existing_geometry_guard": True,
            "same_model_budget_for_methods": True, "physical_execution_authorized": False,
            "P1_admission": False}


def branch(env, store, role, method, receipt, reset, goal_rgb, model, task, model_hash, row):
    arm = role["arm"]
    timeline = history.History()
    goal = np.asarray(receipt["history"]["goal"], np.float64)
    checkpoints = {p["step"]: p for p in receipt["history"]["frames"]}
    gate = live.replay.handoff.ReplayGate(role)
    obs, info = env.reset(seed=prior.SEED)
    require(np.array_equal(np.asarray(obs["state"], np.float64), np.asarray(reset["state"], np.float64))
            and np.array_equal(np.asarray(info["goal_state"], np.float64), goal)
            and np.array_equal(np.asarray(info["goal"]), goal_rgb), "reset/goal mismatch")
    store.emit("reset_verified", arm=arm)
    point = base.capture(store, env, arm, "warmup", 0, checkpoints[0])
    row["observations"].append(point)
    timeline.reset(base.image(store.path / point["file"]), 0)
    step_id = 0

    def step(phase, number, action, expected=None):
        nonlocal step_id
        store.check()
        step_id += 1
        command = np.asarray(action, dtype="<f4")
        timeline.dispatched(command, step=step_id)
        store.emit("action_dispatch", arm=arm, phase=phase, step=number,
                   timeline_step=step_id, action=command.tolist())
        obs, reward, terminated, truncated, info = env.step(command.copy())
        actual = np.asarray(env.latest_action)
        store.emit("action_observed", arm=arm, phase=phase, step=number,
                   timeline_step=step_id, action=actual.tolist(),
                   terminated=bool(terminated), truncated=bool(truncated))
        actual = prior._check_return(env, obs, info, goal, goal_rgb, command)
        require(np.isfinite(float(reward)), "invalid reward")
        transition = {"phase": phase, "step": number, "timeline_step": step_id,
                      "action": actual.tolist(), "state": np.asarray(obs["state"]).tolist(),
                      "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)}
        row["transitions"].append(transition)
        store.emit("action_return", arm=arm, **transition)
        point = base.capture(store, env, arm, phase, number, expected)
        row["observations"].append(point)
        pixels = base.image(store.path / point["file"])
        timeline.returned(actual, pixels, step=step_id)
        return point, pixels, actual, bool(terminated or truncated)

    for i in range(1, 11):
        _, _, _, stopped = step("warmup", i, np.zeros(2, dtype="<f4"), checkpoints.get(i))
        require(not stopped, "warmup stopped")
    for i, action in enumerate(role["prefix"], 1):
        point, _, actual, stopped = step("prefix", i, action, role["prefix_checkpoints"][i-1])
        require(not stopped, "prefix stopped")
        gate.accept_prefix(actual, point)
    gate.fixed_tail()
    require(point["pixel_sha256"] == role["cp5_pixel_sha256"], "cp5 pixel")
    context = timeline.context()
    store.emit("cp5_gate_passed", arm=arm, pixel_sha256=role["cp5_pixel_sha256"],
               frame_ids=context["frame_ids"], pixel_sha256s=context["pixel_sha256"],
               actual_history_sha256=context["actual_history_sha256"])
    for i in range(1, 6):
        scorer = None
        context = timeline.context()
        controller = feedback.Feedback(context["frames"], context["actions"], frame_ids=context["frame_ids"])
        store.emit("planning_started", arm=arm, replan=i, method=method,
                   frame_id=controller.frame_id, rgb_sha256=feedback.sha(controller.frames[-1].tobytes()),
                   actual_history_sha256=feedback.sha(controller.actions.tobytes()),
                   history_frame_ids=context["frame_ids"], history_pixel_sha256s=context["pixel_sha256"],
                   history_actual_actions=context["actions"].tolist())
        try:
            scorer = live.RecordedScorer(model, task, model_hash, store, arm, i)
            choice = decision(controller, scorer, method)
        except Exception:
            if scorer is not None:
                store.emit("planning_partial", arm=arm, replan=i, counts=dict(scorer.counts),
                           hooks=dict(scorer.hooks), rgb_encoder_hook=scorer.rgb_encoder_hook)
            raise
        store.emit("planning_finished", arm=arm, replan=i, decision=choice,
                   counts=dict(scorer.counts), hooks=dict(scorer.hooks),
                   rgb_encoder_hook=scorer.rgb_encoder_hook)
        row["decisions"].append(choice)
        if choice["status"] != "proposal_only_wait_for_real_rgb":
            row["status"] = "rgb_abstained"
            return
        _, pixels, actual, stopped = step("feedback", i, choice["action"])
        if stopped:
            row["status"] = "native_early_stop"
            return
        if method == "model_choice":
            controller.accept(pixels, frame_id=step_id, actual_action=actual)
    row["status"] = "feedback_limit_completed"


def execute(prepared, prepared_sha, output):
    import torch
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments.gpu_preflight import require_cuda_allocation

    prepared = Path(prepared).resolve(strict=True)
    require(base.sha(prepared) == prepared_sha and base.read(prepared) == build(), "prepared contract")
    arrays, receipt, reset = prior.source_data()
    store = base.Store(output)
    store.json("execution_binding.json", {"prepared": str(prepared), "prepare_sha256": prepared_sha})
    store.emit("run_started", prepare_sha256=prepared_sha)
    rows, errors = [], []
    try:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with base.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
            host = require_cuda_allocation("cuda:0")
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            contract = feedback.protocol()
            model = load_compatible_pretrained(contract["policy"], device=torch.device("cuda:0"))
            before = feedback.engine.base.blind.host.signal.model_hash(model)
            require(before == contract["model_hash"], "frozen model")
            store.emit("model_loaded", host=host, model_hash=before)
            source_roles = {r["arm"]: r for r in base.read(prepared)["source_replay"]["source"]["roles"]}
            for spec in base.read(prepared)["arms_in_fixed_order"]:
                arm, method, env = spec["id"], spec["method"], None
                role = dict(source_roles[spec["prefix"]], arm=arm)
                row = {"arm": arm, "prefix": spec["prefix"], "method": method, "status": "running",
                       "transitions": [], "observations": [], "decisions": [], "error": None}
                store.emit("branch_started", arm=arm, prefix=spec["prefix"], method=method)
                try:
                    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
                    require(base.sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "native SHA")
                    env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                    branch(env, store, role, method, receipt, reset, arrays["goal_rgb"],
                           model, contract["task"], before, row)
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                    errors.append({"arm": arm, **row["error"]})
                    store.emit("branch_error", arm=arm, error=row["error"])
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception as exc:
                            errors.append({"arm": arm, "stage": "close", "type": type(exc).__name__,
                                           "message": str(exc)[:1000]})
                rows.append(row)
                store.json(arm + ".row.json", row)
                store.emit("branch_closed", arm=arm, status=row["status"])
            after = feedback.engine.base.blind.host.signal.model_hash(model)
            require(after == before, "postrun frozen model")
            store.emit("model_closed", model_hash=after)
    except Exception as exc:
        error = {"stage": "run", "type": type(exc).__name__, "message": str(exc)[:1000]}
        errors.append(error)
        store.emit("run_error", error=error)
    try:
        require(base.read(prepared) == build(), "postrun source closure")
    except Exception as exc:
        errors.append({"stage": "source_closure", "type": type(exc).__name__, "message": str(exc)[:1000]})
    store.emit("run_finished", errors=errors)
    report = {"schema": NAME + ".report.v1", "prepare_sha256": prepared_sha, "rows": rows,
              "errors": errors, "status": "recorded" if not errors else "partial_or_failed",
              "journal_sha256": base.sha(store.path / "journal.jsonl"),
              "journal_tail": store.events[-1]["sha256"], "full_cost_certified": False,
              "P1_admission": False, "independent_confirmation": False}
    store.json("report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prepare-sha256")
    parser.add_argument("--authorize-development-physics", action="store_true")
    parser.add_argument("--authorize-model-cuda", action="store_true")
    args = parser.parse_args()
    require(args.tag.isascii() and args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_RUN_{args.tag}"
    if args.command == "prepare":
        require(not args.prepare_sha256 and not args.authorize_development_physics
                and not args.authorize_model_cuda and not os.path.lexists(output), "prepare only")
        digest = prior.write_new(prepared, build())
        print(json.dumps({"prepared": str(prepared), "sha256": digest}))
    else:
        require(args.prepare_sha256 and args.authorize_development_physics
                and args.authorize_model_cuda, "separate physics and CUDA authorization")
        report = execute(prepared, args.prepare_sha256, output)
        print(json.dumps({"output": str(output), "status": report["status"],
                          "report_sha256": base.sha(output / "report.json")}))


if __name__ == "__main__":
    main()
