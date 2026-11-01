"""Prospective dev303 feedback runner; execution needs separate authorization."""
from __future__ import annotations

import argparse
import importlib
import inspect
import io
import json
import os
from pathlib import Path

import numpy as np

from . import p1_dev303_cp5_tail_physical_20261102_exclusive as replay
from . import p1_dev303_feedback_history_20261102_codex_exclusive as history
from . import p1_dev303_rgb_feedback_scored_tail_20261102_codex_exclusive as feedback
from . import p1_dev303_feedback_live_saved_20261102_codex_exclusive as saved

NAME = "p1_dev303_feedback_live_20261102_codex_exclusive"
ROOT, base, prior = replay.ROOT, replay.base, replay.prior
require = prior.require
FEEDBACK_PREP = ROOT / "results/p1_dev303_rgb_feedback_scored_tail_20261102_codex_exclusive_PREP_DEV01.json"
FEEDBACK_SHA = "c9d8938a9b6ddb9e4efd2265bcbba00b87114da41be32641b63466f4119b5568"


def build():
    from experiments import checkpoint_compat, gpu_preflight, run_pusht_counterfactual
    from stable_worldmodel.wm.lewm.lewm import LeWM

    require(base.sha(FEEDBACK_PREP) == FEEDBACK_SHA and base.read(FEEDBACK_PREP) == feedback.protocol(),
            "sealed feedback protocol")
    source = replay.build()
    weights, config = checkpoint_compat._resolve_local(base.read(FEEDBACK_PREP)["policy"])
    paths = [Path(__file__), Path(saved.__file__), Path(history.__file__),
             Path(replay.__file__), Path(feedback.__file__), Path(checkpoint_compat.__file__),
             Path(gpu_preflight.__file__), Path(run_pusht_counterfactual.__file__),
             Path(feedback.engine.base.p0.__file__), Path(inspect.getsourcefile(LeWM)),
             Path(weights), Path(config)]
    return {
        "schema": NAME + ".contract.v1", "classification": "post_outcome_development_only",
        "feedback_preparation_sha256": FEEDBACK_SHA,
        "replay": source, "seed": prior.SEED, "arms_in_fixed_order": list(feedback.frozen.ARMS),
        "history": "render and save each actual step 0..15; use steps 5..15 and actual actions 6..15",
        "policy": "after exact original cp5 replay, max five one-step model-scored RGB feedback actions",
        "failure": "record partial model counts, events and frames; no retry or substitution",
        "candidate_and_score_ledger": "each complete V13 variant bank and score array saved with SHA",
        "instrumentation": "separate actual encoder, action_encoder and predictor pre-forward hooks, plus caller sample counters",
        "full_cost_certified": False,
        "selection_input": "actual RGB and actual action history plus frozen task and model only",
        "state_usage": "exact replay guard and audit logging only",
        "max_actions_per_arm": 5, "model_scoring_only_when_geometry_allows": True,
        "physical_execution_authorized_by_contract": False,
        "cuda_execution_authorized_by_contract": False,
        "P1_admission": False, "independent_confirmation": False,
        "implementation_sha256": {str(p.resolve()): base.sha(p) for p in paths},
    }


class RecordedScorer(feedback.CUDARealScorer):
    def __init__(self, model, task, expected_hash, store, arm, replan):
        super().__init__(model, task, expected_hash)
        self.store, self.arm, self.replan, self.iteration = store, arm, replan, 0
        self.rgb_encoder_hook = 0
        try:
            self.handles.append(model.encoder.register_forward_pre_hook(self._rgb_hook))
        except Exception:
            self.close()
            raise

    def _rgb_hook(self, *_args):
        self.rgb_encoder_hook += 1

    def finish(self):
        counts = super().finish()
        require(self.rgb_encoder_hook == counts["rgb_encode_calls"] == 3,
                "actual RGB encoder hook")
        return counts

    def score(self, variants):
        scores = super().score(variants)
        self.store.check()
        name = f"{self.arm}_plan_{self.replan:02d}_iteration_{self.iteration:02d}.npz"
        buffer = io.BytesIO()
        np.savez_compressed(buffer, variants=variants, scores=scores)
        raw = buffer.getvalue()
        with (self.store.path / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.store.emit("score_batch", arm=self.arm, replan=self.replan, iteration=self.iteration,
                        file=name, file_sha256=feedback.sha(raw),
                        counts=dict(self.counts), hooks=dict(self.hooks),
                        rgb_encoder_hook=self.rgb_encoder_hook)
        self.iteration += 1
        return scores


def branch(env, store, role, receipt, reset, goal_rgb, model, task, model_hash, row):
    arm = role["arm"]
    timeline = history.History()
    goal = np.asarray(receipt["history"]["goal"], np.float64)
    checkpoints = {p["step"]: p for p in receipt["history"]["frames"]}
    gate = replay.handoff.ReplayGate(role)
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
    controller = feedback.Feedback(context["frames"], context["actions"], frame_ids=context["frame_ids"])
    for i in range(1, 6):
        scorer = None
        context = timeline.context()
        store.emit("planning_started", arm=arm, replan=i,
                   frame_id=controller.frame_id, rgb_sha256=feedback.sha(controller.frames[-1].tobytes()),
                   actual_history_sha256=feedback.sha(controller.actions.tobytes()),
                   history_frame_ids=context["frame_ids"], history_pixel_sha256s=context["pixel_sha256"],
                   history_actual_actions=context["actions"].tolist())
        try:
            scorer = RecordedScorer(model, task, model_hash, store, arm, i)
            decision = controller.plan(scorer)
        except Exception:
            if scorer is not None:
                store.emit("planning_partial", arm=arm, replan=i,
                           counts=dict(scorer.counts), hooks=dict(scorer.hooks),
                           rgb_encoder_hook=scorer.rgb_encoder_hook)
            raise
        store.emit("planning_finished", arm=arm, replan=i, decision=decision,
                   counts=dict(scorer.counts), hooks=dict(scorer.hooks),
                   rgb_encoder_hook=scorer.rgb_encoder_hook)
        row["decisions"].append(decision)
        if decision["status"] != "proposal_only_wait_for_real_rgb":
            row["status"] = "rgb_abstained"
            return
        _, pixels, actual, stopped = step("feedback", i, decision["action"])
        if stopped:
            row["status"] = "native_early_stop"
            return
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
    model = None
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
            for role in base.read(prepared)["replay"]["source"]["roles"]:
                arm, env = role["arm"], None
                row = {"arm": arm, "status": "running", "transitions": [], "observations": [],
                       "decisions": [], "error": None}
                store.emit("branch_started", arm=arm)
                try:
                    cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
                    require(base.sha(inspect.getsourcefile(cls)) == base.native.ENV_SHA, "native SHA")
                    env = cls(render_mode="rgb_array", resolution=224, with_target=False)
                    branch(env, store, role, receipt, reset, arrays["goal_rgb"], model,
                           contract["task"], before, row)
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
              "journal_tail": store.events[-1]["sha256"], "P1_admission": False,
              "independent_confirmation": False, "classification": "development_only",
              "full_cost_certified": False}
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
