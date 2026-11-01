"""Physical replay of the fixed actual-cp5 feedback CEM plans."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_unified_constrained_cem_physical_20261101_codex as base
from . import p1_unified_cp5_feedback_cem_20261101_codex as planner

ROOT = base.ROOT
NAME = "p1_unified_cp5_feedback_physical_20261101_codex"
PLAN_PREP = ROOT / "results/p1_unified_cp5_feedback_cem_20261101_codex_PREP_MAIN02.json"
PLAN_PREP_SHA = "72a0fe1284b15c0767768ee2c68190279c5ecd6ad6bda3e260d0bbb6e4035c52"
PLAN_RUN = ROOT / "results/p1_unified_cp5_feedback_cem_20261101_codex_RUN_MAIN02"
PLAN_REPORT_SHA = "a2c4c42188e10afd8abdc30b355b8df2da37172c5f626b1bdeba4ce704706c36"
engine = base.engine
require, sha = engine.require, engine.sha


def build():
    require(sha(PLAN_PREP) == PLAN_PREP_SHA and sha(PLAN_RUN / "report.json") == PLAN_REPORT_SHA,
            "feedback plan identity")
    prepared = json.loads(PLAN_PREP.read_bytes())
    report = json.loads((PLAN_RUN / "report.json").read_bytes())
    require(prepared == planner.protocol() and report["status"] == "model_only_complete"
            and report["protocol_sha256"] == PLAN_PREP_SHA
            and report["post_cp5_physical_outcomes_used"] is False, "feedback plan contract")
    original = json.loads(base.CONSUMED.read_bytes())
    original_plans = {row["slot"]: row["candidates"] for row in original["rows"]}
    plan_rows = {(row["slot"], row["role"]): row for row in report["rows"]}
    roots = []
    for slot in (0, 1):
        _, _, expected = base.root_inputs(slot)
        candidates = {}
        decisions = {}
        for role in base.ROLES:
            row = plan_rows[(slot, role)]
            old = np.asarray(original_plans[slot][role], np.float32)
            require(old.shape == (3, 5, 2), "old plan")
            if row["status"] == "early_success_no_replan":
                require(slot == 1 and role == "same_bank_greedy" and row["stop_step"] == 4,
                        "early-success identity")
                plan = old
                decisions[role] = {"kind": "original_early_success", "stop_step_source": 4}
            else:
                prefix = np.asarray(row["prefix"], np.float32)
                tail = np.asarray(row["selected"]["actions"], np.float32)
                require(prefix.shape == (5, 2) and tail.shape == (2, 5, 2)
                        and np.array_equal(prefix, old[0]), "feedback prefix identity")
                plan = np.concatenate((prefix[None], tail), axis=0)
                decisions[role] = {
                    "kind": "actual_cp5_rgb_feedback_cem",
                    "cp5_pixel_sha256": row["cp5_pixel_sha256"],
                    "iteration": row["selected"]["iteration"],
                    "candidate": row["selected"]["candidate"],
                    "nominal_q": row["selected"]["nominal_q"],
                    "robust_score": row["selected"]["robust_score"],
                }
            require(np.isfinite(plan).all() and np.max(np.abs(plan)) <= 1, "bounded plan")
            candidates[role] = plan.tolist()
        roots.append({
            "slot": slot,
            "expected": {key: expected[key] for key in
                         ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")},
            "candidates": candidates,
            "decisions": decisions,
        })
    bindings = {}
    for path in (Path(__file__).resolve(), Path(base.__file__).resolve(), Path(planner.__file__).resolve(),
                 Path(locking.__file__).resolve(), PLAN_PREP, PLAN_RUN / "report.json", base.CONSUMED):
        bindings[str(path)] = sha(path)
    for source, digest in prepared["bindings"].items():
        require(source not in bindings or bindings[source] == digest, "binding collision")
        bindings[source] = digest
    require(all(sha(path) == digest for path, digest in bindings.items()), "source closure")
    return {
        "schema": NAME + ".contract.v1",
        "roots": roots,
        "roles": list(base.ROLES),
        "physical_caps_total": {key: value * 6 for key, value in base.root0.CAPS.items()},
        "wall_seconds": 1200,
        "stop_rule": "native terminated/truncated or original RGB/canvas guard; no padding",
        "feedback_boundary": 5,
        "post_cp5_physical_outcomes_used_for_selection": False,
        "new_neural_calls_during_physical": 0,
        "development_only": True,
        "independent_confirmation": False,
        "fair_total_cost": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def execute(prepared_path, prepared_sha, output):
    contract = build()
    require(sha(prepared_path) == prepared_sha
            and json.loads(prepared_path.read_bytes()) == contract, "contract drift")
    engine.NAME = NAME
    engine.CAPS = contract["physical_caps_total"]
    store = engine.Store(output)
    store.put("execution_binding.json", {"prepared": str(prepared_path), "prepare_sha256": prepared_sha})
    started = time.monotonic()

    def check():
        require(time.monotonic() - started < contract["wall_seconds"], "wall cap")
        store.check()

    meter = engine.Meter(store, check)
    report = {
        "schema": NAME + ".report.v1", "status": "partial_or_failed",
        "rows": [], "errors": [], "prepare_sha256": prepared_sha,
        "new_neural_calls_during_physical": 0, "development_only": True,
        "independent_confirmation": False, "fair_total_cost": False, "P1_admission": False,
    }
    env = None
    try:
        store.emit("run_started", prepare_sha256=prepared_sha, caps=engine.CAPS)
        for root in contract["roots"]:
            slot = root["slot"]
            _, data, expected = base.root_inputs(slot)
            ref = data["references"][slot]
            for role in base.ROLES:
                episode = f"root{slot}_{role}"
                row = deepcopy(engine.p3.roster()[0])
                row.update(episode_id=episode, source_slot=slot, role=role,
                           decision=root["decisions"][role], status="not_started",
                           transitions=[], observations=[], endpoint=None)
                report["rows"].append(row)
                meter.episode, meter.region = episode, "setup"
                env = None
                try:
                    env = meter.call("constructor", lambda: engine.old.native_factory(meter))
                    body = env.unwrapped
                    require(body._live_meter is meter and body._live_constructor_complete
                            and body.with_target is False and body.render_action is False
                            and body.render_size == 224 and body.relative is True
                            and body.action_scale == 100 and body.control_hz == 10 and body.dt == .01,
                            "native invariants")
                    row["status"] = "running"
                    if not engine.moving.setup(env, ref, data["plan"], row, meter, store):
                        row["status"] = "native_stopped_warmup"
                        continue
                    actual_root = row["last_observation"]
                    require(all(actual_root[key] == expected[key] == root["expected"][key]
                                for key in root["expected"]), "exact root/goal/RGB")
                    row["observations"].append(deepcopy(actual_root))
                    row["initial_metrics"] = engine.old.kernel.metric(ref["goal"], actual_root["state"])
                    commands = np.asarray(root["candidates"][role], np.float32).reshape(15, 2)
                    row["planned_actions_sha256"] = hashlib.sha256(
                        commands.astype("<f4").tobytes()).hexdigest()
                    for step, command in enumerate(commands, 1):
                        meter.region = "prefix" if step <= 5 else "suffix"
                        store.emit("branch_dispatch", role=role, slot=slot, step=step,
                                   command=command.tolist())
                        obs, reward, terminated, truncated, info = env.step(command.copy())
                        row["control_returned"] += 1
                        actual = np.asarray(body.latest_action)
                        require(engine.old.kernel.moving.exact(actual, command)
                                and np.array_equal(info["goal_state"], ref["goal"])
                                and np.array_equal(info["goal"], ref["goal_rgb"]), "actual command/goal")
                        state = np.asarray(obs["state"])
                        require(state.shape == (7,) and np.isfinite(state).all()
                                and type(info.get("n_contacts")) is int and info["n_contacts"] >= 0,
                                "native state/contact")
                        engine.old.native_transition(body, ref["goal"], obs, reward,
                                                     terminated, truncated, row)
                        event = {
                            "slot": slot, "role": role, "step": step,
                            "command": actual.tolist(), "state": state.tolist(),
                            "n_contacts": info["n_contacts"], "reward": float(reward),
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "metrics": deepcopy(row["final_metrics"]),
                        }
                        row["executed_raw"].append(actual.tolist())
                        row["transitions"].append(event)
                        store.emit("branch_returned", **event)
                        try:
                            _, receipt = engine.old.capture(env, obs, row, store, "plan", step)
                        except engine.old.kernel.moving.Rejected as exc:
                            require(str(exc) == "fresh RGB/canvas guard" and engine.guard_origin(exc),
                                    "guard exception")
                            row.update(status="censored_out_of_domain",
                                       censor=engine.validate_censor(
                                           {"index": 0, "commands": commands.tolist()},
                                           event, row["last_observation"]), endpoint=None)
                            break
                        row["observations"].append(deepcopy(receipt))
                        if step == 5:
                            row["prefix_metrics"] = deepcopy(row["final_metrics"])
                        if terminated or truncated:
                            row.update(status="native_stopped", stop_step=step, endpoint=None)
                            break
                    else:
                        row.update(status="completed15", full_horizon=True,
                                   endpoint=deepcopy(row["last_observation"]))
                finally:
                    meter.region = "closing"
                    if env is not None:
                        meter.call("close", env.close)
                        env = None
                    store.put(episode + "_row.json", row)
                print(json.dumps({
                    "episode": episode, "status": row["status"],
                    "returned": row["control_returned"],
                    "native_success": row.get("final_metrics", {}).get("native_success"),
                }), flush=True)
        require(all(sha(path) == digest for path, digest in contract["bindings"].items()),
                "post-run source drift")
        report["status"] = "recorded_development"
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
    finally:
        meter.region = "closing"
        store.finalizing = True
        if env is not None:
            try:
                meter.call("close", env.close)
            except BaseException as exc:
                report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
        if meter.pending or meter.secondary:
            report["status"] = "partial_or_failed"
        report.update(counts=meter.counts, pending=meter.pending,
                      secondary_errors=meter.secondary,
                      wall_seconds=time.monotonic() - started)
        store.emit("run_finished", status=report["status"], counts=meter.counts)
        report["journal_sha256"] = sha(store.path / "journal.jsonl")
        report["journal_chain_tail"] = store.previous
        store.put("report.json", report, final=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    output = ROOT / "results" / f"{NAME}_run_{args.tag}"
    require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive outputs")
    digest = planner.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prepared, digest, output)
    print(json.dumps({"status": report["status"], "output": str(output),
                      "report_sha256": sha(output / "report.json"),
                      "errors": report["errors"]}), flush=True)
    return int(report["status"] != "recorded_development")


if __name__ == "__main__":
    raise SystemExit(main())
