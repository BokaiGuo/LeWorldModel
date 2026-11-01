"""Fixed six-branch physical execution of unified constrained CEM programs."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from . import codex_p1_diag32_fixed_physical_20261101_exclusive as root0
from . import codex_p1_root1_robust_feedback_20261101_exclusive as root1
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_unified_constrained_cem_consume_20261101_codex as consumer
from . import p1_unified_constrained_cem_20261101_codex as search

ROOT = search.ROOT
NAME = "p1_unified_constrained_cem_physical_20261101_codex"
CONSUMED = ROOT / "results/p1_unified_constrained_cem_consume_20261101_codex_MAIN01.json"
CONSUMED_SHA = "84a3b6b02657860fe53bb133e505ffc247f94cbbe555490e00919e6b35dd191b"
ROLES = ("selected", "zero_prefix", "same_bank_greedy")
MODEL_SHA = "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546"
engine = root0.engine
require, sha = engine.require, engine.sha


def root_inputs(slot):
    if slot == 0:
        prepared, data, expected, _, _, _, _ = root0.inputs()
        return prepared, data, expected
    if slot == 1:
        prepared, data, expected, _ = root1.inputs()
        return prepared, data, expected
    raise ValueError("unknown slot")


def build():
    require(sha(CONSUMED) == CONSUMED_SHA, "consumer hash")
    consumed = json.loads(CONSUMED.read_bytes())
    require(consumed == consumer.consume(), "consumer reconstruction drift")
    candidates = consumed["rows"]
    require(len(candidates) == 2 and all(row["selected_is_distinct_from_greedy"] for row in candidates),
            "two distinct selected programs required")
    bindings = {
        str(path.resolve()): sha(path.resolve())
        for path in (
            Path(__file__), Path(search.__file__), Path(consumer.__file__),
            Path(root0.__file__), Path(root1.__file__), Path(locking.__file__), CONSUMED,
            consumer.PREP, consumer.RUN / "report.json", consumer.RUN / "events.jsonl",
        )
    }
    roots = []
    for slot in (0, 1):
        prepared, _, expected = root_inputs(slot)
        require(prepared["source"]["expected_model_hash"] == MODEL_SHA, "model identity")
        for path, digest in prepared["sources"].items():
            require(sha(path) == digest, "source drift")
            bindings.setdefault(path, digest)
        roots.append({
            "slot": slot,
            "expected": {key: expected[key] for key in
                         ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")},
            "candidates": candidates[slot]["candidates"],
            "selection": {
                "root_mse": candidates[slot]["root_mse"],
                "selected_q1": candidates[slot]["selected"]["q1"],
                "selected_q3": candidates[slot]["selected"]["q3"],
                "greedy_q1": candidates[slot]["same_bank_greedy"]["q1"],
                "greedy_q3": candidates[slot]["same_bank_greedy"]["q3"],
            },
        })
    caps = {key: value * 6 for key, value in root0.CAPS.items()}
    return {
        "schema": NAME + ".contract.v1",
        "roots": roots,
        "roles": list(ROLES),
        "model_sha256": MODEL_SHA,
        "physical_caps_total": caps,
        "wall_seconds": 1200,
        "stop_rule": "native terminated/truncated or original RGB/canvas guard; no padding",
        "physical_future_used_for_selection": False,
        "new_neural_calls": 0,
        "development_only": True,
        "independent_confirmation": False,
        "fair_total_cost": False,
        "P1_admission": False,
        "bindings": bindings,
    }


def execute(prepared_path, prepared_sha, output):
    require(sha(prepared_path) == prepared_sha and json.loads(prepared_path.read_bytes()) == build(),
            "contract drift")
    contract = json.loads(prepared_path.read_bytes())
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
        "schema": NAME + ".report.v1",
        "status": "partial_or_failed",
        "rows": [],
        "errors": [],
        "prepare_sha256": prepared_sha,
        "new_neural_calls": 0,
        "development_only": True,
        "independent_confirmation": False,
        "fair_total_cost": False,
        "P1_admission": False,
    }
    env = None
    try:
        store.emit("run_started", prepare_sha256=prepared_sha, caps=engine.CAPS)
        for root_contract in contract["roots"]:
            slot = root_contract["slot"]
            _, data, expected = root_inputs(slot)
            ref = data["references"][slot]
            for role in ROLES:
                episode = f"root{slot}_{role}"
                row = deepcopy(engine.p3.roster()[0])
                row.update(
                    episode_id=episode,
                    source_slot=slot,
                    role=role,
                    status="not_started",
                    transitions=[],
                    observations=[],
                    endpoint=None,
                )
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
                    require(np.array_equal(np.asarray(env.action_space.low), [-1, -1])
                            and np.array_equal(np.asarray(env.action_space.high), [1, 1]),
                            "native action space")
                    row["status"] = "running"
                    if not engine.moving.setup(env, ref, data["plan"], row, meter, store):
                        row["status"] = "native_stopped_warmup"
                        continue
                    actual_root = row["last_observation"]
                    require(all(actual_root[key] == expected[key] == root_contract["expected"][key]
                                for key in root_contract["expected"]), "exact root/goal/RGB")
                    row["observations"].append(deepcopy(actual_root))
                    row["initial_metrics"] = engine.old.kernel.metric(ref["goal"], actual_root["state"])
                    success, distance = body.eval_state(ref["goal"], np.asarray(actual_root["state"]))
                    require(not success and bool(success) == row["initial_metrics"]["native_success"]
                            and float(distance) == row["initial_metrics"]["native_distance"],
                            "initial native state")
                    commands = np.asarray(root_contract["candidates"][role], np.float32).reshape(15, 2)
                    require(np.isfinite(commands).all() and np.max(np.abs(commands)) <= 1, "command domain")
                    row["planned_actions_sha256"] = hashlib.sha256(commands.astype("<f4").tobytes()).hexdigest()
                    for step, command in enumerate(commands, 1):
                        meter.region = "prefix" if step <= 5 else "suffix"
                        store.emit("branch_dispatch", role=role, slot=slot, step=step, command=command.tolist())
                        obs, reward, terminated, truncated, info = env.step(command.copy())
                        row["control_returned"] += 1
                        actual = np.asarray(body.latest_action)
                        require(engine.old.kernel.moving.exact(actual, command)
                                and np.array_equal(info["goal_state"], ref["goal"])
                                and np.array_equal(info["goal"], ref["goal_rgb"]), "actual command/goal")
                        state = np.asarray(obs["state"])
                        require(state.shape == (7,) and np.isfinite(state).all()
                                and engine.old.kernel.moving.exact(state, np.asarray(body._get_obs()))
                                and type(info.get("n_contacts")) is int and info["n_contacts"] >= 0,
                                "native state/contact")
                        engine.old.native_transition(body, ref["goal"], obs, reward, terminated, truncated, row)
                        event = {
                            "slot": slot,
                            "role": role,
                            "step": step,
                            "command": actual.tolist(),
                            "state": state.tolist(),
                            "n_contacts": info["n_contacts"],
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
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
                                           event, row["last_observation"]),
                                       endpoint=None)
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
                    "episode": episode,
                    "status": row["status"],
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
                report["status"] = "partial_or_failed"
        if meter.pending or meter.secondary:
            report["status"] = "partial_or_failed"
        report.update(
            counts=meter.counts,
            pending=meter.pending,
            secondary_errors=meter.secondary,
            wall_seconds=time.monotonic() - started,
        )
        store.emit("run_finished", status=report["status"], counts=meter.counts)
        report["journal_sha256"] = sha(store.path / "journal.jsonl")
        report["journal_chain_tail"] = store.previous
        store.put("report.json", report, final=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prepared = ROOT / "results" / f"{NAME}_PREP_{args.tag}.json"
    # engine.Store delegates to the inherited exclusive-output guard, whose
    # contract requires the lowercase ``_run_`` namespace.
    output = ROOT / "results" / f"{NAME}_run_{args.tag}"
    require(not os.path.lexists(prepared) and not os.path.lexists(output), "exclusive outputs")
    digest = search.write_new(prepared, build())
    print(json.dumps({"contract": str(prepared), "sha256": digest}), flush=True)
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prepared, digest, output)
    print(json.dumps({
        "status": report["status"],
        "output": str(output),
        "report_sha256": sha(output / "report.json"),
        "errors": report["errors"],
    }), flush=True)
    return int(report["status"] != "recorded_development")


if __name__ == "__main__":
    raise SystemExit(main())
