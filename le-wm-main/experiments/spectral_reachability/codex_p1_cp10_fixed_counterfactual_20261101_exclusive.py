"""Paired physical last-packet interventions at three already-known cp10 states."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import codex_p1_cp10_rgb_refresh_control_20261101_exclusive as source_plan
from . import codex_p1_cp10_rgb_refresh_verify_20261101_exclusive as source_verify
from . import codex_p1_diag32_fixed_physical_20261101_exclusive as fixed

ROOT, engine = fixed.ROOT, fixed.engine
NAME = "codex_p1_cp10_fixed_counterfactual_20261101_exclusive"
SOURCE = ROOT / "results/codex_p1_cp10_rgb_refresh_runner_20261101_exclusive_run_CODEX01"
SOURCE_SHA = "d22ce41df39c1a5514e31754f0b6cdeb8ab6e425d92385c426c69828d0df990d"
PLAN = ROOT / "results/codex_p1_cp10_rgb_refresh_control_20261101_exclusive_PRESEAL_CODEX01.json"
PLAN_SHA = "e15c1768d3cb32d02a42a78522331fa49beca4859534182a37a6583f37a94459"
ROLES = ("selected", "zero", "same_bank_greedy")
IDENTITY = ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")
require, sha = engine.require, engine.sha
CAPS = {k: v * 6 for k, v in fixed.CAPS.items()}


def raw_actions(value, n):
    return engine.actions(value, n)


def check_cp10(expected, actual, planned, returned):
    require(all(expected[k] == actual[k] for k in IDENTITY), "cp10 state/snapshot/RGB/goal mismatch")
    require(np.array_equal(np.asarray(planned, dtype="<f4"), np.asarray(returned, dtype="<f4")),
            "cp10 first ten actions mismatch")


def schedule(entries):
    require(set(entries) == set(ROLES), "fixed role roster")
    rows = []
    for role in ROLES:
        entry = entries[role]
        first = raw_actions(entry["first10"], 10)
        for tail in ("hold", "optimized"):
            last = raw_actions(entry[tail + "_last"], 5)
            commands = raw_actions(first + last, 15)
            rows.append({"index": len(rows), "role": role, "tail": tail, "commands": commands,
                         "commands_float32_sha256": hashlib.sha256(
                             np.asarray(commands, dtype="<f4").tobytes()).hexdigest(),
                         "cp10": deepcopy(entry["cp10"])})
    return rows


def build():
    source_verify.verify(SOURCE, SOURCE_SHA)
    require(sha(PLAN) == PLAN_SHA and json.loads(PLAN.read_bytes()) == source_plan.contract(), "source plan")
    old_plan, report = json.loads(PLAN.read_bytes()), json.loads((SOURCE / "report.json").read_bytes())
    entries = {}
    for role in ROLES:
        rows = [r for r in report["rows"] if r["role"] == role and r["policy"] == "rgb_refresh"]
        require(len(rows) == 1 and rows[0]["control_returned"] >= 10
                and len(rows[0]["observations"]) > 10, "source cp10 available")
        row, candidates = rows[0], old_plan["candidates"][role]
        require(row["decisions"][0]["choice"] in ("hold", "optimized"), "cp5 source choice")
        cp5choice = row["decisions"][0]["choice"]
        first10 = raw_actions(row["executed_raw"][:10], 10)
        require(first10[:5] == candidates["prefix"] and first10[5:10] == candidates[cp5choice][0],
                "first ten sourced from sealed cp5 choice")
        entries[role] = {"first10": first10,
                         "hold_last": candidates["hold"][1],
                         "optimized_last": candidates["optimized"][1],
                         "cp10": {k: row["observations"][10][k] for k in IDENTITY},
                         "historical_cp10_model_scores": row["decisions"][1]["scores"]}
    prepared, _, root, _, _, _, _ = fixed.inputs()
    sources = {str(path): sha(path) for path in (
        Path(__file__).resolve(), ROOT / "tests" / ("test_" + NAME + ".py"), PLAN,
        ROOT / "experiments/spectral_reachability/codex_p1_cp10_fixed_counterfactual_verify_20261101_exclusive.py",
        SOURCE / "report.json", SOURCE / "journal.jsonl", Path(source_verify.__file__).resolve())}
    return {"schema": NAME + ".contract.v1", "schedule": schedule(entries),
            "scores_are_historical_not_recomputed": {r: entries[r]["historical_cp10_model_scores"] for r in ROLES},
            "root": {k: root[k] for k in IDENTITY}, "goal_binding": prepared["cp10_identity"],
            "sources": sources, "source_report_sha256": SOURCE_SHA, "source_plan_sha256": PLAN_SHA,
            "physical_caps_total": CAPS, "wall_seconds": 1800, "output_bytes_cap": engine.CAP,
            "disk_floor_bytes": engine.FLOOR, "neural_calls_new": 0,
            "stop": "native stop/truncation or original RGB/canvas guard, preserve unobserved remainder",
            "checkpoint_guard": "require same source cp10 state, full snapshot, pixel, goal and actual first 10 before step 11",
            "selection": "none; dispatch both presealed last packets in fixed hold then optimized order",
            "prior_outcomes_known": True, "development_only": True,
            "independent_confirmation": False, "P1_admission": False}


def execute(prep, expected, output):
    require(sha(prep) == expected and json.loads(prep.read_bytes()) == build(), "contract drift")
    data = json.loads(prep.read_bytes())
    prepared, context, root, _, _, _, _ = fixed.inputs()
    require(fixed.cp10.rebind(prepared, context) == data["goal_binding"], "goal binding")
    engine.NAME, engine.CAPS = NAME, CAPS
    store = engine.Store(output)
    store.put("execution_binding.json", {"prepared": str(prep), "sha256": expected})
    started = time.monotonic()
    def check():
        require(time.monotonic() - started < 1800, "wall cap")
        store.check()
    meter = engine.Meter(store, check)
    report = {"status": "partial_or_failed", "rows": [], "errors": [], "neural_calls": 0,
              "contract_sha256": expected, "development_only": True,
              "independent_confirmation": False, "P1_admission": False}
    env = None
    try:
        store.emit("run_started", contract_sha256=expected, caps=CAPS)
        for slot in data["schedule"]:
            ident = f"branch_{slot['index']:02}"
            row = deepcopy(engine.p3.roster()[0])
            row.update(episode_id=ident, branch=slot["index"], role=slot["role"], tail=slot["tail"],
                       status="running", transitions=[], observations=[])
            report["rows"].append(row)
            meter.episode, meter.region = ident, "setup"
            try:
                env = meter.call("constructor", lambda: engine.old.native_factory(meter))
                body, ref = env.unwrapped, context["references"][0]
                require(body._live_meter is meter and body._live_constructor_complete and body.relative
                        and body.action_scale == 100 and body.control_hz == 10 and body.dt == .01
                        and body.with_target is False and not body.render_action and body.render_size == 224
                        and np.array_equal(env.action_space.low, [-1, -1])
                        and np.array_equal(env.action_space.high, [1, 1]), "native ABI")
                if not engine.moving.setup(env, ref, context["plan"], row, meter, store):
                    row["status"] = "native_stopped_warmup"
                    continue
                root = row["last_observation"]
                require(all(root[k] == data["root"][k] for k in IDENTITY), "root identity")
                row["observations"].append(deepcopy(root))
                require(not body.eval_state(ref["goal"], np.asarray(root["state"]))[0], "initial stop")
                commands = np.asarray(slot["commands"], np.float32)
                for step, command in enumerate(commands, 1):
                    meter.region = "prefix" if step <= 10 else "suffix"
                    store.emit("branch_dispatch", branch=slot["index"], step=step, command=command.tolist())
                    obs, reward, terminated, truncated, info = env.step(command.copy())
                    row["control_returned"] += 1
                    actual, state = np.asarray(body.latest_action), np.asarray(obs["state"])
                    require(engine.old.kernel.moving.exact(actual, command)
                            and np.array_equal(info["goal_state"], ref["goal"])
                            and np.array_equal(info["goal"], ref["goal_rgb"]), "action/goal")
                    require(state.shape == (7,) and np.isfinite(state).all()
                            and engine.old.kernel.moving.exact(state, np.asarray(body._get_obs()))
                            and type(info.get("n_contacts")) is int and info["n_contacts"] >= 0, "state ABI")
                    engine.old.native_transition(body, ref["goal"], obs, reward, terminated, truncated, row)
                    event = {"branch": slot["index"], "step": step, "command": actual.tolist(),
                             "state": state.tolist(), "reward": float(reward), "terminated": bool(terminated),
                             "truncated": bool(truncated), "n_contacts": info["n_contacts"],
                             "metrics": deepcopy(row["final_metrics"])}
                    row["executed_raw"].append(actual.tolist())
                    row["transitions"].append(event)
                    store.emit("branch_returned", **event)
                    try:
                        _, receipt = engine.old.capture(env, obs, row, store, "plan", step)
                    except engine.old.kernel.moving.Rejected as exc:
                        require(str(exc) == "fresh RGB/canvas guard" and engine.guard_origin(exc), "guard origin")
                        row.update(status="censored_out_of_domain", censor_step=step)
                        break
                    row["observations"].append(deepcopy(receipt))
                    if terminated or truncated:
                        row.update(status="native_stopped", stop_step=step)
                        break
                    if step == 10:
                        check_cp10(slot["cp10"], receipt, commands[:10], row["executed_raw"])
                        store.emit("cp10_identity_sealed", branch=slot["index"],
                                   pixel_sha256=receipt["pixel_sha256"],
                                   commands_float32_sha256=slot["commands_float32_sha256"])
                else:
                    row["status"] = "completed15"
            finally:
                meter.region = "closing"
                if env is not None:
                    meter.call("close", env.close)
                    env = None
                store.put(ident + "_row.json", row)
            print(json.dumps({"branch": ident, "role": row["role"], "tail": row["tail"],
                              "status": row["status"], "steps": row["control_returned"]}), flush=True)
        require(all(sha(p) == h for p, h in data["sources"].items()) and sha(prep) == expected,
                "source/contract drift")
        require(not meter.pending and not meter.secondary, "physical accounting")
        report["status"] = "recorded_development"
    except BaseException as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
    finally:
        if env is not None:
            meter.region = "closing"
            try:
                meter.call("close", env.close)
            except BaseException as exc:
                report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
        store.finalizing = True
        store.emit("run_finished", status=report["status"], counts=meter.counts)
        report.update(counts=meter.counts, pending=meter.pending, secondary_errors=meter.secondary,
                      journal_sha256=sha(store.path / "journal.jsonl"), journal_chain_tail=store.previous,
                      wall_seconds=time.monotonic() - started)
        store.put("report.json", report, final=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sha256")
    args = parser.parse_args()
    require(args.tag.isalnum(), "tag")
    prep = ROOT / "results" / (NAME + "_PREP_" + args.tag + ".json")
    output = ROOT / "results" / (NAME + "_run_" + args.tag)
    if args.mode == "prepare":
        digest = fixed.blind.write_new(prep, build())
        print(json.dumps({"contract": str(prep), "sha256": digest}), flush=True)
        return 0
    require(args.sha256 is not None and not output.exists(), "explicit contract hash/exclusive output")
    with fixed.locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        result = execute(prep, args.sha256, output)
    print(json.dumps({"status": result["status"], "errors": result["errors"],
                      "report": str(output / "report.json"), "sha256": sha(output / "report.json")}), flush=True)
    return int(result["status"] != "recorded_development")


if __name__ == "__main__":
    raise SystemExit(main())
