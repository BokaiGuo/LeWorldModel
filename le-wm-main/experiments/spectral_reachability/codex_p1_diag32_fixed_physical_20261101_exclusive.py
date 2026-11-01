"""One sealed known-root H3 physical development run; no neural inference."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

from . import codex_four_arm_prefix_intervention_20261030 as engine
from . import codex_p1_blind_uphill_bank_20261101_exclusive as blind
from . import codex_p1_bounded_joint_diag32_20261101_exclusive as diag
from . import codex_p1_bounded_joint_diag32_verify_20261101_exclusive as audit
from . import e3_cp10_four_arm_root0_20261031 as cp10
from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking

NAME = "codex_p1_diag32_fixed_physical_20261101_exclusive"
ROOT = blind.ROOT
DIAG_PREP = ROOT / "results/codex_p1_bounded_joint_diag32_20261101_exclusive_PREP_CODEXDIAG01.json"
DIAG_DIR = ROOT / "results/codex_p1_bounded_joint_diag32_20261101_exclusive_RUN_CODEXDIAG01"
DIAG_SHA = "08a6ca7dad502f80f72758262557199656ff62ad9b3b1b20c1d614ee1216ff05"
SOURCE_PREP = ROOT / "results" / blind.ITEMS[0][0]
SOURCE_REPORT = ROOT / "results/codex_p1_common_structured_cp10_host_20261101_exclusive_RUN_CODEX01/report.json"
SOURCE_REPORT_SHA = "6c49744a051ab327ca91c22f0aa5c19261eacff51c472fdf0b1cd4915d80f4cd"
MODEL_SHA = "4b12c01f5859f4b46bea478f3a8d5e2d3f516df456a622a3feec31bde7e33546"
CAPS = {key: value // 4 for key, value in engine.p3.contract()["physical_caps"].items()}
require, sha = engine.require, engine.sha


def fixed_selection(report):
    require(len(report["rows"]) == 2 and report["model_before"]["original_state_dict"] == MODEL_SHA
            and report["model_before"] == report["model_after"], "model binding")
    feasible = []
    for root in report["rows"]:
        for item in root["evaluations"]:
            if item["q1"] >= root["root_mse"] and item["q3"] < root["root_mse"]:
                feasible.append((root["slot"], item))
    require(len(feasible) == 1 and feasible[0][0] == 0, "unique feasibility")
    item = feasible[0][1]
    require(item["candidate"] == 4 and item["iteration"] == 13
            and report["rows"][0]["selected"] == item, "fixed candidate/iteration")
    commands = engine.actions(np.asarray(item["actions"], np.float32).reshape(15, 2), 15)
    return commands, item


def inputs():
    require(sha(DIAG_DIR / "report.json") == DIAG_SHA and sha(SOURCE_PREP) == blind.ITEMS[0][1]
            and sha(SOURCE_REPORT) == SOURCE_REPORT_SHA, "pinned source")
    audit.verify(DIAG_PREP, DIAG_DIR)
    report = json.loads((DIAG_DIR / "report.json").read_bytes())
    commands, selected = fixed_selection(report)
    task, _, _, _ = blind.load(0)
    prepared = json.loads(SOURCE_PREP.read_bytes())
    data = engine.p3.source_data(prepared["source"])
    rebound = cp10.rebind(deepcopy(prepared), data)
    ref = data["references"][0]
    require(rebound == prepared["cp10_identity"] and task["binding"]["root_sha256"] == rebound["root_sha256"]
            and engine.digest(task["goal"]) == rebound["goal_latent_sha256"]
            and prepared["source"]["expected_model_hash"] == MODEL_SHA, "root/goal/model")
    source_report = json.loads(SOURCE_REPORT.read_bytes())
    source_row = source_report["rows"][0]
    roots = [x for x in source_row["checkpoints"] if x["step"] == 0]
    require(source_row["episode_id"] == "s0_structured" and len(roots) == 1, "source root")
    root = roots[0]
    image_path = Path(root["path"])
    require(image_path.is_file() and not image_path.is_symlink() and sha(image_path) == root["file_sha256"], "root PNG")
    with Image.open(image_path) as image:
        pixels = np.asarray(image)
    require(pixels.shape == (224, 224, 3) and pixels.dtype == np.uint8
            and engine.old.array_sha(pixels) == root["pixel_sha256"]
            and root["pixel_sha256"] == json.loads((ROOT / "results" / blind.ITEMS[0][2] /
                 (blind.ITEMS[0][3] + "_b00_seal.json")).read_bytes())["identity"]["observation"]["pixel_sha256"],
            "root RGB identity")
    require(np.array_equal(np.asarray(root["goal"]), ref["goal"])
            and root["goal_rgb_sha256"] == rebound["goal_pixel_sha256"], "goal state/RGB")
    mean, std = np.asarray(task["action_mean"], np.float32), np.asarray(task["action_std"], np.float32)
    raw = np.asarray(commands, np.float32)
    require(mean.shape == std.shape == (2,) and np.isfinite(mean).all() and np.isfinite(std).all()
            and (std > 0).all() and (np.abs(raw) <= 1).all(), "normalizer/action domain")
    return prepared, data, root, commands, selected, mean, std


def build():
    prepared, data, root, commands, selected, mean, std = inputs()
    raw = np.asarray(commands, dtype="<f4")
    paths = [Path(__file__).resolve(), Path(engine.__file__).resolve(), Path(blind.__file__).resolve(),
             Path(diag.__file__).resolve(), Path(audit.__file__).resolve(), Path(cp10.__file__).resolve(),
             Path(locking.__file__).resolve(), DIAG_PREP, DIAG_DIR / "report.json", DIAG_DIR / "events.jsonl",
             SOURCE_PREP, SOURCE_REPORT, Path(root["path"])]
    bindings = dict(prepared["sources"])
    for path in paths:
        require(path.is_file() and not path.is_symlink(), "source missing")
        h = sha(path)
        require(str(path) not in bindings or bindings[str(path)] == h, "binding collision")
        bindings[str(path)] = h
    require(all(sha(path) == h for path, h in bindings.items()), "source drift")
    return {"schema": NAME + ".contract.v1", "commands": commands,
            "commands_float32_le_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "selection": {"root": 0, "candidate": 4, "iteration": 13,
                          "q0": json.loads((DIAG_DIR / "report.json").read_bytes())["rows"][0]["root_mse"],
                          "q1": selected["q1"], "q3": selected["q3"]},
            "root": {k: root[k] for k in ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")},
            "root_binding": prepared["cp10_identity"], "model_sha256": MODEL_SHA,
            "action_mean": mean.tolist(), "action_std": std.tolist(),
            "normalized_float32_le_sha256": hashlib.sha256(((raw - mean) / std).astype("<f4").tobytes()).hexdigest(),
            "physical_caps": CAPS, "wall_seconds": 600,
            "stop_rule": "stop on native terminated/truncated or original RGB/canvas guard; preserve censor",
            "physical_future_used_for_selection": False, "development_only": True,
            "neural_calls_new": 0, "P1_admission": False, "independent_confirmation": False,
            "prior_model_accounting": {"diag_predict": 1584, "diag_action_encoder": 1584, "diag_reverse": 512,
                                       "earlier_joint_predict": 432, "earlier_joint_action_encoder": 432,
                                       "earlier_joint_reverse": 128, "earlier_bank_and_rgb_cost": "not included"},
            "bindings": bindings}


def execute(prep_path, prep_sha, output):
    require(sha(prep_path) == prep_sha and json.loads(prep_path.read_bytes()) == build(), "contract drift")
    contract = json.loads(prep_path.read_bytes())
    _, data, expected_root, commands, _, _, _ = inputs()
    engine.NAME = NAME
    engine.CAPS = CAPS
    store = engine.Store(output)
    store.put("execution_binding.json", {"prepared": str(prep_path), "prepare_sha256": prep_sha})
    row = deepcopy(engine.p3.roster()[0])
    row.update(episode_id="fixed_root0_c4_i13", status="not_started", transitions=[], observations=[], endpoint=None)
    report = {"schema": NAME + ".report.v1", "status": "partial_or_failed", "row": row,
              "errors": [], "prepare_sha256": prep_sha, "neural_calls": 0, "P1_admission": False}
    start = time.monotonic()
    def check():
        require(time.monotonic() - start < 600, "wall cap")
        store.check()
    meter = engine.Meter(store, check)
    meter.episode = row["episode_id"]
    env = None
    try:
        store.emit("run_started", caps=CAPS, actions_sha256=contract["commands_float32_le_sha256"])
        env = meter.call("constructor", lambda: engine.old.native_factory(meter))
        body = env.unwrapped
        require(body._live_meter is meter and body._live_constructor_complete and body.with_target is False
                and body.render_action is False and body.render_size == 224 and body.relative is True
                and body.action_scale == 100 and body.control_hz == 10 and body.dt == .01, "native invariants")
        require(np.array_equal(np.asarray(env.action_space.low), [-1, -1])
                and np.array_equal(np.asarray(env.action_space.high), [1, 1]), "native action space")
        row["status"] = "running"
        ref = data["references"][0]
        if not engine.moving.setup(env, ref, data["plan"], row, meter, store):
            row["status"] = "native_stopped_warmup"
        else:
            root = row["last_observation"]
            require(all(root[k] == expected_root[k] for k in ("state", "snapshot", "pixel_sha256", "goal", "goal_rgb_sha256")),
                    "exact root/goal/RGB")
            row["observations"].append(deepcopy(root))
            row["initial_metrics"] = engine.old.kernel.metric(ref["goal"], root["state"])
            success, distance = body.eval_state(ref["goal"], np.asarray(root["state"]))
            require(not success and bool(success) == row["initial_metrics"]["native_success"]
                    and float(distance) == row["initial_metrics"]["native_distance"], "initial native stop")
            branch = {"index": 0, "commands": commands}
            for step, command in enumerate(np.asarray(commands, np.float32), 1):
                meter.region = "prefix" if step <= 5 else "suffix"
                store.emit("branch_dispatch", branch=0, step=step, command=command.tolist())
                obs, reward, terminated, truncated, info = env.step(command.copy())
                row["control_returned"] += 1
                actual = np.asarray(body.latest_action)
                require(engine.old.kernel.moving.exact(actual, command) and np.array_equal(info["goal_state"], ref["goal"])
                        and np.array_equal(info["goal"], ref["goal_rgb"]), "actual command/goal")
                state = np.asarray(obs["state"])
                require(state.shape == (7,) and np.isfinite(state).all()
                        and engine.old.kernel.moving.exact(state, np.asarray(body._get_obs()))
                        and type(info.get("n_contacts")) is int and info["n_contacts"] >= 0, "native state/contact")
                engine.old.native_transition(body, ref["goal"], obs, reward, terminated, truncated, row)
                event = {"branch": 0, "step": step, "command": actual.tolist(), "state": state.tolist(),
                         "n_contacts": info["n_contacts"], "reward": float(reward),
                         "terminated": bool(terminated), "truncated": bool(truncated),
                         "metrics": deepcopy(row["final_metrics"])}
                row["executed_raw"].append(actual.tolist())
                row["transitions"].append(event)
                store.emit("branch_returned", **event)
                try:
                    _, receipt = engine.old.capture(env, obs, row, store, "plan", step)
                except engine.old.kernel.moving.Rejected as exc:
                    require(str(exc) == "fresh RGB/canvas guard" and engine.guard_origin(exc), "guard exception")
                    row.update(status="censored_out_of_domain",
                               censor=engine.validate_censor(branch, event, row["last_observation"]), endpoint=None)
                    break
                row["observations"].append(deepcopy(receipt))
                if terminated or truncated:
                    row.update(status="native_stopped", endpoint=None)
                    break
            else:
                row.update(status="completed15", full_horizon=True, endpoint=deepcopy(row["last_observation"]))
        require(all(sha(path) == h for path, h in contract["bindings"].items()), "post-run source drift")
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
        report.update(counts=meter.counts, pending=meter.pending, secondary_errors=meter.secondary,
                      wall_seconds=time.monotonic()-start)
        store.put("row.json", row)
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
    prep = ROOT / "results" / (NAME + "_PREP_" + args.tag + ".json")
    output = ROOT / "results" / (NAME + "_run_" + args.tag)
    require(not os.path.lexists(prep) and not os.path.lexists(output), "exclusive outputs")
    digest = blind.write_new(prep, build())
    print(json.dumps({"contract": str(prep), "sha256": digest}), flush=True)
    with locking.gpu_lock(ROOT / "results/e3_p2_host_cuda_cooperative.lock"):
        report = execute(prep, digest, output)
    print(json.dumps({"status": report["status"], "output": str(output),
                      "report_sha256": sha(output / "report.json"), "errors": report["errors"]}), flush=True)
    return int(report["status"] != "recorded_development")


if __name__ == "__main__":
    raise SystemExit(main())
