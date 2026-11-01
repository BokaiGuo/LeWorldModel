"""Translated-root H4 building blocks; not an executable experiment launcher.

This module never executes a model.  The caller must wire both functions into
the existing runner in one fresh process; see ``wiring_status``.
"""
import base64
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results/p1_centered_witness_20261101_main_RUN_MAIN01/report.json"
TARGET = ROOT / "results/p1_centered_target_cuda_20261101_main_MAIN01.json"
METRIC = ROOT / "results/p1_centered_metric_cuda_20261101_main_MAIN01.json"
TARGET_SHA = "801c3b089cb9af71901c638b9197b25aa84573e0befefec8abe8a3c068d62ef3"
METRIC_SHA = "ef9ae31f143a18d73432fb45bf861db13f9f852baa776fe977962ec9cb2a4818"
NAME = "p1_centered_h4_parfit_20261101_adapter"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array(value, shape):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), f"finite shape {shape}")
    return result


def load():
    require(sha(TARGET) == TARGET_SHA and sha(METRIC) == METRIC_SHA, "target/metric SHA")
    target, metric, report = (json.loads(p.read_bytes()) for p in (TARGET, METRIC, REPORT))
    require(target["schema"] == "p1_centered_target_cuda_20261101_main"
            and target["status"] == "complete" and len(target["rows"]) == 1,
            "centered target receipt")
    require(metric["schema"] == "p1_centered_metric_cuda_20261101_main"
            and metric["status"] == "measured_not_control_validated"
            and metric["target_sha256"] == TARGET_SHA, "centered metric receipt")
    require(target["model_before"] == target["model_after"] == metric["model_before"]
            == metric["model_after"], "encoder/model identity")
    require(target["sources"].get(str(REPORT)) == sha(REPORT), "report source binding")
    require(report["status"] == "recorded" and report["task_admitted"] is True
            and report["warmup_translation_verified"] is True
            and all(report["checks"].values()), "original task admission")
    row = target["rows"][0]
    require(row["root"] == 0 and row["translated_known_root"] is True
            and all(row["task_eligibility"].values()), "target admission")
    require([x["step"] for x in report["history"]] == [0, 5, 10]
            and row["history_pixel_sha256"] == [x["pixel_sha256"] for x in report["history"]],
            "history RGB identity")
    require(np.array_equal(array(row["root_state"], (7,)), array(report["rows"][0]["state"], (7,)))
            and np.array_equal(array(row["goal_state"], (7,)), array(report["rows"][-1]["state"], (7,))),
            "root and goal state identity")
    pixels = np.frombuffer(base64.b64decode(row["pixels_uint8_base64"], validate=True),
                           dtype=np.uint8).reshape(224, 224, 3).copy()
    require(hashlib.sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"]
            == report["rows"][-1]["pixel_sha256"], "target RGB")
    array(row["history_latents"], (3, 192))
    require(np.array_equal(array(row["latent"], (192,)), array(metric["goal_latent"], (192,))),
            "metric goal latent")
    array(metric["sqrt_matrix"], (192, 192))
    array(report["requested_reset"], (7,))
    array(report["placeholder_goal"], (7,))
    array(report["raw_history"], (10, 2))
    return report, row, metric, pixels


def rebind(prepared, data, runner):
    """Bind copied live data and task; caller must also install translated_setup."""
    report, row, metric, pixels = load()
    prepared, data = deepcopy(prepared), deepcopy(data)
    ref, task = data["references"][0], prepared["source"]["tasks"][0]
    ref["goal"] = np.asarray(row["goal_state"], dtype=np.float64)
    ref["goal_rgb"] = pixels
    ref["raw_history"] = np.asarray(report["raw_history"], dtype=np.float32)
    ref["states"] = np.asarray([r["state"] for r in report["history"]], dtype=np.float64)
    ref["root_sha256"] = runner.p3.moving.source.root_hash(ref["states"][-1], ref["goal"])
    task["goal"] = deepcopy(row["latent"])
    task["root_context"]["latent_history"] = deepcopy(row["history_latents"])
    binding = task["binding"]
    binding["goal_sha256"] = runner.digest(task["goal"])
    binding["root_sha256"] = ref["root_sha256"]
    binding["task_id"] = runner.digest([NAME, binding["task_id"], TARGET_SHA])
    binding["task_sha256"] = runner.digest([NAME, "task", binding["task_id"]])
    binding["task_contract_sha256"] = runner.digest([NAME, "contract", TARGET_SHA, METRIC_SHA])
    runner.p3.core.validate_task(task)
    return prepared, data


def translated_setup(env, ref, source_plan, row, meter, store, moving):
    """Real reset and actual warmup, checking every captured 0/5/10 RGB/state."""
    del source_plan
    report, target, _, _ = load()
    old = moving.old
    body = env.unwrapped
    meter.region = "setup"
    env.reset(seed=report["reset_seed"])
    obs, _ = env.reset(seed=0, options={
        "state": np.asarray(report["requested_reset"], dtype=np.float64).copy(),
        "goal_state": np.asarray(report["placeholder_goal"], dtype=np.float64).copy()})

    def capture(step):
        image, receipt = old.capture(env, obs, row, store, "warmup", step)
        expected = report["history"][(0, 5, 10).index(step)]
        require(hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
                == expected["pixel_sha256"] == target["history_pixel_sha256"][(0, 5, 10).index(step)],
                "actual warmup RGB mismatch")
        require(np.allclose(np.asarray(obs["state"], dtype=np.float64),
                            expected["state"], atol=1e-7, rtol=0), "actual warmup state mismatch")
        return receipt

    capture(0)
    meter.region = "warmup"
    for step, command in enumerate(old.kernel.raw_history(ref["raw_history"]), 1):
        obs, _, terminated, truncated, _ = env.step(command.copy())
        row["warmup_returned"] += 1
        value = np.asarray(body.latest_action).copy()
        require(old.kernel.moving.exact(value, command), "actual warmup action")
        store.emit("warmup_return", episode_id=row["episode_id"], step=step,
                   command=value.tolist(), terminated=bool(terminated), truncated=bool(truncated))
        if terminated or truncated:
            row.update(status="native_stopped_warmup", terminated=bool(terminated),
                       truncated=bool(truncated))
            return False
        if step in (5, 10):
            capture(step)
    meter.region = "goal_bind"
    state, pose = old.kernel.moving.snapshot(body), np.asarray(body.goal_pose).copy()
    meter.call("goal_bind", lambda: old.kernel.moving.attach_goal(body, ref["goal"], ref["goal_rgb"]))
    require(old.kernel.moving.snapshot(body) == state and np.array_equal(pose, body.goal_pose),
            "goal bind changed native state")
    _, obs0 = old.capture(env, obs, row, store, "plan", 0)
    require(obs0["pixel_sha256"] == target["history_pixel_sha256"][-1]
            and moving.source.root_hash(moving.source.vector(obs["state"]), ref["goal"])
            == ref["root_sha256"], "translated root/goal mismatch")
    row["initial_exact"] = True
    return True


def wiring_status():
    return {"status": "building_blocks_only_not_runnable",
            "missing": ["fresh-process H4 runner integration",
                        "actual boundary-0 encoding equality to target history_latents[-1]",
                        "runtime metric and source closure binding",
                        "CPU fixture and host CUDA verification"],
            "independent_confirmation": False, "P1_admission": False}


if __name__ == "__main__":
    report, row, _, _ = load()
    print(json.dumps(dict(wiring_status(), task_admitted=report["task_admitted"],
                          target_sha256=TARGET_SHA, metric_sha256=METRIC_SHA,
                          history_pixel_sha256=row["history_pixel_sha256"],
                          cuda_used=False), sort_keys=True))
