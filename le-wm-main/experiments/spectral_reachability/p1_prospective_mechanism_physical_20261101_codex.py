"""Source-bound physical S/G/Z common/zero/own replay seam; not confirmation."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import e3_p2_cooperative_gpu_lock_codex_20261030_v1 as locking
from . import p1_constructive_moving_history_source_20261027_v1 as native_source
from . import p1_return_variants_analysis_20261101_main as evaluator
from . import p1_coherent_intervention_20261101_main as legacy

ROOT = Path(__file__).resolve().parents[2]
NAME = "p1_prospective_mechanism_physical_20261101_codex"
DOC = ROOT.parent / "docs/D2P_P1_PROSPECTIVE_MECHANISM_PHYSICAL_20261101_CODEX.md"
LOCK = ROOT / "results/e3_p2_host_cuda_cooperative.lock"
ROLES = ("S", "G", "Z")
TAILS = ("common", "zero", "own")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vector(value, shape, bounded=False):
    result = np.asarray(value, dtype=np.float32 if bounded else np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("invalid array shape or nonfinite value")
    if bounded and np.any(np.abs(result) > 1):
        raise ValueError("raw action outside [-1,1]")
    return result


def sha_string(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA-256")
    return value


def check_sources(sources):
    if not isinstance(sources, dict) or not sources:
        raise ValueError("source closure required")
    for name, expected in sources.items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or file_sha(path) != sha_string(expected):
            raise ValueError("source missing or hash changed: " + name)


def build_schedule(manifest):
    if manifest.get("schema") != NAME + ".input.v1":
        raise ValueError("input schema")
    root = manifest["root"]
    if type(root["reset_seed"]) is not int:
        raise ValueError("reset seed")
    vector(root["requested_reset"], (7,))
    vector(root["placeholder_goal"], (7,))
    vector(root["goal_state"], (7,))
    vector(root["root_state"], (7,))
    sha_string(root["root_dynamics_sha256"])
    vector(root["warmup_raw"], (10, 2), bounded=True)
    frames = root["warmup_rgb"]
    if not isinstance(frames, list) or len(frames) != 3:
        raise ValueError("three warmup RGB receipts")
    for frame in frames:
        sha_string(frame["pixel_sha256"])
        if frame["path"] not in manifest["sources"]:
            raise ValueError("warmup image absent from source closure")
    if set(manifest["roles"]) != set(ROLES):
        raise ValueError("S/G/Z role roster")
    if not isinstance(manifest.get("source_evidence"), list) or not manifest["source_evidence"]:
        raise ValueError("source evidence references required")
    if not isinstance(manifest.get("planning_evidence"), list) or not manifest["planning_evidence"]:
        raise ValueError("planning evidence references required")
    if any(p not in manifest["sources"] for p in manifest["source_evidence"] + manifest["planning_evidence"]):
        raise ValueError("evidence outside source closure")
    plans = manifest["roles"]
    for role in ROLES:
        row = plans[role]
        vector(row["prefix_raw"], (5, 2), bounded=True)
        vector(row["own_suffix_raw"], (15, 2), bounded=True)
        sha_string(row["cp5_pixel_sha256"])
        sha_string(row["cp5_state_identity_sha256"])
        sha_string(row["cp5_dynamics_sha256"])
    if np.any(vector(plans["Z"]["prefix_raw"], (5, 2), bounded=True) != 0):
        raise ValueError("Z prefix must be zero")
    common = vector(plans["S"]["own_suffix_raw"], (15, 2), bounded=True)
    schedule = []
    for role in ROLES:
        prefix = vector(plans[role]["prefix_raw"], (5, 2), bounded=True)
        for tail in TAILS:
            suffix = (common if tail == "common" else np.zeros((15, 2), np.float32)
                      if tail == "zero" else vector(plans[role]["own_suffix_raw"], (15, 2), bounded=True))
            schedule.append(dict(role=role, tail=tail,
                commands=np.concatenate((prefix, suffix)).tolist(),
                cp5_pixel_sha256=plans[role]["cp5_pixel_sha256"],
                cp5_state_identity_sha256=plans[role]["cp5_state_identity_sha256"],
                cp5_dynamics_sha256=plans[role]["cp5_dynamics_sha256"]))
    return schedule


def load_input(path, digest):
    path = Path(path)
    if not path.is_absolute() or file_sha(path) != sha_string(digest):
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(path.read_bytes())
    check_sources(manifest["sources"])
    schedule = build_schedule(manifest)
    return manifest, schedule


def run_branch(env, root, images, slot, output):
    row = dict(role=slot["role"], tail=slot["tail"], status="failed",
               checkpoints=[], executed_raw=[], physical_step_attempts=0, error=None,
               physical_cost=dict(reset_attempted=0, reset_returned=0,
                                  warmup_attempted=0, warmup_returned=0,
                                  render_attempted=0, render_returned=0))
    cost = row["physical_cost"]
    body = env.unwrapped
    try:
        cost["reset_attempted"] += 1
        env.reset(seed=root["reset_seed"])
        cost["reset_returned"] += 1
        cost["reset_attempted"] += 1
        env.reset(seed=0, options=dict(state=np.asarray(root["requested_reset"], np.float64),
                                      goal_state=np.asarray(root["placeholder_goal"], np.float64)))
        cost["reset_returned"] += 1
        for i in range(11):
            if i:
                action = np.asarray(root["warmup_raw"][i - 1], np.float32)
                cost["warmup_attempted"] += 1
                _, _, stop, trunc, _ = env.step(action.copy())
                cost["warmup_returned"] += 1
                if stop or trunc or not np.array_equal(body.latest_action, action):
                    raise ValueError("warmup stop or action mismatch")
            if i in (0, 5, 10):
                cost["render_attempted"] += 1
                image = env.render()
                cost["render_returned"] += 1
                if not np.array_equal(image, images[i // 5]):
                    raise ValueError("warmup RGB mismatch")
        goal = np.asarray(root["goal_state"], np.float64)
        native_source.attach_goal(body, goal, images[-1])

        def capture(step):
            state = np.asarray(body._get_obs(), np.float64).copy()
            vector(state, (7,))
            dynamics = native_source.snapshot(body)
            cost["render_attempted"] += 1
            pixels = np.asarray(env.render(), np.uint8).copy()
            cost["render_returned"] += 1
            if native_source.snapshot(body) != dynamics:
                raise ValueError("render changed physical dynamics")
            image_path = output / f"{slot['role']}_{slot['tail']}_{step:02}.png"
            Image.fromarray(pixels).save(image_path)
            native = evaluator.qualification.native(goal, state)
            point = dict(step=step, state=state.tolist(),
                state_identity_sha256=legacy.panel.saved.spec.digest(state.tolist()),
                dynamics=dynamics, dynamics_sha256=legacy.panel.saved.spec.digest(dynamics),
                metrics=dict(**evaluator.metrics(state, goal), joint_xy=native["joint_xy_error"],
                             native_success=native["initial_success"]),
                pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                png=str(image_path), file_sha256=file_sha(image_path))
            row["checkpoints"].append(point)
            if step == 0 and point["dynamics_sha256"] != root["root_dynamics_sha256"]:
                raise ValueError("actual root dynamics differ from receipt")
            if step == 5 and (point["pixel_sha256"] != slot["cp5_pixel_sha256"] or
                              point["state_identity_sha256"] != slot["cp5_state_identity_sha256"] or
                              point["dynamics_sha256"] != slot["cp5_dynamics_sha256"]):
                raise ValueError("actual cp5 differs from planning receipt")
            return state

        if not np.array_equal(capture(0), np.asarray(root["root_state"], np.float64)):
            raise ValueError("root state mismatch")
        for step, command in enumerate(np.asarray(slot["commands"], np.float32), 1):
            row["physical_step_attempts"] += 1
            obs, reward, stop, trunc, info = env.step(command.copy())
            row["executed_raw"].append(command.tolist())
            if not np.array_equal(body.latest_action, command) or not np.array_equal(info["goal_state"], goal):
                raise ValueError("executed action or goal mismatch")
            state = vector(obs["state"], (7,))
            outside = not bool(np.all((state[:4] >= 0) & (state[:4] <= 512)))
            if step % 5 == 0 or stop or trunc or outside:
                capture(step)
            if stop or trunc or outside:
                row.update(status="native_stopped" if stop or trunc else "canvas_censored",
                           terminated=bool(stop), truncated=bool(trunc), outside=outside,
                           stop_step=step)
                break
        else:
            row["status"] = "completed20"
    except Exception as exc:
        row["error"] = dict(type=type(exc).__name__, message=str(exc)[:1000])
    row["steps"] = len(row["executed_raw"])
    return row


def run(manifest_path, manifest_sha, output):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    manifest, schedule = load_input(manifest_path, manifest_sha)
    output = Path(output)
    if not output.is_absolute() or output.parent != ROOT / "results" or not output.name.startswith(NAME + "_") or output.exists():
        raise ValueError("exclusive output path")
    sources = dict(manifest["sources"])
    for path in (Path(manifest_path), Path(__file__).resolve(), DOC,
                 Path(native_source.__file__).resolve(), Path(evaluator.__file__).resolve(),
                 Path(locking.__file__).resolve(), Path(legacy.__file__).resolve(),
                 Path(inspect.getfile(evaluator.metrics)).resolve(),
                 Path(evaluator.qualification.__file__).resolve(),
                 Path(legacy.panel.saved.spec.__file__).resolve()):
        sources[str(path)] = file_sha(path)
    check_sources(sources)
    images = []
    for frame in manifest["root"]["warmup_rgb"]:
        with Image.open(frame["path"]) as image:
            data = np.asarray(image.convert("RGB"), np.uint8).copy()
        if data.shape != (224, 224, 3) or hashlib.sha256(data.tobytes()).hexdigest() != frame["pixel_sha256"]:
            raise ValueError("warmup RGB content mismatch")
        images.append(data)
    output.mkdir()
    protocol = dict(schema=NAME + ".protocol.v1", schedule=schedule, sources=sources,
                    source_and_plan_provenance="unverified", P1_admission=False,
                    independent_confirmation=False, neural_calls=0)
    with (output / "protocol.json").open("x") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    rows, incident = [], None
    for slot in schedule:
        if incident is None:
            try:
                check_sources(sources)
            except Exception as exc:
                incident = dict(role=slot["role"], tail=slot["tail"],
                                error=dict(type=type(exc).__name__, message=str(exc)[:1000]))
        if incident is None:
            env = None
            try:
                env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224,
                               with_target=False, render_action=False)
                row = run_branch(env, manifest["root"], images, slot, output)
            except Exception as exc:
                row = dict(role=slot["role"], tail=slot["tail"], status="failed",
                           checkpoints=[], executed_raw=[], physical_step_attempts=0, steps=0,
                           error=dict(type=type(exc).__name__, message=str(exc)[:1000]))
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception as exc:
                        row["status"] = "failed"
                        row["error"] = dict(type=type(exc).__name__, message="close: " + str(exc)[:900])
            if row["status"] == "failed":
                incident = dict(role=slot["role"], tail=slot["tail"], error=row["error"])
        else:
            row = dict(role=slot["role"], tail=slot["tail"], status="not_started",
                       checkpoints=[], executed_raw=[], physical_step_attempts=0, steps=0,
                       error=dict(type="PriorIncident", message="stopped after previous failed branch"))
        rows.append(row)
        with (output / (slot["role"] + "_" + slot["tail"] + ".json")).open("x") as stream:
            json.dump(row, stream, indent=2, allow_nan=False)
        print(json.dumps({k: row.get(k) for k in ("role", "tail", "status", "steps", "error")}), flush=True)
    try:
        check_sources(sources)
    except Exception as exc:
        if incident is None:
            incident = dict(role=None, tail=None, error=dict(type=type(exc).__name__, message=str(exc)[:1000]))
    report = dict(schema=NAME + ".report.v1", status="partial" if incident else "complete",
                  rows=rows, error=incident, sources=sources,
                  protocol_sha256=file_sha(output / "protocol.json"), neural_calls=0,
                  source_and_plan_provenance="unverified", P1_admission=False,
                  independent_confirmation=False)
    with (output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return not incident


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-physical", action="store_true", required=True)
    args = parser.parse_args()
    with locking.gpu_lock(LOCK):
        ok = run(args.manifest, args.manifest_sha256, args.output)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
