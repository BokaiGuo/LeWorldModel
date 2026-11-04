"""Repeat native PushT goal rendering from a fixed seed, without stepping."""
import argparse
import hashlib
import inspect
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_rgb_goal_reacquire_20261102_exclusive as barrier

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "results/p1_dev305_diverse_cem_20261102_CAPTURE_DEV01"
BLIND_SHA = "852a8cf856f61b9118ff8335190b182758e9a36ef098d05c1eb1221af5d348cc"
ENV_SHA = "c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2"
SEED = 301000005


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive_refs():
    raw = (CAPTURE / "blind.json").read_bytes()
    if digest(raw) != BLIND_SHA:
        raise ValueError("archived blind hash mismatch")
    data = json.loads(raw)
    if data["seed"] != SEED or len(data["images"]) != 3:
        raise ValueError("archived roster mismatch")
    return data["goal"]["pixel_sha256"], data["images"][2]["pixel_sha256"]


def rgb(value):
    pixels = np.asarray(value)
    if pixels.shape != (224, 224, 3) or pixels.dtype != np.uint8:
        raise ValueError("expected uint8 RGB224")
    return pixels.copy()


def save_frame(directory, name, pixels):
    path = directory / name
    with path.open("xb") as stream:
        Image.fromarray(pixels).save(stream, format="PNG")
    return {"file": name, "file_sha256": digest(path.read_bytes()),
            "pixel_sha256": digest(pixels.tobytes())}


def run(directory, factory, expected_goal, expected_current, *, clock=time.monotonic_ns):
    """New instances produce goal twice; render current once; never call step."""
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    events, envs, images = [], [], []
    session = uuid.uuid4().hex
    result = {"schema": "p1_dev305_goal_recapture_live_exclusive.v1",
              "status": "engineering_failure", "events": events, "frames": [],
              "seed": SEED, "archive_blind_sha256": BLIND_SHA,
              "scope": "deterministic simulator reset renders, not independent camera samples",
              "physical_steps": 0, "neural_calls": 0, "p1_admission": False}
    started = clock()
    try:
        for n in range(2):
            env = factory()
            envs.append(env)
            events.append({"event": "reset_started", "instance": n + 1})
            _, info = env.reset(seed=SEED)
            goal = rgb(info["goal"])
            stamp = clock()
            images.append(goal)
            events.append({"event": "goal_rendered_by_reset", "instance": n + 1,
                           "monotonic_ns": stamp})
            result["frames"].append(save_frame(directory, f"goal_{n + 1}.png", goal))
        current = rgb(envs[1].render())
        current_time = clock()
        result["frames"].append(save_frame(directory, "current.png", current))
        events.append({"event": "current_rendered", "instance": 2, "monotonic_ns": current_time})
        hashes = [digest(p.tobytes()) for p in (*images, current)]
        result["archive_comparison"] = {"goal_matches": [h == expected_goal for h in hashes[:2]],
                                        "current_matches_warmup10": hashes[2] == expected_current}
        if not all(result["archive_comparison"]["goal_matches"]):
            raise ValueError("reacquired goal differs from archived task RGB")
        if not result["archive_comparison"]["current_matches_warmup10"]:
            raise ValueError("reset observation differs from archived warmup RGB")
        # Metadata is generated at each actual renderer call, not read from the archive.
        common = {"session_id": session, "camera_id": "pusht_rgb224_native",
                  "target_id": "dev305_seed_301000005"}
        packets = [{**common, "pixels": image, "capture_id": f"{session}:{n}",
                    "sequence": n, "captured_ns": stamp}
                   for n, (image, stamp) in enumerate(zip(
                       (*images, current),
                       (events[1]["monotonic_ns"], events[3]["monotonic_ns"], current_time)), 1)]
        calls = []

        def no_candidate(*_):
            calls.append(True)
            raise RuntimeError("physical candidate selection not authorized by this diagnostic")

        decision = barrier.choose(packets[:2], packets[2], now_ns=clock(),
                                  max_age_ns=60_000_000_000, selector=no_candidate)
        result.update(status="completed", decision=decision, selector_calls=len(calls),
                      outcome="no_action_dispatched_not_a_physical_outcome",
                      acquisition_metadata={"session_id": session,
                                            "capture_ids": [p["capture_id"] for p in packets],
                                            "monotonic_ns": [p["captured_ns"] for p in packets],
                                            "ttl_ns": 60_000_000_000,
                                            "ttl_status": "diagnostic_only_not_calibrated"})
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
    finally:
        for env in envs:
            try:
                env.close()
            except Exception as exc:
                result["status"] = "engineering_failure"
                events.append({"event": "close_failed", "type": type(exc).__name__})
        result["elapsed_ns"] = clock() - started
        with (directory / "report.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from stable_worldmodel.envs.pusht.env import PushT

    actual_env_sha = digest(Path(inspect.getsourcefile(PushT)).read_bytes())
    if actual_env_sha != ENV_SHA:
        raise ValueError("native environment source hash mismatch")
    expected_goal, expected_current = archive_refs()
    result = run(args.output, lambda: PushT(render_mode="rgb_array", resolution=224,
                                           with_target=False), expected_goal, expected_current)
    print(json.dumps({"status": result["status"], "decision": result.get("decision"),
                      "error": result.get("error")}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
