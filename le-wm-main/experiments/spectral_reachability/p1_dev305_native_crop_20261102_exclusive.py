"""Fixed 512-render crop diagnostic; no simulator state, actions, or model calls."""
import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import p1_rgb_goal_reacquire_20261102_exclusive as gate
from . import p1_dev305_goal_recapture_live_20261102_exclusive as old

CROP = (32, 64, 480, 512)
PROTOCOL = old.ROOT.parent / "docs/D2P_PUSHT_DEV305_NATIVE_CROP_PROTOCOL_20261102_EXCLUSIVE.md"
PROTOCOL_SHA = "95f53f3d11e8942ea48295df321d2328f75201d877222580d334fc1071279aa1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def image(value, size):
    arr = np.asarray(value)
    if arr.shape != (size, size, 3) or arr.dtype != np.uint8:
        raise ValueError("invalid native RGB")
    return arr.copy()


def masks512(pixels):
    pixels = image(pixels, 512)
    r, g, b = pixels.astype(np.int16).transpose(2, 0, 1)
    return {"object": (g-r > 5) & (b-g > 5) & (b-r < 80) & (r > 70) & (b < 235),
            "pusher": (b-r > 100) & (b-g > 60) & (b > 200)}


def prepare_crop(pixels):
    pixels = image(pixels, 512)
    x0, y0, x1, y1 = CROP
    roi = np.zeros((512, 512), dtype=bool)
    roi[y0:y1, x0:x1] = True
    coverage = {}
    for name, mask in masks512(pixels).items():
        total = int(mask.sum())
        outside = int((mask & ~roi).sum())
        coverage[name] = {"total": total, "outside": outside, "contained": total > 0 and outside == 0}
    cropped = cv2.resize(pixels[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_LINEAR)
    screen = gate.guard.inspect(cropped)
    return cropped, {"coverage": coverage, "guard": screen,
                     "eligible": all(x["contained"] for x in coverage.values()) and screen["eligible"]}


def store(directory, name, pixels):
    path = directory / name
    with path.open("xb") as f:
        Image.fromarray(pixels).save(f, format="PNG")
    return {"file": name, "file_sha256": sha(path.read_bytes()), "pixel_sha256": sha(pixels.tobytes())}


def run(directory, factory, goal_sha, current_sha):
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    report = {"schema": "p1_dev305_native_crop_exclusive.v1", "status": "engineering_failure",
              "seed": old.SEED, "crop_xyxy": list(CROP), "frames": {}, "events": [],
              "action": None, "selector_called": False, "physical_steps": 0,
              "reset_calls": 0, "neural_calls": 0, "p1_admission": False}
    envs = []
    start = time.monotonic_ns()
    try:
        if not gate._guard_intact():
            raise ValueError("guard source mismatch")
        report["protocol_sha256"] = sha(PROTOCOL.read_bytes())
        if report["protocol_sha256"] != PROTOCOL_SHA:
            raise ValueError("protocol hash mismatch")
        for size in (224, 512):
            env = factory(size)
            envs.append(env)
            report["reset_calls"] += 1
            report["events"].append({"event": "reset_started", "resolution": size})
            _, info = env.reset(seed=old.SEED)
            goal = image(info["goal"], size)
            current = image(env.render(), size)
            report["frames"][f"goal_{size}"] = store(directory, f"goal_{size}.png", goal)
            report["frames"][f"current_{size}"] = store(directory, f"current_{size}.png", current)
            if size == 224:
                goal224, current224 = goal, current
            else:
                goal512, current512 = goal, current
        if sha(goal224.tobytes()) != goal_sha or sha(current224.tobytes()) != current_sha:
            raise ValueError("native 224 does not match archived dev305")
        for low, high in ((goal224, goal512), (current224, current512)):
            if not np.array_equal(low, cv2.resize(high, (224, 224), interpolation=cv2.INTER_LINEAR)):
                raise ValueError("native resolutions mismatch")
        for name, frame in (("goal", goal512), ("current", current512)):
            cropped, check = prepare_crop(frame)
            report["frames"][name + "_crop"] = store(directory, name + "_crop.png", cropped)
            report[name + "_crop_check"] = check
        report["baseline"] = {"goal": gate.guard.inspect(goal224), "current": gate.guard.inspect(current224)}
        ok = report["goal_crop_check"]["eligible"] and report["current_crop_check"]["eligible"]
        report.update(status="completed", archive_binding="matched",
                      decision="crop_observation_eligible_unvalidated" if ok else "blocked",
                      reason=None if ok else "crop_coverage_or_original_guard_failed",
                      scope="deterministic simulator RGB development; no planning or physical outcome")
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
    finally:
        for env in envs:
            try:
                env.close()
            except Exception as exc:
                report["status"] = "engineering_failure"
                report["events"].append({"event": "close_failed", "type": type(exc).__name__})
        report["elapsed_ns"] = time.monotonic_ns() - start
        with (directory / "report.json").open("x") as f:
            json.dump(report, f, indent=2, allow_nan=False)
            f.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from stable_worldmodel.envs.pusht.env import PushT
    if sha(Path(inspect.getsourcefile(PushT)).read_bytes()) != old.ENV_SHA:
        raise ValueError("environment source mismatch")
    goal, current = old.archive_refs()
    result = run(args.output, lambda n: PushT(resolution=n, render_mode="rgb_array",
                                              with_target=False), goal, current)
    print(json.dumps({"status": result["status"], "decision": result.get("decision"),
                      "error": result.get("error")}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
