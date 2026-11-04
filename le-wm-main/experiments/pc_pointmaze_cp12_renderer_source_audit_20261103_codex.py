"""Compare sealed RGB to current native replay, with active OpenGL identity."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from experiments import pc_causal_history_h10_suffix_20261103 as h
from experiments.pc_pointmaze_cp12_prefix_replay_preflight_20261103_codex import (
    SOURCE, SOURCE_SHA, compare_frame,
)
from experiments.pointmaze_native_rgb_bridge_20261102 import LEGACY, RUNTIME

ROOT = h.ROOT


def gl_info():
    code = r"""import ctypes,json,platform,sys,mujoco_py
m=mujoco_py.load_model_from_xml('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
s=mujoco_py.MjSim(m); s.render(32,32)
lib=ctypes.CDLL('libGL.so.1'); lib.glGetString.argtypes=[ctypes.c_uint]; lib.glGetString.restype=ctypes.c_char_p
def g(n):
    v=lib.glGetString(n)
    return v.decode() if v else None
print('AUDIT_JSON='+json.dumps({'gl_vendor':g(0x1f00),'gl_renderer':g(0x1f01),'gl_version':g(0x1f02),
 'mujoco_py_version':mujoco_py.__version__,'mujoco_py_runtime_version':mujoco_py.get_version(),
 'cymj':mujoco_py.cymj.__file__,'python':sys.version,'platform':platform.platform()}))"""
    env = os.environ.copy()
    env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
    env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                       "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
    p = subprocess.run([str(LEGACY / "bin/python"), "-c", code], env=env,
                       capture_output=True, text=True, check=True, timeout=30)
    lines = [x[len("AUDIT_JSON="):] for x in p.stdout.splitlines() if x.startswith("AUDIT_JSON=")]
    if len(lines) != 1:
        raise ValueError("missing OpenGL provenance")
    info = json.loads(lines[0])
    info["cymj_sha256"] = h.sha256_file(Path(info["cymj"]))
    info["runtime_library_sha256"] = h.sha256_file(RUNTIME / "bin/libmujoco210.so")
    info["parent_environment"] = {k: os.environ.get(k) for k in
                                  ("DISPLAY", "MUJOCO_GL", "PYOPENGL_PLATFORM", "LIBGL_ALWAYS_SOFTWARE",
                                   "LD_LIBRARY_PATH", "EGL_DEVICE_ID")}
    return info


def pixel_stats(actual, saved):
    row = compare_frame(actual, saved)
    signed = actual.astype(np.int16) - saved.astype(np.int16)
    mask = np.any(signed != 0, axis=2)
    ys, xs = np.nonzero(mask)
    row.update({"different_pixels": int(mask.sum()), "pixel_fraction": float(mask.mean()),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None,
                "signed_channel_mean": signed.mean(axis=(0, 1)).tolist(),
                "actual_rgb_mean": actual.mean(axis=(0, 1)).tolist(),
                "saved_rgb_mean": saved.mean(axis=(0, 1)).tolist()})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", choices=("sandbox", "host"), required=True)
    label = p.parse_args().label
    output = ROOT / f"results/pc_pointmaze_cp12_renderer_source_audit_{label}_20261103_CODEX.json"
    if output.exists():
        raise FileExistsError("refusing overwrite")
    for path, expected in ((SOURCE, SOURCE_SHA), (h.CAMERA, h.CAMERA_SHA), (h.OLD, h.OLD_SHA)):
        if h.sha256_file(path) != expected:
            raise ValueError("source changed")
    source, camera, old = (json.loads(p.read_text()) for p in (SOURCE, h.CAMERA, h.OLD))
    if h.sha256_file(SOURCE.with_suffix(".npz")) != source["arrays_sha256"]:
        raise ValueError("source arrays changed")
    info, rows = gl_info(), []
    with np.load(SOURCE.with_suffix(".npz"), allow_pickle=False) as data:
        for i, root in enumerate((8343, 8345, 8349, 8351)):
            if source["rows"][i]["root"] != root:
                raise ValueError("root drift")
            key = f"r{root}_a0_p2"
            saved, actions, start = data[key + "_history"], data[key + "_actions"], data["starts"][8*i]
            with h.NativeRGBBridge(start) as env:
                now = [env.initial_rgb]
                for action in actions[:12]:
                    now.append(env.step(action[None]))
            with h.NativeRGBBridge(start) as repeat:
                repeat_equal = np.array_equal(now[0], repeat.initial_rgb)
            initial = pixel_stats(now[0], saved[0])
            pos, _ = h.estimate_state(now[0], camera["metadata"], old["body_offset_from_static_scene"])
            old_pos, _ = h.estimate_state(saved[0], camera["metadata"], old["body_offset_from_static_scene"])
            states = []
            for t in (2, 7, 12):
                x = h.causal_state(now[t-1], now[t], camera["metadata"], old["body_offset_from_static_scene"], .05)
                y = h.causal_state(saved[t-1], saved[t], camera["metadata"], old["body_offset_from_static_scene"], .05)
                states.append({"t": t, "current": x.tolist(), "saved": y.tolist(), "difference": (x-y).tolist()})
            frames = [compare_frame(a, b) for a, b in zip(now, saved[:13], strict=True)]
            rows.append({"root": root, "branch": key, "initial_pixel_stats": initial,
                         "initial_current_state": pos.tolist(), "initial_saved_state": old_pos.tolist(),
                         "current_initial_repeat_exact": bool(repeat_equal), "frames": frames,
                         "causal_states": states})
            print(f"{label} {root}: {sum(r['equal'] for r in frames)}/13 exact", flush=True)
    report = {"label": label, "gl": info, "rows": rows, "source_sha256": SOURCE_SHA,
              "source_npz_sha256": source["arrays_sha256"], "bridge_sha256": h.sha256_file(
                  ROOT / "experiments/pointmaze_native_rgb_bridge_20261102.py"),
              "script_sha256": h.sha256_file(Path(__file__)),
              "counts": {"prefix_commands": 48, "extra_reset_repeat_tests": 4, "model_calls": 0},
              "scope": "renderer diagnostic, no physical failure claim"}
    with output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"{output} sha256={h.sha256_file(output)}", flush=True)


if __name__ == "__main__":
    main()
