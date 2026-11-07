"""Replay the historical rendering backend exactly and compare preserved results."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_pointmaze_heldout_amplitude_encoding_20261107_codex as prior

ROOT = prior.ROOT
HISTORICAL = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.npz"
HISTORICAL_META = ROOT / "results/pc_real_camera_effect_dev_20261106_codex.json"
BACKEND = ROOT / "experiments/pc_real_camera_effect_dev_20261106_codex.py"
PROTOCOL = ROOT.parent / "docs/PC_POINTMAZE_ORIGINAL_RENDERER_REPLAY_PROTOCOL_20261107_CODEX.md"
REPLAY = ROOT / "results/pc_pointmaze_original_renderer_replay_20261107_codex.npz"
OUTPUT = ROOT / "results/pc_pointmaze_original_renderer_replay_20261107_codex.json"


def pixel_diff(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("pixel contract mismatch")
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {"mean": float(difference.mean()), "max": int(difference.max()),
            "nonzero": int(np.count_nonzero(difference)), "elements": int(difference.size)}


def run():
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    from experiments.a0_pointmaze_legacy_render_probe_20261102 import LEGACY, RUNTIME

    if OUTPUT.exists():
        raise FileExistsError("output exists")
    meta = json.loads(HISTORICAL_META.read_text())
    if (meta["payload_sha256"] != file_hash(HISTORICAL) or meta["code_sha256"] != file_hash(BACKEND)):
        raise ValueError("historical source or code drift")
    xml_text = official_xml()
    xml_sha = hashlib.sha256(xml_text.encode()).hexdigest()
    if xml_sha != meta["xml_sha256"]:
        raise ValueError("historical XML drift")
    with np.load(prior.prior.PHYSICAL, allow_pickle=False) as physical:
        states = physical["states"].copy()
    if not REPLAY.exists():
        with tempfile.TemporaryDirectory(prefix="pc_old_render_") as tmp:
            directory = Path(tmp)
            request, xml = directory / "request.npz", directory / "scene.xml"
            np.savez_compressed(request, states=states)
            xml.write_text(xml_text)
            env = os.environ.copy()
            env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
            env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                               "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
            subprocess.run([str(LEGACY / "bin/python"), str(BACKEND), "--backend", str(request), str(xml), str(REPLAY)],
                           env=env, check=True, timeout=240)
    with (np.load(HISTORICAL, allow_pickle=False) as old, np.load(REPLAY, allow_pickle=False) as replay,
          np.load(prior.FRAMES, allow_pickle=False) as new):
        if replay["traces"].shape != old["traces"].shape or replay["images"].shape != old["images"].shape:
            raise ValueError("replay shape mismatch")
        max_state = float(np.max(np.abs(replay["traces"] - old["traces"])))
        max_time = float(np.max(np.abs(replay["times"] - old["times"])))
        if max_state > 1e-9 or max_time > 1e-12:
            raise ValueError("physical trajectory changed")
        camera_diff = float(np.max(np.abs(replay["camera_default"] - old["camera_default"])))
        # Original action order is [0,+.1,-.1], new is [-.1,-.03,+.03,+.1].
        replay_selected = np.take(replay["images"][::2], [2, 1], axis=1)[:, :, :, -1]
        old_selected = np.take(old["images"][::2], [2, 1], axis=1)[:, :, :, -1]
        replay_to_new = pixel_diff(replay_selected, new["frames"][:, [0, 3]])
        replay_to_old = pixel_diff(replay["images"], old["images"])
        selected_to_old = pixel_diff(replay_selected, old_selected)
        metrics = {"all_historical_frames": replay_to_old, "selected_historical_frames": selected_to_old,
                   "selected_new_frames": replay_to_new, "max_state_error": max_state,
                   "max_time_error": max_time, "max_default_camera_error": camera_diff,
                   "historical_binding": str(old["binding"]), "replay_binding": str(replay["binding"]),
                   "historical_runtime": str(old["runtime"]), "replay_runtime": str(replay["runtime"])}
    output = {"metrics": metrics, "source_sha256": file_hash(HISTORICAL), "replay_sha256": file_hash(REPLAY),
              "backend_sha256": file_hash(BACKEND), "xml_sha256": xml_sha,
              "physical_sha256": file_hash(prior.prior.PHYSICAL), "new_frames_sha256": file_hash(prior.FRAMES),
              "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(Path(__file__)),
              "scope": "state-assisted rendering diagnostic, not independent environment certification"}
    with OUTPUT.open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
