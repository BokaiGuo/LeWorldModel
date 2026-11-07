"""Persistent legacy PointMaze executor exposing RGB and state-assisted proprio.

Development bridge: camera derived from an exposed archive frame. This is
not an authenticated benchmark, success evaluator, or planning algorithm.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / ".legacy_mujoco210_py310"
RUNTIME = LEGACY / "vendor/mujoco-2.1.2"


def _worker(xml_path: Path) -> None:
    import mujoco_py

    model = mujoco_py.load_model_from_xml(xml_path.read_text())
    sim = mujoco_py.MjSim(model)
    sim.render(224, 224)
    camera = sim.render_contexts[0].cam
    camera.azimuth, camera.elevation = 90, -90
    camera.lookat[0], camera.lookat[1] = 3.0, 3.0
    target = model.site_name2id("target_site")
    count = 0
    initialized = False

    def observation() -> dict[str, Any]:
        image = np.ascontiguousarray(sim.render(224, 224), dtype=np.uint8)
        proprio = np.concatenate((sim.data.qpos[:2], sim.data.qvel[:2])).tolist()
        return {"step": count, "proprio": proprio, "rgb_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                "rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest()}

    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request.get("op")
            if op == "reset" and not initialized:
                state = np.asarray(request["initial_state"], dtype=np.float64)
                if state.shape != (4,) or not np.isfinite(state).all():
                    raise ValueError("initial_state must be four finite numbers")
                sim.reset()
                sim.data.qpos[:], sim.data.qvel[:] = state[:2], state[2:]
                sim.forward()
                sim.data.site_xpos[target] = [11, 11, 0]
                initialized, count = True, 0
                response = observation()
            elif op == "step" and initialized:
                actions = np.asarray(request["actions"], dtype=np.float64)
                if actions.ndim != 2 or actions.shape[1] != 2 or not 1 <= len(actions) <= 5:
                    raise ValueError("actions must have shape [1..5,2]")
                if not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                    raise ValueError("actions must be finite in [-1,1]")
                for action in actions:
                    sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                    sim.forward()
                    sim.data.ctrl[:] = action
                    for _ in range(5):
                        sim.step()
                    sim.data.site_xpos[target] = [11, 11, 0]
                    count += 1
                # Do not forward before capturing post-step RGB.
                response = observation()
            elif op == "close":
                print(json.dumps({"ok": True, "closed": True}), flush=True)
                return
            else:
                raise ValueError("invalid operation or worker state")
            print(json.dumps({"ok": True, **response}, allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
            return


class NativeRGBBridge:
    """Environment-side worker; proprio is permitted only on the state-assisted track."""

    def __init__(self, initial_state: np.ndarray, timeout: float = 30.0):
        from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml

        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (4,) or not np.isfinite(state).all() or timeout <= 0:
            raise ValueError("invalid initial state or timeout")
        self.timeout = timeout
        self._temporary = tempfile.TemporaryDirectory(prefix="pc_pointmaze_rgb_worker_")
        xml_path = Path(self._temporary.name) / "scene.xml"
        xml = official_xml()
        xml_path.write_text(xml)
        self.xml_sha256 = hashlib.sha256(xml.encode()).hexdigest()
        env = os.environ.copy()
        env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
        env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                           "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
        self._stderr = tempfile.TemporaryFile(mode="w+t")
        self._process = subprocess.Popen(
            [str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "--worker", str(xml_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1, env=env,
        )
        try:
            self.initial_rgb, self.initial_proprio, step = self._exchange({"op": "reset", "initial_state": state.tolist()})
            if step != 0:
                raise RuntimeError("invalid initial step")
            self.step_count = 0
        except BaseException:
            self.close()
            raise

    def _exchange(self, request: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
        process = self._process
        if process.poll() is not None or process.stdin is None or process.stdout is None:
            raise RuntimeError("environment worker is not running")
        process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        process.stdin.flush()
        if not select.select([process.stdout], [], [], self.timeout)[0]:
            raise TimeoutError("environment worker did not respond")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("environment worker closed its output")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"environment worker rejected request: {response.get('error')}")
        data = base64.b64decode(response["rgb_b64"], validate=True)
        if len(data) != 224 * 224 * 3 or hashlib.sha256(data).hexdigest() != response["rgb_sha256"]:
            raise RuntimeError("invalid or corrupted RGB response")
        proprio = np.asarray(response["proprio"], dtype=np.float32)
        if proprio.shape != (4,) or not np.isfinite(proprio).all():
            raise RuntimeError("invalid proprio response")
        return np.frombuffer(data, dtype=np.uint8).reshape(224, 224, 3).copy(), proprio, int(response["step"])

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        packet = np.asarray(actions, dtype=np.float64)
        if packet.ndim != 2 or packet.shape[1] != 2 or not 1 <= len(packet) <= 5:
            raise ValueError("actions must have shape [1..5,2]")
        if not np.isfinite(packet).all() or np.any(np.abs(packet) > 1):
            raise ValueError("actions must be finite in [-1,1]")
        image, proprio, step = self._exchange({"op": "step", "actions": packet.tolist()})
        if step != self.step_count + len(packet):
            raise RuntimeError("environment action-clock mismatch")
        self.step_count = step
        return image, proprio

    def close(self) -> None:
        process = self._process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        self._stderr.close()
        self._temporary.cleanup()

    def __enter__(self) -> "NativeRGBBridge":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        _worker(Path(sys.argv[2]))
    else:
        raise SystemExit("worker invoked by NativeRGBBridge")
