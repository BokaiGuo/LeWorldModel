"""Opt-in RGB-only PointMaze bridge with a process-isolated shadow renderer.

Initial state is environment setup data, not a planner observation. This is a
development bridge for the one audited XML and legacy runtime, not a benchmark.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import tempfile
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / ".legacy_mujoco210_py310"
RUNTIME = LEGACY / "vendor/mujoco-2.1.2"
XML_SHA256 = "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d"
MAX_RESPONSE = 210_000


def _scene():
    from experiments.a0_pointmaze_native_render_probe_20261102 import official_xml
    xml = official_xml().encode()
    if hashlib.sha256(xml).hexdigest() != XML_SHA256:
        raise RuntimeError("unsupported PointMaze XML")
    return xml


def _launch_worker(xml_path):
    parent, child = socket.socketpair()
    env = os.environ.copy()
    env["MUJOCO_PY_MUJOCO_PATH"] = str(RUNTIME.resolve())
    env["LD_LIBRARY_PATH"] = ":".join((str(RUNTIME / "lib"), str(RUNTIME / "bin"),
                                       "/usr/lib/nvidia", env.get("LD_LIBRARY_PATH", "")))
    try:
        process = subprocess.Popen(
            [str(LEGACY / "bin/python"), str(Path(__file__).resolve()), "--executor",
             str(xml_path), str(child.fileno())],
            pass_fds=(child.fileno(),), start_new_session=True, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except BaseException:
        parent.close()
        child.close()
        raise
    child.close()
    return process, parent


class ShadowRGBBridge:
    """Single-caller RGB-only environment; setup state is not an observation."""

    def __init__(self, initial_state, timeout=30.0):
        state = np.asarray(initial_state, dtype=np.float64)
        if (state.shape != (4,) or not np.isfinite(state).all()
                or isinstance(timeout, bool) or not math.isfinite(float(timeout)) or timeout <= 0):
            raise ValueError("invalid initial state or timeout")
        self.timeout = float(timeout)
        self._setup_state = state.copy()
        self._sequence = 0
        self.step_count = 0
        self._closed = False
        self._lock = threading.Lock()
        self._process = None
        self._socket = None
        self._temporary = tempfile.TemporaryDirectory(prefix="pointmaze_shadow_rgb_")
        try:
            path = Path(self._temporary.name) / "scene.xml"
            path.write_bytes(_scene())
            self._process, self._socket = _launch_worker(path)
            self._socket.setblocking(False)
            self.initial_rgb = self.reset()
        except BaseException:
            self.close()
            raise

    def _transaction(self, payload, expected_step):
        if self._closed:
            raise RuntimeError("bridge is closed or failed")
        self._sequence += 1
        payload = {**payload, "seq": self._sequence}
        outgoing = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode() + b"\n"
        deadline = time.monotonic() + self.timeout
        incoming = bytearray()
        try:
            while outgoing:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self._socket], [], remaining)[1]:
                    raise TimeoutError("worker transaction timed out")
                sent = self._socket.send(outgoing)
                if sent <= 0:
                    raise RuntimeError("worker connection closed")
                outgoing = outgoing[sent:]
            while b"\n" not in incoming:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self._socket], [], [], remaining)[0]:
                    raise TimeoutError("worker transaction timed out")
                chunk = self._socket.recv(65536)
                if not chunk:
                    raise RuntimeError("worker connection closed")
                incoming.extend(chunk)
                if len(incoming) > MAX_RESPONSE:
                    raise RuntimeError("oversized worker response")
            if not incoming.endswith(b"\n") or incoming.count(b"\n") != 1:
                raise RuntimeError("invalid response framing")
            response = json.loads(incoming)
            if not isinstance(response, dict) or set(response) != {"seq", "step", "rgb", "sha256"}:
                raise RuntimeError("invalid worker response schema")
            if (type(response["seq"]) is not int or response["seq"] != self._sequence
                    or type(response["step"]) is not int or response["step"] != expected_step
                    or not isinstance(response["rgb"], str) or not isinstance(response["sha256"], str)):
                raise RuntimeError("worker sequence or action-clock mismatch")
            raw = base64.b64decode(response["rgb"], validate=True)
            if len(raw) != 224*224*3 or hashlib.sha256(raw).hexdigest() != response["sha256"]:
                raise RuntimeError("corrupt RGB response")
            return np.frombuffer(raw, dtype=np.uint8).reshape(224, 224, 3).copy()
        except (OSError, ValueError, TypeError, KeyError, binascii.Error, RuntimeError, TimeoutError) as exc:
            self.close()
            if isinstance(exc, TimeoutError):
                raise
            raise RuntimeError("shadow RGB worker failed; bridge closed") from exc

    def reset(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent bridge operations are not supported")
        try:
            image = self._transaction({"op": "reset", "initial_state": self._setup_state.tolist()}, 0)
            self.step_count = 0
            return image
        finally:
            self._lock.release()

    def step(self, actions):
        packet = np.asarray(actions, dtype=np.float64)
        if (packet.ndim != 2 or packet.shape[1] != 2 or not 1 <= len(packet) <= 5
                or not np.isfinite(packet).all() or np.any(np.abs(packet) > 1)):
            raise ValueError("actions must be finite [1..5,2] in [-1,1]")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("concurrent bridge operations are not supported")
        try:
            next_step = self.step_count + len(packet)
            image = self._transaction({"op": "step", "actions": packet.tolist()}, next_step)
            self.step_count = next_step
            return image
        finally:
            self._lock.release()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            self._socket.close()
        if self._process is not None:
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait(timeout=5)
        self._temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4 or sys.argv[1] not in ("--executor", "--shadow"):
        raise SystemExit("worker only")
    from pc_pointmaze_shadow_rgb_bridge_worker_20261103_codex import run_worker
    run_worker(sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]))
