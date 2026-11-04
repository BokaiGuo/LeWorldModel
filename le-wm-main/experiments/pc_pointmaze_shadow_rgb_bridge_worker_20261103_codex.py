"""Internal executor and isolated shadow worker. Never expose state to caller."""
import base64
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np

from pc_pointmaze_shadow_renderer_worker_v4_20261103_codex import EXTRA, camera, same, snapshot

XML_SHA256 = "64d83955572d62b90d8641f50c0751cb91a39d593d8398e324ab16b4d895168d"
EXPECTED_DIMS = {"nq": 2, "nv": 2, "na": 0, "nu": 2, "nmocap": 0,
                 "nuserdata": 0, "neq": 0, "nbody": 2, "ngeom": 20, "nsite": 2, "ntendon": 0}


def _model(path):
    import mujoco_py
    xml = path.read_bytes()
    if hashlib.sha256(xml).hexdigest() != XML_SHA256:
        raise ValueError("unsupported model")
    model = mujoco_py.load_model_from_xml(xml.decode())
    if {key: int(getattr(model, key)) for key in EXPECTED_DIMS} != EXPECTED_DIMS:
        raise ValueError("unsupported model dimensions")
    return mujoco_py.MjSim(model)


def _encode(image):
    raw = np.ascontiguousarray(image, dtype=np.uint8).tobytes()
    if len(raw) != 224*224*3:
        raise ValueError("bad image size")
    return {"rgb": base64.b64encode(raw).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _shadow(path, fd):
    conn = socket.socket(fileno=fd)
    stream = conn.makefile("rb")
    sim = _model(path)
    camera(sim)
    target = sim.model.site_name2id("target_site")
    for line in stream:
        if len(line) > 100_000:
            raise ValueError("oversized state")
        request = json.loads(line)
        if set(request) != {"state"}:
            raise ValueError("invalid shadow request")
        wanted = {key: np.asarray(value) for key, value in request["state"].items()}
        if set(wanted) != {"time", "qpos", "qvel", "act", *EXTRA}:
            raise ValueError("invalid state fields")
        sim.data.time = float(wanted["time"])
        sim.data.qpos[:] = wanted["qpos"]
        sim.data.qvel[:] = wanted["qvel"]
        for field in EXTRA:
            dst, src = getattr(sim.data, field), wanted[field]
            if dst is None:
                if src.size:
                    raise ValueError("unsupported state field")
            else:
                dst[:] = src.reshape(dst.shape)
        before = snapshot(sim)
        if not same(before, wanted):
            raise ValueError("state transfer mismatch")
        sim.forward()
        after = snapshot(sim)
        if any(not np.array_equal(before[k], after[k]) for k in ("time", "qpos", "qvel", "act")):
            raise ValueError("shadow forward changed physical state")
        sim.data.site_xpos[target] = [11, 11, 0]
        conn.sendall(json.dumps(_encode(sim.render(224, 224)), allow_nan=False).encode() + b"\n")


def _executor(path, fd):
    conn = socket.socket(fileno=fd)
    stream = conn.makefile("rb")
    sim = _model(path)
    camera(sim)
    target = sim.model.site_name2id("target_site")
    local, remote = socket.socketpair()
    entry = Path(__file__).with_name("pc_pointmaze_shadow_rgb_bridge_20261103_codex.py")
    try:
        child = subprocess.Popen([sys.executable, str(entry), "--shadow", str(path), str(remote.fileno())],
                                 pass_fds=(remote.fileno(),), stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        remote.close()
    shadow_in = local.makefile("rb")
    initialized, count, last_seq = False, 0, 0
    try:
        for line in stream:
            if len(line) > 4096:
                raise ValueError("oversized request")
            req = json.loads(line)
            seq = req.get("seq")
            if type(seq) is not int or seq != last_seq + 1:
                raise ValueError("bad request sequence")
            last_seq = seq
            if req.get("op") == "reset" and set(req) == {"seq", "op", "initial_state"}:
                state = np.asarray(req["initial_state"], dtype=np.float64)
                if state.shape != (4,) or not np.isfinite(state).all():
                    raise ValueError("bad initial state")
                sim.reset()
                sim.data.qpos[:], sim.data.qvel[:] = state[:2], state[2:]
                sim.forward()
                sim.data.site_xpos[target] = [11, 11, 0]
                count, initialized = 0, True
            elif req.get("op") == "step" and set(req) == {"seq", "op", "actions"} and initialized:
                actions = np.asarray(req["actions"], dtype=np.float64)
                if (actions.ndim != 2 or actions.shape[1] != 2 or not 1 <= len(actions) <= 5
                        or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
                    raise ValueError("bad actions")
                for action in actions:
                    sim.data.qvel[:] = np.clip(sim.data.qvel, -5, 5)
                    sim.forward()  # The original pre-command forward, never an extra render forward.
                    sim.data.ctrl[:] = action
                    for _ in range(5):
                        sim.step()
                    sim.data.site_xpos[target] = [11, 11, 0]
                    count += 1
            else:
                raise ValueError("invalid request")
            before = snapshot(sim)
            payload = {key: val.tolist() for key, val in before.items()}
            local.sendall(json.dumps({"state": payload}, allow_nan=False).encode() + b"\n")
            response_line = shadow_in.readline(210_001)
            if not response_line.endswith(b"\n"):
                raise RuntimeError("shadow response unavailable or oversized")
            response = json.loads(response_line)
            if set(response) != {"rgb", "sha256"} or not same(before, snapshot(sim)):
                raise RuntimeError("shadow violated observation contract")
            conn.sendall(json.dumps({"seq": seq, "step": count, **response}, allow_nan=False).encode() + b"\n")
    finally:
        local.close()
        shadow_in.close()
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=3)


def run_worker(mode, path, fd):
    if mode == "--executor":
        _executor(path, fd)
    elif mode == "--shadow":
        _shadow(path, fd)
    else:
        raise ValueError("invalid worker mode")
