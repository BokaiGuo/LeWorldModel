"""CPU-only integration of the existing live join and real dispatcher.

Native packets, encoder and solver are fixtures; this is not a physical run.
"""
from contextlib import nullcontext
from types import SimpleNamespace
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import agent_e3_live_episode_join_20261029_v1 as live
from test_agent_e3_gd_host_cuda_20261029_v1 import body


@pytest.mark.parametrize("corrupt", [False, True])
def test_live_join_dispatches_encoded_boundary_into_next_plan(monkeypatch, corrupt):
    calls, events, frames, planned, terminal = [], [], {}, [], []
    class Env:
        def close(self):
            calls.append(("close",))
    env = Env()
    row = {"episode_id": "cell10"}
    meter = SimpleNamespace(region="initial")
    store = SimpleNamespace(output="fixture", check=lambda *args: None)
    class Bridge:
        def __init__(self, unified):
            assert unified is store
        def emit(self, kind, **kwargs):
            events.append(kind)
        def descriptor(self, observation):
            return {"file": observation["file"], "sha256": observation["file_sha256"]}
    monkeypatch.setattr(live.capture, "CaptureStore", Bridge)
    b = body()
    prior = np.full((5, 2), .1, np.float32)
    mean = np.asarray(b["action_mean"], np.float32)
    std = np.asarray(b["action_std"], np.float32)
    b["root_context"]["action_history"][-1] = ((prior - mean) / std).reshape(10).tolist()
    setup = {"prior_raw5": prior, "goal": np.zeros(7), "goal_rgb": np.zeros((224, 224, 3), np.uint8)}
    monkeypatch.setattr(live.setup, "setup", lambda actual, **kwargs: setup)
    def packet(actual, action, *, start, **kwargs):
        assert actual is env and meter.region == "initial"
        calls.append(("packet", start))
        data = ("frame" + str(start + 5)).encode()
        name = str(start + 5)
        frames[name] = data
        return {"returned_raw": action.copy(), "end": start + 5, "stopped": False,
                "complete": True, "observation": {"step": start + 5, "file": name,
                "file_sha256": hashlib.sha256(data).hexdigest()}}
    monkeypatch.setattr(live.region, "packet", packet)
    def encode(model, torch, device, unified, frame, boundary, check):
        assert meter.region == "encode"
        calls.append(("encode", boundary))
        check()
        return np.full(192, boundary, np.float32), None
    monkeypatch.setattr(live.encoding, "encode", encode)
    monkeypatch.setattr(live.encoding.u, "read_bytes",
                        lambda path, cap: b"wrong" if corrupt else frames[str(path).split("/")[-1]])
    class NodeBridge:
        def __init__(self, unified, boundary):
            self.boundary = boundary
        def event(self, *args, **kwargs):
            pass
        def finish(self, record):
            assert record["result"] is not None
    class Backend:
        def __init__(self, model, device, torch, adapter, task, record):
            assert meter.region == "compile"
            self.task = task
        def observe(self):
            return nullcontext()
    def solve(task, backend, event):
        calls.append(("plan", task["feedback_boundary"]))
        planned.append(task["root_context"]["latent_history"][-1])
        return {"first5": np.full((5, 2), .2, np.float32).tolist()}
    monkeypatch.setattr(live.nodes, "NodeBridge", NodeBridge)
    monkeypatch.setattr(live.nodes.bridge.host, "HostBackend", Backend)
    monkeypatch.setattr(live.nodes.node, "solve", solve)
    monkeypatch.setattr(live.terminal, "write", lambda unified, result: terminal.append(dict(result)))
    # Path joining is part of the production callback's byte-read seam.
    from pathlib import Path
    store.output = Path("fixture")
    args = (env, {}, b, np.full((5, 2), .3, np.float32), None, None, None, store, meter, row)
    if corrupt:
        with pytest.raises(ValueError):
            live.execute(*args)
        assert row["status"] == "failed" and not planned
        assert calls == [("packet", 5), ("encode", 10), ("close",)]
    else:
        live.execute(*args)
        assert row["status"] == "completed20"
        assert calls == [("packet", 5), ("encode", 10), ("plan", 10), ("packet", 10),
                         ("encode", 15), ("plan", 15), ("packet", 15), ("close",)]
        assert planned == [[10.] * 192, [15.] * 192]
    assert meter.region == "initial" and terminal[0]["status"] == row["status"]
