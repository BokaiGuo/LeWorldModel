"""Replay exposed RGB feedback decisions through a mount-isolated planner process."""
import base64
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments import pc_rgb_gradient_public_plan_20261107_codex as public
from experiments import pc_rgb_planner_mount_sandbox_20261107_codex as sandbox
from experiments.pc_input_permissions import file_hash

ROOT = public.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RGB_ISOLATED_FEEDBACK_REPLAY_PROTOCOL_20261107_CODEX.md"
OUT = ROOT / "results/pc_rgb_isolated_feedback_replay_20261107_codex.json"
EPISODES = (9000, 9013, 9022, 9032)
FIELDS = {"episode", "iteration", "arm", "remaining", "current", "goal", "old_tail"}
SHAPE = (224, 224, 3)
MARKER = "PC_RGB_RESPONSE:"


def decode_image(value):
    if not isinstance(value, str):
        raise ValueError("image must be base64")
    raw = base64.b64decode(value, validate=True)
    if len(raw) != int(np.prod(SHAPE)):
        raise ValueError("invalid image size")
    return np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE).copy()


def validate_request(req):
    if not isinstance(req, dict) or set(req) != FIELDS:
        raise ValueError("unexpected request fields")
    ep, it, remaining = req["episode"], req["iteration"], req["remaining"]
    if type(ep) is not int or ep not in EPISODES or type(it) is not int or it not in range(5):
        raise ValueError("invalid task or iteration")
    if type(remaining) is not int or remaining != 25 - 5 * it or req["arm"] not in ("effect", "cem16"):
        raise ValueError("invalid clock or arm")
    current, goal = decode_image(req["current"]), decode_image(req["goal"])
    tail = None if req["old_tail"] is None else np.asarray(req["old_tail"], dtype=np.float32)
    if it == 0 and tail is not None:
        raise ValueError("unexpected initial tail")
    if it > 0 and (tail is None or tail.shape != (remaining, 2) or not np.isfinite(tail).all()
                   or np.abs(tail).max(initial=0) > 1.000001):
        raise ValueError("invalid old tail")
    return current, goal, tail


def worker():
    import torch
    from sklearn.preprocessing import StandardScaler
    from eval import img_transform
    from experiments.checkpoint_compat import load_compatible_pretrained
    from experiments import pc_rgb_temporal_final_search_20261107_codex as program
    from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as prior
    from experiments.pc_input_permissions import RGB, RGBLeWM, freeze_audit
    from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

    denied = json.loads(os.environ["PC_DENIED_PATHS"])
    checks = {p: os.access(p, os.R_OK) for p in denied}
    if any(checks.values()):
        raise RuntimeError("protected path readable")
    with np.load(public.PUBLIC, allow_pickle=False) as data:
        public.validate_public(data)
        scaler = StandardScaler()
        scaler.mean_, scaler.scale_, scaler.n_features_in_ = data["action_mean"].copy(), data["action_scale"].copy(), 2
    model = load_compatible_pretrained(str(public.CHECKPOINT), device="cuda").eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    fingerprint = model_fingerprint(model)
    audit = freeze_audit(RGB, "lewm", [public.PUBLIC, public.PROTOCOL, public.CHECKPOINT / "weights.pt",
                         public.CHECKPOINT / "config.json"], preprocessing="ImageNet 224; fixed action statistics",
                         clock="five physical steps per packet")
    transform = img_transform(SimpleNamespace(eval=SimpleNamespace(img_size=224)))

    def emit(obj):
        sys.__stdout__.write(MARKER + json.dumps(obj, allow_nan=False, separators=(",", ":")) + "\n")
        sys.__stdout__.flush()

    emit({"ready": True, "denied": checks, "model_sha256": fingerprint,
          "normalizer_mean": scaler.mean_.tolist(), "normalizer_scale": scaler.scale_.tolist()})
    for line in sys.stdin:
        req = json.loads(line)
        if req == {"stop": True}:
            if model_fingerprint(model) != fingerprint:
                raise ValueError("model changed")
            emit({"stopped": True})
            return
        image, goal, tail = validate_request(req)
        current = {"pixels": transform(image)[None, None, None].cuda(),
                   "goal": transform(goal)[None, None, None].cuda(),
                   "action": torch.empty((1, 1, 0, 10), device="cuda")}
        boundary = RGBLeWM(model, audit)
        with torch.inference_mode():
            if req["arm"] == "effect":
                res = program.compile_rgb(boundary, current, scaler=scaler, old_tail=tail,
                    candidate_seed=27121000 + req["episode"] * 10 + req["iteration"],
                    strategy_seed=27122000 + req["episode"] * 10 + req["iteration"],
                    strategy="effect", remaining_steps=req["remaining"], final_rule="temporal")
                choice = {"source": res["choice"].source, "index": res["choice"].index,
                          "cost": res["choice"].cost}
                queries, packets = res["model_queries"], res["model_packet_steps"]
            else:
                res = prior.cem_temporal(boundary, current, scaler, tail, req["remaining"],
                    8140000 + req["episode"] * 10 + req["iteration"], 16)
                choice = {"index": res["chosen"], "cost": res["chosen_cost"]}
                queries, packets = res["queries"], res["packet_steps"]
        emit({"actions": res["actions"].tolist(), "choice": choice, "queries": queries,
              "packet_steps": packets, "forward_calls": res["forward_calls"]})
    raise RuntimeError("worker disconnected without stop")


def encode_image(image):
    image = np.asarray(image)
    if image.shape != SHAPE or image.dtype != np.uint8:
        raise ValueError("invalid RGB image")
    return base64.b64encode(image.tobytes()).decode("ascii")


def receive(proc):
    for line in proc.stdout:
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER):])
        print(f"worker log: {line.rstrip()}", file=sys.stderr, flush=True)
    raise RuntimeError("worker exited before response")


def main():
    import gymnasium as gym
    import h5py
    import stable_worldmodel  # noqa: F401
    from sklearn.preprocessing import StandardScaler
    from experiments import pc_tworoom_temporal_new_episodes_20261107_codex as prior

    if OUT.exists():
        raise FileExistsError(OUT)
    historic = json.loads(prior.OUTPUT.read_text())
    if (historic["runner_sha256"] != file_hash(prior.__file__)
            or historic["protocol_sha256"] != file_hash(prior.PROTOCOL)
            or historic["input_sha256"] != file_hash(prior.INPUT)):
        raise ValueError("reference provenance drift")
    tmp = Path(tempfile.mkdtemp(prefix="pc-rgb-feedback-sandbox-"))
    command, denied, mounts = sandbox.make_command(tmp)
    command[-2:] = ["-u", "-m", "experiments.pc_rgb_isolated_feedback_replay_20261107_codex", "worker"]
    rows = []
    print("Replay exposed effect and CEM16 feedback with a mount-isolated RGB planner.", flush=True)
    with (tmp / "worker.stderr").open("w") as errors:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                                text=True, close_fds=True, bufsize=1)
        try:
            ready = receive(proc)
            if not ready["ready"] or set(ready["denied"]) != set(denied) or any(ready["denied"].values()):
                raise ValueError("worker path probe failure")
            if ready["model_sha256"] != historic["model_sha256"]:
                raise ValueError("checkpoint mismatch")
            with h5py.File(prior.common.DATA, "r") as h5:
                source_actions = np.asarray(h5["action"][:])
            source_actions = source_actions[~np.isnan(source_actions).any(axis=1)]
            scaler = StandardScaler().fit(source_actions)
            np.testing.assert_allclose(ready["normalizer_mean"], scaler.mean_, atol=0, rtol=0)
            np.testing.assert_allclose(ready["normalizer_scale"], scaler.scale_, atol=0, rtol=0)
            env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
            try:
                with np.load(prior.INPUT, allow_pickle=False) as data, h5py.File(prior.common.DATA, "r") as h5:
                    for ep in EPISODES:
                        index = int(np.flatnonzero(data["episodes"] == ep).item())
                        row = next(r for r in historic["rows"] if r["episode"] == ep)
                        if int(data["rows"][index]) != row["row"]:
                            raise ValueError("task identity drift")
                        for arm in ("effect", "cem16"):
                            env.reset(seed=42)
                            actor = env.unwrapped
                            actor._set_state(h5["proprio"][row["row"]])
                            actor._set_goal_state(h5["proprio"][row["row"] + 25])
                            tail, cursor = None, 0
                            for iteration, saved in enumerate(row["arms"][arm]["decisions"]):
                                req = {"episode": ep, "iteration": iteration, "arm": arm,
                                       "remaining": 25 - cursor, "current": encode_image(actor.render().copy()),
                                       "goal": encode_image(data["goal"][index]),
                                       "old_tail": None if tail is None else tail.tolist()}
                                proc.stdin.write(json.dumps(req, allow_nan=False) + "\n")
                                proc.stdin.flush()
                                result = receive(proc)
                                actions = np.asarray(result["actions"], dtype=np.float32)
                                np.testing.assert_allclose(actions, saved["actions"], atol=1e-6, rtol=0)
                                expected = saved["choice"]["index"] if arm == "effect" else saved["chosen"]
                                if (result["choice"]["index"] != expected or result["queries"] != saved["queries"]
                                        or result["packet_steps"] != saved["packet_steps"]
                                        or result["forward_calls"] != saved["forward_calls"]
                                        or (arm == "effect" and result["choice"]["source"] != saved["choice"]["source"])):
                                    raise ValueError("selection or accounting drift")
                                dispatched = []
                                for action in actions[:5]:
                                    _, _, term, trunc, info = actor.step(action.copy())
                                    old = row["arms"][arm]["trace"][cursor]
                                    if (abs(float(info["distance_to_target"]) - old["distance"]) > 1e-5
                                            or bool(term) != old["terminated"] or bool(trunc) != old["truncated"]):
                                        raise ValueError("physical trace drift")
                                    np.testing.assert_allclose(action, old["action"], atol=1e-6, rtol=0)
                                    dispatched.append(action.tolist())
                                    cursor += 1
                                    if term or trunc:
                                        break
                                np.testing.assert_allclose(dispatched, saved["dispatched"], atol=1e-6, rtol=0)
                                tail = actions[len(dispatched):].copy()
                                rows.append({"episode": ep, "arm": arm, "iteration": iteration,
                                             "remaining": req["remaining"], "choice": result["choice"],
                                             "queries": result["queries"], "packet_steps": result["packet_steps"],
                                             "actions": actions.tolist(), "dispatched": dispatched})
                            if cursor != len(row["arms"][arm]["trace"]):
                                raise ValueError("incomplete trace")
                            print(f"episode={ep} arm={arm} steps={cursor}", flush=True)
            finally:
                env.close()
            proc.stdin.write('{"stop":true}\n')
            proc.stdin.flush()
            if receive(proc) != {"stopped": True} or proc.wait(timeout=30) != 0:
                raise RuntimeError("worker failed at shutdown")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
    with OUT.open("x") as stream:
        json.dump({"scope": "exposed paired feedback replay; not independent confirmation",
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "historic_sha256": file_hash(prior.OUTPUT), "input_sha256": file_hash(prior.INPUT),
                   "public_sha256": file_hash(public.PUBLIC), "model_sha256": ready["model_sha256"],
                   "denied_paths_readable": ready["denied"], "network_namespace": False,
                   "mounts": mounts, "rows": rows}, stream, indent=2, allow_nan=False)
    print(f"sealed {len(rows)} decisions; stderr={tmp / 'worker.stderr'}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["worker"]:
        with contextlib.redirect_stdout(sys.stderr):
            worker()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("expected no arguments or worker")
