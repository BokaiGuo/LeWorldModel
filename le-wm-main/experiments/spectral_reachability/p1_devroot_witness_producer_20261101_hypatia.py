"""CPU-only, fixed two-development-root witness producer; never a fresh-root release."""
import argparse
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_devroot_witness_ledger_20261101_hypatia as accounting
from experiments.spectral_reachability import p1_exploratory_rgb_history_v4_20261101_hypatia as v4
from experiments.spectral_reachability import p1_constructive_moving_history_source_20261027_v1 as moving

NAME = "p1_devroot_witness_producer_20261101_hypatia"
ROOT = v4.ROOT
PROTOCOL = ROOT.parent / "docs/D2P_P1_DEVROOT_WITNESS_PREREG_20261101_HYPATIA.md"
TEST = ROOT / ("tests/test_" + NAME + ".py")
PREP = ROOT / ("results/" + NAME + "_PREP01.json")
OUT = ROOT / ("results/" + NAME + "_RUN01")
PIN_PROTOCOL = "c7a7322b998ab6e76f8b4b5a1da07e3d3874e17e00e7d14ab21ea5b3a1331e00"
PIN_V4_PREP = "9cbab1d0eef2fcf9c067d332e89f12b77178f17c8f3b0c37cce30aa0e39c6dca"


def runtime_binding():
    for module in ("gymnasium", "stable_worldmodel", "stable_worldmodel.envs.pusht.env",
                   "pymunk", "pygame", "cv2", "PIL.Image"):
        importlib.import_module(module)
    prefixes = ("experiments.spectral_reachability.", "stable_worldmodel.", "pymunk.",
                "pygame.", "gymnasium.", "cv2", "numpy.", "PIL.")
    files = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and (name.startswith(prefixes) or name in ("numpy", "pymunk", "pygame",
                                                            "gymnasium", "stable_worldmodel", "PIL")):
            path = Path(path).resolve()
            if path.is_file() and path.suffix != ".pyc":
                files[str(path)] = v4.sha(path)
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "gymnasium", "pymunk", "pygame", "Pillow", "stable-worldmodel")}
    return {"loaded_module_files": files, "versions": versions,
            "limitation": "loaded Python/extension modules, not complete OS shared-library closure"}


def need(ok, text):
    if not ok:
        raise ValueError(text)


def prepared():
    need(v4.sha(PROTOCOL) == PIN_PROTOCOL and v4.sha(v4.PREPARED) == PIN_V4_PREP,
         "preregistration or v4 prepare drift")
    old = v4.prepare()
    files = dict(old["files"])
    for f in (PROTOCOL, Path(__file__), Path(accounting.__file__), Path(moving.__file__),
              Path(moving.old.__file__), TEST, v4.PREPARED,
              ROOT / "tests/test_p1_devroot_witness_ledger_20261101_hypatia.py",
              ROOT / "tests/test_p1_devroot_witness_prereg_20261101_hypatia.py"):
        files[str(f.resolve())] = v4.sha(f)
    return {"schema": NAME + ".prepare.v1", "files": files, "slots": [0, 1],
            "signs": [1, -1], "phases": ["construction", "replay0", "replay1"],
            "caps": accounting.CAPS, "output_cap": 32 * 1024**2,
            "final_reserve": 512 * 1024, "disk_floor": 200 * 1024**2,
            "wall_seconds": 1200, "training_disjoint": "unknown",
            "freshness": "unknown", "development_roots": True,
            "planner_authorized": False, "P1_admission": False,
            "python": sys.version, "runtime": runtime_binding()}


def save_prepare():
    b = accounting.canonical(prepared()) + b"\n"
    with PREP.open("xb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    return hashlib.sha256(b).hexdigest()


class SlotFailure(Exception):
    pass


def frame_bytes(frame):
    b = io.BytesIO()
    Image.fromarray(frame).save(b, format="PNG")
    return b.getvalue()


def equal(a, b):
    return moving.exact(a, b)


def hooks(env, ledger):
    body = env.unwrapped
    original_set, original_goal, original_render = body._set_state, body._set_goal_state, body.render
    seen_spaces = []
    reset_mode = [False]

    def set_state(x):
        if not any(space is body.space for space in seen_spaces):
            space = body.space
            step = space.step
            space.step = lambda *args, **kwargs: ledger.call(
                "physics_step", lambda: step(*args, **kwargs))
            seen_spaces.append(space)
        return ledger.call("set_state", lambda: original_set(x))

    body._set_state = set_state
    body._set_goal_state = lambda x: ledger.call("goal_setter", lambda: original_goal(x))
    body.render = lambda: ledger.call("goal_render" if reset_mode[0] else "explicit_render",
                                      original_render)

    def reset(**kwargs):
        reset_mode[0] = True
        try:
            return ledger.call("reset", lambda: env.reset(**kwargs))
        finally:
            reset_mode[0] = False
    return reset


def phase(slot_id, name, p, packet, ledger, factory, reference=None):
    env = factory()
    body = env.unwrapped
    try:
        moving.validate_env(body)
        need(body.render_action is False and body.render_size == 224, "render contract")
        reset = hooks(env, ledger)
        meta = packet["metadata"]
        _, info = reset(seed=meta["selected_attempt"]["random_reset_seed"])
        goal = v4.source.vector(info["goal_state"])
        slot = p["slots"][slot_id]
        request = v4.source.sample_state(goal, list(body.block.shapes), slot,
                                         meta["source"]["proposal_index"],
                                         meta["selected_attempt"]["attempt_index"], p)
        obs, info = reset(seed=0, options={"state": request.copy(), "goal_state": goal.copy()})
        need(equal(obs["state"], packet["state"]) and equal(info["goal_state"], packet["goal"]),
             "source state/goal binding")
        captured, actual = [], []

        def capture(step):
            snap = moving.snapshot(body)
            rgb = moving.image(body.render())
            need(snap == moving.snapshot(body) and equal(obs["state"], body._get_obs()),
                 "render changed state")
            filename = f"s{slot_id}_{name}_{step:02}.png"
            digest = ledger.write(filename, frame_bytes(rgb))
            event = {"step": step, "snapshot": snap, "state": v4.source.vector(obs["state"]).tolist(),
                     "file": filename, "file_sha256": digest,
                     "pixel_sha256": moving.image_sha(rgb),
                     "angles": v4.angles(goal if reference is None else reference["goal"],
                                          obs["state"], body.eval_state(
                                              goal if reference is None else reference["goal"], obs["state"]))}
            ledger.event("observation", slot=slot_id, phase=name, **event)
            moving.guard(rgb, obs["state"])
            return {"state": np.array(obs["state"], copy=True), "snapshot": snap, "rgb": rgb}

        def dispatch(action, kind, step):
            nonlocal obs
            ledger.event("command", slot=slot_id, phase=name, step=step,
                         raw_float32_bits=action.tobytes().hex())
            obs, reward, terminated, truncated, info = ledger.call(
                kind, lambda: env.step(action.copy()), slot=slot_id, phase=name, step=step)
            need(equal(body.latest_action, action), "executed action mismatch")
            ledger.event("return", slot=slot_id, phase=name, step=step,
                         reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))
            return bool(terminated), bool(truncated)

        captured.append(capture(0))
        for i, action in enumerate(moving.warmup_commands(), 1):
            if body.eval_state(goal, obs["state"])[0]:
                raise SlotFailure("native terminal before warmup dispatch")
            stop, trunc = dispatch(action, "warmup_step", i)
            actual.append(np.array(body.latest_action, copy=True))
            captured.append(capture(i))
            if stop or trunc:
                raise SlotFailure("warmup stopped or truncated at step %d" % i)
        root = np.array(obs["state"], copy=True)
        logits, commands = moving.witness(root, 1 if slot_id == 0 else -1, .75 * np.sqrt(5))
        ledger.write(f"s{slot_id}_{name}_commit.json", accounting.canonical(
            {"logits": logits.tolist(), "commands_bits": commands.tobytes().hex(),
             "root_snapshot": moving.snapshot(body)}) + b"\n")
        if reference is not None:
            for step in (0, 5, 10):
                for key in ("state", "rgb"):
                    need(equal(captured[step][key], reference["warm"][step][key]),
                         "warmup replay " + key)
                need(captured[step]["snapshot"] == reference["warm"][step]["snapshot"],
                     "warmup snapshot mismatch")
            need(equal(np.asarray(actual), reference["actions"]) and equal(commands, reference["commands"]),
                 "replayed action mismatch")
            before_pose = np.array(body.goal_pose, copy=True)
            ledger.call("goal_bind", lambda: moving.attach_goal(
                body, reference["goal"], reference["goal_rgb"]), slot=slot_id, phase=name)
            need(equal(before_pose, body.goal_pose), "goal pose changed")
            if body.eval_state(reference["goal"], root)[0]:
                raise SlotFailure("degenerate_goal")
        future = []
        for i, command in enumerate(commands, 1):
            stop, trunc = dispatch(command, "witness_step", i)
            record = capture(10 + i)
            future.append(record)
            if reference is not None:
                ref = reference["future"][i - 1]
                need(equal(record["state"], ref["state"]) and equal(record["rgb"], ref["rgb"])
                     and record["snapshot"] == ref["snapshot"], "witness replay mismatch")
            if stop or trunc:
                if reference is None or trunc or not body.eval_state(reference["goal"], obs["state"])[0]:
                    raise SlotFailure("witness stopped without valid derived-goal success at %d" % i)
                break
        if reference is None:
            return {"warm": captured, "actions": np.asarray(actual), "commands": commands,
                    "goal": future[-1]["state"], "goal_rgb": future[-1]["rgb"], "future": future}
        if not stop:
            raise SlotFailure("replay did not reach derived goal")
        return {"steps": len(future)}
    finally:
        env.close()


def execute(prep_sha):
    need(v4.sha(PREP) == prep_sha, "prepare hash mismatch")
    need(json.loads(PREP.read_bytes()) == prepared(), "prepared binding drift")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    ledger = accounting.Ledger(OUT)
    ledger.event("started", prepare_sha256=prep_sha, fixed_slots=[0, 1])
    plan = json.loads(v4.REQUEST.read_bytes())["plan"]
    rows = [{"slot": s, "status": "not_started", "phases": []} for s in (0, 1)]
    start = time.monotonic()
    for row in rows:
        slot_id = row["slot"]
        try:
            packet = v4.source.load_raw_source(v4.RUN / f"source_{slot_id}.npz", plan)
            need(packet["metadata"]["candidate_id"] == slot_id, "source slot binding")
            factory = lambda: gym.make("swm/PushT-v1", render_mode="rgb_array",
                                       resolution=224, with_target=False, render_action=False)
            row["status"] = "running"
            c = phase(slot_id, "construction", plan, packet, ledger, factory)
            ledger.write(f"s{slot_id}_construction_goal.json", accounting.canonical({
                "goal_state": c["goal"].tolist(), "goal_rgb_file": f"s{slot_id}_construction_20.png",
                "goal_pixel_sha256": moving.image_sha(c["goal_rgb"])}) + b"\n")
            row["phases"].append("construction")
            r0 = phase(slot_id, "replay0", plan, packet, ledger, factory, c)
            row["phases"].append("replay0")
            r1 = phase(slot_id, "replay1", plan, packet, ledger, factory, c)
            row.update(status="construction_and_replays_completed", replay_steps=[r0["steps"], r1["steps"]])
            row["phases"].append("replay1")
        except Exception as exc:
            row.update(status="failed", first_error=f"{type(exc).__name__}: {exc}")
            if ledger.failed or isinstance(exc, (accounting.GuardError, OSError)):
                break
        if not ledger.failed:
            ledger.event("slot_finished", row=row)
    events = [json.loads(line) for line in (OUT / "journal.jsonl").read_bytes().splitlines()]
    for row in rows:
        row["missing_checkpoints"] = {
            name: [step for step in range(21) if not any(
                e.get("kind") == "observation" and e.get("slot") == row["slot"]
                and e.get("phase") == name and e.get("step") == step for e in events)]
            for name in ("construction", "replay0", "replay1")}
    report = {"schema": NAME + ".receipt.v1", "prepare_sha256": prep_sha, "rows": rows,
              "counts": ledger.counts, "elapsed_seconds": time.monotonic() - start,
              "freshness": "unknown", "training_disjoint": "unknown", "development_roots": True,
              "planner_authorized": False, "P1_admission": False, "solver_cache": "unknown"}
    if not ledger.failed:
        ledger.write("receipt.json", accounting.canonical(report) + b"\n", final=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "run"])
    ap.add_argument("--prepare-sha256")
    a = ap.parse_args()
    if a.action == "prepare":
        print(save_prepare())
    else:
        need(a.prepare_sha256 is not None, "explicit prepare hash required")
        print(json.dumps(execute(a.prepare_sha256), indent=2))


if __name__ == "__main__":
    main()
