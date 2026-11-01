"""Development-only queue-bound actual goal RGB, reset and warmup history."""
import argparse
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from . import p1_future_root_queue_20261101_codex as q
from . import p1_dev_reset_bound_20261101_codex as native

NAME = "p1_queue_goal_history_20261101_codex"
DOC = q.ROOT.parent / "docs/D2P_P1_QUEUE_GOAL_HISTORY_20261101_CODEX.md"
require = q.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bindings():
    return dict(producer_sha256=sha(Path(__file__)), environment_sha256=native.ENV_SHA,
                protocol_sha256=sha(DOC))


def events(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    require(len(raw) <= q.MAX_BYTES and (not raw or raw.endswith(b"\n")), "incomplete ledger")
    return [json.loads(line) for line in raw.splitlines()]


class Stop(Exception):
    pass


def run(queue_path, queue_sha, slot_id, attempt, *, factory=None):
    path = Path(queue_path).resolve(strict=True)
    require(sha(path) == queue_sha, "queue SHA")
    queue = q.read_json(path)
    require(queue == q.freeze(queue.get("spec")) and
            all(queue["spec"][k] == v for k, v in bindings().items()), "queue/source binding")
    slot = next((s for s in queue["spec"]["slots"] if s["slot_id"] == slot_id), None)
    require(slot is not None and slot["role"] == "development", "development only")
    require(type(attempt) is int and 0 <= attempt < len(slot["reset_seeds"]), "attempt")
    seed = slot["reset_seeds"][attempt]
    base = path.parent / (path.name + ".native_reset")
    base.mkdir(mode=0o700, exist_ok=True)
    require(base.is_dir() and not base.is_symlink(), "bridge namespace")
    fd = os.open(base / "bridge.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        require(sha(path) == queue_sha and queue == q.freeze(q.read_json(path)["spec"])
                and all(queue["spec"][k] == v for k, v in bindings().items()), "post-lock binding")
        ledger = base / "queue_ledger.jsonl"
        prior = events(ledger)
        q.validate(queue, prior)
        require(not any(e["slot_id"] == slot_id and e["attempt"] == attempt for e in prior),
                "replay prohibited")
        run_dir = base / f"{slot_id}_{attempt:04d}.warmup"
        run_dir.mkdir(mode=0o700)  # exclusive, including interrupted attempts
        deps = native.dependencies()
        reservation = dict(schema=NAME + ".reservation.v1", queue_file_sha256=queue_sha,
                           queue_digest=q.digest(queue), slot_id=slot_id, attempt=attempt,
                           seed=seed, source_sha256=deps, independent_confirmation=False, P1_admission=False)
        q.write_new(run_dir / "reservation.json", reservation)
        start = q.append(path, ledger, dict(kind="reset_attempt", slot_id=slot_id, attempt=attempt,
                                           seed=seed, payload={}))
        journal = run_dir / "journal.jsonl"
        log = []
        native.emit(journal, log, "reset_authorized", queue_attempt_sha256=start, seed=seed)
        env = None
        status, error, returned, frames, actions = "partial", None, 0, [], []
        stage, initial, goal = "constructor", None, None
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(sha(inspect.getsourcefile(cls)) == native.ENV_SHA, "environment source")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False) if factory is None else factory()
            require(type(env) is cls if factory is None else True, "native class")
            stage = "reset"
            native.emit(journal, log, "reset_dispatch", seed=seed)
            obs, info = env.reset(seed=seed)
            state = np.asarray(obs["state"], np.float64)
            goal = np.asarray(info["goal_state"], np.float64).copy()
            goal_rgb = np.asarray(info["goal"]).copy()
            initial = native.snapshot(env)
            require(state.shape == goal.shape == (7,) and np.isfinite(state).all()
                    and np.isfinite(goal).all() and np.array_equal(state, initial["state"]), "reset observation")
            require(goal_rgb.shape == (224, 224, 3) and goal_rgb.dtype == np.uint8
                    and np.array_equal(goal_rgb, env._goal)
                    and np.array_equal(goal, env.goal_state), "actual native goal state/RGB")
            goal_file = run_dir / "goal.png"
            with goal_file.open("xb") as stream:
                Image.fromarray(goal_rgb).save(stream, format="PNG")
                stream.flush()
                os.fsync(stream.fileno())
            goal_binding = dict(file=goal_file.name, file_sha256=sha(goal_file),
                                pixel_sha256=hashlib.sha256(np.ascontiguousarray(goal_rgb).tobytes()).hexdigest(),
                                source="same_instance_reset_info_goal")
            reset = dict(state=state.tolist(), goal=goal.tolist(), goal_rgb=goal_binding, snapshot=initial)
            reset_sha = q.write_new(run_dir / "reset_receipt.json", reset)
            native.emit(journal, log, "reset_return", reset_receipt_sha256=reset_sha)

            def capture(step):
                before = native.snapshot(env)
                pixels = np.asarray(env.render())
                after = native.snapshot(env)
                require(before == after and pixels.shape == (224, 224, 3)
                        and pixels.dtype == np.uint8, "actual RGB/snapshot")
                file = run_dir / f"rgb_{step:02d}.png"
                with file.open("xb") as stream:
                    Image.fromarray(pixels).save(stream, format="PNG")
                    stream.flush()
                    os.fsync(stream.fileno())
                entry = dict(step=step, file=file.name, file_sha256=sha(file),
                             pixel_sha256=hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
                             state=after["state"], snapshot=after)
                frames.append(entry)
                native.emit(journal, log, "rgb_captured", **entry)

            stage = "warmup"
            capture(0)
            initial_success, _ = env.eval_state(goal, state)
            if initial_success:
                raise Stop("native_success_at_reset")
            for step in range(1, 11):
                action = np.zeros(2, dtype=np.float32)
                native.emit(journal, log, "step_dispatch", step=step, action=action.tolist())
                obs, reward, terminated, truncated, info = env.step(action.copy())
                returned += 1
                native.emit(journal, log, "step_native_return", step=step,
                            terminated=bool(terminated), truncated=bool(truncated))
                actual = np.asarray(env.latest_action)
                state = np.asarray(obs["state"], np.float64)
                require(np.array_equal(actual, action) and np.array_equal(info["goal_state"], goal)
                        and np.array_equal(info["goal"], goal_rgb) and np.array_equal(env._goal, goal_rgb)
                        and np.array_equal(state, env._get_obs()) and np.isfinite(state).all(), "native step")
                actions.append(actual.tolist())
                native.emit(journal, log, "step_return", step=step, action=actual.tolist(), state=state.tolist(),
                            reward=float(reward), terminated=bool(terminated), truncated=bool(truncated))
                if step in (5, 10):
                    capture(step)
                if terminated or truncated:
                    raise Stop("native_terminated" if terminated else "native_truncated")
            require(sha(path) == queue_sha and deps == native.dependencies()
                    and all(queue["spec"][k] == v for k, v in bindings().items()), "source drift")
            final = native.snapshot(env)
            history = dict(reset_receipt_sha256=reset_sha, frames=frames, actions=actions,
                           final_state=final["state"], goal=goal.tolist(), goal_rgb=goal_binding)
            history_sha = q.digest(history)
            source = dict(schema=NAME + ".source.v1", queue_attempt_sha256=start, history=history,
                          history_sha256=history_sha, final_snapshot=final, steps_returned=returned,
                          history_semantics="actual_native_rgb_at_0_5_10_and_10_actual_raw_zero_actions",
                          training_membership="unknown", historical_coverage="unknown",
                          independent_confirmation=False, P1_admission=False)
            source_sha = q.write_new(run_dir / "source_receipt.json", source)
            native.emit(journal, log, "source_sealed", source_sha256=source_sha, history_sha256=history_sha)
            stage = "queue_completion"
            completion = q.append(path, ledger, dict(kind="reset_return", slot_id=slot_id, attempt=attempt,
                                  seed=seed, payload=dict(state_sha256=q.digest(final["state"]),
                                  goal_sha256=q.digest(goal.tolist()), history_sha256=history_sha,
                                  snapshot_sha256=q.digest(final))))
            native.emit(journal, log, "queue_completed", event_sha256=completion)
            status = "complete_development"
        except Exception as exc:
            error = dict(type=type(exc).__name__, message=str(exc)[:500], stage=stage)
            native.emit(journal, log, "stopped", **error, steps_returned=returned)
            # Never overwrite a potentially durable completion after a crash
            # or ledger I/O error; leave pending for manual adjudication.
            if stage != "queue_completion":
                q.append(path, ledger, dict(kind="reset_failed", slot_id=slot_id, attempt=attempt,
                                          seed=seed, payload={"error_class": type(exc).__name__}))
            status = "stopped_or_failed"
        finally:
            if env is not None:
                env.close()
        result = dict(schema=NAME + ".report.v1", status=status, error=error, steps_returned=returned,
                      frames_recorded=len(frames), queue_attempt_sha256=start,
                      training_membership="unknown", historical_coverage="unknown", freshness="unknown",
                      independent_confirmation=False, P1_admission=False, neural_calls=0)
        native.emit(journal, log, "run_finished", status=status)
        result["journal_sha256"] = sha(journal)
        result["journal_tail"] = log[-1]["sha256"]
        q.write_new(run_dir / "report.json", result)
        return result
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--authorize-development", action="store_true", required=True)
    args = parser.parse_args()
    with native.locking.gpu_lock(native.LOCK):
        result = run(args.queue, args.queue_sha256, args.slot, args.attempt)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "complete_development":
        sys.exit(1)


if __name__ == "__main__":
    main()
