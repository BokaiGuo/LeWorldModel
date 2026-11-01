"""Development-only native reset dispatch bound to the frozen queue ledger."""
import argparse
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import stat
import sys

import numpy as np

from . import p1_future_root_queue_20261101_codex as q
from . import p1_dev_reset_bound_20261101_codex as native

NAME = "p1_queue_native_reset_bridge_20261101_codex"
ROOT = q.ROOT
DOC = ROOT.parent / "docs/D2P_P1_QUEUE_NATIVE_RESET_BRIDGE_20261101_CODEX.md"
require = q.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_spec_bindings():
    return dict(producer_sha256=sha(Path(__file__)), environment_sha256=native.ENV_SHA,
                protocol_sha256=sha(DOC))


def inspect_queue(queue_path, expected_sha, slot_id, attempt):
    path = Path(queue_path).resolve(strict=True)
    require(path.is_file() and not path.is_symlink() and sha(path) == expected_sha,
            "external queue file SHA")
    queue = q.read_json(path)
    require(queue == q.freeze(queue.get("spec")), "queue/source drift")
    require(all(queue["spec"][k] == v for k, v in expected_spec_bindings().items()),
            "producer/environment/protocol SHA mismatch")
    slots = [slot for slot in queue["spec"]["slots"] if slot["slot_id"] == slot_id]
    require(len(slots) == 1 and slots[0]["role"] == "development", "development slot only")
    require(type(attempt) is int and 0 <= attempt < len(slots[0]["reset_seeds"]), "attempt roster")
    return path, queue, slots[0]["reset_seeds"][attempt]


def read_events(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    require(len(raw) <= q.MAX_BYTES and (not raw or raw.endswith(b"\n")), "incomplete ledger")
    return [json.loads(line) for line in raw.splitlines()]


def run(queue_path, expected_sha, slot_id, attempt, *, factory=None):
    path, queue, seed = inspect_queue(queue_path, expected_sha, slot_id, attempt)
    # One namespace per canonical queue path. The exclusive operation receipt
    # is reserved before the queue attempt. An interrupted reservation is not
    # retried automatically, even when the reset itself may not have run.
    workspace = path.parent / (path.name + ".native_reset")
    workspace.mkdir(mode=0o700, exist_ok=True)
    require(workspace.is_dir() and not workspace.is_symlink(), "workspace type")
    lock_path = workspace / "bridge.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode) and os.fstat(fd).st_nlink == 1, "bridge lock")
        fcntl.flock(fd, fcntl.LOCK_EX)
        # Re-check after acquiring the producer-wide lock.
        path, queue, seed = inspect_queue(path, expected_sha, slot_id, attempt)
        ledger = workspace / "queue_ledger.jsonl"
        events = read_events(ledger)
        q.validate(queue, events)
        require(not any(e["slot_id"] == slot_id and e["attempt"] == attempt for e in events),
                "attempt already used: replay prohibited")
        receipt_path = workspace / f"{slot_id}_{attempt:04d}.json"
        reservation = dict(schema=NAME + ".reservation.v1", queue_file_sha256=expected_sha,
                           queue_digest=q.digest(queue), slot_id=slot_id, attempt=attempt, seed=seed,
                           source_sha256=native.dependencies(), training_membership="unknown",
                           historical_coverage="unknown", independent_confirmation=False, P1_admission=False)
        # A separate immutable reservation ensures a crash can only block, not
        # silently re-dispatch. It is not a trusted external timestamp.
        reservation_path = workspace / f"{slot_id}_{attempt:04d}.reservation.json"
        q.write_new(reservation_path, reservation)
        start = q.append(path, ledger, dict(kind="reset_attempt", slot_id=slot_id,
                                           attempt=attempt, seed=seed, payload={}))
        env = None
        phase = "construction"
        try:
            cls = importlib.import_module("stable_worldmodel.envs.pusht.env").PushT
            require(sha(inspect.getsourcefile(cls)) == native.ENV_SHA, "native class source drift")
            env = cls(render_mode="rgb_array", resolution=224, with_target=False) if factory is None else factory()
            require(type(env) is cls if factory is None else True, "native class identity")
            phase = "reset_dispatched"
            obs, info = env.reset(seed=seed)
            phase = "reset_returned_capture"
            state, goal = np.asarray(obs["state"], np.float64), np.asarray(info["goal_state"], np.float64)
            require(state.shape == goal.shape == (7,) and np.isfinite(state).all()
                    and np.isfinite(goal).all(), "native observation")
            snapshot = native.snapshot(env)
            require(np.array_equal(state, np.asarray(snapshot["state"])), "actual state mismatch")
            receipt = dict(schema=NAME + ".receipt.v1", status="reset_returned_history_pending",
                           reservation_sha256=sha(reservation_path), queue_file_sha256=expected_sha,
                           queue_digest=q.digest(queue), queue_attempt_event_sha256=start,
                           slot_id=slot_id, attempt=attempt, seed=seed, state=state.tolist(),
                           goal=goal.tolist(), snapshot=snapshot,
                           state_sha256=q.digest(state.tolist()), goal_sha256=q.digest(goal.tolist()),
                           snapshot_sha256=q.digest(snapshot), history_sha256=None,
                           history_status="not_collected", independent_confirmation=False,
                           historical_coverage="unknown", training_membership="unknown", P1_admission=False)
            require(queue == q.read_json(path) and sha(path) == expected_sha
                    and all(queue["spec"][k] == v for k, v in expected_spec_bindings().items())
                    and reservation["source_sha256"] == native.dependencies(), "post-reset drift")
            q.write_new(receipt_path, receipt)
            # The old queue requires a real history hash. Keep the attempt
            # pending instead of passing a manufactured or unrelated hash.
            return receipt
        except Exception as exc:
            failure = dict(schema=NAME + ".receipt.v1", status="failed",
                           reservation_sha256=sha(reservation_path), queue_file_sha256=expected_sha,
                           queue_digest=q.digest(queue), queue_attempt_event_sha256=start,
                           slot_id=slot_id, attempt=attempt, seed=seed, phase=phase,
                           error_class=type(exc).__name__, message=str(exc)[:500],
                           independent_confirmation=False, historical_coverage="unknown",
                           training_membership="unknown", P1_admission=False)
            q.write_new(receipt_path, failure)
            q.append(path, ledger, dict(kind="reset_failed", slot_id=slot_id, attempt=attempt,
                                       seed=seed, payload={"error_class": type(exc).__name__}))
            return failure
        finally:
            if env is not None:
                env.close()
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--queue-sha256", required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--authorize-development-reset", action="store_true", required=True)
    args = parser.parse_args()
    with native.locking.gpu_lock(native.LOCK):
        result = run(args.queue, args.queue_sha256, args.slot, args.attempt)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "reset_returned_history_pending":
        sys.exit(1)


if __name__ == "__main__":
    main()
