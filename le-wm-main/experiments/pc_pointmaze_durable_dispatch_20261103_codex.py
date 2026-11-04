"""Synchronous, fail-closed audit records for independent bridge commands."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_snapshot(path, **arrays):
    path = Path(path)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(path.parent)
    return sha(path)


class DurableJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("x")
        try:
            _sync_dir(self.path.parent)
        except BaseException:
            self.file.close()
            raise
        self.index = 0

    def record(self, event, **fields):
        entry = {"index": self.index, "event": event, "time_ns": time.time_ns(), **fields}
        self.file.write(json.dumps(entry, sort_keys=True, allow_nan=False)+"\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.index += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()


def _verify_snapshot(path, digest, clock, action=None):
    if sha(path) != digest:
        raise ValueError("snapshot hash mismatch")
    with np.load(path, allow_pickle=False) as snap:
        if int(snap["clock"]) != clock:
            raise ValueError("snapshot clock mismatch")
        if action is not None and not np.array_equal(snap["action"], action):
            raise ValueError("snapshot action mismatch")


def _rgb(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != (224, 224, 3):
        raise RuntimeError("invalid bridge RGB")
    return image


def dispatch_reset(role, create, snapshot, digest, journal, output_prefix, branch):
    _verify_snapshot(snapshot, digest, 0)
    journal.record("bridge_attempt", bridge=role, phase="reset", branch=branch, clock=0,
                   snapshot_path=str(snapshot), snapshot_sha256=digest)
    env = create()
    try:
        if env.step_count != 0:
            raise RuntimeError("reset clock mismatch")
        image = _rgb(env.initial_rgb)
        path = Path(str(output_prefix)+"_"+role+".npz")
        image_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        response_sha = save_snapshot(path, rgb=image, clock=np.asarray(0))
        journal.record("bridge_returned", bridge=role, phase="reset", branch=branch, clock=0,
                       snapshot_path=str(snapshot), snapshot_sha256=digest,
                       rgb_sha256=image_sha, rgb_path=str(path), artifact_sha256=response_sha)
        return env
    except BaseException:
        env.close()
        raise


def dispatch_pair(shadow, legacy, action, snapshot, digest, journal, output_prefix,
                  branch, clock, phase, after_return=None):
    action = np.asarray(action)
    if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError("invalid action")
    images, timings = [], []
    for role, bridge in (("shadow", shadow), ("legacy", legacy)):
        _verify_snapshot(snapshot, digest, clock, action)
        journal.record("bridge_attempt", bridge=role, phase=phase, branch=branch, clock=clock,
                       snapshot_path=str(snapshot), snapshot_sha256=digest, action=action.tolist())
        began = time.monotonic()
        image = _rgb(bridge.advance(action.copy()))
        elapsed = time.monotonic()-began
        if bridge.clock != clock:
            raise RuntimeError("bridge clock mismatch")
        path = Path(str(output_prefix)+"_"+role+".npz")
        image_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        response_sha = save_snapshot(path, rgb=image, clock=np.asarray(clock))
        journal.record("bridge_returned", bridge=role, phase=phase, branch=branch, clock=clock,
                       snapshot_path=str(snapshot), snapshot_sha256=digest, action=action.tolist(),
                       rgb_sha256=image_sha, rgb_path=str(path), artifact_sha256=response_sha,
                       seconds=elapsed)
        images.append(image.copy())
        timings.append(elapsed)
        if after_return is not None:
            after_return(role)
    return images[0], images[1], timings[0], timings[1]
