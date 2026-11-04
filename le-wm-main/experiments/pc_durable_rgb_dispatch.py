"""Exclusive, per-action RGB dispatch journal for simulated feedback runs."""
import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_frame(path, frame):
    frame = np.asarray(frame)
    if frame.dtype != np.uint8 or frame.shape != (224, 224, 3):
        raise ValueError("invalid RGB frame")
    with Path(path).open("xb") as stream:
        np.save(stream, frame, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def dispatch(env, actions, directory):
    """Persist each attempt before dispatch and its returned RGB immediately.

    An attempt without a return is deliberately ambiguous: it must not be
    replayed automatically because the environment may have executed it.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if (actions.ndim != 2 or actions.shape[1] != 2 or len(actions) == 0
            or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
        raise ValueError("invalid actions")
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    frames, times = [], []
    for index, action in enumerate(actions):
        before = time.monotonic_ns()
        write_json(directory / f"{index:03d}_attempt.json",
                   {"index": index, "action": action.tolist(), "before_ns": before})
        frame = np.asarray(env.step(action.reshape(1, 2))).copy()
        after = time.monotonic_ns()
        image_path = directory / f"{index:03d}_rgb.npy"
        digest = write_frame(image_path, frame)
        write_json(directory / f"{index:03d}_returned.json",
                   {"index": index, "after_ns": after, "rgb_file": image_path.name, "rgb_sha256": digest})
        frames.append(frame)
        times.append([before, after])
    return frames, times


def inspect(directory):
    """Validate a journal and report completion or a non-retriable attempt."""
    directory = Path(directory)
    attempts = sorted(directory.glob("*_attempt.json"))
    expected_names = {f"{i:03d}_attempt.json" for i in range(len(attempts))}
    if {p.name for p in attempts} != expected_names:
        raise ValueError("noncontiguous journal attempts")
    allowed = set(expected_names)
    pending = None
    for index, path in enumerate(attempts):
        record = json.loads(path.read_text())
        action = np.asarray(record.get("action"), dtype=np.float64)
        if (record.get("index") != index or type(record.get("before_ns")) is not int
                or action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1)):
            raise ValueError("invalid attempt record")
        image = directory / f"{index:03d}_rgb.npy"
        returned = directory / f"{index:03d}_returned.json"
        if not returned.exists():
            if index != len(attempts) - 1:
                raise ValueError("actions recorded after an incomplete attempt")
            pending = index
            if image.exists():
                allowed.add(image.name)
            continue
        allowed.update((image.name, returned.name))
        event = json.loads(returned.read_text())
        if (event.get("index") != index or event.get("rgb_file") != image.name
                or type(event.get("after_ns")) is not int or event["after_ns"] < record["before_ns"]):
            raise ValueError("invalid returned record")
        frame = np.load(image, allow_pickle=False)
        if (frame.dtype != np.uint8 or frame.shape != (224, 224, 3)
                or hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest() != event.get("rgb_sha256")):
            raise ValueError("returned RGB mismatch")
    if {path.name for path in directory.iterdir()} != allowed:
        raise ValueError("unexpected journal files")
    return {"attempts": len(attempts), "returned": len(attempts) - int(pending is not None),
            "pending": pending, "safe_to_retry": False}
