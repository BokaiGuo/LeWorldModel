"""Append-only prediction-call journal, including attempted failures."""
import hashlib
import os
from pathlib import Path
import time

import numpy as np

from experiments.pc_durable_rgb_dispatch import write_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_arrays(path, **arrays):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


class PredictionJournal:
    def __init__(self, directory, predict, *, action_dim, packet_length, token_cap):
        if min(action_dim, packet_length, token_cap) < 1:
            raise ValueError("invalid prediction contract")
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.predict, self.action_dim, self.packet_length = predict, action_dim, packet_length
        self.token_cap, self.tokens, self.calls = token_cap, 0, 0
        write_json(self.directory / "contract.json", {"action_dim": action_dim,
                   "packet_length": packet_length, "token_cap": token_cap})

    def __call__(self, actions):
        actions = np.asarray(actions, np.float32)
        if (actions.ndim != 3 or not len(actions) or not actions.shape[1]
                or actions.shape[2] != self.action_dim or actions.shape[1] % self.packet_length
                or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1)):
            raise ValueError("invalid prediction actions")
        cost = len(actions) * (actions.shape[1] // self.packet_length)
        if self.tokens + cost > self.token_cap:
            raise RuntimeError("prediction budget exceeded")
        index = self.calls
        action_path = self.directory / f"{index:06d}_actions.npz"
        write_arrays(action_path, actions=actions)
        write_json(self.directory / f"{index:06d}_attempt.json",
                   {"index": index, "cost": cost, "actions": action_path.name,
                    "sha256": digest(action_path), "before_ns": time.monotonic_ns()})
        self.calls += 1
        self.tokens += cost
        try:
            features, scores = self.predict(actions.copy())
            features, scores = np.asarray(features), np.asarray(scores)
            if (features.ndim != 2 or len(features) != len(actions)
                    or scores.shape != (len(actions),) or not np.isfinite(features).all()
                    or not np.isfinite(scores).all()):
                raise ValueError("invalid predictor output")
            result_path = self.directory / f"{index:06d}_result.npz"
            write_arrays(result_path, features=features, scores=scores)
            write_json(self.directory / f"{index:06d}_returned.json",
                       {"index": index, "result": result_path.name, "sha256": digest(result_path),
                        "after_ns": time.monotonic_ns()})
        except Exception as error:
            write_json(self.directory / f"{index:06d}_failed.json",
                       {"index": index, "type": type(error).__name__, "message": str(error),
                        "after_ns": time.monotonic_ns()})
            raise
        return features, scores
