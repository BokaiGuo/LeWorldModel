"""Abrupt process-stop fixture; no real environment or CUDA."""
import os
from pathlib import Path
import sys

import numpy as np
from experiments import pc_pointmaze_durable_dispatch_20261103_codex as d

root, stage = Path(sys.argv[1]), sys.argv[2]

class Fake:
    def __init__(self, crash=False):
        self.clock = 0
        self.crash = crash
    def advance(self, action):
        if self.crash:
            os._exit(73)
        self.clock += 1
        return np.zeros((224, 224, 3), dtype=np.uint8)

with d.DurableJournal(root/"events.jsonl") as journal:
    action = np.array([0.25, -0.5])
    path = root/"decision.npz"
    digest = d.save_snapshot(path, action=action, clock=np.asarray(1))
    def hook(role):
        if role == "shadow" and stage == "after_shadow":
            os._exit(73)
    d.dispatch_pair(Fake(), Fake(stage == "during_legacy"), action, path, digest, journal,
                    root/"response", "root", 1, "prefix", after_return=hook)
