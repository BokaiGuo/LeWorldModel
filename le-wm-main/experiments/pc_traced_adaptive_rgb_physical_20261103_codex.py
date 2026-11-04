"""Validate full model-input traces and replay sealed RGB trajectories."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_fixed_terminal_fair_feedback_physical_20261103_codex as replay
from experiments import pc_traced_adaptive_rgb_dev_20261103_codex as pilot


def main():
    run = replay.run
    run.PROTOCOL, run.PREFIX, run.ARMS = pilot.PROTOCOL, pilot.PREFIX, pilot.ARMS
    replay.OUTPUT = pilot.PREFIX.with_name(pilot.PREFIX.name + "_physical.json")
    sha = run.old.prior.sha
    for root in range(4):
        for arm in pilot.ARMS:
            paths, _, _ = run.paths(root, arm)
            for path in paths:
                decision = json.loads(path.read_text())
                result = decision["result"]
                if (result["adaptive_source_sha256"] != sha(Path(pilot.__file__))
                        or result["base_policy_sha256"] != sha(Path(pilot.policy.__file__))):
                    raise ValueError("policy source hash mismatch")
                chain = np.asarray(result["selected"]["chain"])
                found, tokens, samples = False, 0, 0
                for batch in result["prediction_trace"]:
                    requested = np.asarray(batch["requested_actions"])
                    actual = np.asarray(batch["model_actions"])
                    if not np.array_equal(requested.astype(np.float32), actual.astype(np.float32)):
                        raise ValueError("model input conversion mismatch")
                    if requested.shape[1:] == chain.shape:
                        indices = np.flatnonzero(np.all(requested == chain, axis=(1, 2)))
                        for index in indices:
                            if abs(batch["scores"][int(index)] - result["selected"]["terminal_score"]) < 1e-7:
                                found = True
                    cost = len(requested) * requested.shape[1] // 5
                    if cost != batch["tokens"]:
                        raise ValueError("trace token mismatch")
                    tokens += cost
                    samples += len(requested)
                counts = decision["counts"]
                if (not found or tokens != counts["prediction_token_samples"]
                        or samples != counts["trajectory_samples"]
                        or len(result["prediction_trace"]) != counts["prediction_calls"]):
                    raise ValueError("model trace/decision mismatch")
    replay.main()
    print("validated all model-input batches and 16 physical trajectories", flush=True)


if __name__ == "__main__":
    main()
