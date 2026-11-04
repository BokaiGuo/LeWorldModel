"""Trace audit and native replay of successor RGB development decisions."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_fixed_terminal_fair_feedback_physical_20261103_codex as replay
from experiments import pc_successor_rgb_execution_dev_20261103_codex as pilot


def main():
    run, sha = replay.run, replay.run.old.prior.sha
    run.PROTOCOL, run.PREFIX, run.ARMS = pilot.PROTOCOL, pilot.PREFIX, pilot.ARMS
    replay.OUTPUT = pilot.PREFIX.with_name(pilot.PREFIX.name + "_physical.json")
    for root in range(4):
        for arm in pilot.ARMS:
            decisions, _, _ = run.paths(root, arm)
            for path in decisions:
                row = json.loads(path.read_text())
                result = row["result"]
                if result["policy_sha256"] != sha(Path(pilot.__file__)):
                    raise ValueError("policy source drift")
                selected = np.asarray(result["selected"]["chain"])
                check = result["final_prefix_comparison"]
                original = np.asarray(check["precomparison_chain"])
                comparison = np.asarray(check["comparison_chain"])
                expected = original.copy()
                if arm == "successor":
                    expected[:5] = 0.
                np.testing.assert_array_equal(expected, comparison)
                abstain = arm == "successor" and check["comparison_score"] <= check["precomparison_score"]
                if result["abstained_reconfiguration"] != (bool(abstain) if arm == "successor" else None):
                    raise ValueError("abstention mismatch")
                np.testing.assert_array_equal(selected, comparison if abstain else original)
                tokens, samples, found_original, found_comparison = 0, 0, False, False
                for batch in result["prediction_trace"]:
                    requested = np.asarray(batch["requested_actions"])
                    np.testing.assert_array_equal(requested.astype(np.float32), np.asarray(batch["model_actions"]))
                    cost = len(requested) * requested.shape[1] // 5
                    if cost != batch["tokens"]:
                        raise ValueError("token mismatch")
                    tokens += cost
                    samples += len(requested)
                    if requested.shape[1:] == original.shape:
                        for chain, key in ((original, "precomparison_score"), (comparison, "comparison_score")):
                            indices = np.flatnonzero(np.all(requested == chain, axis=(1, 2)))
                            found = any(abs(batch["scores"][i] - check[key]) < 1e-7 for i in indices)
                            if key == "precomparison_score":
                                found_original |= found
                            else:
                                found_comparison |= found
                counts = row["counts"]
                if (not found_original or not found_comparison or tokens != counts["prediction_token_samples"]
                        or samples != counts["trajectory_samples"]
                        or len(result["prediction_trace"]) != counts["prediction_calls"]):
                    raise ValueError("trace mismatch")
                if arm == "successor" and row["step"] == 0:
                    if len(result["bank"]) != 4 or any(x["samples"] < 13 for x in result["programs"]):
                        raise ValueError("successor bank or suffix search mismatch")
    replay.main()
    print("Validated 24 decision traces and replayed eight trajectories.", flush=True)


if __name__ == "__main__":
    main()
