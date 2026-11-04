"""Audit final-prefix counterfactuals and replay the sealed feedback actions."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_final_chain_abstention_rgb_dev_20261103_codex as pilot
from experiments import pc_fixed_terminal_fair_feedback_physical_20261103_codex as replay
from experiments import pc_traced_adaptive_rgb_dev_20261103_codex as adaptive


def main():
    run, sha = replay.run, replay.run.old.prior.sha
    run.PROTOCOL, run.PREFIX, run.ARMS = pilot.PROTOCOL, pilot.PREFIX, pilot.ARMS
    replay.OUTPUT = pilot.PREFIX.with_name(pilot.PREFIX.name + "_physical.json")
    for root in range(4):
        for arm in pilot.ARMS:
            decisions, _, _ = run.paths(root, arm)
            for path in decisions:
                row = json.loads(path.read_text())
                result, counts = row["result"], row["counts"]
                if (result["final_abstention_source_sha256"] != sha(Path(pilot.__file__))
                        or result["adaptive_source_sha256"] != sha(Path(adaptive.__file__))
                        or result["base_policy_sha256"] != sha(Path(adaptive.policy.__file__))):
                    raise ValueError("policy source drift")
                check = result["final_prefix_comparison"]
                original = np.asarray(check["precomparison_chain"])
                comparison = np.asarray(check["comparison_chain"])
                selected = np.asarray(result["selected"]["chain"])
                expected = original.copy()
                if arm != "cem":
                    expected[:5] = 0.
                np.testing.assert_array_equal(comparison, expected)
                abstain = check["comparison_score"] <= check["precomparison_score"]
                if arm != "cem" and result["abstained_reconfiguration"] != abstain:
                    raise ValueError("incorrect final abstention")
                np.testing.assert_array_equal(selected, comparison if arm != "cem" and abstain else original)
                tokens, samples, found = 0, 0, False
                for batch in result["prediction_trace"]:
                    requested = np.asarray(batch["requested_actions"])
                    np.testing.assert_array_equal(requested.astype(np.float32), np.asarray(batch["model_actions"]))
                    cost = len(requested) * requested.shape[1] // 5
                    if cost != batch["tokens"]:
                        raise ValueError("trace token mismatch")
                    tokens += cost
                    samples += len(requested)
                    if requested.shape[1:] == original.shape:
                        indices = np.flatnonzero(np.all(requested == original, axis=(1, 2)))
                        found |= any(abs(batch["scores"][i] - check["precomparison_score"]) < 1e-7
                                     for i in indices)
                reserve = len(original) // 5
                if (not found or check["additional_prediction_tokens"] != reserve
                        or tokens + reserve != counts["prediction_token_samples"]
                        or samples + 1 != counts["trajectory_samples"]
                        or len(result["prediction_trace"]) + 1 != counts["prediction_calls"]):
                    raise ValueError("prediction accounting or selection mismatch")
    replay.main()
    print("Validated 48 final-chain decisions and replayed 16 trajectories.", flush=True)


if __name__ == "__main__":
    main()
