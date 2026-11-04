"""Post-outcome random-bridge/lookahead interventions on final selected chains."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_rgb_successor_exact_controls_dev_20261103_codex as controls

prior = controls.prior
PROTOCOL = prior.ROOT.parent / "docs/PC_RGB_SUCCESSOR_EXACT_EXTRA_INTERVENTION_DIAG_PROTOCOL_20261103_CODEX.md"
OUTPUT = prior.PREFIX.with_name(prior.PREFIX.name + "_extra_intervention_diagnostic.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plans = json.loads(prior.PLANS.read_text())
    previous = json.loads(prior.INTERVENTION.read_text())
    comparators = json.loads(controls.PLANS.read_text())
    audited = json.loads(controls.AUDIT.read_text())
    if (plans["dependencies"] != prior.dependencies() or previous["plans_sha256"] != prior.sha(prior.PLANS)
            or comparators["dependencies"] != controls.dependencies()
            or audited["plans_sha256"] != prior.sha(controls.PLANS)):
        raise ValueError("source seal mismatch")
    index = {(row["root"], row["arm"]): row for row in comparators["rows"]}
    rows = []
    for row in plans["rows"]:
        root, task = row["root"], row["task"]
        chain = np.asarray(row["result"]["selected"]["chain"])
        prefixes = {"successor": chain[:5]}
        for arm in ("random_bridge", "lookahead"):
            prefixes[arm] = np.asarray(index[root, arm]["selected"]["chain"])[:5]
        suffixes = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        measurements = {}
        for name, prefix in prefixes.items():
            actions = np.asarray([np.concatenate((prefix, tail)) for tail in suffixes])
            values = np.linalg.norm(prior.exact.endpoints(task, actions) - task["goal"], axis=1)
            best, score, scores = controls.optimize(
                prefix, 15, 120, np.random.default_rng(2026121400 + root),
                lambda a: np.linalg.norm(prior.exact.endpoints(task, a) - task["goal"], axis=1))
            measurements[name] = {"prefix": prefix.tolist(), "common_scores": values.tolist(),
                                  "relative_changes": (values - values[0]).tolist(),
                                  "own_chain": best.tolist(), "own_score": score, "own_scores": scores}
        rows.append({"root": root, "selected_chain": chain.tolist(), "measurements": measurements})
    with OUTPUT.open("x") as stream:
        json.dump({"protocol_sha256": prior.sha(PROTOCOL), "runner_sha256": prior.sha(Path(__file__)),
                   "plans_sha256": prior.sha(prior.PLANS), "previous_intervention_sha256": prior.sha(prior.INTERVENTION),
                   "control_plans_sha256": prior.sha(controls.PLANS), "control_audit_sha256": prior.sha(controls.AUDIT),
                   "rows": rows}, stream, indent=2, allow_nan=False)
    print(f"Computed extra interventions on {len(rows)} actual successor chains.", flush=True)


if __name__ == "__main__":
    main()
