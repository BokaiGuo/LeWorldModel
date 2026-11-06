"""Replay both stopped ablation arms against their recorded predictor calls."""
import json

import numpy as np

from experiments import pc_v25_successor_probe_cost_e0_dev_20261104_codex as panel


def main():
    output = panel.OUT / "optimizer_replay_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    results = []
    for root, task in enumerate(panel.tasks()):
        for arm in panel.ARMS:
            row = json.loads((panel.OUT / f"root{root:02d}_{arm}.json").read_text())
            if row["status"] != "completed":
                results.append({"root": root, "arm": arm, "status": "failed"})
                continue
            module = panel.frozen if arm == "response" else panel.ablation
            calls = iter(panel.prior.read_calls(panel.OUT / f"root{root:02d}_{arm}_calls"))

            def replay(actions):
                recorded, features, scores = next(calls)
                np.testing.assert_array_equal(actions, recorded)
                return features, scores

            saved = row["result"]
            reproduced = module.compile_program(replay, np.asarray(task["goal"]),
                module.Config(**saved["config"]), score_scale=1., score_target=0.)
            if next(calls, None) is not None:
                raise ValueError("unconsumed prediction calls")
            for key in ("selected", "selected_score", "incumbent", "comparison", "paired_scores", "abstained"):
                if reproduced[key] != saved[key]:
                    raise ValueError("selection replay mismatch")
            for key in ("layers", "refinement", "refinement_starts"):
                if reproduced["child"][key] != saved["child"][key]:
                    raise ValueError(f"compiler replay mismatch: {key}")
            results.append({"root": root, "arm": arm, "status": "passed",
                            "prediction_calls": row["journal_calls"]})
    panel.seal.verify(manifest["dependencies"])
    panel.OLD.writer(output, {"status": "passed", "rows": results,
        "scope": "same-code deterministic replay, not independent replication"})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
