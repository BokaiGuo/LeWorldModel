"""Attribute saved v25 calls to search stages via prediction-record replay."""
import json
from pathlib import Path
import sys

import numpy as np

from experiments import pc_v25_e0_prospective_same_chain_dev_20261104_codex as panel


def caller_source(frame):
    while frame is not None:
        if (frame.f_code.co_name == "evaluate"
                and frame.f_globals.get("__name__") == panel.compiler.core.__name__):
            return frame.f_locals["source"]
        frame = frame.f_back
    return "global_proposals"


def add_batch(stages, source, actions, scores):
    entry = stages.setdefault(source, {"calls": 0, "samples": 0, "tokens": 0, "best_full_score": None})
    entry["calls"] += 1
    entry["samples"] += len(actions)
    entry["tokens"] += len(actions) * (actions.shape[1] // 5)
    if actions.shape[1] == 15:
        best = float(np.min(scores))
        if entry["best_full_score"] is None or best < entry["best_full_score"]:
            entry["best_full_score"] = best


def main():
    output = panel.OUT / "search_loss_posthoc.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    plans = json.loads((panel.OUT / "plans.json").read_text())
    plan_audit = json.loads((panel.OUT / "plans_audit.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    if plan_audit["status"] != "passed" or plan_audit["plans_sha256"] != panel.seal.digest(panel.OUT / "plans.json"):
        raise ValueError("unsealed source plans")
    results = []
    for row in plans["rows"]:
        root = row["root"]
        if row["status"] != "completed":
            results.append({"root": root, "status": "failed"})
            continue
        calls = iter(panel.fixed.read_calls(panel.OUT / f"plan_root{root:02d}_calls"))
        stages = {}

        def replay(actions):
            source = caller_source(sys._getframe(1))
            actual, features, scores = next(calls)
            np.testing.assert_array_equal(actions, actual)
            add_batch(stages, source, actual, scores)
            return features, scores

        saved = row["result"]
        rebuilt = panel.compiler.compile_program(replay, np.asarray(row["task"]["goal"]),
            panel.compiler.Config(**saved["config"]), score_scale=1., score_target=0.)
        if next(calls, None) is not None:
            raise ValueError("unconsumed recorded prediction")
        for key in ("selected", "selected_score", "incumbent", "comparison", "paired_scores", "abstained"):
            if saved[key] != rebuilt[key]:
                raise ValueError("selected-chain replay mismatch")
        for key in ("layers", "refinement", "refinement_starts", "incumbent_source"):
            if saved["child"][key] != rebuilt["child"][key]:
                raise ValueError(f"search replay mismatch: {key}")
        if sum(x["tokens"] for x in stages.values()) != row["journal_tokens"]:
            raise ValueError("stage cost mismatch")
        layer = saved["child"]["layers"][0]
        greedy_index = int(np.argmin([candidate["score"] for candidate in layer["candidates"]]))
        greedy_prefix = np.asarray(layer["candidates"][greedy_index]["chain"], np.float32)
        selected_prefix = np.asarray(saved["selected"], np.float32)[:5]
        results.append({"root": root, "status": "completed", "stages": stages,
            "incumbent_source": saved["child"]["incumbent_source"],
            "selected_score": saved["selected_score"],
            "selected_prefix_equals_first_layer_greedy": bool(np.array_equal(greedy_prefix, selected_prefix)),
            "first_layer_response_added": layer["response_added"],
            "first_layer_response_removed": layer["response_removed"],
            "refinement_starts": saved["child"]["refinement_starts"]})
    panel.seal.verify(manifest["dependencies"])
    panel.journal.write_json(output, {"scope": "post-hoc deterministic replay of recorded predictions; no new physics",
        "plans_sha256": panel.seal.digest(panel.OUT / "plans.json"),
        "analysis_sha256": panel.seal.digest(Path(__file__)), "rows": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
