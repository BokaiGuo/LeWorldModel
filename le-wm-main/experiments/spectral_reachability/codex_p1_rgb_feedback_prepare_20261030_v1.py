"""Predeclare a known-root RGB feedback candidate; does not authorize a run."""
import argparse
import json
import os
from pathlib import Path

from . import codex_p1_rgb_feedback_candidate_20261030_v1 as candidate
from . import codex_p1_two_stage_launcher_20261030_v1 as old

NAME = "codex_p1_rgb_feedback_prepare_20261030_v1"
PRIOR = old.ROOT/"results/codex_p1_two_stage_launcher_20261030_v1_prepare_CODEX01.json"
PRIOR_SHA = "edea86663dd55caa9a7c5a6f48862030fb84248164df1691f38a82b2ca9672c5"
ENV = old.ROOT/".venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py"
MODEL = old.ROOT/".venv/lib/python3.12/site-packages/stable_worldmodel/wm/lewm/lewm.py"


def build():
    source = old.verify(PRIOR, PRIOR_SHA)
    roster = source["roster"]
    old.need(len(roster) == 24 and [r["index"] for r in roster] == list(range(24)), "fixed roster")
    source_hashes = dict(source["sources"])
    for p in (Path(__file__), Path(candidate.__file__), Path(candidate.rgb.__file__), ENV, MODEL,
              old.ROOT/"tests/test_codex_p1_rgb_feedback_candidate_20261030_v1.py",
              old.ROOT/"tests/test_codex_p1_rgb_feedback_prepare_20261030_v1.py", PRIOR):
        source_hashes[str(p.resolve())] = old.sha(p)
    old.need(all(old.sha(path) == digest for path, digest in source_hashes.items()), "source drift")
    # This roster binds identity and known prefix, not observed future outcomes.
    fixed = [{**{k: row[k] for k in ("index", "logical_pair", "root_slot", "condition", "arm",
                                     "role", "task_id", "physical_root_sha256", "prefix5_float32_hex")},
              "cp5_file_sha256": row["checkpoint"]["file_sha256"],
              "cp5_pixel_sha256": row["checkpoint"]["pixel_sha256"]} for row in roster]
    return {"schema": NAME, "fixed_rows": 24, "roster": fixed, "policy": candidate.POLICY,
            "neural_caps_each_stage": source["stage_caps"], "encoding_caps_each_stage": source["encoding_caps"],
            "physical_caps_each_row": source["physical_caps_per_row"],
            "rgb_evaluations_cap_each_row": 1, "stages_max": 2,
            "resource_policy": "same caps per fixed row; actual consumption recorded, no call padding",
            "failure_policy": "RGB rejection means no second-stage action; other errors fail closed",
            "native_stop_policy": "retain stop, no later plan or action; no fabricated endpoint",
            "prior_roster_prepare_sha256": PRIOR_SHA, "sources": source_hashes,
            "future_rgb_prediction_available": False, "frozen_lewm_has_rgb_decoder": False,
            "planning_input_whitelist": ["current_rgb", "goal_rgb", "current_model_packet"],
            "known_roots": True, "post_outcome_exploration": True, "os_isolation_certified": False,
            "dependency_closure_complete": False, "cooperative_lock_not_global": True,
            "physical_execution_authorized": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    path = Path(parser.parse_args().output)
    old.need(path.is_absolute() and path.parent == old.ROOT/"results" and
             path.name.startswith(NAME+"_") and not os.path.lexists(path), "exclusive output")
    result = build()
    print(json.dumps({"sha256": old.h10._write_new(path, result), "rows": result["fixed_rows"]}))


if __name__ == "__main__":
    main()
