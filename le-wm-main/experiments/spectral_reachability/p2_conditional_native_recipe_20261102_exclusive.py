"""Construct candidate-specific native recipe; no physical executor is implied."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p2_conditional_physical_handoff_20261102_exclusive as handoff

NAME = "p2_conditional_native_recipe_20261102_exclusive"
PLAN = handoff.RESULTS / (handoff.NAME + "_PLAN_01.json")
PLAN_SHA = "df6c41689dc75340b16e297bc12a3267bb554e1dae82c0854468d2c1b69fa035"


def build():
    plan = handoff.verify(PLAN, PLAN_SHA)
    original = json.loads(handoff.checked(handoff.old.PREP, handoff.old.PREP_SHA))
    template = original["plans"][0]
    recipes = []
    for index, entry in enumerate(plan["arms"]):
        action = bytes.fromhex(entry["full25_raw_hex"])
        if (len(action) != 200 or hashlib.sha256(action).hexdigest() != entry["full25_sha256"]
                or hashlib.sha256(action[120:]).hexdigest() != entry["suffix_sha256"]):
            raise ValueError("new action hash")
        raw = np.frombuffer(action, dtype="<f4").reshape(25, 2)
        handoff.old.physical.raw25(raw.astype(float).tolist())
        witness = {"schema": NAME + ".winner_reference_v1", "arm": entry["arm"],
                   "model_report_sha256": handoff.REPORT_SHA, "model_plan_sha256": handoff.MODEL_PLAN_SHA,
                   "handoff_sha256": PLAN_SHA, "candidate_index": entry["candidate_index"],
                   "iteration": entry["iteration"], "suffix_sha256": entry["suffix_sha256"],
                   "input_context": deepcopy(original["observed_context"]),
                   "scope": "new optimized winner reference, not historical model witness"}
        witness["new_witness_sha256"] = handoff.old.physical.digest(witness)
        new = deepcopy(template)
        new.update(index=index, policy=entry["arm"], commands=raw.astype(float).tolist(),
                   raw_float32_hex=action.hex(), raw_float32_sha256=entry["full25_sha256"],
                   new_model_plan=witness)
        new.pop("selection_digest")
        new["selection_digest"] = handoff.old.physical.digest(new)
        recipes.append(new)
    return {"schema": NAME + ".recipe_v1", "handoff_sha256": PLAN_SHA,
            "recipe_source_sha256": handoff.host.file_sha(Path(__file__)),
            "old_prepared_sha256": handoff.old.PREP_SHA, "plans": recipes,
            "prefix_reference_sha256": plan["prefix_reference_sha256"],
            "execution_authorized": False,
            "blocker": "new-witness-aware native execution, postcheck and saved consumer not implemented",
            "known_root": True, "equal_total_compute": False,
            "independent_confirmation": False, "P1_admission": False, "P2_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if (not path.is_absolute() or path.parent != handoff.RESULTS or
            path.parent.resolve() != handoff.RESULTS or not path.name.startswith(NAME + "_")):
        raise ValueError("exclusive recipe output path")
    print(json.dumps({"recipe": str(path), "sha256": handoff.host.write_new(path, build())}))


if __name__ == "__main__":
    main()
