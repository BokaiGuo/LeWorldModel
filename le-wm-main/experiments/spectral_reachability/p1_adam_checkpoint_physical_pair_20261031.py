"""Join fixed 5/10/20-update physical outcomes by the same executed prefix."""
import argparse
import json
from pathlib import Path

from . import p1_adam_checkpoint_physical_20261031 as physical
from . import p1_uphill_multistart_pairing_20261030_main as pairing

ROOT = physical.ROOT
NAME = "p1_adam_checkpoint_physical_pair_20261031"
SOURCES = {
    physical.PLAN: physical.PLAN_SHA,
    physical.prior.PLAN: physical.prior.PLAN_SHA,
    pairing.ANCHORS: pairing.ANCHORS_SHA,
    pairing.ROOT / "results/p1_uphill_multistart_pairing_20261030_main_RUN01.json":
        "b69b9d38b7f926c31648dc00d0445e710a1014bdff2dff216a979c43ac397ff4",
    pairing.ROOT / "results/p1_adam_checkpoint_physical_5_20261031_prepare_RUN01.json":
        "eaec5459a4aa5755941ef2749fb0f6e4264600eb3c33f379b5266860a8b938af",
    pairing.ROOT / "results/p1_adam_checkpoint_physical_10_20261031_prepare_RUN01.json":
        "5be37dd41e7edfb228390a8924ecfaf9688ea0e54cd93386a3d042aaf43b9202",
    pairing.ROOT / "results/p1_adam_checkpoint_physical_5_20261031_consume_RUN01.json":
        "7b413749a1a792bab575a6574e1027f237bc38f376cbe61de5bc01599595dd2f",
    pairing.ROOT / "results/p1_adam_checkpoint_physical_10_20261031_consume_RUN01.json":
        "21158d681fe62d9a42cbfd046ea5b7080403a410b50e733ffd7d78093e4ecedc",
}


def join(anchors, curve, prepared, consumed, twenty, old_plan):
    pairing.need(len(twenty["rows"]) == 24 and twenty["P1_admission"] is False,
                 "twenty roster")
    levels = {}
    for step in (5, 10):
        plan = physical.convert(curve, step)
        for row in plan["rows"]:
            # These fields are unavailable per checkpoint in the shared CUDA run.
            row["calls"] = {}
            row["planning_wall_seconds"] = None
        levels[step] = pairing.join(anchors, prepared[step], plan, consumed[step])["rows"]
    rows = []
    for index in range(24):
        new = [levels[step][index] for step in (5, 10)]
        old = twenty["rows"][index]
        for row in new:
            pairing.need(all(row[k] == old[k] for k in
                             ("index", "logical_pair", "root_slot", "physical_root_sha256",
                              "condition", "arm", "role", "task_id", "root", "prefix")),
                         "cross-budget prefix identity")
        saved = curve["rows"][index]["checkpoints"]["20"]
        pairing.need(saved["winner"] == old["chosen_start"]
                     and saved["scores"] == old["model_all_scores"]
                     and old_plan["rows"][index]["index"] == index
                     and saved["candidate_sha256"][saved["winner"]]
                     == old_plan["rows"][index]["selected_action_sha256"], "twenty score and action identity")
        rows.append({"index": index, "logical_pair": old["logical_pair"], "root_slot": old["root_slot"],
                     "condition": old["condition"], "role": old["role"], "task_id": old["task_id"],
                     "physical_root_sha256": old["physical_root_sha256"], "root": old["root"],
                     "prefix": old["prefix"], "prefix_minus_root": old["prefix_minus_root"],
                     "budgets": {str(step): {
                         "model_best_score": row["model_best_score"], "chosen_start": row["chosen_start"],
                         "status": row["status"], "returned_suffix": row["returned_suffix"],
                         "last_native_success": row["last_native_success"], "endpoint": row["endpoint"],
                         "endpoint_minus_zero_prefix": row["endpoint_minus_zero_prefix"],
                     } for step, row in ((5, new[0]), (10, new[1]), (20, old))}})
    return {"schema": NAME, "rows": rows, "P1_admission": False, "independent_confirmation": False,
            "equal_budget_certified": False, "limitations": "Shared optimization trajectory and known roots; "
            "no independently measured per-budget runtime or total prefix cost; no imputation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    pairing.need(out.is_absolute() and out.parent == ROOT / "results" and out.name.startswith(NAME + "_")
                 and not out.exists() and not out.is_symlink(), "exclusive output")
    values = {path: pairing.read(path, digest) for path, digest in SOURCES.items()}
    def path(name):
        return ROOT / "results" / name
    plan, anchors = values[physical.PLAN], values[pairing.ANCHORS]
    prepared = {step: values[path(f"p1_adam_checkpoint_physical_{step}_20261031_prepare_RUN01.json")]
                for step in (5, 10)}
    consumed = {step: values[path(f"p1_adam_checkpoint_physical_{step}_20261031_consume_RUN01.json")]
                for step in (5, 10)}
    old = values[path("p1_uphill_multistart_pairing_20261030_main_RUN01.json")]
    result = join(anchors, plan, prepared, consumed, old, values[physical.prior.PLAN])
    result["source_hashes"] = {str(path): digest for path, digest in SOURCES.items()}
    result["analysis_sha256"] = physical.sha(Path(__file__).resolve())
    pairing.need(all(physical.sha(path) == digest for path, digest in SOURCES.items()), "source drift")
    digest = physical.prior.physical.design.run._write_new(out, result)
    print(json.dumps({"rows": len(result["rows"]), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
