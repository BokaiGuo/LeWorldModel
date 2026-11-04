"""Read-only cross-run and guard-predicate audit; never dispatches physics."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_all30_cpu_consume_20261102_codexexclusive as all30
from . import p3_venv_winner_cpu_consume_20261102_codexexclusive as four

ROOT = all30.replay.ROOT
NAME = "p3_all30_stop_crosscheck_20261102_codexexclusive"
ALL = ROOT / "results/p3_all30_cpu_physical_20261102_codexexclusive_RUN_01"
FOUR = ROOT / "results/p3_venv_winner_cpu_physical_20261102_codexexclusive_RUN_01"
ALL_PREP = ROOT / "results/p3_all30_cpu_physical_20261102_codexexclusive_PREP_01.json"
FOUR_PREP = ROOT / "results/p3_venv_winner_cpu_physical_20261102_codexexclusive_PREP_01.json"
ALL_PREP_SHA = "375da1e30a17cde98fb60879d0649c42a8e5f283ec2d0c56c702efdc27e4aadb"
ALL_REPORT_SHA = "8dfa0ece3f13972cf305eacb6f53e265b1b0713feca3b6038ca5473526dba704"
FOUR_PREP_SHA = "7d5cb06276b9c1fb2d86344a38f3baf4f67b08a6740a60bc321e3e033033df08"
FOUR_REPORT_SHA = "672b020c82b289481f528c23722a41741b7fed72e7f7407e039492d8e5f4335f"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def verify_records(all_prep, four_prep, a, b, directory):
    first, second = all_prep["plans"][0], four_prep["plans"][2]
    require(first["root_slot"] == second["root_slot"] == 0 and first["candidate_index"] ==
            second["candidate_index"] == 0 and first["raw_float32_sha256"] ==
            second["raw_float32_sha256"] and first["commands"] == second["commands"], "action binding")
    def frames(events, branch):
        return [e["receipt"] for e in events if e.get("branch") == branch and
                e["kind"] == "fresh_observation" and e["receipt"]["phase"] == "plan"]
    left, right = frames(a, 0), frames(b, 2)
    require([e["step"] for e in left] == [e["step"] for e in right] == list(range(11)),
            "frame sequence")
    require(all(x[k] == y[k] for x, y in zip(left, right, strict=True)
                for k in ("pixel_sha256", "state", "snapshot")), "cross-run divergence")
    failed_obs = [e["receipt"] for e in a if e.get("branch") == 7 and e["kind"] == "fresh_observation"
                  and e["receipt"]["phase"] == "plan" and e["receipt"]["step"] == 8]
    failed_guard = [e for e in a if e.get("branch") == 7 and e["kind"] == "guard_receipt"
                    and e["phase"] == "plan" and e["step"] == 8]
    require(len(failed_obs) == len(failed_guard) == 1, "failed guard receipt")
    obs, guard = failed_obs[0], failed_guard[0]
    image = np.asarray(Image.open(directory / obs["png"]).convert("RGB"))
    require(image.shape == (224, 224, 3) and
            all30.moving.moving.image_sha(image) == obs["pixel_sha256"], "failed frame pixels")
    pixels = int(np.count_nonzero(np.all(image == [65, 105, 225], axis=-1)))
    state = np.asarray(obs["state"])
    within = bool(((state[:4] >= 0) & (state[:4] <= 512)).all())
    same_state = obs["state"] == obs["observation_state"]
    require(pixels == guard["canonical_agent_pixels"] and pixels > 0 and not within and
            same_state and guard["render_preserved_dynamics"] and not guard["accepted"], "guard predicate")
    plan_steps = [e["step"] for e in a if e.get("branch") == 7 and
                  e["kind"] == "returned_transition" and e["phase"] == "plan"]
    require(plan_steps == list(range(1, 9)), "failure dispatch stop")
    return {"same_candidate": {"root_slot": 0, "candidate_index": 0,
                               "identical_plan_frames": 11, "identical_state_and_snapshot": True},
            "failed_branch": {"root_slot": 0, "candidate_index": 7, "step": 8,
                              "agent_xy": obs["state"][:2], "object_xy": obs["state"][2:4],
                              "all_xy_within_0_512": within, "agent_pixels": pixels,
                              "state_matches_observation": same_state,
                              "logged_render_preserved_dynamics": guard["render_preserved_dynamics"],
                              "accepted": guard["accepted"], "plan_steps_returned": plan_steps,
                              "image": obs["png"]}}


def build():
    ac = all30.consume(ALL_PREP, ALL_PREP_SHA, ALL, ALL_REPORT_SHA)
    bc = four.consume(FOUR_PREP, FOUR_PREP_SHA, FOUR)
    require(bc["report_sha256"] == FOUR_REPORT_SHA and ac["status"] == "partial_record_verified",
            "saved record authentication")
    ap, bp = all30.moving.read(ALL_PREP), all30.moving.read(FOUR_PREP)
    ar, br = all30.moving.read(ALL / "report.json"), all30.moving.read(FOUR / "report.json")
    a = all30.chain(ALL / "journal.jsonl", ar["journal_chain_tail"])
    b = all30.moving.journal(FOUR / "journal.jsonl", br["journal_chain_tail"])
    return {"schema": NAME + ".v1", "status": "saved_records_crosschecked",
            "all30_report_sha256": ALL_REPORT_SHA, "four_report_sha256": FOUR_REPORT_SHA,
            "checks": verify_records(ap, bp, a, b, ALL),
            "interpretation": "same root and action repeated, not independent generalization",
            "guard_changed": False, "new_physics": False, "gpu_used": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    require(out.is_absolute() and out.parent == ROOT / "results" and
            out.name.startswith(NAME + "_") and out.suffix == ".json", "exclusive path")
    result = build()
    with out.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"output": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                      "checks": result["checks"]}))


if __name__ == "__main__":
    main()
