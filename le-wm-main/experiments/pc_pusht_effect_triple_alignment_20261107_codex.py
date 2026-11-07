"""Tie-aware post-hoc decomposition of previously replayed PushT action effects."""
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.pc_input_permissions import file_hash

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ROOT / "results/pc_osfal_pusht_heldout_geometry_20261107_codex"
PHYSICAL = PREFIX.with_name(PREFIX.name + "_physical.json")
MODEL = PREFIX.with_name(PREFIX.name + "_model.json")
REPLAY = PREFIX.with_name(PREFIX.name + "_replay.json")
PROTOCOL = ROOT.parent / "docs/PC_PUSHT_EFFECT_TRIPLE_ALIGNMENT_POSTHOC_PROTOCOL_20261107_CODEX.md"
OUTPUT = ROOT / "results/pc_pusht_effect_triple_alignment_20261107_codex.json"
ACTIONS = ("zero", "plus", "minus")
PAIRS = tuple(itertools.combinations(ACTIONS, 2))


def sign(value, tolerance):
    if not np.isfinite(value):
        raise ValueError("non-finite difference")
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def classify(physical_delta, encoded_delta, predicted_delta):
    p = sign(physical_delta, 1e-9)
    e, m = sign(encoded_delta, 1e-4), sign(predicted_delta, 1e-4)
    if not p:
        category = "physical_tie"
    elif not e or not m:
        category = "score_tie"
    else:
        category = ("both_correct" if e == p and m == p else
                    "encoded_only_correct" if e == p else
                    "predicted_only_correct" if m == p else "both_wrong")
    return {"physical": p, "encoded": e, "predicted": m, "category": category}


def count(rows):
    names = ("physical_tie", "score_tie", "both_correct", "encoded_only_correct",
             "predicted_only_correct", "both_wrong")
    counts = {name: sum(row["classification"]["category"] == name for row in rows) for name in names}
    decisive = [row for row in rows if row["classification"]["physical"] != 0]
    counts.update(total=len(rows), physical_decisive=len(decisive),
                  predicted_sign_correct=sum(row["classification"]["predicted"] == row["classification"]["physical"]
                                             for row in decisive),
                  encoded_sign_correct=sum(row["classification"]["encoded"] == row["classification"]["physical"]
                                           for row in decisive),
                  predicted_encoded_disagree=sum(
                      row["classification"]["predicted"] != 0 and row["classification"]["encoded"] != 0
                      and row["classification"]["predicted"] != row["classification"]["encoded"]
                      for row in decisive))
    return counts


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    physical = json.loads(PHYSICAL.read_text())
    model = json.loads(MODEL.read_text())
    replay = json.loads(REPLAY.read_text())
    if (model["physical_sha256"] != file_hash(PHYSICAL)
            or replay["physical_sha256"] != file_hash(PHYSICAL)
            or replay["model_sha256"] != file_hash(MODEL)
            or replay["valid_roots"] != 16):
        raise ValueError("source binding or valid roots mismatch")
    if len(physical["rows"]) != 16 or len(model["rows"]) != 16:
        raise ValueError("unexpected root counts")
    rows = []
    for truth, pred in zip(physical["rows"], model["rows"], strict=True):
        if truth["root"] != pred["root"] or not truth["valid"] or not pred["valid"]:
            raise ValueError("invalid root pairing")
        for tc, mc in zip(truth["conditions"], pred["conditions"], strict=True):
            if (tc["appearance"], tc["side"]) != (mc["appearance"], mc["side"]):
                raise ValueError("condition identity mismatch")
            for a, b in PAIRS:
                physical_delta = tc["actions"][a]["object_pose_cost"] - tc["actions"][b]["object_pose_cost"]
                encoded_delta = mc["actions"][a]["real_encoded_cost"] - mc["actions"][b]["real_encoded_cost"]
                predicted_delta = mc["actions"][a]["predicted_cost"] - mc["actions"][b]["predicted_cost"]
                contact = [any(step["agent_block_contacts"] > 0 for step in tc["actions"][action]["trace"])
                           for action in (a, b)]
                rows.append({"root": truth["root"], "appearance": tc["appearance"], "side": tc["side"],
                             "pair": [a, b], "contact": contact, "physical_delta": physical_delta,
                             "encoded_delta": encoded_delta, "predicted_delta": predicted_delta,
                             "classification": classify(physical_delta, encoded_delta, predicted_delta)})
    if len(rows) != 16 * 2 * 2 * 3:
        raise ValueError("incomplete panel")
    by_root = [{"root": root, **count([row for row in rows if row["root"] == root])} for root in range(16)]
    strata = {"/".join(pair): count([row for row in rows if row["pair"] == list(pair)]) for pair in PAIRS}
    strata["contact_vs_no_contact"] = count([row for row in rows if sum(row["contact"]) == 1])
    strata["neither_contact"] = count([row for row in rows if sum(row["contact"]) == 0])
    strata["both_contact"] = count([row for row in rows if sum(row["contact"]) == 2])
    report = {"scope": "post-hoc exposed geometric development panel; no causal or gate claim",
              "overall": count(rows), "by_root": by_root, "strata": strata, "rows": rows,
              "sources": {str(path): file_hash(path) for path in (PHYSICAL, MODEL, REPLAY, PROTOCOL, Path(__file__))}}
    with OUTPUT.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"overall": report["overall"], "strata": strata, "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
