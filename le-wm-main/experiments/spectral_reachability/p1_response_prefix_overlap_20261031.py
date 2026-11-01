"""Audit exact prefix overlap; this does not inspect physical outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

from . import p1_uphill_complete_pairing_20261030 as pairing

ROOT = pairing.ROOT
PANEL = ROOT / "results/p1_response_common_suffix20_cuda_20261030_main_RUN01.json"
PANEL_SHA = "9b29bfd041bd0ed73ba03c3fe2add8bc22522de6a319326a02ea68b9c0f6a053"
PAIRS = ROOT / "results/p1_uphill_complete_pairing_20261030_RUN01.json"
PAIRS_SHA = "e3efb9815f5f4154036e34331d8da87b8a2dab007fd5db3288cfaeec6a8538ed"


def audit(panel, pairs):
    pairing.require(panel["physical_execution"] is False and len(panel["rows"]) == 8,
                    "model panel contract")
    anchors = pairs["anchors"]
    pairing.require(len(anchors) == 24 and pairs["P1_admission"] is False, "anchor contract")
    rows = []
    for item in panel["rows"]:
        slot, arm = item["slot"], item["arm"]
        condition = ("off", "on")[slot % 2]
        matches = [a for a in anchors if a["identity"]["root_slot"] == slot // 2
                   and a["identity"]["condition"] == condition and a["identity"]["arm"] == arm]
        pairing.require(len(matches) == 3, "slot and arm binding")
        by_digest = {}
        for anchor in matches:
            by_digest.setdefault(anchor["prefix_action_sha256"], []).append(anchor["identity"]["role"])
        candidates = item["candidates"]
        for index in sorted(candidates, key=int):
            candidate = candidates[index]
            raw = bytes.fromhex(candidate["prefix_float32_hex"])
            pairing.require(len(raw) == 40, "five-step float32 action shape")
            digest = hashlib.sha256(raw).hexdigest()
            rows.append({"slot": slot, "arm": arm, "candidate_index": int(index),
                         "prefix_sha256": digest, "matching_anchor_roles": sorted(by_digest.get(digest, [])),
                         "candidate_sets": sorted(name for name, indices in item["candidate_sets"].items()
                                                  if int(index) in indices)})
    pairing.require(len(rows) == 54, "candidate count")
    return {"schema": "p1_response_prefix_overlap_20261031",
            "model_panel_sha256": PANEL_SHA, "paired_panel_sha256": PAIRS_SHA,
            "rows": rows, "matched_count": sum(bool(row["matching_anchor_roles"]) for row in rows),
            "scope": "Exact prefix bytes within matching slot and arm only; no outcome selection. "
                     "A match does not certify equivalent complete experiment or fresh roots.",
            "physical_execution": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    pairing.require(out.is_absolute() and out.parent == ROOT / "results"
                    and out.name.startswith("p1_response_prefix_overlap_")
                    and not out.exists() and not out.is_symlink(), "exclusive output")
    result = audit(pairing.read_checked(PANEL, PANEL_SHA), pairing.read_checked(PAIRS, PAIRS_SHA))
    data = json.dumps(result, sort_keys=True, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    with out.open("xb") as stream:
        stream.write(data)
    print(json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "rows": len(result["rows"]),
                      "matched_count": result["matched_count"]}))


if __name__ == "__main__":
    main()
