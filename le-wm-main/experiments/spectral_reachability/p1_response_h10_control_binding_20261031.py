"""Preflight historical zero/greedy controls for the response-prefix H10 panel."""
import hashlib
import json
from pathlib import Path

from . import p1_response_prefix_h10_20261031 as panel

need, sha = panel.need, panel.sha
NAME = "p1_response_h10_control_binding_20261031"
NEW = panel.ROOT / "results/p1_response_prefix_h10_20261031_prepare_RUN01.json"
NEW_SHA = "8577190effc65a61e1b69cee6dcdbd1c2814a3da79b8383d468a76e03f0c05bc"
OLD = panel.BANKS
OLD_SHA = panel.BANKS_SHA
PAIR = panel.ROOT / "results/p1_uphill_complete_pairing_20261030_RUN01.json"
PAIR_SHA = "e3efb9815f5f4154036e34331d8da87b8a2dab007fd5db3288cfaeec6a8538ed"
KEYS = ("root_slot", "condition", "arm", "task_id", "root_state", "goal")
FRAME_KEYS = ("phase", "step", "condition", "overlay_index", "file_sha256",
              "pixel_sha256", "state", "snapshot")


def verify(new, old, pairing):
    need(len(new["anchors"]) == 14 and len(old["anchors"]) == 24
         and len(pairing["common_suffix_pairs"]) == 336, "roster")
    matched = []
    for anchor in new["anchors"]:
        pair = anchor["logical_pair"]
        probes = new["banks"][pair]
        need(probes == old["probe_banks"][pair] and len(probes) == 42, "bank bytes")
        for role, offset in (("zero", 1), ("greedy", 2)):
            control = old["anchors"][3 * pair + offset]
            need(control["role"] == role and all(anchor[k] == control[k] for k in KEYS),
                 "control task identity")
            need(len(anchor["trace"]) >= 33 and len(control["trace"]) >= 33, "warmup length")
            for a, b in zip(anchor["trace"][:33], control["trace"][:33], strict=True):
                need(a["pre"] == b["pre"] and a["post"] == b["post"]
                     and all(a["receipt"][k] == b["receipt"][k] for k in FRAME_KEYS),
                     "warmup frame and simulator snapshot")
            for probe in probes:
                record = pairing["common_suffix_pairs"][pair * 42 + probe["index"]]
                digest = hashlib.sha256(bytes.fromhex(probe["raw10_float32_hex"])).hexdigest()
                need(record["logical_pair"] == pair and record["probe"] == probe["name"]
                     and record["probe_sha256"] == digest
                     and record["roles"][role]["anchor_index"] == control["index"], "saved probe identity")
            matched.append({"new_anchor": anchor["index"], "logical_pair": pair,
                            "control": role, "old_anchor": control["index"],
                            "matched_warmup_frames": 33, "matched_suffixes": 42})
    return {"schema": NAME, "status": "identity_preflight_passed",
            "matches": matched, "candidate_prefixes": 14, "control_links": len(matched),
            "limitations": "Identity only; new physical panel still requires complete saved verification. "
                           "Historical known roots and shared suffixes are not independent confirmation.",
            "P1_admission": False, "equal_budget_certified": False}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == panel.ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists()
         and not output.is_symlink(), "exclusive output")
    sources = ((NEW, NEW_SHA), (OLD, OLD_SHA), (PAIR, PAIR_SHA))
    need(all(sha(p) == d for p, d in sources), "source hashes")
    read = panel.physical.bundle.physical.prior.native.read
    new, old, pairing = (read(p, cap=64 * 1024**2) for p, _ in sources)
    value = verify(new, old, pairing)
    need(all(sha(p) == d for p, d in sources), "source drift")
    value["source_sha256"] = {str(p): d for p, d in sources}
    value["code_sha256"] = sha(__file__)
    digest = panel.physical.bundle.physical.write(output, value)
    print(json.dumps({"status": value["status"], "links": value["control_links"], "sha256": digest}))


if __name__ == "__main__":
    main()
