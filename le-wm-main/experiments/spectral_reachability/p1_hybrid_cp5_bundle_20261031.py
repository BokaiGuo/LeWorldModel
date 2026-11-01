"""Build the two fixed hybrid real-cp5 inputs for the common optimizer."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import codex_p1_three_prefix_blind_inputs_20261030_v1 as blind
from . import p1_hybrid_h10_20261031 as hybrid

ROOT, need, sha = hybrid.ROOT, hybrid.need, hybrid.sha
NAME = "p1_hybrid_cp5_bundle_20261031"
PREPARE = ROOT / "results/p1_hybrid_h10_20261031_prepare_RUN01.json"
PREPARE_SHA = "724a6ec10566b31e50f2287678ac926986a4bab4b1951a13aee694c5735f7433"
JOIN = ROOT / "results/p1_backtrack_physical_source_join_20261030_MAIN02.json"
JOIN_SHA = "cb82a39b8a464f16b7e9f71cd9d072f2d48c0887bf36d57c91ca50baf0f2af84"
PROTOCOL = ROOT.parent / "docs/D2P_P1_HYBRID_CP5_COMMON_OPTIMIZER_PROTOCOL_20261031.md"


def build():
    need(sha(PREPARE) == PREPARE_SHA and sha(JOIN) == JOIN_SHA, "pinned sources")
    p = hybrid.read(PREPARE)
    need(p == hybrid.build(), "reconstructed prepare")
    _, _, _, old = blind.replay.sources()
    tasks = hybrid.read(JOIN)["task_rows"]
    bindings = dict(p["bindings"])
    for path in (PREPARE, JOIN, PROTOCOL, Path(__file__).resolve(), Path(blind.__file__).resolve()):
        bindings[str(path)] = sha(path)
    rows = []
    for anchor in p["anchors"]:
        slot = 2 * anchor["root_slot"] + (anchor["condition"] == "on")
        request = old["requests"][slot]
        need(request["root_slot"] == anchor["root_slot"] and
             request["condition"] == anchor["condition"], "task identity")
        matches = [t for t in tasks if t["root_slot"] == anchor["root_slot"]
                   and t["condition"] == anchor["condition"]]
        need(len(matches) == 1, "unique warmup source")
        raw = np.frombuffer(bytes.fromhex(matches[0]["warmup_raw_float32_hex"]), "<f4").reshape(10, 2)
        cp = anchor["checkpoint"]
        image = Path(cp["path"])
        need(bindings.get(str(image)) == cp["file_sha256"] and sha(image) == cp["file_sha256"],
             "source image binding")
        input_row = blind.make_row(anchor, request["task"], raw, image.read_bytes())
        rows.append({"index": len(rows), "logical_pair": anchor["logical_pair"],
                     "root_slot": anchor["root_slot"], "blind": input_row})
    need(len(rows) == 2 and all(sha(path) == digest for path, digest in bindings.items()), "source drift")
    return {"schema": NAME, "rows": rows, "bindings": bindings,
            "fixed_rows": 2, "independent_confirmation": False, "P1_admission": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    need(output.is_absolute() and output.parent == ROOT / "results"
         and output.name.startswith(NAME + "_") and not output.exists(), "exclusive output")
    with output.open("x") as stream:
        json.dump(build(), stream, sort_keys=True, allow_nan=False)
    print(json.dumps({"rows": 2, "sha256": sha(output)}))


if __name__ == "__main__":
    main()
