"""Audit whether selected prefixes are numerically distinct from zero and each other."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import p1_diverse_h10_20261030_main as h10

NAME = "p1_prefix_nontriviality_20261030_main"


def classify(raw_hex):
    raw = bytes.fromhex(raw_hex)
    if len(raw) != 40:
        raise ValueError("five-step prefix byte count")
    values = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(values).all() or (np.abs(values) > 1).any():
        raise ValueError("finite bounded prefix")
    return dict(exact_numerical_zero=bool(np.all(values == 0)),
                nonzero_components=int(np.count_nonzero(values)),
                l2_norm=float(np.linalg.norm(values)),
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                signed_zero_components=int(np.sum((values == 0) & np.signbit(values))))


def build(prepared):
    rows = []
    for anchor in prepared["anchors"]:
        rows.append(dict(logical_pair=anchor["logical_pair"], root_slot=anchor["root_slot"],
                         condition=anchor["condition"], arm=anchor["arm"],
                         prefix_float32_hex=anchor["prefix5_float32_hex"],
                         classification=classify(anchor["prefix5_float32_hex"])))
    return dict(schema=NAME, rows=rows,
                input_sha256="1beac317193419c0e1ff38e4b6b921251c5b2732f9c7e5ea4a6ae129e796c5d1",
                numerical_zero_is_reconfiguration=False, changes_selection=False, P1_admission=False,
                note="Numerical nonzero alone does not prove physical change or non-greedy mechanism.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    h10.need(output.is_absolute() and output.parent == h10.ROOT / "results"
             and output.name.startswith(NAME+"_") and not output.exists(), "exclusive output")
    path = h10.ROOT / "results/p1_diverse_h10_20261030_main_prepare_RUN01.json"
    digest = "1beac317193419c0e1ff38e4b6b921251c5b2732f9c7e5ea4a6ae129e796c5d1"
    prepared, _ = h10.verify(str(path), digest)
    result = build(prepared)
    print(json.dumps({"sha256": h10.runner._write_new(output, result), "rows": len(result["rows"])}))


if __name__ == "__main__":
    main()
