"""Export fixed model-selected and incumbent action bytes; not a physical authorization."""
import argparse
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_bounded_backtrack_consume_20261030 as verifier

SHA = "bd829508ac62df40a82dd219c220367ae3925ecda77075b05deda2a18a7423fb"
NAME = "p1_backtrack_action_inventory_20261030"


def build(result, report, prepared):
    verifier.verify(result, report, prepared)
    rows = []
    for arm_index in range(8):
        group = result["rows"][20 * arm_index:20 * (arm_index + 1)]
        winner = min(range(20), key=lambda i: (group[i]["best_regularized"], i))
        for role, index in (("incumbent", 0), ("model_selected", winner)):
            saved = group[index]
            source = report["rows"][arm_index]["result"]["candidates"][index]
            prefix = np.asarray(source["prefix"], dtype="<f4")
            suffix = np.asarray(saved["final"]["raw"], dtype="<f4").reshape(10, 2)
            if prefix.shape != (5, 2) or not np.isfinite(prefix).all() or not np.isfinite(suffix).all():
                raise ValueError("action shape or finiteness")
            raw = np.concatenate((prefix, suffix))
            rows.append(dict(slot=saved["slot"], arm=saved["arm"], role=role,
                             candidate_index=index, start=saved["start"],
                             prefix_iteration=saved["iteration"],
                             model_stage_uphill=saved["stage_uphill"], raw15=raw.tolist(),
                             raw15_le_float32_hex=raw.tobytes().hex(),
                             task_binding=prepared["requests"][arm_index // 2]["task"]["binding"],
                             task_slot=prepared["requests"][arm_index // 2]["slot"]))
    return dict(schema=NAME, source_result_sha256=SHA,
                source_report_sha256=result["source_report_sha256"],
                rows=rows, counts=dict(arms=8, records=16), known_roots_only=True,
                physical_authorized=False, fresh_confirmation=False, P1_admission=False,
                note="Inventory only; raw normalization, reset/history guard and physical execution are not certified.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    one = verifier.producer.base.one
    output = args.output
    if (not output.is_absolute() or output.parent != one.source.ROOT / "results"
            or output.resolve() != output or not output.name.startswith(NAME + "_")):
        raise ValueError("exclusive results path required")
    result = one.saved.read_verified(one.source.ROOT / "results/p1_bounded_backtrack_host_20261030_MAIN01.json", SHA)
    report = one.saved.read_verified(one.source.REPORT, one.source.REPORT_SHA)
    receipt = one.saved.read_verified(one.source.RECEIPT, one.source.RECEIPT_SHA)
    if receipt["report_sha256"] != one.source.REPORT_SHA:
        raise ValueError("source receipt mismatch")
    prepared = one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    one.saved.analyze(report, receipt, prepared)
    value = build(result, report, prepared)
    digest = one.saved.write(output, value)
    print(json.dumps(dict(output=str(output), sha256=digest, count=len(value["rows"]),
                          physical_authorized=False)))


if __name__ == "__main__":
    main()
