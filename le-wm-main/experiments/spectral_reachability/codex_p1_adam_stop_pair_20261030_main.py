"""Compare physical observations at original native-stop times without imputation."""
import json
from pathlib import Path

from . import codex_p1_adam_feedback_compare_20261030_main as compare
from . import codex_p1_h10_paired_table_20261030_main as metrics

base = compare.base
NAME = "codex_p1_adam_stop_pair_20261030_main"


def build():
    # Reconstruct the complete 24-row paired source before selecting stopped rows.
    pairs = compare.build()
    original, _ = compare.plan.gap.prior.gap.read(compare.plan.gap.PHYSICAL / "run_report.json", compare.plan.gap.REPORT_SHA)
    feedback, _ = compare.plan.gap.prior.gap.read(compare.DIR / "run_report.json", compare.REPORT_SHA)
    from . import codex_p1_multistart_physical_20261030_main as executor
    source = executor.old.h10.native.read(executor.old.h10.INPUT, cap=64 * 1024**2)["anchors"]
    if len(source) != 24:
        raise ValueError("anchor roster")
    rows = []
    for pair in pairs["rows"]:
        if pair["open_status"] != "native_stopped":
            continue
        i = pair["index"]
        anchor = source[i]
        if anchor["index"] != i or anchor["role"] != pair["role"] or anchor["logical_pair"] != pair["logical_pair"]:
            raise ValueError("anchor identity")
        a = compare.trace.transitions(compare.plan.gap.PHYSICAL / f"row_{i:02d}", original["shards"][i]["report_sha256"])
        b = compare.trace.transitions(compare.DIR / f"row_{i:02d}", feedback["shards"][i]["report_sha256"])
        stop = pair["open_steps"]
        if len(a) != stop or len(b) < stop or not a[-1]["terminated"] or a[-1]["truncated"]:
            raise ValueError("stop observation")
        steps = []
        for t in range(5, stop + 1):
            x, y = a[t-1], b[t-1]
            if x["step"] != t or y["step"] != t:
                raise ValueError("time identity")
            ma, mb = metrics.physical_metrics(x["state"], anchor["goal"]), metrics.physical_metrics(y["state"], anchor["goal"])
            steps.append({"step": t, "open_metrics": ma, "feedback_metrics": mb,
                          "feedback_minus_open": metrics.paired_difference(mb, ma),
                          "open_terminated": x["terminated"], "feedback_terminated": y["terminated"],
                          "open_action_hex": x["actual_bits"], "feedback_action_hex": y["actual_bits"]})
        if not steps[-1]["open_metrics"]["native_success"]:
            raise ValueError("success recomputation disagreement")
        rows.append({"index": i, "root_slot": pair["root_slot"], "condition": pair["condition"],
                     "logical_pair": pair["logical_pair"], "role": pair["role"],
                     "original_stop_step": stop, "feedback_status": pair["feedback_status"],
                     "feedback_steps": pair["feedback_steps"], "steps": steps})
    if len(rows) != 5:
        raise ValueError("five stopped rows")
    return {"schema": NAME, "status": "complete", "rows": rows, "source_roster": 24,
            "P1_admission": False, "independent_confirmation": False,
            "limitations": "post-outcome descriptive same-time physical observation; two known roots and duplicates; "
                           "no imputed ten-step endpoint or deployment use of simulator state"}


def main():
    out = base.ROOT / f"results/{NAME}_RUN01.json"
    result = build()
    result["code_sha256"] = base.sha(Path(__file__).resolve())
    print(json.dumps({"rows": len(result["rows"]), "sha256": base.write(out, result)}), flush=True)


if __name__ == "__main__":
    main()
