"""Pair original and real-midpoint reoptimized Adam physical executions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import codex_p1_adam_feedback_plan_20261030_main as plan
from . import codex_p1_same_first5_feedback_compare_20261030_main as trace

base = plan.base
NAME = "codex_p1_adam_feedback_compare_20261030_main"
DIR = base.ROOT / "results/codex_p1_adam_feedback_physical_20261030_main_RUN01"
REPORT_SHA = "9100c09b72f4289b642619e9a21ec8879b4ecdd801e187f1215c66a3c7783c59"
SAVED = base.ROOT / "results/codex_p1_adam_feedback_physical_20261030_main_SAVED_RUN01.json"
SAVED_SHA = "0d77f3409afadf4f0029721ccf1b8aa543f742d26c8a78c808925610db45eb78"
FIELDS = ("object_position_error", "object_angle_error_wrapped", "pusher_position_error", "native_joint_xy_error")


def build():
    entries = plan.inputs()
    saved, _ = plan.gap.prior.gap.read(SAVED, SAVED_SHA)
    report, _ = plan.gap.prior.gap.read(DIR / "run_report.json", REPORT_SHA)
    old_report, _ = plan.gap.prior.gap.read(plan.gap.PHYSICAL / "run_report.json", plan.gap.REPORT_SHA)
    if not (len(entries) == len(saved["rows"]) == len(report["shards"]) == len(old_report["shards"]) == 24):
        raise ValueError("roster")
    rows = []
    for i, (entry, new, shard, old_shard) in enumerate(zip(entries, saved["rows"], report["shards"], old_report["shards"], strict=True)):
        row, old, raw, rgb, mid = entry
        if any(new[k] != old[k] for k in ("index", "logical_pair", "root_slot", "condition", "role", "arm")) or i != new["index"]:
            raise ValueError("identity")
        folder = DIR / f"row_{i:02d}"
        old_folder = plan.gap.PHYSICAL / f"row_{i:02d}"
        a = trace.transitions(old_folder, old_shard["report_sha256"])
        b = trace.transitions(folder, shard["report_sha256"])
        if len(a) < 5 or len(b) < 5 or any(any(a[j][key] != b[j][key] for key in
                ("step", "actual_bits", "state", "snapshot", "terminated", "truncated")) for j in range(5)):
            raise ValueError("first-five replay mismatch")
        physical, _ = plan.gap.prior.gap.read(folder / "report.json", shard["report_sha256"])
        frames = [f for f in physical["row"]["trajectory"] if f["phase"] == "probe" and f["step"] == 5
                  and f["condition"] == old["condition"] and f["overlay_index"] == (1 if old["condition"] == "on" else 0)]
        if len(frames) != 1 or frames[0]["pixel_sha256"] != mid["pixel_sha256"]:
            raise ValueError("midpoint pixel mismatch")
        frame = frames[0]
        png = folder / frame["png"]
        if Path(frame["png"]).name != frame["png"] or base.sha(png) != frame["file_sha256"]:
            raise ValueError("new midpoint file mismatch")
        with Image.open(png) as image:
            pixels = np.asarray(image).copy()
        if pixels.shape != (224, 224, 3) or hashlib.sha256(pixels.tobytes()).hexdigest() != frame["pixel_sha256"]:
            raise ValueError("new midpoint pixel mismatch")
        x, y = old["endpoint_metrics"], new["endpoint_metrics"]
        diff = None if x is None or y is None else {k: y[k] - x[k] for k in FIELDS}
        rows.append({"index": i, "logical_pair": old["logical_pair"], "root_slot": old["root_slot"],
                     "condition": old["condition"], "role": old["role"], "midpoint_sha256": mid["pixel_sha256"],
                     "open_status": old["status"], "feedback_status": new["status"],
                     "open_steps": old["returned_suffix"], "feedback_steps": new["returned_suffix"],
                     "open_last_success": old["last_accepted_state_metrics"]["native_success"],
                     "feedback_last_success": new["last_accepted_state_metrics"]["native_success"],
                     "open_endpoint": x, "feedback_endpoint": y, "feedback_minus_open": diff})
    return {"schema": NAME, "rows": rows, "status": "complete", "P1_admission": False,
            "independent_confirmation": False, "equal_total_budget_certified": False,
            "limitations": "known roots and repeated trajectories; offline midpoint planning; feedback adds compute; "
                           "no equal budget or online control claim; original censor semantics preserved"}


def main():
    out = base.ROOT / f"results/{NAME}_RUN03.json"
    result = build()
    result["code_sha256"] = base.sha(Path(__file__).resolve())
    digest = base.write(out, result)
    print(json.dumps({"rows": len(result["rows"]), "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
