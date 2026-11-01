"""Audit physical cp5 identity for already-consumed complete P3 H1 pairs."""
import hashlib
import json
from pathlib import Path

from . import p3_raw_h1_physical_consume_20261031_main as previous

ROOT = previous.ROOT
RECEIPT = previous.OUT
RECEIPT_SHA = "5ab22eba2370a9ad1187ab0c468efac3defa99605bd8f8e7edf6a9236b89c9a8"
OUT = ROOT / "results/p3_raw_h1_prefix_pair_audit_20261031_RUN_01.json"
require = previous.require


def checkpoint(row, step):
    matches = [c for c in row["checkpoints"] if c["phase"] == "plan" and c["step"] == step]
    require(len(matches) == 1, "unique checkpoint")
    return matches[0]


def compare_pair(old, new):
    a, b = old["selected_plan"], new["selected_plan"]
    require(a["variant"] == "old" and b["variant"] == "new", "variant order")
    require(all(a[k] == b[k] for k in ("decision_index", "policy", "candidate",
                                       "root_slot", "root_sha256", "task_id"))
            and a["commands"][:5] == b["commands"][:5], "pair identity")
    require(old["root_state"] == new["root_state"] and old["goal"] == new["goal"],
            "physical root and goal")
    checks = []
    for step in (0, 5):
        x, y = checkpoint(old, step), checkpoint(new, step)
        require(all(x[k] == y[k] for k in ("pixel_sha256", "file_sha256", "state",
                                            "observation_state", "snapshot")),
                f"physical checkpoint mismatch at {step}")
        checks.append({"step": step, "old_png": x["png"], "new_png": y["png"],
                       "png_sha256": x["file_sha256"], "pixel_sha256": x["pixel_sha256"]})
    return {"decision_index": a["decision_index"], "candidate": a["candidate"],
            "root_sha256": a["root_sha256"], "task_id": a["task_id"], "checks": checks}


def run():
    host = previous.host
    require(host.sha(RECEIPT) == RECEIPT_SHA, "consumed receipt sha")
    receipt = json.loads(RECEIPT.read_bytes())
    require(receipt["schema"] == "p3_raw_h1_physical_consume_20261031_main"
            and len(receipt["complete_batch_rows"]) == 48, "receipt roster")
    sources = ((previous.continuation.PREVIOUS, previous.continuation.PREVIOUS_SHA),
               (previous.LATER, previous.LATER_SHA))
    reports = {}
    for directory, digest in sources:
        require(host.sha(directory / "batch_report.json") == digest, "batch authority")
        authority = json.loads((directory / "batch_report.json").read_bytes())
        for entry in authority["batches"]:
            batch = entry["batch"]
            require(batch not in reports, "duplicate batch")
            reports[batch] = (directory / f"batch_{batch:02}", entry["report_sha256"])
    pairs = []
    for batch in sorted({row["batch"] for row in receipt["complete_batch_rows"]}):
        directory, digest = reports[batch]
        require(host.sha(directory / "report.json") == digest, "report sha")
        report = json.loads((directory / "report.json").read_bytes())
        expected = [row for row in receipt["complete_batch_rows"] if row["batch"] == batch]
        require(len(expected) == len(report["rows"]) == 6, "complete batch roster")
        for actual, sealed in zip(report["rows"], expected, strict=True):
            require(actual["selected_plan"] == sealed["plan"]
                    and actual["status"] == sealed["status"]
                    and actual["own_returned"] == sealed["returned"], "receipt row binding")
        for old, new in zip(report["rows"][::2], report["rows"][1::2], strict=True):
            item = compare_pair(old, new)
            for cp in item["checks"]:
                for name in ("old_png", "new_png"):
                    path = directory / cp[name]
                    require(Path(cp[name]).name == cp[name] and host.sha(path) == cp["png_sha256"],
                            "checkpoint file hash")
            item["batch"] = batch
            pairs.append(item)
    require(len(pairs) == 24, "complete pair count")
    require(host.sha(RECEIPT) == RECEIPT_SHA, "receipt drift")
    result = {"schema": "p3_raw_h1_prefix_pair_audit_20261031",
              "source_receipt_sha256": RECEIPT_SHA, "pairs": pairs,
              "partial_batches_included": False, "independent_reexecution": False,
              "P3_admission": False, "equal_budget_certified": False}
    require(not OUT.exists() and not OUT.is_symlink(), "exclusive output")
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"pairs": len(pairs), "output": str(OUT),
                      "sha256": hashlib.sha256(raw).hexdigest()}))


if __name__ == "__main__":
    run()
