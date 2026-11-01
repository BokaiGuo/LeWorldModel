"""Audit saved partial H1 batches without admitting censored endpoints."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p3_raw_h1_physical_consume_20261031_main as prior

NAME = "p3_raw_h1_partial_audit_20261031"
OUT = prior.ROOT / f"results/{NAME}_RUN02.json"
require, host = prior.require, prior.host
kernel = prior.saved.kernel


def verify_guard(pixels, image, guard):
    count = int(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1)))
    state = np.asarray(image["state"], dtype=np.float64)
    require(state.shape == (7,) and np.isfinite(state).all(), "finite state")
    in_canvas = bool(((state[:4] >= 0) & (state[:4] <= 512)).all())
    require(count == guard["canonical_agent_pixels"], "guard pixel count")
    require(image["state"] == image["observation_state"]
            and guard["render_preserved_dynamics"] is True, "guard state/render")
    require(guard["accepted"] == (count > 0 and in_canvas), "recomputed guard")
    return {"agent_pixels": count, "in_canvas": in_canvas,
            "state_xy": state[:4].tolist()}


def audit_batch(directory, record, plans, bindings, contract, data):
    require(host.sha(directory / "report.json") == record["report_sha256"], "report hash")
    report = json.loads((directory / "report.json").read_bytes())
    prepare = json.loads((directory / "prepare.json").read_bytes())
    require(prepare["plans"] == plans and prepare["contract"] == contract
            and all(prepare["bindings"].get(k) == v and host.sha(k) == v for k, v in bindings.items()),
            "prepare binding")
    require(report["status"] == "partial_or_failed" and report["fixed_denominator"] == 6
            and report["scientific_pass"] is False and report["new_neural_calls"] == 0, "partial status")
    require(host.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal hash")
    events = kernel.journal(directory / "journal.jsonl", report["journal_chain_tail"])
    require(events[0]["kind"] == "run_started" and events[0]["prepare"] == prepare
            and events[-1]["kind"] == "run_finished" and events[-1]["status"] == report["status"]
            and events[-1]["counts"] == report["counts"], "journal boundary")
    rows = report["rows"]
    require(len(rows) == 6 and [e["receipt"] for e in events if e["kind"] == "branch_closed"] == rows,
            "row closure")
    outputs, frames = [], []
    for i, (row, plan) in enumerate(zip(rows, plans, strict=True)):
        require(row["branch_index"] == i and row["selected_plan"] == plan
                and row["arm"] == plan["arm"] and row["root_slot"] == plan["root_slot"]
                and row["initial_exact"] is True and row["warmup_returned"] == 10, "identity")
        ref = data["references"][row["root_slot"]]
        require(kernel.source.root_hash(kernel.source.vector(row["root_state"]),
                                        kernel.source.vector(row["goal"])) == ref["root_sha256"]
                and np.array_equal(np.asarray(row["goal"]), ref["goal"]), "root binding")
        ev = [e for e in events if e.get("branch") == i]
        ack = [e["receipt"] for e in ev if e["kind"] == "step_ack_observed"]
        trans = [e for e in ev if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        n = row["own_returned"]
        require(len(trans) == n == row["own_attempted"] and len(ack) == 10+n
                and [e["step"] for e in trans] == list(range(1, n+1)), "transition census")
        warm = kernel.moving.warmup_commands()
        commands = np.asarray(plan["commands"], dtype="<f4")
        for j, item in enumerate(ack):
            action = warm[j] if j < 10 else commands[j-10]
            require(item["ordinal"] == j and item["sent_dtype"] == item["actual_dtype"] == "float32"
                    and item["sent_shape"] == item["actual_shape"] == [2]
                    and item["sent_hex"] == item["actual_hex"] == action.tobytes().hex(), "ack")
        for j, event in enumerate(trans):
            require(event["command"] == plan["commands"][j] and event["returned_goal_state"] == row["goal"]
                    and event["terminated"] == ack[j+10]["terminated"]
                    and event["truncated"] == ack[j+10]["truncated"], "transition")
        require(trans and row["final_state"] == trans[-1]["state"]
                and row["final_metrics"] == kernel.metric(row["goal"], row["final_state"])
                and row["native_success"] == row["final_metrics"]["native_success"], "last metrics")
        images = [e["receipt"] for e in ev if e["kind"] == "fresh_observation"]
        guards = [e for e in ev if e["kind"] == "guard_receipt"]
        require(len(images) == len(guards) == row["PNG"], "frame count")
        guard_details = []
        for image, guard in zip(images, guards, strict=True):
            require(image["phase"] == guard["phase"] and image["step"] == guard["step"]
                    and guard["render_preserved_dynamics"] is True, "guard alignment")
            path = directory / image["png"]
            require(path.name == image["png"] and host.sha(path) == image["file_sha256"], "frame file")
            with Image.open(path) as im:
                pixels = np.asarray(im.convert("RGB")).copy()
            require(pixels.shape == (224, 224, 3)
                    and kernel.moving.image_sha(pixels) == image["pixel_sha256"]
                    and image["state"] == image["observation_state"] == image["snapshot"]["state"],
                    "frame pixel or state")
            guard_details.append(verify_guard(pixels, image, guard))
            frames.append(image["png"])
        warm_images = [x for x in images if x["phase"] == "warmup" and x["step"] in (0, 5, 10)]
        require([x["step"] for x in warm_images] == [0, 5, 10]
                and all(x["pixel_sha256"] == kernel.moving.image_sha(ref["history"][j])
                        and x["state"] == ref["states"][j].tolist()
                        and x["snapshot"] == ref["dynamics"][j] for j, x in enumerate(warm_images)),
                "warmup reference")
        failed = [g for g in guards if not g["accepted"]]
        if row["status"] == "failed":
            require(len(failed) == 1 and failed[0]["phase"] == "plan" and failed[0]["step"] == n
                    and row["first_error"]["message"] == "RGB/canvas/state/pure-render guard; failed image retained"
                    and not row["full_horizon"], "guard failure")
        else:
            require(row["status"] == "horizon_completed" and n == 10 and not failed
                    and row["first_error"] is None and row["full_horizon"], "completed row")
        outputs.append({"arm": row["arm"], "status": row["status"], "returned": n,
                        "failed_guard": failed[0] if failed else None,
                        "failed_guard_recomputed": guard_details[-1] if failed else None,
                        "endpoint": row["final_metrics"] if row["status"] == "horizon_completed" else None})
    require(len(frames) == len(set(frames)) and {p.name for p in directory.iterdir()} ==
            set(frames) | {"prepare.json", "report.json", "journal.jsonl"}, "artifact census")
    require(all(host.sha(k) == v for k, v in bindings.items()), "binding drift")
    return outputs


def main():
    require(not OUT.exists() and not OUT.is_symlink(), "exclusive output")
    require(host.sha(prior.continuation.PREVIOUS / "batch_report.json") == prior.continuation.PREVIOUS_SHA
            and host.sha(prior.LATER / "batch_report.json") == prior.LATER_SHA, "batch hashes")
    first = json.loads((prior.continuation.PREVIOUS / "batch_report.json").read_bytes())
    later = json.loads((prior.LATER / "batch_report.json").read_bytes())
    records = first["batches"] + later["batches"]
    require([r["batch"] for r in records] == list(range(12)), "attempt roster")
    plans, bindings, contract, data = prior.initial.prepare()
    results = []
    for record in records:
        batch = record["batch"]
        directory = (prior.continuation.PREVIOUS if batch < 2 else prior.LATER) / f"batch_{batch:02}"
        report = json.loads((directory / "report.json").read_bytes())
        if report["status"] == "partial_or_failed":
            results.append({"batch": batch, "report_sha256": record["report_sha256"],
                            "rows": audit_batch(directory, record, plans[6*batch:6*batch+6],
                                                bindings, contract, data)})
    require([r["batch"] for r in results] == [1, 2, 4, 5], "partial roster")
    value = {"schema": NAME, "batches": results, "P3_admission": False, "P1_admission": False,
             "limitations": "Saved identity checks only, not reexecution or scientific admission; "
                            "guard-failed endpoints excluded.", "code_sha256": host.sha(Path(__file__).resolve())}
    print(json.dumps({"sha256": host.bind.host.write_new(OUT, value), "batches": len(results)}))


if __name__ == "__main__":
    main()
