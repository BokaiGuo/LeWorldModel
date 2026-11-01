"""Verify saved fixed-roster transitions, images and censor evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import p1_response_prefix_censored_20261031 as run
from . import codex_p1_fixed_suffix_censored_consume_20261030 as prior

need, sha, native = run.need, run.sha, run.prior.native


def check(directory, report, prepared):
    need(report["schema"] == run.NAME + ".report" and
         report["status"] == "physical_recorded_not_independently_consumed" and
         report["fixed_rows"] == len(report["rows"]) == len(prepared["plans"]) == 16 and
         not report["errors"] and report["P1_admission"] is False and
         report["independent_confirmation"] is False and report["new_neural_calls"] == 0,
         "fixed complete roster")
    need(sha(directory / "journal.jsonl") == report["journal_sha256"], "journal hash")
    events = prior.events(directory / "journal.jsonl", report["journal_chain_tail"])
    need(events[-1]["status"] == report["status"] and events[-1]["counts"] == report["counts"],
         "terminal journal")
    returned = sorted((e["receipt"] for e in events if e["kind"] == "operation_returned"),
                      key=lambda x: x["index"])
    intents = [e["receipt"] for e in events if e["kind"] == "operation_dispatch_intent"]
    need(returned == report["receipts"] and [r["index"] for r in intents] == list(range(len(returned))),
         "operation receipts")
    run.prior.census(report["rows"], report["counts"])
    for i, (row, plan) in enumerate(zip(report["rows"], prepared["plans"], strict=True)):
        need(row["index"] == i and row["plan"] == plan and row["initial_exact"] is True and
             row["status"] in ("horizon_completed", "native_stopped", "censored_out_of_domain") and
             not row["first_error"] and not row["secondary_errors"], "row binding")
        branch = [e for e in events if e.get("branch") == i]
        steps = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        warm = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "warmup"]
        actions = np.frombuffer(bytes.fromhex(plan["raw15_float32_hex"]), "<f4").reshape(15, 2)
        need(len(warm) == 10 and len(steps) == row["own_returned"], "action denominator")
        for event, action, step in ([(e, native.moving.warmup_commands()[j], j + 1) for j, e in enumerate(warm)]
                                    + [(e, actions[j], j + 1) for j, e in enumerate(steps)]):
            need(event["step"] == step and event["requested_bits"] == event["actual_bits"] ==
                 action.tobytes().hex() and event["actual_dtype"] == "<f4", "actual action")
        censored = row["status"] == "censored_out_of_domain"
        accepted = len(steps) - int(censored)
        need([x["step"] for x in row["trajectory"] if x["phase"] == "plan"] ==
             list(range(accepted + 1)), "accepted observation sequence")
        need([c["step"] for c in row["checkpoints"]] == [n for n in (0, 5, 10, 15) if n <= accepted]
             and row["missing_checkpoints"] == [n for n in (0, 5, 10, 15) if n > accepted],
             "checkpoint censoring")
        frames = [e["receipt"] for e in branch if e["kind"] == "fresh_observation"]
        need(len(frames) == row["PNG"], "frame count")
        for frame in frames:
            path = native.paired.flat(directory, frame["png"])
            need(sha(path) == frame["file_sha256"], "file hash")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            need(pixels.shape == (224, 224, 3) and
                 native.moving.image_sha(pixels) == frame["pixel_sha256"], "pixel hash")
        proofs = [e for e in branch if e["kind"] == "censored_out_of_domain"]
        if censored:
            proof, step = row["censor"], len(steps)
            matching = [f for f in frames if f["phase"] == "plan" and f["step"] == step
                        and f["overlay_index"] == 0]
            need(step > 0 and len(proofs) == len(matching) == 1 and proofs[0]["proof"] == proof
                 and proof["step"] == step and proof["returned_state"] == steps[-1]["state"]
                 and proof["action_float32_hex"] == actions[step - 1].tobytes().hex()
                 and proof["out_of_bounds"] == [dict(coordinate=k, value=proof["returned_state"][k])
                                                 for k in range(4)
                                                 if not 0 <= proof["returned_state"][k] <= 512]
                 and proof["out_of_bounds"] and matching[0]["file_sha256"] == proof["file_sha256"]
                 and matching[0]["pixel_sha256"] == proof["pixel_sha256"], "censor proof")
            with Image.open(native.paired.flat(directory, matching[0]["png"])) as image:
                pixels = np.asarray(image.convert("RGB"))
            need(int(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1))) == proof["blue_pixels"],
                 "censor image")
        else:
            need(not proofs, "no spurious censor")
        if len(steps) >= 6:
            guard = [e for e in branch if e["kind"] == "cp5_guard_verified"]
            six = [e for e in events if e["kind"] == "operation_dispatch_intent" and
                   e["receipt"].get("branch") == i and e["receipt"]["resource"] == "public_step" and
                   e["receipt"].get("phase") == "plan" and e["receipt"].get("step") == 6]
            need(len(guard) == len(six) == 1 and guard[0]["index"] < six[0]["index"], "cp5 guard order")
        need(len(row["responses"]) == accepted, "response count")
        for response, transition in zip(row["responses"], steps[:accepted], strict=True):
            metric = native.base.previous.metric(row["goal"], transition["state"])
            need(response["state"] == transition["state"] and response["step"] == transition["step"] and
                 response["native_metrics"] == metric and response["reward"] == -metric["native_distance"] ==
                 transition["reward"] and response["native_success"] == metric["native_success"],
                 "native metric")
    return {"status": "saved_checks_passed_not_admitted", "rows": 16,
            "censored": sum(r["status"] == "censored_out_of_domain" for r in report["rows"]),
            "P1_admission": False, "independent_confirmation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory, output = Path(args.directory), Path(args.output)
    need(directory.is_absolute() and directory.is_dir() and not directory.is_symlink()
         and output.is_absolute() and output.parent == run.ROOT / "results"
         and output.name.startswith(run.NAME + "_consume_") and not output.exists()
         and not output.is_symlink(), "output paths")
    need(sha(directory / "report.json") == args.report_sha, "report hash")
    report = native.read(directory / "report.json", cap=64 * 1024**2)
    binding = native.read(directory / "execution_binding.json")
    need(binding["prepared"] == report["prepared"] and
         binding["prepare_sha256"] == report["prepare_sha256"] and
         sha(binding["prepared"]) == binding["prepare_sha256"], "prepare binding")
    prepared = native.read(binding["prepared"], cap=32 * 1024**2)
    need(all(sha(p) == d for p, d in prepared["bindings"].items()), "source hash")
    result = check(directory, report, prepared)
    result.update(report_sha256=args.report_sha, prepare_sha256=binding["prepare_sha256"])
    need(sha(directory / "report.json") == args.report_sha and
         all(sha(p) == d for p, d in prepared["bindings"].items()), "final drift")
    print(json.dumps({"sha256": run.write(output, result), **result}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
