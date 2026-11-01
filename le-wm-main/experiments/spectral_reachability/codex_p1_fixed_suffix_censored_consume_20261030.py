"""Independent saved-artifact checks for the fixed 32-row P1 physical run.

Checks report, journal, literal actions, RGB files, and native metric arithmetic.
It does not establish freshness, causal capability improvement or P1 admission.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import codex_p1_fixed_suffix_censored_runner_20261030 as runner

require = runner.need


def events(path, expected_tail):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 128 * 1024**2,
            "bounded regular journal")
    result, previous = [], None
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            require(line.endswith(b"\n"), "complete journal line")
            event = json.loads(line)
            require(event["index"] == index and event["previous_sha256"] == previous and
                    event["event_sha256"] == runner.native.source.digest(
                        {k: v for k, v in event.items() if k != "event_sha256"}), "journal hash chain")
            previous = event["event_sha256"]
            result.append(event)
    require(result and previous == expected_tail and result[-1]["kind"] == "run_finished", "journal terminal")
    return result


def check(directory, report, prepared):
    require(report["schema"] == runner.NAME + ".report" and
            report["status"] == "physical_recorded_not_independently_consumed" and
            report["errors"] == [] and
            report["fixed_rows"] == 32 and report["P1_admission"] is False and
            report["independent_confirmation"] is False and report["new_neural_calls"] == 0 and
            len(prepared["plans"]) == len(report["rows"]) == 32, "complete unadmitted fixed32 report")
    require(runner.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    journal = events(directory / "journal.jsonl", report["journal_chain_tail"])
    require(journal[-1]["status"] == report["status"] and journal[-1]["counts"] == report["counts"],
            "terminal count binding")
    returned = sorted((e["receipt"] for e in journal if e["kind"] == "operation_returned"),
                      key=lambda receipt: receipt["index"])
    intents = [e["receipt"] for e in journal if e["kind"] == "operation_dispatch_intent"]
    require(returned == report["receipts"] and
            [r["index"] for r in intents] == list(range(len(returned))),
            "actual dispatch and nested return receipts")
    counts = report["counts"]
    require(all(v["attempted"] == v["completed"] for v in counts.values()), "completed operation census")
    c = {k: v["completed"] for k, v in counts.items()}
    require(c["constructor"] == c["close"] == 32 and c["reset"] == 64 and
            c["physics_step"] == 2 * c["reset"] + 10 * c["public_step"] and
            c["set_state"] == 2 * c["reset"] and c["goal_render"] == c["reset"] and
            c["goal_setter"] == c["reset"] + c["goal_bind"] and
            c["public_step"] == sum(r["warmup_returned"] + r["own_returned"] for r in report["rows"]) and
            c["explicit_render"] == sum(r["PNG"] for r in report["rows"]), "32-branch census")
    for i, (plan, row) in enumerate(zip(prepared["plans"], report["rows"], strict=True)):
        require(row["index"] == i and row["plan"] == plan and row["initial_exact"] is True and
                row["status"] in ("horizon_completed", "native_stopped", "censored_out_of_domain") and
                not row["first_error"] and not row["secondary_errors"], "row and task identity")
        branch = [e for e in journal if e.get("branch") == i]
        plan_transitions = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "plan"]
        warm_transitions = [e for e in branch if e["kind"] == "returned_transition" and e["phase"] == "warmup"]
        commands = np.frombuffer(bytes.fromhex(plan["raw15_float32_hex"]), dtype="<f4").reshape(15, 2)
        warm = runner.native.moving.warmup_commands()
        require(len(warm_transitions) == 10 and len(plan_transitions) == row["own_returned"],
                "returned action denominator")
        for trans, cmd, step in [(e, warm[j], j + 1) for j, e in enumerate(warm_transitions)] + [
                (e, commands[j], j + 1) for j, e in enumerate(plan_transitions)]:
            require(trans["step"] == step and trans["requested_bits"] == cmd.tobytes().hex() ==
                    trans["actual_bits"] and trans["actual_dtype"] == "<f4", "literal action dispatch")
        censored = row["status"] == "censored_out_of_domain"
        accepted_steps = len(plan_transitions) - int(censored)
        expected_steps = [0] + list(range(1, accepted_steps + 1))
        require([r["step"] for r in row["trajectory"] if r["phase"] == "plan"] == expected_steps,
                "complete same-step selected observation")
        expected_cp = [n for n in (0, 5, 10, 15) if n <= accepted_steps]
        require([r["step"] for r in row["checkpoints"]] == expected_cp and
                row["missing_checkpoints"] == [n for n in (0, 5, 10, 15) if n not in expected_cp],
                "checkpoints and no padding")
        guards = [e for e in branch if e["kind"] == "cp5_guard_verified"]
        if len(plan_transitions) >= 6:
            matches = [e for e in journal if e["kind"] == "operation_dispatch_intent" and
                       e["receipt"].get("branch") == i and e["receipt"]["resource"] == "public_step" and
                       e["receipt"].get("phase") == "plan" and e["receipt"].get("step") == 6]
            require(len(matches) == 1, "single action6 intent")
            six = matches[0]
            require(len(guards) == 1 and guards[0]["index"] < six["index"], "cp5 guard before suffix")
        frames = [e["receipt"] for e in branch if e["kind"] == "fresh_observation"]
        require(len(frames) == row["PNG"], "physical frame count")
        for frame in frames:
            path = runner.native.paired.flat(directory, frame["png"])
            require(runner.sha(path) == frame["file_sha256"], "frame file SHA")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            require(pixels.shape == (224, 224, 3) and
                    runner.native.moving.image_sha(pixels) == frame["pixel_sha256"], "pixel SHA")
        if censored:
            proof = row["censor"]
            step = len(plan_transitions)
            require(step > 0 and proof["step"] == step and
                    proof["action_float32_hex"] == commands[step-1].tobytes().hex() and
                    proof["returned_state"] == plan_transitions[-1]["state"] and
                    proof["out_of_bounds"] == [dict(coordinate=k, value=proof["returned_state"][k])
                                              for k in range(4) if not 0 <= proof["returned_state"][k] <= 512]
                    and bool(proof["out_of_bounds"]), "censor action and state")
            events_censor = [e for e in branch if e["kind"] == "censored_out_of_domain"]
            matching_frames = [f for f in frames if f["phase"] == "plan" and f["step"] == step
                               and f["overlay_index"] == 0]
            require(len(events_censor) == len(matching_frames) == 1 and events_censor[0]["proof"] == proof
                    and matching_frames[0]["file_sha256"] == proof["file_sha256"]
                    and matching_frames[0]["pixel_sha256"] == proof["pixel_sha256"], "censor frame identity")
            image_path = runner.native.paired.flat(directory, matching_frames[0]["png"])
            with Image.open(image_path) as image:
                pixels = np.asarray(image.convert("RGB"))
            require(int(np.count_nonzero(np.all(pixels == [65, 105, 225], axis=-1))) ==
                    proof["blue_pixels"], "censor pixel count")
            require(not any(e["kind"] == "returned_transition" and e["phase"] == "plan"
                            and e["step"] > step for e in branch), "no post-censor action")
        else:
            require(not any(e["kind"] == "censored_out_of_domain" for e in branch), "no spurious censor")
        require(len(row["responses"]) == accepted_steps, "response denominator")
        for response, trans in zip(row["responses"], plan_transitions[:accepted_steps], strict=True):
            metric = runner.native.base.previous.metric(row["goal"], trans["state"])
            require(response["step"] == trans["step"] and response["state"] == trans["state"] and
                    response["reward"] == trans["reward"] == -metric["native_distance"] and
                    response["native_metrics"] == metric and
                    response["native_success"] == metric["native_success"], "saved native metric recomputation")
    return dict(status="saved_checks_passed_not_admitted", rows=32, known_roots=2,
                scientific_P1_admission=False, freshness=False, capability_probe=False,
                checks="journal, receipts, literal actions, frame hashes, native metrics; no new physics or neural calls")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-sha", required=True)
    args = parser.parse_args()
    directory = Path(args.output)
    require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), "output directory")
    report_path = directory / "report.json"
    require(runner.sha(report_path) == args.report_sha, "externally pinned report SHA")
    report = runner.native.read(report_path, cap=64 * 1024**2)
    binding = runner.native.read(directory / "execution_binding.json")
    prepared_path = Path(binding["prepared"])
    require(prepared_path.is_file() and not prepared_path.is_symlink() and
            runner.sha(prepared_path) == binding["prepare_sha256"] == report["prepare_sha256"] and
            str(prepared_path) == report["prepared"], "prepared binding")
    prepared = runner.native.read(prepared_path, cap=64 * 1024**2)
    require(all(runner.sha(p) == h for p, h in prepared["bindings"].items()), "source binding")
    require(binding == runner.native.read(directory / "execution_binding.json") and
            runner.sha(report_path) == args.report_sha, "execution binding")
    result = check(directory, report, prepared)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
