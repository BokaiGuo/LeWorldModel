"""Check and summarize completed fixed common-prefix probe shards.

This saved-data analyzer does not authorize P1, claim fresh roots, validate
unserialized simulator caches, or select candidate actions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability import p1_backtrack_capability_prepare_20261030 as preparation
from experiments.spectral_reachability import p1_backtrack_capability_runner_20261030 as runner
from experiments.spectral_reachability import p1_backtrack_physical_consume_20261030 as prior

require = runner.require


def check_shard(directory, anchor, probes):
    report = runner.physical.native.read(directory / "report.json", cap=64 * 1024**2)
    require(report["schema"] == runner.NAME + ".shard" and report["anchor"] == anchor["index"] and
            report["status"] == "recorded_not_independently_consumed" and not report["errors"] and
            report["fixed_rows"] == len(report["rows"]) == 21 and
            report["P1_admission"] is False and report["new_neural_calls"] == 0, "completed fixed shard")
    require(runner.physical.sha(directory / "journal.jsonl") == report["journal_sha256"], "journal SHA")
    journal = prior.events(directory / "journal.jsonl", report["journal_chain_tail"])
    require(journal[-1]["status"] == report["status"] and journal[-1]["counts"] == report["counts"],
            "terminal counts")
    returned = sorted((e["receipt"] for e in journal if e["kind"] == "operation_returned"),
                      key=lambda receipt: receipt["index"])
    intents = [e["receipt"] for e in journal if e["kind"] == "operation_dispatch_intent"]
    require(returned == report["receipts"] and [e["index"] for e in intents] == list(range(len(returned))),
            "dispatch and return ledger")
    c = {k: v["completed"] for k, v in report["counts"].items()}
    require(all(v["attempted"] == v["completed"] and v["attempted"] <= runner.CAPS[k]
                for k, v in report["counts"].items()) and
            c["constructor"] == c["close"] == 21 and c["reset"] == 42 and
            c["set_state"] == 2 * c["reset"] and c["physics_step"] == 2*c["reset"]+10*c["public_step"] and
            c["goal_render"] == c["reset"] and c["goal_setter"] == c["reset"] + c["goal_bind"] and
            c["goal_bind"] == 21 and c["explicit_render"] == sum(row["PNG"] for row in report["rows"]) and
            c["public_step"] == sum(row["warmup_returned"] + row["own_returned"] for row in report["rows"]),
            "resource census")
    endpoints = []
    warm = runner.physical.native.moving.warmup_commands()
    prefix = np.frombuffer(bytes.fromhex(anchor["prefix5_float32_hex"]), dtype="<f4").reshape(5, 2)
    for row, probe in zip(report["rows"], probes, strict=True):
        index = probe["index"]
        require(row["index"] == index and row["anchor_index"] == anchor["index"] and
                row["task_id"] == anchor["task_id"] and row["probe"] == probe["name"] and
                row["start_verified"] is True and row["status"] in ("probe_completed", "native_stopped") and
                not row["secondary_errors"], "probe row and identity")
        branch = [e for e in journal if e.get("branch") == index]
        frames = [e for e in branch if e["kind"] == "fresh_observation"]
        require(len(frames) == row["PNG"] and len(frames) >= 51, "complete anchor frame triplets")
        for actual, reference in zip(frames[:51], anchor["trace"], strict=True):
            require(actual["pre"] == actual["post"] == reference["pre"] == reference["post"] and
                    all(actual["receipt"][k] == reference["receipt"][k] for k in
                        ("phase", "step", "condition", "overlay_index", "file_sha256",
                         "pixel_sha256", "state", "snapshot")), "exact prefix replay")
        for event in frames:
            frame = event["receipt"]
            path = runner.physical.native.paired.flat(directory, frame["png"])
            require(runner.physical.sha(path) == frame["file_sha256"], "image file SHA")
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            require(pixels.shape == (224, 224, 3) and
                    runner.physical.native.moving.image_sha(pixels) == frame["pixel_sha256"], "pixel hash")
        start = [e for e in branch if e["kind"] == "suffix_start_verified"]
        require(len(start) == 1, "single start verification")
        transitions = [e for e in branch if e["kind"] == "returned_transition"]
        suffix = np.frombuffer(bytes.fromhex(probe["raw5_float32_hex"]), dtype="<f4").reshape(5, 2)
        for phase, commands in (("warmup", warm), ("plan", prefix), ("probe", suffix)):
            events = [e for e in transitions if e["phase"] == phase]
            require(len(events) == (len(row["responses"]) if phase == "probe" else len(commands)),
                    "action denominator")
            for step, (event, command) in enumerate(zip(events, commands), 1):
                require(event["step"] == step and event["requested_bits"] == event["actual_bits"] ==
                        command.tobytes().hex() and event["actual_dtype"] == "<f4", "literal action identity")
                if phase == "probe" and step == 1:
                    intent = [e for e in journal if e["kind"] == "operation_dispatch_intent" and
                              e["receipt"].get("branch") == index and
                              e["receipt"].get("phase") == "probe" and e["receipt"].get("step") == 1 and
                              e["receipt"]["resource"] == "public_step"]
                    require(len(intent) == 1 and start[0]["index"] < intent[0]["index"] < event["index"],
                            "verified before suffix dispatch")
        require(row["start_state"] == anchor["checkpoint"]["state"], "anchor state")
        for response, event in zip(row["responses"], [e for e in transitions if e["phase"] == "probe"], strict=True):
            delta = np.asarray(event["state"]) - np.asarray(row["start_state"])
            require(response["step"] == event["step"] and response["state"] == event["state"] and
                    response["delta_native7"] == delta.tolist() and
                    response["terminated"] == event["terminated"] and response["truncated"] == event["truncated"],
                    "saved response")
        endpoints.append(row["responses"][-1]["state"] if len(row["responses"]) == 5 else None)
    return endpoints


def summarize(anchor, endpoints):
    origin = np.asarray(anchor["root_state"][:4], dtype=np.float64)
    goal = np.asarray(anchor["goal"][:4], dtype=np.float64)
    direction = goal - origin
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm else None
    zero = None if endpoints[0] is None else np.asarray(endpoints[0][:4], dtype=np.float64)
    deltas = [None if state is None or zero is None else (np.asarray(state[:4]) - zero) for state in endpoints]
    projections = [None if d is None or direction is None else float(direction @ d) for d in deltas]
    finite_difference, asymmetry = [], []
    for coordinate in range(10):
        plus, minus = deltas[1 + 2 * coordinate:3 + 2 * coordinate]
        finite_difference.append(None if plus is None or minus is None else ((plus - minus) / .5).tolist())
        asymmetry.append(None if plus is None or minus is None else (plus + minus).tolist())
    return dict(anchor=anchor["index"], root_slot=anchor["root_slot"], condition=anchor["condition"],
                role=anchor["role"], complete_endpoints=sum(x is not None for x in endpoints),
                direction=None if direction is None else direction.tolist(), relative_delta=[
                    None if x is None else x.tolist() for x in deltas], projection=projections,
                central_difference=finite_difference, asymmetry=asymmetry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--prepare-sha", required=True)
    args = parser.parse_args()
    source = Path(args.prepared)
    require(runner.physical.sha(source) == args.prepare_sha, "prepared SHA")
    prepared = runner.physical.native.read(source, cap=32 * 1024**2)
    require(prepared == preparation.build(), "source and input closure")
    directory = Path(args.directory)
    results = []
    for anchor in prepared["anchors"]:
        shard = directory / f"anchor_{anchor['index']:02d}"
        require(shard.is_dir(), "all 16 shards required")
        results.append(summarize(anchor, check_shard(shard, anchor, prepared["probes"])))
    print(json.dumps(dict(status="saved_probe_checks_passed_not_admitted", known_roots=2,
                          fixed_rows=336, P1_admission=False, summaries=results), sort_keys=True))


if __name__ == "__main__":
    main()
