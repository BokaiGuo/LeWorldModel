"""Strict saved-only verifier for the matched P2 HOST CUDA comparison."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import p2_matched_program_host_20261101_parallel as host

kernel = host.kernel
need = kernel.old.require
NAME = "p2_matched_program_saved_20261101_parallel"


def _resource_events(scoped, branch):
    attempts = dict.fromkeys(kernel.CAPS, 0)
    returned = dict.fromkeys(kernel.CAPS, 0)
    stack = []
    permitted = {
        "branch_start",
        "branch_complete",
        "packet_intent",
        "packet_return",
        "candidate_return",
        "candidate_prefix_return",
        "prefix_return",
        "joint_anchor_return",
        "winner_replayed",
    }
    for event in scoped:
        kind = event["kind"]
        if kind == "resource_attempt":
            axis = event["resource"]
            need(
                axis in attempts
                and event["ordinal"] == attempts[axis] + 1
                and event["depth"] == len(stack),
                "ordered resource attempt",
            )
            attempts[axis] += 1
            stack.append((axis, event["ordinal"]))
        elif kind == "resource_return":
            axis = event["resource"]
            need(
                stack
                and stack[-1] == (axis, event["ordinal"])
                and event["depth"] == len(stack) - 1,
                "nested resource return",
            )
            stack.pop()
            returned[axis] += 1
        else:
            need(kind in permitted, "unexpected failure or event")
    counts = branch["resource_counts"]
    need(
        not stack
        and attempts == returned == counts["attempted"] == counts["returned"]
        and attempts == kernel.CAPS
        and counts["failed"] is False
        and 0 <= counts["wall_seconds"] < kernel.contract()["wall_seconds"],
        "exact measured resource vector",
    )
    return counts


def validate_branch(case, branch, events):
    arm = branch["arm"]
    base = case["base"]
    goal = np.asarray(case["goal"], dtype=np.float64)
    need(
        arm in kernel.ARMS
        and branch["base"] == base
        and branch["schema"] == kernel.NAME + ".branch"
        and branch["context_sha256"] == kernel.old.digest(case["context"])
        and branch["goal_sha256"] == kernel.old.digest(case["goal"])
        and branch["physical"] is False
        and branch["P2_admission"] is False
        and branch["measured_resource_vector"] == kernel.CAPS
        and branch["measured_resource_match"] is True
        and branch["equal_total_compute"] is False,
        "branch identity and claim boundary",
    )
    root_singular, root_basis = kernel.old.canonical_basis(
        branch["root_jacobian"]
    )
    need(
        np.array_equal(root_singular, branch["root_singular_values"])
        and np.array_equal(root_basis, branch["root_basis"]),
        "saved root Jacobian and SVD",
    )
    library, _ = kernel.old.candidates(root_basis)
    rows = branch["candidates"]
    need(len(rows) == 16, "sixteen candidates")
    scoped = [
        event["event"]
        for event in events
        if event.get("base") == base
        and event.get("arm") == arm
        and "event" in event
    ]
    need(
        len(scoped) == len(events)
        and scoped[0]["kind"] == "branch_start"
        and scoped[-1]["kind"] == "branch_complete"
        and scoped[-1]["result_sha256"] == kernel.old.digest(branch)
        and scoped[0]["context_sha256"] == branch["context_sha256"]
        and scoped[0]["goal_sha256"] == branch["goal_sha256"]
        and scoped[0]["caps"] == kernel.CAPS
        and scoped[-1]["resource_counts"] == branch["resource_counts"],
        "journal branch bookends",
    )
    counts = _resource_events(scoped, branch)
    candidates = [event for event in scoped if event["kind"] == "candidate_return"]
    need(len(candidates) == 16, "candidate event count")

    if arm == "joint_multistart":
        anchors = [
            event for event in scoped if event["kind"] == "joint_anchor_return"
        ]
        need(len(anchors) == 4, "joint anchor count")
        for prefix_index, context in enumerate(branch["prefix_contexts"]):
            first = library[0][prefix_index]
            anchor = np.asarray(first["logits"] + [0.0] * 10, dtype=np.float64)
            singular, local = kernel.old.canonical_basis(context["jacobian"])
            need(
                context["prefix_index"] == prefix_index
                and context["anchor_logits"] == anchor.tolist()
                and np.array_equal(singular, context["singular_values"])
                and np.array_equal(local, context["basis"])
                and anchors[prefix_index]["anchor_logits"] == anchor.tolist(),
                "joint anchor witness",
            )
            for offset in range(4):
                row = rows[4 * prefix_index + offset]
                mode = offset // 2
                sign = (-1, 1)[offset % 2]
                logits = anchor + sign * (
                    kernel.old.CAP / math.sqrt(2)
                ) * local[mode]
                norm = float(np.linalg.norm(logits))
                if norm > kernel.old.CAP:
                    logits = logits * (kernel.old.CAP / norm)
                raw = np.tanh(logits).astype("<f4").reshape(10, 2)
                terminal = np.asarray(
                    row["terminal_context"]["latent_history"][-1],
                    dtype=np.float64,
                )
                need(
                    row["index"] == 4 * prefix_index + offset
                    and row["prefix_index"] == prefix_index
                    and row["mode"] == mode
                    and row["sign"] == sign
                    and row["logits"] == logits.tolist()
                    and row["raw_float32_hex"] == raw.tobytes().hex()
                    and row["prefix_context_sha256"]
                    == kernel.old.digest(row["prefix_context"])
                    and row["terminal_context_sha256"]
                    == kernel.old.digest(row["terminal_context"])
                    and row["cost"] == float(np.mean((terminal - goal) ** 2))
                    and candidates[row["index"]]["candidate"] == row,
                    "joint candidate exact",
                )
    else:
        prefixes = [event for event in scoped if event["kind"] == "prefix_return"]
        candidate_prefixes = [
            event
            for event in scoped
            if event["kind"] == "candidate_prefix_return"
        ]
        need(
            len(prefixes) == 4 and len(candidate_prefixes) == 16,
            "compiled and per-candidate prefix events",
        )
        for prefix_index, context in enumerate(branch["prefix_contexts"]):
            first = library[0][prefix_index]
            singular, local = kernel.local_basis(context["jacobian"])
            need(
                context["prefix_index"] == prefix_index
                and context["context_sha256"]
                == kernel.old.digest(context["context"])
                and np.array_equal(singular, context["singular_values"])
                and np.array_equal(local, context["basis"])
                and prefixes[prefix_index]["program_id"] == first["program_id"]
                and prefixes[prefix_index]["raw_float32_hex"]
                == first["raw_float32_hex"]
                and prefixes[prefix_index]["context_sha256"]
                == context["context_sha256"],
                "compiled prefix witness",
            )
            for offset in range(4):
                row = rows[4 * prefix_index + offset]
                mode = offset // 2
                sign = (-1, 1)[offset % 2]
                second = np.tanh(
                    sign * (kernel.old.CAP / math.sqrt(2)) * local[mode]
                ).astype("<f4").reshape(5, 2)
                event = candidate_prefixes[row["index"]]
                terminal = np.asarray(
                    row["terminal_context"]["latent_history"][-1],
                    dtype=np.float64,
                )
                need(
                    row["index"] == 4 * prefix_index + offset
                    and row["prefix_index"] == prefix_index
                    and row["mode"] == mode
                    and row["sign"] == sign
                    and row["first_raw_float32_hex"]
                    == first["raw_float32_hex"]
                    and row["second_raw_float32_hex"] == second.tobytes().hex()
                    and row["prefix_context_sha256"]
                    == context["context_sha256"]
                    and row["candidate_prefix_context_sha256"]
                    == context["context_sha256"]
                    and event["candidate_index"] == row["index"]
                    and event["prefix_index"] == prefix_index
                    and event["raw_float32_hex"] == first["raw_float32_hex"]
                    and event["context_sha256"] == context["context_sha256"]
                    and row["terminal_context_sha256"]
                    == kernel.old.digest(row["terminal_context"])
                    and row["cost"] == float(np.mean((terminal - goal) ** 2))
                    and candidates[row["index"]]["candidate"] == row,
                    "composition candidate full replay exact",
                )

    winner = min(rows, key=lambda item: (item["cost"], item["index"]))
    need(branch["winner"] == winner, "winner selection")
    replay = [event for event in scoped if event["kind"] == "winner_replayed"]
    need(
        len(replay) == 1
        and replay[0]["index"] == winner["index"]
        and replay[0]["replay"] == branch["replay"]
        and branch["replay"]["prefix_context_sha256"]
        == winner["prefix_context_sha256"]
        and branch["replay"]["terminal_context_sha256"]
        == winner["terminal_context_sha256"]
        and branch["replay"]["cost"] == winner["cost"],
        "winner replay",
    )
    return dict(
        base=base,
        arm=arm,
        status="saved_structure_and_measured_resources_verified",
        counts=counts,
        winner_index=winner["index"],
        winner_cost=winner["cost"],
        P2_admission=False,
    )


def read_events(path):
    raw = path.read_bytes()
    need(raw.endswith(b"\n") and len(raw) <= host.CAP, "bounded journal")
    return host.journal.read(
        path, json.loads(raw.splitlines()[-1])["sha256"], host.CAP
    )


def consume(directory, report_sha, supervisor_sha, prepared, prepare_sha):
    out = Path(directory)
    need(
        out.is_absolute() and out.is_dir() and not out.is_symlink(),
        "saved output",
    )
    manifest = host.verify(prepared, prepare_sha)
    need(
        host.prep.original.previous.sha(out / "report.json") == report_sha
        and host.prep.original.previous.sha(out / "supervisor_events.jsonl")
        == supervisor_sha,
        "external report hashes",
    )
    report = json.loads((out / "report.json").read_bytes())
    supervisor = read_events(out / "supervisor_events.jsonl")
    need(
        [event["kind"] for event in supervisor]
        == [
            "lock_acquired",
            "child_spawn_intent",
            "child_spawned",
            "child_exited",
            "supervisor_terminal",
        ]
        and supervisor[-1]["returncode"]
        == supervisor[-2]["returncode"]
        == 0,
        "supervisor lifecycle",
    )
    proof = report["lock_proof"]
    need(
        (proof["device"], proof["inode"])
        == (supervisor[0]["device"], supervisor[0]["inode"]),
        "cooperative lock identity",
    )
    plan = manifest["parent"]
    panel = json.loads(Path(plan["panel_path"]).read_bytes())
    need(
        report["schema"] == host.NAME + ".report"
        and report["status"] == "exploratory_complete"
        and report["prepare_sha256"] == prepare_sha
        and report["error"] is None
        and report["device"]["uuid"] == plan["gpu_uuid"]
        and report["model_before"]
        == report["model_after"]
        == plan["expected_model_hash"]
        and report["physical"] is False
        and report["P1_admission"] is False
        and report["P2_admission"] is False
        and report["measured_resource_match"] is True
        and report["equal_total_compute"] is False,
        "CUDA report and claim boundary",
    )
    host.identity.verify(report["loaded_sources"])
    events = host.journal.read(
        out / "child_events.jsonl", report["journal_tail"], host.CAP
    )
    need(
        events[0]["kind"] == "child_start"
        and events[0]["lock_proof"] == proof
        and events[0]["order"] == kernel.contract()["order"]
        and events[-1]["kind"] == "child_terminal"
        and events[-1]["status"] == "exploratory_complete"
        and events[-1]["error"] is None,
        "child lifecycle",
    )
    loaded = [event for event in events if event.get("kind") == "model_loaded"]
    need(
        len(loaded) == 1
        and loaded[0]["device"] == report["device"]
        and loaded[0]["model_sha256"] == report["model_before"],
        "model event",
    )
    rows = report["rows"]
    need(len(rows) == 8, "eight fixed slots")
    checked_rows = []
    names = {"report.json", "child_events.jsonl", "supervisor_events.jsonl"}
    for index, ((base, arm), row) in enumerate(
        zip(kernel.contract()["order"], rows, strict=True)
    ):
        name = f"branch_{index}.json"
        need(
            (row["base"], row["arm"], row["status"], row["file"])
            == (base, arm, "complete", name)
            and host.prep.original.previous.sha(out / name) == row["sha256"],
            "saved branch hash",
        )
        branch = json.loads((out / name).read_bytes())
        bound = [
            event
            for event in events
            if event.get("base") == base
            and event.get("arm") == arm
            and "event" in event
        ]
        checked = validate_branch(panel["cases"][base], branch, bound)
        file_event = [
            event
            for event in events
            if event.get("kind") == "branch_file"
            and event.get("index") == index
        ]
        need(
            len(file_event) == 1
            and file_event[0]["file"] == name
            and file_event[0]["sha256"] == row["sha256"]
            and checked["counts"] == row["resource_counts"],
            "branch journal binding",
        )
        checked_rows.append(checked)
        names.add(name)
    need(
        {path.name for path in out.iterdir()} == names
        and all(path.is_file() and not path.is_symlink() for path in out.iterdir()),
        "exact saved artifact inventory",
    )
    pairs = []
    for base in range(4):
        pair = [row for row in checked_rows if row["base"] == base]
        need(
            [row["arm"] for row in pair] == list(kernel.ARMS)
            and pair[0]["counts"]["attempted"]
            == pair[1]["counts"]["attempted"]
            == kernel.CAPS,
            "same-root exact measured resource pairing",
        )
        pairs.append(
            dict(
                base=base,
                measured_resource_vector=kernel.CAPS,
                joint_winner_cost=pair[0]["winner_cost"],
                composition_winner_cost=pair[1]["winner_cost"],
                model_preference=(
                    "joint_multistart"
                    if pair[0]["winner_cost"] < pair[1]["winner_cost"]
                    else "prefix_chain"
                    if pair[1]["winner_cost"] < pair[0]["winner_cost"]
                    else "tie"
                ),
            )
        )
    return dict(
        schema=NAME + ".receipt",
        status="saved_cuda_comparison_verified_not_scientific",
        rows=checked_rows,
        pairs=pairs,
        measured_resource_match=True,
        measured_axes=list(kernel.MEASURED_AXES),
        equal_total_compute=False,
        total_flops=None,
        report_sha256=report_sha,
        supervisor_sha256=supervisor_sha,
        prepare_sha256=prepare_sha,
        physical=False,
        independent_confirmation=False,
        known_roots=True,
        P1_admission=False,
        P2_admission=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in (
        "output",
        "report-sha",
        "supervisor-sha",
        "prepared",
        "prepare-sha",
        "receipt",
    ):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    receipt = Path(args.receipt)
    need(
        receipt.is_absolute()
        and receipt.parent == host.ROOT / "results"
        and receipt.name.startswith(NAME + "_RECEIPT_"),
        "exclusive receipt",
    )
    value = consume(
        args.output,
        args.report_sha,
        args.supervisor_sha,
        args.prepared,
        args.prepare_sha,
    )
    print(
        json.dumps(
            dict(
                path=str(receipt),
                sha256=host.prep.original.previous.write_new(receipt, value),
                status=value["status"],
            )
        )
    )


if __name__ == "__main__":
    main()
