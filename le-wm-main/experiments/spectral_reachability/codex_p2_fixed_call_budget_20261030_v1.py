"""Fixed, known-root P2 model-call comparison; not equal total compute."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time

from experiments.spectral_reachability import codex_p2_knownroot_host_cuda_20261030_v1 as host

NAME = "codex_p2_fixed_call_budget_20261030_v1"
ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / "results/codex_p2_knownroot_host_cuda_20261030_v1_prepared01"
PREP_SHA = "455038931b3db1a89f34cd5e95f9e40662b01291b84bfd4d4bd523d54892605e"
ACK = "KNOWN_ROOT_REPEAT_NOT_PREREGISTERED_NOT_EQUAL_TOTAL_COMPUTE"
ARMS = host.frontier.ARMS
SHARED = host.bank.gate_module.SHARED_ARM
need = host.need


def contract():
    return dict(schema=NAME + ".contract", roots=list(host.frontier.ROOTS),
                arms=list(ARMS), candidate_counts=[80, 40],
                shared_compilation_entries_per_kind=4,
                exclusive_entries_per_kind=164,
                standalone_entries_per_kind=168, caps=deepcopy(host.CAPS),
                entry_kinds=["predict", "action_encoder"], shared_charge="once to each standalone arm; once in unique execution",
                repetitions=1, train=False, physical=False, known_roots=True,
                independent_confirmation=False, pre_existing_results_known=True,
                preregistered=False, equal_total_compute=False, P1_admission=False,
                P2_admission=False, physical_authorized=False)


def inspect_events(events):
    counts = {owner: {kind: 0 for kind in contract()["entry_kinds"]}
              for owner in (*ARMS, SHARED)}
    for event in events:
        if event.get("kind") != "entry_terminal":
            continue
        entry = event["entry"]
        owner, kind = entry["scope"]["arm"], entry["kind"]
        need(owner in counts and kind in counts[owner] and entry["completed"] is True and
             entry["samples"] == 1, "unexpected or failed model entry")
        counts[owner][kind] += 1
    need(all(counts[SHARED][kind] == 4 and
             all(counts[arm][kind] == 164 for arm in ARMS)
             for kind in contract()["entry_kinds"]), "fixed exact call census")
    return dict(exclusive=counts, attributed={arm: {
        kind: counts[arm][kind] + counts[SHARED][kind]
        for kind in contract()["entry_kinds"]} for arm in ARMS},
        unique={kind: sum(counts[arm][kind] for arm in counts)
                for kind in contract()["entry_kinds"]})


def audit_run(directory):
    directory = Path(directory)
    report = host.frontier.read(directory / "run_report.json")
    need(report["status"] == "exploratory_complete" and
         report["P1_admission"] is False and report["P2_admission"] is False and
         report["equal_total_compute"] is False and len(report["roots"]) == 4,
         "complete unadmitted run")
    rows = []
    for i, receipt in enumerate(report["roots"]):
        path = directory / ("root_" + str(i))
        need(receipt["root_index"] == i and receipt["root_sha256"] == host.frontier.ROOTS[i]
             and Path(receipt["directory"]) == path and
             host.frontier.file_sha(path / "report.json") == receipt["report_sha256"], "root identity")
        root = host.frontier.read(path / "report.json")
        need([r["arm"] for r in root["results"]] == list(ARMS) and
             [r["selection_evaluations"] for r in root["results"]] == [80, 40] and
             all(r["complete"] is True for r in root["results"]), "candidate census")
        raw = (path / "events.jsonl").read_bytes()
        need(raw.endswith(b"\n") and host.frontier.file_sha(path / "events.jsonl") ==
             root["events_sha256"], "journal integrity")
        counts = inspect_events([json.loads(line) for line in raw.splitlines()])
        budget = root["budget"]
        need(budget["failed"] is False and all(
             budget["attempted"][arm][kind] == counts["attributed"][arm][kind] == 168
             for arm in ARMS for kind in contract()["entry_kinds"]), "gate/count mismatch")
        rows.append(dict(root_index=i, root_sha256=receipt["root_sha256"], calls=counts,
                         attempted_budget=budget["attempted"],
                         selection_evaluations=[80, 40], report_sha256=receipt["report_sha256"]))
    return dict(status="fixed_model_entry_counts_checked", rows=rows,
                unique_calls_per_kind=sum(r["calls"]["unique"]["predict"] for r in rows),
                equal_total_compute=False, independent_confirmation=False,
                P1_admission=False, P2_admission=False)


def prepare(output):
    host.validate_prepare(PREP, PREP_SHA)
    output = Path(output)
    need(output.is_absolute() and output.name.startswith(NAME + "_") and
         not os.path.lexists(output), "exclusive manifest")
    document = dict(contract=contract(), upstream_prepare=str(PREP),
                    upstream_prepare_sha256=PREP_SHA,
                    implementation_sha256=host.frontier.file_sha(__file__),
                    prepared_before_this_run=True,
                    status="fixed_for_future_execution_after_prior_outcomes_known")
    host.bank._write(output, document)
    return host.frontier.file_sha(output)


def load_manifest(path, expected):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink() and
         host.frontier.file_sha(path) == expected, "manifest SHA")
    m = host.frontier.read(path)
    need(m == dict(contract=contract(), upstream_prepare=str(PREP),
                   upstream_prepare_sha256=PREP_SHA,
                   implementation_sha256=host.frontier.file_sha(__file__),
                   prepared_before_this_run=True,
                   status="fixed_for_future_execution_after_prior_outcomes_known"), "manifest drift")
    return host.validate_prepare(PREP, PREP_SHA)


def run(manifest, expected, output, ack):
    need(ack == ACK, "explicit known-root repeat acknowledgement required")
    m, protocol, prepared = load_manifest(manifest, expected)
    output = Path(output)
    need(output.is_absolute() and output.parent.resolve() == output.parent and
         output.name.startswith(NAME + "_") and not os.path.lexists(output), "exclusive output")
    output.mkdir(mode=0o700)
    status = dict(schema=NAME + ".run", status="partial_or_failed",
                  manifest_sha256=expected, started_ns=time.time_ns(),
                  independent_confirmation=False, equal_total_compute=False,
                  P1_admission=False, P2_admission=False)
    host.bank._write(output / "started.json", status)
    try:
        host._run(manifest=m, protocol=protocol, prepared=prepared,
                  output=output / "planner", backend=host.HostCuda())
        checked = audit_run(output / "planner")
        load_manifest(manifest, expected)
        status.update(status="fixed_model_entry_counts_checked", audit=checked)
    except BaseException as exc:
        status["error"] = dict(type=type(exc).__name__, message=str(exc)[:1500])
        raise
    finally:
        status["finished_ns"] = time.time_ns()
        host.bank._write(output / "report.json", status)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "execute", "audit"))
    p.add_argument("--manifest")
    p.add_argument("--manifest-sha256")
    p.add_argument("--output", required=True)
    p.add_argument("--ack")
    a = p.parse_args()
    if a.mode == "prepare":
        result = {"manifest_sha256": prepare(a.output)}
    elif a.mode == "execute":
        need(a.manifest and a.manifest_sha256, "manifest and SHA required")
        value = run(a.manifest, a.manifest_sha256, a.output, a.ack)
        result = {"status": value["status"], "report_sha256": host.frontier.file_sha(Path(a.output) / "report.json")}
    else:
        result = audit_run(a.output)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
