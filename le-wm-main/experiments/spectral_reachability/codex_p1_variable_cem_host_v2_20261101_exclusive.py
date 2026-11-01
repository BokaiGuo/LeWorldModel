"""Variable CEM host v2: bind a closed complete CEM journal into the report."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
from unittest.mock import patch

from . import codex_p1_variable_cem_host_20261101_exclusive as previous
from .codex_e1_proposal_durable_events_20261101 import read_events

NAME = "codex_p1_variable_cem_host_v2_20261101_exclusive"
AUTH = "MAIN_AUTHORIZES_P1_VARIABLE_CEM_V2_CUDA_AND_PHYSICS"
HERE = Path(__file__).resolve()
DOC = HERE.parents[3] / "docs/D2P_P1_VARIABLE_CEM_HOST_V2_20261101_EXCLUSIVE.md"
TEST = HERE.parents[2] / "tests/test_codex_p1_variable_cem_host_v2_20261101_exclusive.py"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def anchor_journal(journal):
    require(journal is not None and journal.fd is None and not journal.poisoned,
            "CEM journal missing, open, or poisoned")
    raw = journal.path.read_bytes()
    rows = read_events(journal.path)
    require(bool(rows) and len(rows) == journal.index and rows[-1]["sha256"] == journal.tail,
            "CEM journal incomplete or changed")
    return {"status": "closed", "path": "full_cem_events.jsonl",
            "sha256": hashlib.sha256(raw).hexdigest(), "event_count": len(rows),
            "chain_tail": journal.tail}


def verify_anchor(path, anchor):
    require(anchor.get("status") == "closed" and anchor.get("path") == "full_cem_events.jsonl",
            "invalid CEM anchor")
    path = Path(path)
    require(path.name == anchor["path"], "CEM journal path")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == anchor.get("sha256"), "CEM journal SHA")
    rows = read_events(path)
    require(bool(rows) and len(rows) == anchor.get("event_count")
            and rows[-1]["sha256"] == anchor.get("chain_tail"), "CEM journal chain")


def finalize_report(report, captured, path):
    try:
        require(len(captured) == 1, "CEM controller not created exactly once")
        report["cem_journal"] = anchor_journal(captured[0].journal)
        verify_anchor(Path(path) / "full_cem_events.jsonl", report["cem_journal"])
    except (ValueError, OSError) as exc:
        report["cem_journal"] = {"status": "unavailable_or_invalid", "reason": str(exc)}
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
        report["status"] = "partial_or_failed"
    return report


@contextmanager
def configured(root):
    with previous.configured(root) as runner:
        parent_build, parent_protocol = runner.build, runner.protocol

        def protocol():
            value = deepcopy(parent_protocol())
            value.update(schema=f"{NAME}_root{root}.protocol.v2",
                         cem_journal_anchor={"required": True, "algorithm": "sha256",
                                             "closed_writer_required": True,
                                             "external_report_anchor_required": True})
            return value

        def build():
            value = parent_build()
            value["schema"] = f"{NAME}_root{root}.prepare.v2"
            value["protocol"] = protocol()
            for path in (HERE, DOC, TEST):
                value["sources"][str(path)] = runner.sha(path)
            return value

        def execute(prepared, data, store, visual, runtime, factory):
            captured = []
            base = previous.variable.Controller

            class TrackedController(base):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    captured.append(self)

            env = dict(previous.old.execute_one.__globals__)
            env.update(NAME=NAME, node_cost=previous.node_cost,
                       node_contract=previous.node_contract,
                       verify_node_events=previous.verify_node_events,
                       cem=SimpleNamespace(**{**vars(previous.old.cem), "Controller": TrackedController}))
            fn = FunctionType(previous.old.execute_one.__code__, env, "v2_execute_one",
                              previous.old.execute_one.__defaults__, previous.old.execute_one.__closure__)
            target = (previous.old.root0, previous.old.root1)[root]
            report = fn(prepared, data, store, visual, runtime, factory, runner, target.rebind, root)
            return finalize_report(report, captured, store.path)

        with patch.object(runner, "NAME", f"{NAME}_root{root}"), \
             patch.object(runner, "AUTH", AUTH), patch.object(runner, "protocol", protocol), \
             patch.object(runner, "build", build), patch.object(runner, "execute", execute):
            yield runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=int, choices=(0, 1), required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-prepare")
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", required=True)
    run = sub.add_parser("run")
    for key in ("prepared", "sha256", "output", "authorization"):
        run.add_argument("--" + key, required=True)
    run.add_argument("--authorize-host-cuda-and-physics", action="store_true", required=True)
    args = parser.parse_args()
    prefix = f"{NAME}_root{args.root}"
    if args.command in ("prepare", "run"):
        path = Path(args.output if args.command == "prepare" else args.prepared)
        require(path.name.startswith(prefix + "_PREP_") and path.name.endswith(".json"), "prepare name")
    if args.command == "run":
        require(Path(args.output).name.startswith(prefix + "_RUN_")
                and args.authorization == AUTH, "output/authorization")
    with configured(args.root) as runner:
        if args.command == "dry-prepare":
            value = runner.build()
            result = {"status": "dry_prepared_no_model_or_physics", "root": args.root,
                      "digest": runner.digest(value), "protocol": value["protocol"]}
        elif args.command == "prepare":
            result = runner.prepare(args.output)
        else:
            with previous.old.gpu.locking.gpu_lock(previous.old.gpu.LOCK):
                result = runner.run(args.prepared, args.sha256, args.output, args.authorization)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.command == "run" and result["status"] != "recorded_known_root_not_admitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
