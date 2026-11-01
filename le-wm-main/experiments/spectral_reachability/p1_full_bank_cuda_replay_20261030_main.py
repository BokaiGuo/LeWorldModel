"""Replay all fixed full-bank selections with the existing CUDA replay implementation."""
import argparse
import json
from pathlib import Path

from . import p1_diverse_cuda_replay_20261030_main as replay
from . import p1_full_bank_selector_20261030_main as selector

NAME = "p1_full_bank_cuda_replay_20261030_main"
ROOT = replay.ROOT
SELECTED = ROOT / "results/p1_full_bank_selector_20261030_main_RUN01.json"
SELECTED_SHA = "f2c4e481287223c43488d692b7f7c20f4704a6c454dd2ebbc61d0444a2b7b32d"
PROTOCOL = ROOT.parent / "docs/D2P_P1_FULL_BANK_CUDA_PROTOCOL_20261030_MAIN.md"
original_sources = replay.sources
original_bindings = replay.bindings


def sources():
    previous, backtrack, report, prepared = original_sources()
    selected = replay.prior.one.saved.read_verified(SELECTED, SELECTED_SHA)
    prior = replay.prior.one.saved.read_verified(replay.JOIN, replay.JOIN_SHA)
    rebuilt = selector.build(backtrack, report, prepared, prior)
    extras = {"code_sha256", "source_join_sha256", "source_backtrack_sha256", "source_report_sha256"}
    if {k: v for k, v in selected.items() if k not in extras} != rebuilt:
        raise ValueError("full-bank selection reconstruction")
    rows = []
    for i, item in enumerate(selected["rows"]):
        incumbent = previous[2 * i]
        if (incumbent["slot"], incumbent["arm"], incumbent["task_binding"]) != (
                item["slot"], item["arm"], item["task_binding"]):
            raise ValueError("task identity")
        rows.append(incumbent)
        rows.append(dict(slot=item["slot"], arm=item["arm"], role="diverse_selected",
                         candidate_index=item["selected_index"], task_binding=item["task_binding"],
                         raw15_le_float32_hex=item["raw15_float32_hex"]))
    return rows, backtrack, report, prepared


def bindings():
    value = original_bindings()
    for path in (Path(__file__), Path(selector.__file__), SELECTED, PROTOCOL):
        path = path.resolve()
        value["additional"][str(path)] = replay.prior.digest(path)
    return value


def configure():
    replay.NAME = NAME
    replay.sources = sources
    replay.bindings = bindings


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output", required=True)
    r = sub.add_parser("run")
    r.add_argument("--prepared", required=True)
    r.add_argument("--sha256", required=True)
    r.add_argument("--output", required=True)
    args = parser.parse_args()
    configure()
    if args.command == "prepare":
        print(json.dumps({"sha256": replay.prepare(args.output), "physical_authorized": False}))
    else:
        print("Replay all full-bank and incumbent actions on host CUDA; no physical release.", flush=True)
        digest, complete = replay.run(args.prepared, args.sha256, args.output)
        print(json.dumps({"sha256": digest, "complete": complete}))
        if not complete:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
