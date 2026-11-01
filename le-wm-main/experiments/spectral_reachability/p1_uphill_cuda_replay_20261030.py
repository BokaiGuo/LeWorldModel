"""Replay all paired original and latent-uphill joint winners on host CUDA."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import p1_joint_cuda_replay_20261030 as engine
from . import p1_joint_cuda_replay_consume_20261030 as consumer
from . import p1_model_uphill_rerank_20261030 as rerank

ROOT = engine.ROOT
NAME = "p1_uphill_cuda_replay_20261030"
RERANK = ROOT / "results/p1_model_uphill_rerank_20261030_RUN01.json"
RERANK_SHA = "5bd30892cca302e6b98a456b7c50b2364bd70e442ec5b77497d02cafb8e80033"
PROTOCOL = ROOT.parent / "docs/D2P_P1_UPHILL_REPLAY_PROTOCOL_20261030_MAIN.md"
original_sources, original_files = engine.sources, engine.files


def sources():
    baseline, joint, backtrack, report, prepared = original_sources()
    saved = rerank.read(RERANK, RERANK_SHA)
    need = rerank.need
    need(saved["source_sha256"] == rerank.SOURCE_SHA and len(saved["groups"]) == 8
         and saved["P1_admission"] is False and saved["physical_release"] is False, "rerank source")
    rows = []
    for i, group in enumerate(saved["groups"]):
        old = dict(baseline[2 * i + 1])
        selected = group["selected"]
        need(selected is not None and (group["slot"], group["arm"]) == (old["slot"], old["arm"]),
             "selected roster")
        candidate = joint["rows"][20*i + selected["candidate_index"]]
        record = candidate["trajectory"][selected["iteration"]]
        need(candidate["best_iteration"] == selected["iteration"]
             and record["actions"] == selected["actions"], "candidate iteration")
        raw = np.asarray(record["actions"], dtype="<f4").reshape(15, 2)
        need(np.isfinite(raw).all() and (np.abs(raw) <= 1).all(), "action bounds")
        new = dict(old, candidate_index=selected["candidate_index"], raw15_le_float32_hex=raw.tobytes().hex())
        rows.extend((old, new))
    return rows, joint, backtrack, report, prepared


def files():
    data = original_files()
    for path in (Path(__file__).resolve(), Path(rerank.__file__).resolve(), RERANK,
                 PROTOCOL, rerank.PROTOCOL):
        data[str(path)] = engine.old.digest(path)
    return data


def configure():
    engine.NAME = NAME
    engine.sources = sources
    engine.files = files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "consume"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--sha256")
    parser.add_argument("--result")
    parser.add_argument("--result-sha")
    args = parser.parse_args()
    configure()
    if args.command == "prepare":
        print(json.dumps({"sha256": engine.prepare(args.output)}))
    elif args.command == "run":
        print("Replay all 16 paired actions on host CUDA, without physical execution.", flush=True)
        digest, complete = engine.run(args.prepared, args.sha256, args.output)
        print(json.dumps({"sha256": digest, "complete": complete}))
        if not complete:
            raise SystemExit(1)
    else:
        out = engine.target(args.output)
        value = engine.old.one.saved.read_verified(args.result, args.result_sha)
        manifest = engine.old.one.saved.read_verified(args.prepared, args.sha256)
        rerank.need(value["manifest"] == args.prepared and value["manifest_sha256"] == args.sha256,
                    "manifest identity")
        rows, joint, backtrack, report, prepared = sources()
        checked = consumer.verify(value, manifest, rows, joint, backtrack, report, prepared)
        checked.update(result_sha256=args.result_sha, prepare_sha256=args.sha256,
                       wrapper_sha256=engine.old.digest(Path(__file__)))
        print(json.dumps({"sha256": engine.old.one.saved.write(out, checked), **checked}))


if __name__ == "__main__":
    main()
