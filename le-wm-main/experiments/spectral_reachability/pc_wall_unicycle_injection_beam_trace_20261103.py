"""Trace actual beam decisions for privileged wall-unicycle injection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from . import pc_wall_unicycle_injection_20261103 as inj
from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_composition_witness_v2 import digest
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_atlas_depth3_pruning_trace_20261103_codex import beam_assignment_line

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/pc_wall_unicycle_injection_20261103_DEV01.json"
PROTOCOL = ROOT.parent / "docs/PC_WALL_UNICYCLE_INJECTION_BEAM_TRACE_PROTOCOL_20261103.md"


def run(seed, actions):
    wall = inj.wall
    backend = wall.Backend()
    compiler = LocalCompiler(backend, goal=wall.GOAL, sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[.3, np.pi / 2], low=wall.LOW, high=wall.HIGH,
                             config=CompilerConfig(packet_steps=4, spectral_modes=2, prefix_radius=.3,
                                                   max_rollout_operations=10000, max_jacobian_operations=2000))
    search = inj.Injection(compiler, AtlasConfig(depth=4, beam_width=2, random_bridges=2, seed=seed), actions)
    if sys.gettrace() is not None:
        raise RuntimeError("exclusive tracing required")
    target = beam_assignment_line()
    pending, snapshots = set(), []

    def trace(frame, event, arg):
        if frame.f_code is not AtlasSearch.search.__code__:
            return trace
        if event == "line":
            if frame in pending:
                pending.remove(frame)
                local = frame.f_locals
                if local["root"]["root_name"] == "injected_0":
                    expanded, beam = local["expanded"], local["beam"]
                    expected = sorted(expanded, key=lambda node: compiler._cost(node["context"]))[:2]
                    if len(beam) != len(expected) or any(a is not b for a, b in zip(beam, expected)):
                        raise RuntimeError("beam is not stable sorted slice")
                    ordered = sorted(enumerate(expanded), key=lambda item: compiler._cost(item[1]["context"]))
                    ranks = {id(node): rank for rank, (_, node) in enumerate(ordered, 1)}
                    snapshots.append({"depth": len(expanded[0]["edges"]),
                                      "expanded": [{"names": [e["name"] for e in node["edges"]],
                                                    "context_sha256": digest(node["context"]),
                                                    "cost": compiler._cost(node["context"]),
                                                    "rank": ranks[id(node)],
                                                    "retained": any(node is kept for kept in beam)}
                                                   for node in expanded]})
            if frame.f_lineno == target:
                pending.add(frame)
        return trace

    try:
        sys.settrace(trace)
        result = search.search(wall.ROOT)
    finally:
        sys.settrace(None)
    if pending or [item["depth"] for item in snapshots] != [2, 3, 4]:
        raise ValueError("missing beam snapshots")
    exact = [n for n in snapshots[0]["expanded"] if n["names"] == ["injected_0", "injected_1"]]
    if len(exact) != 1:
        raise ValueError("expected unique named two-packet path")
    builder = wall.Backend()
    after_two = builder.rollout(wall.ROOT, np.asarray(actions)[:8])
    if exact[0]["context_sha256"] != digest(after_two):
        raise ValueError("two-packet witness mismatch")
    chosen = result["programs"][result["selected_index"]]
    return {"seed": seed, "exact_two_packet_node": exact[0], "snapshots": snapshots,
            "offered": search.offered, "selected_actions": chosen["actions"],
            "compiler_counts": result["operation_counts"], "backend": vars(backend),
            "binding_transitions": search.construction_transitions, "two_packet_check_transitions": 8}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    old, prior = json.loads(SOURCE.read_text()), json.loads(inj.SOURCE.read_text())
    if old["source_sha256"] != sha(inj.SOURCE):
        raise ValueError("injection source changed")
    print("Trace actual width-two beam on privileged wall-unicycle witnesses.", flush=True)
    rows = []
    for ref, origin in zip(old["rows"], prior["rows"], strict=True):
        if ref["seed"] != origin["seed"]:
            raise ValueError("source seed mismatch")
        row = run(ref["seed"], origin["selected"]["actions"])
        for name in ("offered", "selected_actions", "compiler_counts", "backend"):
            if row[name] != ref[name]:
                raise ValueError(f"injection replay mismatch: {name}")
        rows.append(row)
        x = row["exact_two_packet_node"]
        print(f"seed={row['seed']} rank={x['rank']} retained={x['retained']} cost={x['cost']:.6g}", flush=True)
    report = {"schema": "wall_unicycle_injection_beam_trace.v1", "rows": rows,
              "source_sha256": sha(SOURCE), "witness_source_sha256": sha(inj.SOURCE),
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "scope": "privileged exact development injection; not independent or planner performance"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
