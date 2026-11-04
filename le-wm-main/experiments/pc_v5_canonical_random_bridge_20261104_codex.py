"""Fixed random-bridge comparison for sealed V5 selected prefixes."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v5_canonical_response_local_mechanism_20261104_codex as previous
from experiments.spectral_reachability import pc_response_temporal_search_v1 as core
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = previous.bank.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V5_CANONICAL_RANDOM_BRIDGE_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v5_canonical_random_bridge_20261104_codex"
MANIFEST, DATA, AUDIT = [PREFIX.with_name(PREFIX.name + part + ".json")
                         for part in ("_manifest", "_data", "_audit")]
sha, write, exact = previous.sha, previous.write, previous.exact


def build(row):
    root, task, goal = row["root"], row["task"], np.asarray(row["task"]["goal"])
    prefix = np.clip(core.interpolate(np.random.default_rng(2027112300 + root).normal(size=(1, 4, 2)), 5)[0], -1, 1)
    trace = []

    def predict(actions):
        raw = np.asarray(actions, dtype=np.float32)
        positions = exact.endpoints(task, raw)
        scores = np.linalg.norm(positions-goal, axis=1)
        trace.append({"actions": raw.tolist(), "positions": positions.tolist(), "scores": scores.tolist()})
        return positions, scores

    zero = np.concatenate((prefix, np.zeros((10, 2))))[None]
    zero_score = float(predict(zero)[1][0])
    chain, value, generations = core.cem(np.zeros((10, 2)), prefix, 119,
        np.random.default_rng(2027111400 + root), predict, zero_score, 4)
    tails = [np.tile(value, (10, 1)) for value in
             ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
    tails.append(np.asarray(row["selected_tail"]))
    full = np.asarray([np.concatenate((prefix, tail)) for tail in tails], np.float32)
    _, scores = predict(full)
    modified = dict(row)
    modified["candidates"] = row["candidates"][:-1] + [{"name": "random_bridge", "prefix": prefix.tolist()}] + row["candidates"][-1:]
    modified["lookahead_index"] = len(modified["candidates"]) - 2
    response = previous.local.build(modified)
    bridge = response["controls"]["lookahead"]
    selected = response["controls"]["selected"]
    return {"root": root, "prefix": prefix.tolist(), "seed": 2027112300+root,
        "zero_distance": zero_score, "optimized_chain": chain.tolist(), "optimized_distance": value,
        "generations": generations, "trace": trace, "common_distances": scores.tolist(),
        "relative_to_zero": (scores-scores[0]).tolist(), "selected_local": selected,
        "bridge_local": bridge, "response_gap": response["gaps"]["lookahead"],
        "selected_nonprogress": response["nonprogress"], "selected_tail_nonzero": response["selected_tail_nonzero"],
        "tail_clip_linf": response["tail_clip_linf"]}


def main(mode):
    if mode == "plan":
        if any(p.exists() for p in (MANIFEST, DATA, AUDIT)):
            raise FileExistsError("existing or partial diagnostic")
        local_audit = json.loads(previous.AUDIT.read_text())
        if local_audit["data_sha256"] != sha(previous.DATA):
            raise ValueError("local audit mismatch")
        deps = seal.capture([Path(__file__), PROTOCOL, previous.BANK, previous.BANK_AUDIT,
            previous.DATA, previous.AUDIT, Path(previous.local.__file__), Path(core.__file__),
            Path(exact.__file__), Path(exact.wall.__file__)])
        write(MANIFEST, {"seal": deps, "scope": "post-hoc one fixed random bridge per exposed root"})
        print("Fixed random bridge, same suffix optimizer, and task-local response.", flush=True)
        rows = [build(row) for row in json.loads(previous.BANK.read_text())["rows"]]
        if len(rows) != 24:
            raise ValueError("incomplete panel")
        seal.verify(deps)
        write(DATA, {"manifest_sha256": sha(MANIFEST), "rows": rows})
    else:
        if AUDIT.exists():
            raise FileExistsError(AUDIT)
        manifest, data = json.loads(MANIFEST.read_text()), json.loads(DATA.read_text())
        seal.verify(manifest["seal"])
        if data["manifest_sha256"] != sha(MANIFEST):
            raise ValueError("manifest mismatch")
        rebuilt = [build(row) for row in json.loads(previous.BANK.read_text())["rows"]]
        if rebuilt != data["rows"] or len(rebuilt) != 24:
            raise ValueError("random bridge rebuild mismatch")
        seal.verify(manifest["seal"])
        write(AUDIT, {"data_sha256": sha(DATA), "verified_roots": len(rebuilt),
            "suffix_evaluation_chains": 24*126, "local_response_chains": 24*4*2*21,
            "scope": "same-code exact dynamics deterministic reconstruction"})
        print("Rebuilt all random bridge trajectories and local responses.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    main(sys.argv[1])
