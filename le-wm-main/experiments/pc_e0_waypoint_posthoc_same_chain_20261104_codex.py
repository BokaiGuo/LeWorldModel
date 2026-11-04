"""Post-hoc reuse of the fixed V5 intervention on sealed waypoint plans."""
import json
from pathlib import Path
import sys

import numpy as np

from experiments import pc_e0_goal_waypoint_search_dev_20261104_codex as source
from experiments import pc_v5_e0_prospective_interventions_dev_20261104_codex as prior
from experiments.pc_e0_wall_independent_branch_replay_dev_20261104_codex import independent
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT, SOURCE = source.ROOT, source.OUT
OUT = ROOT / "results/pc_e0_waypoint_posthoc_same_chain_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_E0_WAYPOINT_POSTHOC_SAME_CHAIN_PROTOCOL_20261104_CODEX.md"
sha = source.prior.previous.sha


def convert(row):
    if row["arm"] != "waypoint_cem" or row["status"] != "completed":
        raise ValueError("invalid source row")
    bank = [{"name": "zero", "prefix": np.zeros((5, 2), np.float32).tolist()}]
    bank.extend({"name": f"waypoint_{i}", "prefix": item["prefix"]}
                for i, item in enumerate(row["result"]["waypoints"]))
    prefixes = np.asarray([item["prefix"] for item in bank], np.float32)
    positions = prior.exact.endpoints(row["task"], prefixes)
    scores = np.linalg.norm(positions - np.asarray(row["task"]["goal"]), axis=1)
    return {"root": row["root"], "task": row["task"], "result": {
        "selected": {"chain": row["selected"]}, "first_bank": bank, "prefix_scores": scores.tolist()}}


def calculate(row):
    old_local = prior.local

    class Adapter:
        @staticmethod
        def build(argument):
            result = old_local.build(argument)
            result["controls"]["selected"]["candidate_index"] = -1
            return result
    try:
        prior.local = Adapter
        return prior.calculate(convert(row))
    finally:
        prior.local = old_local


def source_rows():
    result = json.loads((SOURCE / "result.json").read_text())
    audit = json.loads((SOURCE / "audit.json").read_text())
    geometry = json.loads((SOURCE / "independent_geometry_replay.json").read_text())
    digest = sha(SOURCE / "result.json")
    if (audit["status"] != "passed" or geometry["status"] != "passed"
            or audit["result_sha256"] != digest or geometry["result_sha256"] != digest):
        raise ValueError("source is not fully audited")
    rows = [row for row in result["rows"] if row["arm"] == "waypoint_cem"]
    if len(rows) != 8 or [row["root"] for row in rows] != list(range(8)):
        raise ValueError("incomplete source")
    return rows


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = source_rows()
    dependencies = seal.capture([Path(__file__), PROTOCOL, Path(source.__file__), Path(prior.__file__),
        Path(prior.local.__file__), Path(prior.core.__file__), Path(prior.exact.__file__),
        Path(prior.exact.wall.__file__), SOURCE / "result.json", SOURCE / "audit.json",
        SOURCE / "independent_geometry_replay.json"])
    OUT.mkdir()
    source.prior.previous.write(OUT / "manifest.json", {"source_sha256": sha(SOURCE / "result.json"),
        "dependencies": dependencies, "roots": 8,
        "scope": "post-hoc hand-designed waypoint baseline, not v11 or PC2 confirmation"})
    bindings = []
    for row in rows:
        data, arrays = calculate(row)
        index = row["root"]
        apath, mpath = OUT / f"root{index:02d}.npz", OUT / f"root{index:02d}.json"
        with apath.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        data["arrays_sha256"] = sha(apath)
        source.prior.previous.write(mpath, data)
        bindings.append({"root": index, "arrays_sha256": sha(apath), "metadata_sha256": sha(mpath)})
        print(f"root={index} base={data['base_intersection']} extended={data['all_intersection']}", flush=True)
    seal.verify(dependencies)
    source.prior.previous.write(OUT / "result.json", {"manifest_sha256": sha(OUT / "manifest.json"), "rows": bindings})


def audit():
    destination = OUT / "audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    rows = source_rows()
    manifest = json.loads((OUT / "manifest.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    seal.verify(manifest["dependencies"])
    if manifest["source_sha256"] != sha(SOURCE / "result.json") or result["manifest_sha256"] != sha(OUT / "manifest.json"):
        raise ValueError("manifest binding")
    chain_count = 0
    for row, binding in zip(rows, result["rows"], strict=True):
        index = row["root"]
        apath, mpath = OUT / f"root{index:02d}.npz", OUT / f"root{index:02d}.json"
        if binding != {"root": index, "arrays_sha256": sha(apath), "metadata_sha256": sha(mpath)}:
            raise ValueError("hash binding")
        data, arrays = calculate(row)
        data["arrays_sha256"] = sha(apath)
        if data != json.loads(mpath.read_text()):
            raise ValueError("metadata rebuild mismatch")
        np.testing.assert_array_equal(np.asarray(data["selected_chain"], np.float32),
                                      np.asarray(row["selected"], np.float32))
        with np.load(apath, allow_pickle=False) as saved:
            for key, value in arrays.items():
                np.testing.assert_array_equal(saved[key], value)
            own = saved["own_actions"].reshape(-1, 15, 2)
            expected = saved["own_positions"].reshape(-1, 2)
            for offset in range(0, len(own), 1024):
                path, _ = independent(row["task"]["start"], own[offset:offset + 1024])
                np.testing.assert_allclose(path[:, -1, :2], expected[offset:offset + 1024], atol=1e-12, rtol=0)
            common = saved["common_actions"].reshape(-1, 15, 2)
            paths, _ = independent(row["task"]["start"], common)
            scores = np.linalg.norm(paths[:, -1, :2] - np.asarray(row["task"]["goal"]), axis=1)
            expected_scores = np.asarray([data["controls"][name]["common_distances"] for name in prior.NAMES]).reshape(-1)
            np.testing.assert_allclose(scores, expected_scores, atol=1e-12, rtol=0)
            chain_count += len(own) + len(common)
        print(f"root={index} audited", flush=True)
    seal.verify(manifest["dependencies"])
    source.prior.previous.write(destination, {"status": "passed", "result_sha256": sha(OUT / "result.json"),
        "roots": 8, "independent_geometry_chains": chain_count,
        "scope": "deterministic same-code reconstruction plus separate geometry; not independent environment"})


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
