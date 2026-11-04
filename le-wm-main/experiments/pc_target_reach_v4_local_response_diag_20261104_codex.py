"""Task-related tail-local responses for v4's sealed selected prefix."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_target_reach_v4_same_bank_mechanism_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PREFIX = source.PREFIX.with_name(source.PREFIX.name + "_local_response_v2")
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
DATA = PREFIX.with_name(PREFIX.name + "_data.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
BANK = source.PREFIX.with_name(source.PREFIX.name + "_plans.json")
BANK_AUDIT = source.PREFIX.with_name(source.PREFIX.name + "_audit.json")
NAMES = ("selected", "zero", "greedy", "lookahead")


def build(row):
    wall = source.harness.src.prior.base.exact.wall
    exact = source.harness.src.prior.base.exact
    candidates = row["candidates"]
    zero_index = next(i for i, item in enumerate(candidates[:-1]) if item["name"] == "zero")
    indices = {"selected": len(candidates) - 1, "zero": zero_index,
               "greedy": row["greedy_index"], "lookahead": row["lookahead_index"]}
    goal = np.asarray(row["task"]["goal"])
    start = np.asarray(row["task"]["start"])
    tail = np.asarray(row["selected_tail"])
    nominal = tail.copy()
    nominal[:5] = np.clip(nominal[:5], -.9, .9)
    axes = np.eye(10).reshape(10, 5, 2)
    controls = {}
    for name, index in indices.items():
        prefix = np.asarray(candidates[index]["prefix"])
        initial_position = exact.endpoints(row["task"], prefix[None].astype(np.float32))[0]
        prefix_distance = float(np.linalg.norm(initial_position - goal))
        scales = {}
        for epsilon in (.01, .05):
            tails = np.broadcast_to(nominal, (21, 10, 2)).copy()
            tails[1:11, :5] += epsilon * axes
            tails[11:21, :5] -= epsilon * axes
            joined = np.concatenate((np.broadcast_to(prefix, (21, 5, 2)), tails), axis=1).astype(np.float32)
            if np.max(np.abs(joined)) > 1 + 1e-6:
                raise ValueError("out of range probe")
            endpoint = exact.endpoints(row["task"], joined)
            trajectory, blocked = wall.rollout(row["task"]["start"],
                np.pad(joined, ((0, 0), (0, 1), (0, 0))) * wall.HIGH)
            np.testing.assert_allclose(trajectory[:, 15, :2], endpoint, rtol=0, atol=1e-12)
            blocked = np.asarray(blocked)[:, :15]
            matrix = ((endpoint[1:11] - endpoint[11:21]) / (2 * epsilon)).T
            distance = float(np.linalg.norm(endpoint[0] - goal))
            gradient = None if distance <= 1e-10 else (matrix.T @ ((endpoint[0] - goal) / distance)).tolist()
            scales[str(epsilon)] = {"endpoints": endpoint.tolist(), "blocked": blocked.tolist(),
                                    "branch_stable": bool(np.all(blocked == blocked[0])),
                                    "matrix": matrix.tolist(), "distance_gradient": gradient}
        controls[name] = {"candidate_index": index, "prefix_distance": prefix_distance, "scales": scales}
    gaps = {}
    for name in NAMES[1:]:
        gaps[name] = {}
        for eps in (".01", ".05"):
            key = str(float(eps))
            left, right = controls["selected"]["scales"][key], controls[name]["scales"][key]
            lg, rg = left["distance_gradient"], right["distance_gradient"]
            gaps[name][key] = {"matrix": float(np.linalg.norm(np.asarray(left["matrix"]) - right["matrix"])),
                               "goal_gradient": None if lg is None or rg is None else float(
                                   np.linalg.norm(np.asarray(lg) - rg))}
    return {"root": row["root"], "controls": controls, "gaps": gaps,
            "selected_prefix_nonzero": bool(np.any(np.asarray(candidates[-1]["prefix"]) != 0)),
            "selected_tail_nonzero": bool(np.any(tail != 0)),
            "nonprogress": bool(controls["selected"]["prefix_distance"] >= np.linalg.norm(start[:2] - goal) - 1e-10),
            "tail_clip_linf": float(np.max(np.abs(tail - nominal)))}


def plan():
    if any(p.exists() for p in (MANIFEST, DATA, AUDIT)):
        raise FileExistsError("existing or partial response diagnostic")
    sha = source.harness.src.prior.base.sha
    bank = json.loads(BANK.read_text())
    if json.loads(BANK_AUDIT.read_text())["plans_sha256"] != sha(BANK):
        raise ValueError("bank audit mismatch")
    dependencies = seal.capture([Path(__file__), source.PROTOCOL, BANK, BANK_AUDIT,
                                 Path(source.harness.src.prior.base.exact.__file__)])
    source.harness.src.prior.base.write(MANIFEST, {"seal": dependencies, "bank_sha256": sha(BANK)})
    rows = [build(row) for row in bank["rows"]]
    if len(rows) != 24:
        raise ValueError("incomplete panel")
    seal.verify(dependencies)
    source.harness.src.prior.base.write(DATA, {"manifest_sha256": sha(MANIFEST), "rows": rows})
    print("Sealed all v4 local response interventions.", flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    sha = source.harness.src.prior.base.sha
    manifest, data, bank = (json.loads(p.read_text()) for p in (MANIFEST, DATA, BANK))
    seal.verify(manifest["seal"])
    if manifest["bank_sha256"] != sha(BANK) or data["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("manifest mismatch")
    rebuilt = [build(row) for row in bank["rows"]]
    if data["rows"] != rebuilt:
        raise ValueError("response replay mismatch")
    source.harness.src.prior.base.write(AUDIT, {"data_sha256": sha(DATA), "verified_roots": len(rebuilt),
                                               "chains": len(rebuilt) * 4 * 2 * 21,
                                               "scope": "same-code same-backend reconstruction"})
    print("Reconstructed all v4 local response interventions.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    {"plan": plan, "audit": audit}[sys.argv[1]]()
