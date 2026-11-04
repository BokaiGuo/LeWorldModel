"""V5 selected-chain common-tail and task-local response interventions."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_target_reach_v4_local_response_diag_20261104_codex as local
from experiments import pc_v5_canonical_response_same_bank_20261104_codex as bank
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

PREFIX = bank.PREFIX.with_name(bank.PREFIX.name + "_local")
MANIFEST, DATA, AUDIT = [PREFIX.with_name(PREFIX.name + part + ".json")
                         for part in ("_manifest", "_data", "_audit")]
BANK = bank.PREFIX.with_name(bank.PREFIX.name + "_plans.json")
BANK_AUDIT = bank.PREFIX.with_name(bank.PREFIX.name + "_audit.json")
sha, write = bank.harness.src.prior.base.sha, bank.harness.src.prior.base.write
exact = bank.harness.src.prior.base.exact


def calculate(row):
    response = local.build(row)
    candidates, goal = row["candidates"], np.asarray(row["task"]["goal"])
    selected_tail = np.asarray(row["selected_tail"], dtype=np.float32)
    indices = {"selected": len(candidates)-1,
               "zero": next(i for i, x in enumerate(candidates[:-1]) if x["name"] == "zero"),
               "greedy": row["greedy_index"], "lookahead": row["lookahead_index"]}
    tails = [np.tile(value, (10, 1)) for value in
             ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
    tails.append(selected_tail)
    fixed = {}
    for name, index in indices.items():
        item = candidates[index]
        prefix = np.asarray(item["prefix"])
        chains = np.asarray([np.concatenate((prefix, tail)) for tail in tails], np.float32)
        endpoints = exact.endpoints(row["task"], chains)
        values = np.linalg.norm(endpoints-goal, axis=1)
        fixed[name] = {"distance": values.tolist(), "relative_to_zero": (values-values[0]).tolist(),
                       "own_optimized_distance": item["optimized_distance"]}
    others = ("zero", "greedy")
    other_look = ("zero", "greedy", "lookahead")

    def intersection(names):
        controls = response["controls"]
        return bool(response["nonprogress"] and response["selected_prefix_nonzero"]
            and response["selected_tail_nonzero"]
            and all(fixed["selected"]["relative_to_zero"][-1] < fixed[n]["relative_to_zero"][-1] - 1e-10
                    and fixed["selected"]["own_optimized_distance"] < fixed[n]["own_optimized_distance"] - 1e-10
                    for n in names)
            and all(response["gaps"][n][eps]["matrix"] > 1e-6 for n in names for eps in ("0.01", "0.05"))
            and all(controls[n]["scales"][eps]["branch_stable"]
                    for n in ("selected", *names) for eps in ("0.01", "0.05")))
    return {"root": row["root"], "response": response, "fixed_tails": fixed,
            "base_intersection": intersection(others), "lookahead_intersection": intersection(other_look)}


def main(mode):
    if mode == "plan":
        if any(p.exists() for p in (MANIFEST, DATA, AUDIT)):
            raise FileExistsError("existing or partial result")
        if json.loads(BANK_AUDIT.read_text())["plans_sha256"] != sha(BANK):
            raise ValueError("same-bank audit mismatch")
        deps = seal.capture([Path(__file__), bank.PROTOCOL, BANK, BANK_AUDIT,
            Path(local.__file__), Path(exact.__file__), Path(exact.wall.__file__)])
        write(MANIFEST, {"seal": deps, "scope": "post-hoc exposed v5 selected-chain mechanism"})
        rows = [calculate(row) for row in json.loads(BANK.read_text())["rows"]]
        if len(rows) != 24:
            raise ValueError("incomplete task panel")
        seal.verify(deps)
        write(DATA, {"manifest_sha256": sha(MANIFEST), "rows": rows})
        print("Saved 24 common-tail and local-response interventions.", flush=True)
    else:
        if AUDIT.exists():
            raise FileExistsError(AUDIT)
        manifest, data = json.loads(MANIFEST.read_text()), json.loads(DATA.read_text())
        seal.verify(manifest["seal"])
        if data["manifest_sha256"] != sha(MANIFEST):
            raise ValueError("manifest mismatch")
        rebuilt = [calculate(row) for row in json.loads(BANK.read_text())["rows"]]
        if rebuilt != data["rows"] or len(rebuilt) != 24:
            raise ValueError("intervention reconstruction mismatch")
        write(AUDIT, {"data_sha256": sha(DATA), "verified_roots": len(rebuilt),
            "local_response_probe_chains": 24*4*2*21, "common_tail_chains": 24*4*6,
            "scope": "same-code exact dynamics reconstruction, not independent backend"})
        print("Rebuilt all local-response and common-tail interventions.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    main(sys.argv[1])
