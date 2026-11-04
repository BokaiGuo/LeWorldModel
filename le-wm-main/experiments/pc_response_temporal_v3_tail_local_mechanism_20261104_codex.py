"""Prospective tail-local, branch-stable same-chain v3 mechanism development."""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_prospective_mechanism_20261104_codex as prior

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_RESPONSE_TEMPORAL_V3_TAIL_LOCAL_MECHANISM_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_response_temporal_v3_tail_local_mechanism_20261104_codex"
MANIFEST = PREFIX.with_name(PREFIX.name + "_manifest.json")
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")
REBUILD = PREFIX.with_name(PREFIX.name + "_intervention_rebuild.json")
original_dependencies = prior.dependencies
original_tasks = prior.tasks


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.35, -.05, .25, .50), (-1., -.05, 1.), (-.25, .25))]


def dependencies():
    result = original_dependencies()
    result[str(Path(__file__).resolve())] = prior.base.sha(Path(__file__))
    return result


def configure():
    prior.PROTOCOL, prior.MANIFEST, prior.PLANS, prior.AUDIT = PROTOCOL, MANIFEST, PLANS, AUDIT
    prior.INTERVENTION, prior.REBUILD, prior.tasks, prior.dependencies = INTERVENTION, REBUILD, tasks, dependencies


def plan():
    if any(task in original_tasks() or task in prior.base.tasks.tasks() for task in tasks()):
        raise ValueError("new grid overlaps either cited grid")
    prior.plan()


def calculate():
    rows = prior.calculate()
    axes = np.eye(10).reshape(10, 5, 2)
    wall = prior.base.exact.wall
    for row in rows:
        chain = np.asarray(row["selected_chain_float32"])
        tail = chain[5:].copy()
        nominal = tail.copy()
        nominal[:5] = np.clip(nominal[:5], -.9, .9)
        row["nominal_tail"] = nominal.tolist()
        row["tail_clip_linf"] = float(np.max(np.abs(nominal - tail)))
        for name, control in row["controls"].items():
            prefix = np.asarray(control["prefix"])
            output = {}
            for epsilon in (.01, .05):
                tails = np.broadcast_to(nominal, (21, 10, 2)).copy()
                tails[1:11, :5] += epsilon * axes
                tails[11:21, :5] -= epsilon * axes
                joined = np.concatenate((np.broadcast_to(prefix, (21, 5, 2)), tails), axis=1).astype(np.float32)
                if np.max(np.abs(joined)) > 1.00000001:
                    raise ValueError("actions outside domain")
                padded = np.pad(joined, ((0, 0), (0, 1), (0, 0))) * wall.HIGH
                trajectory, blocked = wall.rollout(row["task"]["start"], padded)
                endpoints, blocked = trajectory[:, 15, :2], np.asarray(blocked)[:, :15]
                exact = prior.base.exact.endpoints(row["task"], joined)
                np.testing.assert_allclose(endpoints, exact, rtol=0, atol=1e-12)
                matrix = ((endpoints[1:11] - endpoints[11:21]) / (2 * epsilon)).T
                output[str(epsilon)] = {"center_and_probe_endpoints": endpoints.tolist(),
                    "blocked": blocked.tolist(), "branch_stable": bool(np.all(blocked == blocked[0])),
                    "response": matrix.tolist()}
            control["tail_local_response"] = output
            control["diagnostic_rollouts"] += 42
        names = ("zero", "greedy", "random_bridge", "lookahead")
        selected = row["controls"]["selected"]
        row["tail_response_gaps"] = {
            name: {str(eps): float(np.linalg.norm(
                np.asarray(selected["tail_local_response"][str(eps)]["response"])
                - np.asarray(row["controls"][name]["tail_local_response"][str(eps)]["response"])))
                for eps in (.01, .05)} for name in names}

        def passes(others):
            involved = ("selected", *others)
            return bool(row["nonprogress"]
                        and all(row["active_better"][name] and row["own_better"][name] for name in others)
                        and all(row["tail_response_gaps"][name][str(eps)] > 1e-6
                                for name in others for eps in (.01, .05))
                        and all(row["controls"][name]["tail_local_response"][str(eps)]["branch_stable"]
                                for name in involved for eps in (.01, .05)))
        row["tail_base_intersection"] = passes(("zero", "greedy"))
        row["tail_all_intersection"] = passes(names)
    return rows


def intervene(rebuild=False):
    output = REBUILD if rebuild else INTERVENTION
    if output.exists() or (not rebuild and REBUILD.exists()):
        raise FileExistsError(output)
    hashes = dependencies()
    print("Tail-local same-chain interventions; source plans sealed before inspection.", flush=True)
    rows = calculate()
    if dependencies() != hashes:
        raise ValueError("dependency drift")
    if rebuild:
        old = json.loads(INTERVENTION.read_text())
        if old["dependencies"] != hashes or old["rows"] != rows:
            raise ValueError("deterministic reconstruction mismatch")
        prior.base.write(output, {"intervention_sha256": prior.base.sha(INTERVENTION),
                                  "verified_roots": len(rows), "scope": "same-code deterministic reconstruction"})
    else:
        prior.base.write(output, {"dependencies": hashes, "plans_sha256": prior.base.sha(PLANS), "rows": rows})
    print(f"{'Rebuilt' if rebuild else 'Sealed'} {len(rows)} tail-local interventions.", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": prior.audit, "intervene": intervene, "rebuild": lambda: intervene(True)}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan, audit, intervene or rebuild")
    configure()
    commands[sys.argv[1]]()
