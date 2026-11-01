"""Audit saved P3 goal/velocity semantics; no simulator or neural execution."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct

NAME = "main_p3_velocity_semantics_audit_20261029_v1"
ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/agent_p3_capability_probe_result_summary_20261029_v1_MAIN01.json"
SUMMARY_SHA = "e59eeadc6e6de711e60e50709162c4310ce0e69e9b3fdb8a5db838fd74797b19"
ENV = ROOT / ".venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py"
ENV_SHA = "c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2"
ORDER = ((0, "C"), (0, "Z"), (0, "P"), (1, "Z"), (1, "P"))
CAP = 32 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_contract(source):
    """Check the pinned implementation, not a generic simulator assumption."""
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PushT")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    obs = ast.get_source_segment(source, methods["_get_obs"])
    evaluate = ast.get_source_segment(source, methods["eval_state"])
    require("tuple(self.agent.velocity)" in obs and "tuple(self.block.position)" in obs,
            "state7 uses agent velocity and block position")
    require("np.linalg.norm(goal_state[:4] - cur_state[:4])" in evaluate
            and "np.linalg.norm(goal_state - cur_state)" in evaluate,
            "source success XY and reward seven-state semantics")
    render = methods["_render_frame"]
    require(not any(isinstance(n, ast.Attribute) and n.attr == "velocity" for n in ast.walk(render)),
            "render method must not directly consume a velocity field")
    return dict(state7_last_two="agent_velocity", success_uses_velocity=False,
                source_reward_uses_all_seven=True, renderer_directly_reads_velocity=False,
                qualification="a single rendered image is not direct velocity input; history may contain motion information")


def audit(data):
    require(data["schema"] == "agent_p3_capability_probe_result_summary_20261029_v1.summary"
            and data["known_roots"] == 2 and data["fresh_roots"] == 0,
            "fixed actual authority and independent-root boundary")
    rows = data["original_rows"]
    branches = data["arithmetic"]["branches"]
    require(len(rows) == len(branches) == 10, "retain all ten original branches")
    result = []
    for i, (row, branch) in enumerate(zip(rows, branches, strict=True)):
        root, role = ORDER[i // 2]
        probe = "U" if i % 2 == 0 else "Z2"
        require((branch["index"], branch["root_slot"], branch["role"], branch["probe"])
                == (i, root, role, probe), "fixed root/role/probe order")
        endpoint = row["endpoint15"]
        require(endpoint is not None and row["status"] == "horizon_completed", "actual endpoint only")
        snapshot = endpoint["snapshot"]
        state = endpoint["state"]
        goal = branch["metrics"]["goal7"]
        require(len(state) == len(goal) == 7 and state == snapshot["state"]
                and all(type(v) in (int, float) and math.isfinite(v) for v in state + goal), "finite state/goal7")
        require(struct.pack("<7d", *state).hex() == snapshot["state_float64_bits"], "literal native state bits")
        agent, block = snapshot["bodies"]["agent"], snapshot["bodies"]["block"]
        require(state[:2] == agent["position"] and state[2:4] == block["position"]
                and state[5:] == agent["velocity"] and state[4] == block["unwrapped_angle"] % (2 * math.pi),
                "full state7 matches actual native bodies")
        last = row["observations"][-1]
        require(last["receipt"]["snapshot"] == snapshot and last["post"]["snapshot"] == snapshot
                and last["post"]["goal_state"] == goal and row["responses"][-1]["state"] == state,
                "same final time, actual goal and physical snapshot")
        metrics = branch["metrics"]
        velocity_error = math.hypot(state[5] - goal[5], state[6] - goal[6])
        require(math.isclose(velocity_error, metrics["velocity_error"], rel_tol=1e-15, abs_tol=1e-12),
                "original seven-state velocity term")
        # Only a descriptive metric sensitivity: original goal, success and gate remain unchanged.
        goal_zero_distance = math.sqrt(math.fsum([(state[j] - goal[j]) ** 2 for j in range(5)]
                                                + [state[5] ** 2, state[6] ** 2]))
        result.append(dict(index=i, root=root, role=role, probe=probe,
            goal_velocity=goal[5:], goal_velocity_norm=math.hypot(*goal[5:]),
            actual_agent_velocity=state[5:], actual_agent_speed=math.hypot(*state[5:]),
            original_velocity_error=velocity_error, original_native7_distance=branch["D_original_reward"],
            zero_goal_velocity_counterfactual_distance=goal_zero_distance,
            original_native_success=branch["original_native_success"], goal_rgb_sha256=last["post"]["goal_rgb_sha256"]))
    pairs = []
    for i, (root, role) in enumerate(ORDER):
        u, z = result[2*i:2*i+2]
        require(u["goal_velocity"] == z["goal_velocity"] and u["goal_rgb_sha256"] == z["goal_rgb_sha256"],
                "both probes share original goal state velocity and RGB")
        original = z["original_native7_distance"] - u["original_native7_distance"]
        zero = z["zero_goal_velocity_counterfactual_distance"] - u["zero_goal_velocity_counterfactual_distance"]
        pairs.append(dict(root=root, role=role, original_G=original, zero_goal_velocity_sensitivity_G=zero,
                          zero_goal_velocity_is_new_gate=False))
    return dict(branches=result, pairs=pairs, total=10, known_roots=2, fresh_roots=0,
                all_endpoint_snapshot_state_matches=True,
                interpretation="fixed observed trajectories; zero-velocity calculation is post-hoc sensitivity, not a new execution or success test")


def build():
    require(SUMMARY.is_file() and SUMMARY.stat().st_size <= CAP and sha(SUMMARY) == SUMMARY_SHA,
            "frozen complete MAIN physical summary")
    require(ENV.is_file() and sha(ENV) == ENV_SHA, "actual frozen native source")
    source = ENV.read_text()
    data = json.loads(SUMMARY.read_bytes())
    result = dict(schema=NAME + ".summary", source_contract=source_contract(source), audit=audit(data),
                  inputs={str(SUMMARY): SUMMARY_SHA, str(ENV): ENV_SHA},
                  own_source_sha256=sha(__file__),
                  new_neural_calls=0, new_physics_calls=0, new_fullstrict_calls=0,
                  scientific_admission=False, P1_admission=False)
    require(sha(SUMMARY) == SUMMARY_SHA and sha(ENV) == ENV_SHA, "inputs unchanged during audit")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--output", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--summary", required=True)
    p.add_argument("--sha", required=True)
    args = parser.parse_args()
    if args.command == "build":
        out = Path(args.output)
        require(out.is_absolute() and out.resolve() == out and out.name.startswith(NAME + "_") and not out.exists(),
                "new canonical output")
        data = encoded(build())
        require(len(data) <= CAP, "bounded output")
        with out.open("xb") as f:
            require(f.write(data) == len(data), "complete output")
            f.flush()
            os.fsync(f.fileno())
        fd = os.open(out.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        print(encoded(dict(path=str(out), sha256=sha(out))).decode(), end="")
    else:
        p = Path(args.summary)
        require(p.is_absolute() and p.resolve() == p and p.is_file() and p.stat().st_size <= CAP
                and sha(p) == args.sha and encoded(json.loads(p.read_bytes())) == encoded(build()),
                "whole deterministic audit verification")
        print(encoded(dict(verified=True, sha256=args.sha)).decode(), end="")


if __name__ == "__main__":
    main()
