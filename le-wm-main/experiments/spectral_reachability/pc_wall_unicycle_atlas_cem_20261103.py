"""Exact wall-unicycle development comparison; not a neural or gate result."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .d2p_atlas_search_v2 import AtlasConfig, AtlasSearch
from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_exact_wall_kill import step_batch

START = np.array([-.5, 0., 0.])
GOAL = np.array([.5, 0., 0.])
LOW = np.array([-.3, -np.pi / 2])
HIGH = -LOW
HORIZON = 16
EPS = 1e-5
ROOT = {"layout": "states_with_between_state_actions_v1", "context_length": 2,
        "latent_dim": 3, "action_dim": 2, "latent_history": [START.tolist()] * 2,
        "action_history": [[0., 0.]]}


class Backend:
    def __init__(self):
        self.transitions = 0
        self.rollouts = 0
        self.fd_rollouts = 0
        self.jacobians = 0

    def advance(self, state, action):
        if np.any(action < LOW - 1e-12) or np.any(action > HIGH + 1e-12):
            raise ValueError("action out of domain")
        displacement = action[0] * np.array([np.cos(state[2]), np.sin(state[2])])
        pos, blocked = step_batch(state[:2], displacement)
        self.transitions += 1
        return np.array([pos[0], pos[1], state[2] + action[1]]), bool(blocked)

    def _roll(self, context, actions):
        states, history = list(context["latent_history"]), list(context["action_history"])
        state = np.asarray(states[-1], dtype=float)
        for action in np.asarray(actions):
            state, _ = self.advance(state, action)
            states = (states + [state.tolist()])[-2:]
            history = (history + [action.tolist()])[-1:]
        return {**context, "latent_history": states, "action_history": history}

    def rollout(self, context, actions):
        self.rollouts += 1
        return self._roll(context, actions)

    def jacobian(self, context, actions):
        self.jacobians += 1
        actions = np.asarray(actions, dtype=float)
        result = np.empty((3, actions.size))
        for i in range(actions.size):
            plus, minus = actions.copy(), actions.copy()
            plus.flat[i] += EPS
            minus.flat[i] -= EPS
            if np.any(plus > HIGH) or np.any(minus < LOW):
                raise ValueError("finite difference outside domain")
            a, b = self._roll(context, plus), self._roll(context, minus)
            self.fd_rollouts += 2
            result[:, i] = (np.asarray(a["latent_history"][-1]) - b["latent_history"][-1]) / (2 * EPS)
        return result


def replay(actions):
    if np.asarray(actions).shape != (HORIZON, 2):
        raise ValueError("wrong program horizon")
    backend, state, trajectory, rejected = Backend(), START.copy(), [START.tolist()], []
    for action in np.asarray(actions):
        state, blocked = backend.advance(state, action)
        trajectory.append(state.tolist())
        rejected.append(blocked)
    return {"actions": np.asarray(actions).tolist(), "trajectory": trajectory, "blocked": rejected,
            "terminal_error": float(np.linalg.norm(state[:2] - GOAL[:2])),
            "max_endpoint_distance": float(np.max(np.linalg.norm(np.asarray(trajectory)[:, :2] - GOAL[:2], axis=1)))}


def witness():
    target = np.array([0., 1.1])
    angle1 = np.arctan2(target[1] - START[1], target[0] - START[0])
    angle2 = np.arctan2(GOAL[1] - target[1], GOAL[0] - target[0])
    a = [[0., angle1]] + [[np.linalg.norm(target - START[:2]) / 5, 0.]] * 5
    a += [[0., (angle2 - angle1) / 2]] * 2
    a += [[np.linalg.norm(GOAL[:2] - target) / 5, 0.]] * 5
    return np.asarray(a + [[0., 0.]] * (HORIZON - len(a)))


def run_atlas(seed):
    backend = Backend()
    compiler = LocalCompiler(backend, goal=GOAL, sqrt_q=np.diag([1., 1., 0.]),
                             action_scale=[.3, np.pi / 2], low=LOW, high=HIGH,
                             config=CompilerConfig(packet_steps=4, spectral_modes=2, prefix_radius=.3,
                                                   max_rollout_operations=10000, max_jacobian_operations=2000))
    start = perf_counter()
    out = AtlasSearch(compiler, AtlasConfig(depth=4, beam_width=2, random_bridges=2, seed=seed)).search(ROOT)
    elapsed = perf_counter() - start
    program = np.asarray(out["programs"][out["selected_index"]]["actions"])
    evaluation = replay(program)
    selected = out["programs"][out["selected_index"]]
    expected = np.asarray(selected["output_context"]["latent_history"][-1])
    if not np.allclose(evaluation["trajectory"][-1], expected, atol=1e-12, rtol=0):
        raise ValueError("atlas replay mismatch")
    return {"seed": seed, "selected": evaluation, "root_cost": out["root_cost"],
            "qualified_reconfiguration_index": out["qualified_reconfiguration_index"],
            "compiler_counts": out["operation_counts"], "backend": vars(backend).copy(),
            "wall_seconds": elapsed}


def run_cem(seed, atlas_transitions):
    population = atlas_transitions // (4 * HORIZON)
    if population < 8:
        raise ValueError("insufficient CEM population")
    elites = max(1, population // 8)
    rng = np.random.default_rng(seed)
    mean = np.zeros((HORIZON, 2))
    std = np.broadcast_to(HIGH / 2, (HORIZON, 2)).copy()
    backend, best, best_cost, trace = Backend(), None, float("inf"), []
    start = perf_counter()
    for _ in range(4):
        actions = np.clip(rng.normal(mean, std, (population, HORIZON, 2)), LOW, HIGH)
        scores = np.empty(population)
        for i, program in enumerate(actions):
            end = backend.rollout(ROOT, program)["latent_history"][-1]
            scores[i] = np.linalg.norm(np.asarray(end)[:2] - GOAL[:2])
        idx = int(np.argmin(scores))
        if scores[idx] < best_cost:
            best, best_cost = actions[idx].copy(), float(scores[idx])
        selected = actions[np.argsort(scores, kind="stable")[:elites]]
        mean, std = selected.mean(axis=0), np.maximum(selected.std(axis=0), 1e-5)
        trace.append({"best_in_round": float(scores[idx]), "best_so_far": best_cost})
    elapsed = perf_counter() - start
    evaluated = replay(best)
    if not np.isclose(evaluated["terminal_error"], best_cost, rtol=0, atol=1e-12):
        raise ValueError("CEM replay mismatch")
    return {"seed": seed, "population": population, "elites": elites, "trace": trace,
            "selected": evaluated, "backend": vars(backend).copy(), "wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    witness_result = replay(witness())
    if witness_result["terminal_error"] > 1e-12 or any(witness_result["blocked"]):
        raise ValueError("invalid feasibility witness")
    print("Exact wall-unicycle: bounded Atlas vs transition-capped CEM; exposed development task.", flush=True)
    rows = []
    for seed in (20261103, 20261104, 20261105):
        atlas = run_atlas(seed)
        cem = run_cem(seed, atlas["backend"]["transitions"])
        if cem["backend"]["transitions"] > atlas["backend"]["transitions"]:
            raise ValueError("transition budget")
        rows.append({"seed": seed, "atlas": atlas, "cem": cem})
        print(f"seed={seed}, Atlas={atlas['selected']['terminal_error']:.6f}, CEM={cem['selected']['terminal_error']:.6f}", flush=True)
    protocol = Path(__file__).resolve().parents[3] / "docs/PC_WALL_UNICYCLE_ATLAS_CEM_PROTOCOL_20261103.md"
    sources = [Path(__file__), Path(__file__).with_name("pc_exact_wall_kill.py"),
               Path(__file__).with_name("d2p_local_compiler.py"),
               Path(__file__).with_name("d2p_atlas_search_v2.py")]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"schema": "wall_unicycle_atlas_cem.v1",
              "scope": "single exact-dynamics development task; not neural physical independent or admission",
              "protocol_sha256": digest(protocol), "source_sha256": {p.name: digest(p) for p in sources},
              "witness": witness_result, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
