"""Common multistart program suffix search for Atlas-style wall prefixes."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .d2p_local_compiler import CompilerConfig, LocalCompiler
from .pc_wall_unicycle_atlas_cem_20261103 import Backend, GOAL, HIGH, LOW, ROOT, replay

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = REPO / "docs/PC_WALL_SHARED_MULTISTART_PROGRAM_PROTOCOL_20261103_CODEX.md"
SEEDS = (20261103, 20261104, 20261105)
NAMES = ("zero", "progress", "spectral_minus", "spectral_plus", "random_0", "random_1")


def program(params, horizon):
    p = np.asarray(params, dtype=float)
    if p.shape != (4,) or not np.isfinite(p).all() or np.any(np.abs(p) > 1):
        raise ValueError("invalid program parameters")
    actions = []
    for turn, amount in ((p[0] * np.pi, (p[1] + 1) * .75),
                         (p[2] * np.pi, (p[3] + 1) * .75)):
        turns = int(np.ceil(abs(turn) / HIGH[1]))
        moves = int(np.ceil(amount / HIGH[0]))
        if turns:
            actions.extend([[0., turn / turns]] * turns)
        if moves:
            actions.extend([[amount / moves, 0.]] * moves)
    actions = actions[:horizon]
    actions.extend([[0., 0.]] * (horizon - len(actions)))
    result = np.asarray(actions, dtype=float)
    if np.any(result < LOW - 1e-12) or np.any(result > HIGH + 1e-12):
        raise ValueError("program outside bounds")
    return result


def search(backend, context, horizon, seed, kind, population):
    if kind not in ("program", "flat") or horizon not in (12, 16):
        raise ValueError("invalid optimizer contract")
    shape = (4,) if kind == "program" else (horizon, 2)
    bounds = np.ones(shape) if kind == "program" else np.broadcast_to(HIGH, shape)
    rng = np.random.default_rng(seed)
    best_score, best, trace = float("inf"), None, []
    for start in range(2):
        mean, std = np.zeros(shape), bounds * .5
        for iteration in range(3):
            samples = np.clip(mean + std * rng.standard_normal((population, *shape)), -bounds, bounds)
            costs = []
            for sample in samples:
                actions = program(sample, horizon) if kind == "program" else sample
                successor = backend.rollout(context, actions)
                costs.append(float(np.linalg.norm(np.asarray(successor["latent_history"][-1])[:2] - GOAL[:2])))
            costs = np.asarray(costs)
            if not np.isfinite(costs).all():
                raise ValueError("invalid terminal cost")
            order = np.argsort(costs, kind="stable")
            index = int(order[0])
            if costs[index] < best_score:
                best_score = float(costs[index])
                best = (program(samples[index], horizon) if kind == "program" else samples[index]).copy()
            elites = samples[order[:4]]
            mean, std = elites.mean(axis=0), np.maximum(elites.std(axis=0), 1e-5)
            trace.append({"start": start, "round": iteration, "samples": samples.tolist(),
                          "scores": costs.tolist(), "elite_indices": order[:4].tolist(),
                          "best_so_far": best_score})
    return best, best_score, trace


def make_bank(seed):
    backend = Backend()
    compiler = LocalCompiler(
        backend, goal=GOAL, sqrt_q=np.diag([1., 1., 0.]),
        action_scale=[.3, np.pi / 2], low=LOW, high=HIGH,
        config=CompilerConfig(packet_steps=4, spectral_modes=2, prefix_radius=.3,
                              max_rollout_operations=10000, max_jacobian_operations=2000),
    )
    progress = compiler.progress(ROOT, packets=1)
    directions = progress["geometry"]["raw_spectral_directions"]
    if not directions:
        raise ValueError("no task spectral direction")
    direction = np.asarray(directions[0]).reshape(4, 2)
    size = np.max(np.abs(direction) / HIGH)
    if not np.isfinite(size) or size <= 1e-12:
        raise ValueError("degenerate spectral direction")
    direction = .3 * direction / size
    rng = np.random.default_rng(seed)
    bank = [np.zeros((4, 2)), np.asarray(progress["actions"]), -direction, direction,
            rng.uniform(LOW, HIGH, (4, 2)), rng.uniform(LOW, HIGH, (4, 2))]
    contexts = [backend.rollout(ROOT, prefix) for prefix in bank]
    costs = [compiler._cost(context) for context in contexts]
    root_cost = compiler._cost(ROOT)
    return bank, contexts, costs, root_cost, vars(backend).copy(), compiler.counts.copy()


def select(prefix_costs, terminal_distances, root_cost):
    if len(prefix_costs) != 6 or len(terminal_distances) != 6:
        raise ValueError("invalid candidate count")
    values = np.asarray(prefix_costs + terminal_distances + [root_cost], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite selection cost")
    greedy = min(range(6), key=lambda i: (prefix_costs[i], i))
    selected = min(range(6), key=lambda i: (terminal_distances[i], i))
    reference = min(terminal_distances[0], terminal_distances[greedy])
    qualified = [i for i in range(2, 6) if prefix_costs[i] >= root_cost
                 and terminal_distances[i] < reference]
    eligible = min(qualified, key=lambda i: (terminal_distances[i], i)) if qualified else None
    return {"selected_index": selected, "same_bank_greedy_index": greedy,
            "qualified_reconfiguration_index": eligible,
            "selected_is_qualified": eligible is not None and selected == eligible,
            "abstained_reconfiguration": eligible is None}


def run(seed):
    start = perf_counter()
    bank, contexts, costs, root_cost, generation, compiler_counts = make_bank(seed)
    choices = []
    for name, prefix, context, prefix_cost in zip(NAMES, bank, contexts, costs):
        backend = Backend()
        tail, score, trace = search(backend, context, 12, seed + 100000, "program", 34)
        checked = replay(np.concatenate((prefix, tail)))
        if not np.isclose(checked["terminal_error"], score, atol=1e-12, rtol=0):
            raise ValueError("prefix/tail replay mismatch")
        if backend.transitions != 2448 or backend.rollouts != 204:
            raise ValueError("suffix budget mismatch")
        choices.append({"name": name, "prefix_cost": prefix_cost, "suffix": tail.tolist(),
                        "replay": checked, "backend": vars(backend).copy(), "trace": trace})
    decision = select(costs, [item["replay"]["terminal_error"] for item in choices], root_cost)
    controls = {}
    for kind in ("flat", "program"):
        backend = Backend()
        actions, cost, trace = search(backend, ROOT, 16, seed + 200000, kind, 153)
        checked = replay(actions)
        if (not np.isclose(checked["terminal_error"], cost, atol=1e-12, rtol=0)
                or backend.transitions != 14688 or backend.rollouts != 918):
            raise ValueError("direct control budget or replay mismatch")
        controls[kind] = {"replay": checked, "backend": vars(backend).copy(), "trace": trace}
    return {"seed": seed, "bank": [x.tolist() for x in bank], "root_cost": root_cost,
            "generation_backend": generation, "compiler_counts": compiler_counts,
            "choices": choices, "decision": decision, "direct_controls": controls,
            "wall_seconds": perf_counter() - start}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = [Path(__file__), PROTOCOL, Path(__file__).with_name("d2p_local_compiler.py"),
               Path(__file__).with_name("d2p_composition_witness_v2.py"),
               Path(__file__).with_name("pc_wall_unicycle_atlas_cem_20261103.py"),
               Path(__file__).with_name("pc_exact_wall_kill.py")]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p.relative_to(REPO)): digest(p) for p in sources}
    print("Exposed wall: shared multistart suffix versus direct planning.", flush=True)
    rows = []
    for seed in SEEDS:
        row = run(seed)
        rows.append(row)
        choice = row["choices"][row["decision"]["selected_index"]]
        print(f"seed={seed}, selected={choice['name']} "
              f"error={choice['replay']['terminal_error']:.6f}, "
              f"direct_flat={row['direct_controls']['flat']['replay']['terminal_error']:.6f}, "
              f"direct_program={row['direct_controls']['program']['replay']['terminal_error']:.6f}",
              flush=True)
    if hashes != {str(p.relative_to(REPO)): digest(p) for p in sources}:
        raise ValueError("source changed during run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump({"schema": "pc_wall_shared_multistart_program.v1",
                   "scope": "one exposed exact task, not RGB, independent, physical or admission",
                   "source_sha256": hashes, "numpy_version": np.__version__, "rows": rows},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
