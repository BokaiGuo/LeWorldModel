"""Sealed automatic response-prefix programs and separate exact interventions."""
import json
from pathlib import Path

import numpy as np

from . import pc_exact_unicycle_development as dynamics
from .d2p_goal_mode_rank_20261102 import rank_goal_modes

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC2_AUTO_RESPONSE_UNICYCLE_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = ROOT / "results/pc2_auto_response_unicycle_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")
NOMINALS = ((-.4, -.5), (-.4, .5), (.4, -.5), (.4, .5))
GRID = np.linspace(-1., 1., 33)
FIXED = (-1., -.5, 0., .5, 1.)


def prefix_from_normalized(point):
    point = np.asarray(point, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or np.any(np.abs(point) > 1):
        raise ValueError("invalid normalized prefix")
    return np.array([point[0], point[1] * dynamics.TURN])


def endpoint(point, suffix):
    successor = dynamics.step(np.zeros(3), prefix_from_normalized(point))
    return dynamics.step(successor, [suffix, 0.])[:2]


def build_bank(goal, index, mode):
    if mode not in ("goal", "random"):
        raise ValueError("invalid generator")
    bank = [{"name": "zero", "prefix": [0., 0.], "source": "zero"}]
    records = []
    rng = np.random.default_rng(2026112700 + index)
    for i, values in enumerate(NOMINALS):
        nominal = np.asarray(values)
        if mode == "goal":
            eps = .01
            nominal_end = endpoint(nominal, .5)
            matrix = np.stack([(endpoint(nominal + eps * v, .5) -
                                endpoint(nominal - eps * v, .5)) / (2 * eps)
                               for v in np.eye(2)], axis=1)
            residual = goal - nominal_end
            ranked = rank_goal_modes(matrix, residual, .001)
            k = int(ranked["order"][0])
            delta = ranked["right_t"][k] * ranked["coefficients"][k]
            delta *= min(1., .5 / max(float(np.max(np.abs(delta))), 1e-30))
            point = np.clip(nominal + delta, -1, 1)
            records.append({"nominal": nominal.tolist(), "endpoint": nominal_end.tolist(),
                            "jacobian": matrix.tolist(), "residual": residual.tolist(),
                            "singular_values": ranked["singular"].tolist(),
                            "goal_contributions": ranked["contributions"].tolist(), "mode": k,
                            "delta": delta.tolist(), "candidate": point.tolist()})
        else:
            point = rng.uniform(-1., 1., 2)
        bank.append({"name": f"{mode}_{i}", "prefix": prefix_from_normalized(point).tolist(), "source": mode})
    return bank, records


def planned(goal, bank):
    programs = []
    for item in bank:
        prefix = np.asarray(item["prefix"])
        successor = dynamics.step(np.zeros(3), prefix)
        prefix_score = float(np.linalg.norm(successor[:2] - goal))
        scores = [dynamics.evaluate([prefix, [velocity, 0.]], goal)[0] for velocity in GRID]
        best = int(np.argmin(scores))
        probe_scores = [scores[int(round((velocity + 1) * 16))] for velocity in FIXED]
        programs.append({**item, "prefix_score": prefix_score, "suffix": [float(GRID[best]), 0.],
                         "terminal_score": float(scores[best]), "suffix_scores": scores,
                         "lookahead_score": float(min(probe_scores))})
    return programs


def select(programs, key):
    if key == "terminal_score":
        best = min(range(len(programs)), key=lambda i: (programs[i][key], i))
        return 0 if programs[0][key] <= programs[best][key] else best
    return min(range(len(programs)), key=lambda i: (programs[i][key], i))


def plan_stage():
    if PLANS.exists() or INTERVENTION.exists():
        raise FileExistsError("existing plans or interventions")
    paths = (PROTOCOL, Path(__file__), Path(dynamics.__file__), Path(rank_goal_modes.__code__.co_filename))
    hashes = {str(p): dynamics.hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows = []
    print("Automatic response prefixes with complete-chain selection; exact development.", flush=True)
    for index in range(16):
        theta = 2 * np.pi * (index + .37) / 16
        goal = .9 * np.array([np.cos(theta), np.sin(theta)])
        goal_bank, responses = build_bank(goal, index, "goal")
        random_bank, _ = build_bank(goal, index, "random")
        response_programs, random_programs = planned(goal, goal_bank), planned(goal, random_bank)
        methods = {}
        for name, programs, criterion in (("response", response_programs, "terminal_score"),
                                          ("same_bank_greedy", response_programs, "prefix_score"),
                                          ("lookahead", response_programs, "lookahead_score"),
                                          ("random_bridge", random_programs, "terminal_score")):
            selected = select(programs, criterion)
            item = programs[selected]
            methods[name] = {"selected_index": selected, "chain": [item["prefix"], item["suffix"]],
                             "terminal_score": item["terminal_score"],
                             "generation_rollouts": 20 if name in ("response", "same_bank_greedy", "lookahead") else 0,
                             "svd_calls": 4 if name in ("response", "same_bank_greedy", "lookahead") else 0,
                             "suffix_rollouts": 165, "prefix_evaluations": 5}
        rows.append({"index": index, "goal": goal.tolist(), "responses": responses,
                     "response_bank": response_programs, "random_bank": random_programs, "methods": methods})
    if any(dynamics.hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
        raise ValueError("source drift")
    with PLANS.open("x") as file:
        json.dump({"dependencies": hashes, "scope": "exposed exact-model dev, not RGB",
                   "rows": rows, "PC2_pass": False, "PC3_pass": False}, file, indent=2, allow_nan=False)
    print("All programs sealed before interventions.", flush=True)


def intervention_stage():
    if INTERVENTION.exists():
        raise FileExistsError(INTERVENTION)
    sha = lambda p: dynamics.hashlib.sha256(p.read_bytes()).hexdigest()
    source = json.loads(PLANS.read_text())
    if any(sha(Path(p)) != h for p, h in source["dependencies"].items()):
        raise ValueError("plan source drift")
    rows = []
    for row in source["rows"]:
        goal = np.asarray(row["goal"])
        methods = {}
        for name, item in row["methods"].items():
            chain = np.asarray(item["chain"])
            bank = row["random_bank"] if name == "random_bridge" else row["response_bank"]
            selected = bank[item["selected_index"]]
            if not np.array_equal(chain, [selected["prefix"], selected["suffix"]]):
                raise ValueError("selected-chain mismatch")
            greedy = select(bank, "prefix_score")
            prefixes = {"selected": chain[0], "zero": np.zeros(2), "greedy": np.asarray(bank[greedy]["prefix"])}
            controls = {}
            for role, prefix in prefixes.items():
                succ = dynamics.step(np.zeros(3), prefix)
                values = [dynamics.evaluate([prefix, [v, 0]], goal)[0] for v in (*FIXED, float(chain[1, 0]))]
                zero = values[2]
                grid_scores = [dynamics.evaluate([prefix, [v, 0]], goal)[0] for v in GRID]
                controls[role] = {"prefix": prefix.tolist(), "prefix_distance": float(np.linalg.norm(succ[:2] - goal)),
                                  "common_suffix_distances": values, "active_changes": [x - zero for x in values],
                                  "own_optimized_distance": float(min(grid_scores)),
                                  "own_suffix": [float(GRID[int(np.argmin(grid_scores))]), 0.]}
            chosen = controls["selected"]
            nonprogress = chosen["prefix_distance"] >= np.linalg.norm(goal) - 1e-12
            response = all(chosen["active_changes"][-1] < controls[c]["active_changes"][-1] - 1e-12
                           for c in ("zero", "greedy"))
            own = all(chosen["own_optimized_distance"] < controls[c]["own_optimized_distance"] - 1e-12
                      for c in ("zero", "greedy"))
            methods[name] = {"chain": chain.tolist(), "controls": controls, "nonprogress": bool(nonprogress),
                             "active_response": bool(response), "own_plan_advantage": bool(own),
                             "limited_witness": bool(nonprogress and response and own)}
        rows.append({"index": row["index"], "goal": row["goal"], "methods": methods})
    summary = {method: {key: sum(row["methods"][method][key] for row in rows)
                        for key in ("nonprogress", "active_response", "own_plan_advantage", "limited_witness")}
               for method in rows[0]["methods"]}
    with INTERVENTION.open("x") as file:
        json.dump({"plans_sha256": sha(PLANS), "protocol_sha256": sha(PROTOCOL), "rows": rows, "summary": summary,
                   "scope": "exact post-seal dev intervention, no RGB or independent confirmation",
                   "PC2_pass": False, "PC3_pass": False}, file, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "intervene"):
        raise SystemExit("choose plan or intervene")
    (plan_stage if sys.argv[1] == "plan" else intervention_stage)()
