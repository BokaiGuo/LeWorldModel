"""Coverage/pruning/continuation diagnostic; exact exposed tasks only."""
import json
from pathlib import Path

import numpy as np

from . import pc_wall_response_side_20261103 as base

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT.parent / "docs/PC_WALL_RETENTION_SIDE_PROTOCOL_20261103.md"
OUTPUT = ROOT / "results/pc_wall_retention_side_20261103.json"
CAPS = (128, 256)
ARMS = ("greedy", "lookahead", "random")


def prefix_bank():
    bank = [np.zeros((6, 2))]
    for turn in -np.pi + np.arange(16) * np.pi / 8:
        for speed in (.15, .3):
            bank.append(np.array([[0., turn / 2]] * 2 + [[speed, 0.]] * 4))
    return np.asarray(bank)


def suffixes(parameters):
    parameters = np.asarray(parameters, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 2 or not np.isfinite(parameters).all() or np.any(np.abs(parameters) > 1):
        raise ValueError("bounded two-coordinate suffix required")
    actions = np.zeros((len(parameters), 10, 2))
    actions[:, :2, 1] = parameters[:, :1] * np.pi / 2
    actions[:, 2:, 0] = parameters[:, 1:] * .3
    return actions


def probe_family():
    return suffixes([[0., 0.]] + [[turn, speed] for turn in (-1., -.5, 0., .5, 1.)
                                 for speed in (.5, 1.)])


def evaluate(task, prefix, tails):
    chains = np.concatenate((np.broadcast_to(prefix, (len(tails), 6, 2)), tails), axis=1)
    trace, _ = base.rollout(task["start"], chains)
    return np.linalg.norm(trace[:, -1, :2] - task["goal"], axis=1)


def screen(task, index):
    bank, probes = prefix_bank(), probe_family()
    zeros = np.zeros((len(bank), 10, 2))
    trace, _ = base.rollout(task["start"], np.concatenate((bank, zeros), axis=1))
    prefix_cost = np.linalg.norm(trace[:, 6, :2] - task["goal"], axis=1)
    probe_cost = np.asarray([evaluate(task, prefix, probes) for prefix in bank])
    retained = {
        "greedy": np.argsort(prefix_cost, kind="stable")[:8].tolist(),
        "lookahead": np.argsort(probe_cost.min(axis=1), kind="stable")[:8].tolist(),
        "random": np.random.default_rng(2026111600 + index).choice(len(bank), 8, replace=False).tolist()}
    return {"prefix_costs": prefix_cost.tolist(), "probe_costs": probe_cost.tolist(),
            "retained": retained, "same_bank_greedy": int(np.argmin(prefix_cost)),
            "screen_calls": len(bank) * (1 + len(probes))}


def optimize(task, index, prefix, probe_costs, cap):
    if cap not in CAPS:
        raise ValueError("unregistered budget")
    best_index = int(np.argmin(probe_costs))
    best, best_cost = probe_family()[best_index].copy(), float(probe_costs[best_index])
    rng = np.random.default_rng(2026111700 + index)
    trace = []
    for start in range(2):
        mean, std = np.zeros(2), np.full(2, .5)
        for iteration in range(4):
            parameters = np.clip(mean + std * rng.standard_normal((cap // 8, 2)), -1., 1.)
            proposals = suffixes(parameters)
            scores = evaluate(task, prefix, proposals)
            order = np.argsort(scores, kind="stable")
            if scores[order[0]] < best_cost:
                best, best_cost = proposals[order[0]].copy(), float(scores[order[0]])
            elites = parameters[order[:max(2, len(order) // 8)]]
            mean, std = elites.mean(axis=0), np.maximum(elites.std(axis=0), .02)
            trace.append({"start": start, "round": iteration, "parameters": parameters.tolist(),
                          "scores": scores.tolist(), "best_so_far": best_cost})
    return {"suffix": best.tolist(), "terminal_distance": best_cost,
            "optimizer_calls": cap, "incumbent_probe_calls": 11, "trace": trace}


def intervention(task, bank, screen_data, census, selected):
    greedy = screen_data["same_bank_greedy"]
    tail = np.asarray(census[selected]["suffix"])
    records = {}
    for label, idx in (("selected", selected), ("zero", 0), ("same_bank_greedy", greedy)):
        error = float(evaluate(task, bank[idx], tail[None])[0])
        zero = screen_data["probe_costs"][idx][0]
        records[label] = {"index": idx, "prefix_distance": screen_data["prefix_costs"][idx],
                          "common_suffix_distance": error, "zero_suffix_distance": zero,
                          "active_change": error - zero,
                          "own_planned_distance": census[idx]["terminal_distance"]}
    initial = float(np.linalg.norm(np.asarray(task["start"])[:2] - task["goal"]))
    own = records["selected"]
    witness = own["prefix_distance"] >= initial - 1e-12 and all(
        own["active_change"] < records[c]["active_change"] - 1e-12
        and own["own_planned_distance"] < records[c]["own_planned_distance"] - 1e-12
        for c in ("zero", "same_bank_greedy"))
    return {"controls": records, "initial_distance": initial,
            "limited_witness": bool(witness), "diagnostic_calls": 3}


def run_task(task, index):
    bank = prefix_bank()
    screened = screen(task, index)
    # Retention is finalized before any optimized-suffix outcomes exist.
    sealed_retention = json.dumps(screened["retained"], sort_keys=True)
    budgets = []
    for cap in CAPS:
        census = [optimize(task, index, prefix, screened["probe_costs"][i], cap)
                  for i, prefix in enumerate(bank)]
        successful = [i for i, x in enumerate(census) if x["terminal_distance"] < .05]
        arms = {}
        for name, retained in screened["retained"].items():
            selected = min(retained, key=lambda i: (census[i]["terminal_distance"], i))
            chains = np.concatenate((bank[selected], census[selected]["suffix"]))
            trajectory, blocked = base.rollout(task["start"], chains[None])
            calls = 33 + (33 * 11 if name == "lookahead" else 8 * 11) + 8 * cap
            arms[name] = {
                "selected_index": selected, "terminal_distance": census[selected]["terminal_distance"],
                "retained_successful_indices": [i for i in retained if i in successful],
                "actions": chains.tolist(), "trajectory": trajectory[0].tolist(),
                "blocked": blocked[0].tolist(),
                "hypothetical_unique_planner_calls": calls,
                "hypothetical_planner_transitions": 16 * calls,
                "intervention": intervention(task, bank, screened, census, selected)}
        budgets.append({"cap": cap, "census": census, "successful_indices": successful, "arms": arms,
                        "census_optimizer_calls": 33 * cap,
                        "diagnostic_selected_replay_calls": 3,
                        "diagnostic_intervention_calls": 9})
    if json.dumps(screened["retained"], sort_keys=True) != sealed_retention:
        raise AssertionError("retention changed after outcomes")
    return {"index": index, "task": task, "bank": bank.tolist(),
            "screen": screened, "sealed_retention": sealed_retention, "budgets": budgets}


def summary(rows):
    return {str(cap): {arm: {
        "successes": sum(b["arms"][arm]["terminal_distance"] < .05
                         for row in rows for b in row["budgets"] if b["cap"] == cap),
        "mean_distance": float(np.mean([b["arms"][arm]["terminal_distance"]
                                       for row in rows for b in row["budgets"] if b["cap"] == cap])),
        "limited_witnesses": sum(b["arms"][arm]["intervention"]["limited_witness"]
                                 for row in rows for b in row["budgets"] if b["cap"] == cap)}
        for arm in ARMS} for cap in CAPS}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    paths = [PROTOCOL, Path(__file__), Path(base.__file__), Path(base.physics.__file__)]
    hashes = {str(p): base.digest(p) for p in paths}
    print("Exact fixed-bank coverage/pruning/continuation; no RGB or P1 claim.", flush=True)
    rows = []
    for index, task in enumerate(base.tasks()):
        rows.append(run_task(task, index))
        print(f"completed task {index}", flush=True)
    if hashes != {str(p): base.digest(p) for p in paths}:
        raise RuntimeError("source changed during execution")
    report = {"dependencies": hashes, "rows": rows, "summary": summary(rows),
              "scope": "exposed exact-model fixed-bank diagnostic, not spectral or fair total resource",
              "P1_admission": False, "full_cost_certified": False}
    with OUTPUT.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
