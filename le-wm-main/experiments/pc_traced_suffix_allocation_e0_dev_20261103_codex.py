"""Same-bank allocation development comparison with full prediction traces."""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_multi_nominal_rgb_dev_20261103_codex as policy
from experiments import pc_same_policy_exact_mechanism_dev_20261103_codex as exact
from experiments.pc_budget_allocation_20261103_codex import optimize
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability.pc_spectral_bank_planner_20261103_codex import Meter as StructureMeter

PROTOCOL = policy.ROOT.parent / "docs/PC_TRACED_SUFFIX_ALLOCATION_E0_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = policy.ROOT / "results/pc_traced_suffix_allocation_e0_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
ARMS = ("uniform", "adaptive", "greedy", "cem")
BUDGETS = (480, 720)


def hashes():
    paths = (PROTOCOL, Path(__file__), Path(policy.__file__), Path(policy.generator.__file__),
             Path(exact.__file__), Path(optimize.__code__.co_filename), Path(inspect.getfile(Meter)),
             Path(inspect.getfile(StructureMeter)))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


class CEMState:
    """Persistent suffix CEM; a call never resets a candidate's RNG or elites."""

    def __init__(self, prefix, seed):
        self.prefix = np.asarray(prefix, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.loc = np.zeros((10, 2))
        self.scale = np.full((10, 2), 1 / 3)
        self.best, self.score, self.count = None, float("inf"), 0

    def step(self, size, predict):
        if not 1 <= size <= 12:
            raise ValueError("invalid batch size")
        suffix = np.clip(self.loc + self.rng.normal(size=(size, 10, 2)) * self.scale, -1, 1)
        suffix[0] = self.loc
        chains = np.concatenate((np.broadcast_to(self.prefix, (size, 5, 2)), suffix), axis=1)
        values = np.asarray(predict(chains)[1])
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError("invalid scores")
        best = int(np.argmin(values))
        if values[best] < self.score:
            self.best, self.score = chains[best].copy(), float(values[best])
        if size > 1:
            elite = suffix[np.argsort(values, kind="stable")[:min(4, size)]]
            self.loc = elite.mean(axis=0)
            self.scale = np.maximum(elite.std(axis=0), 1e-5)
        self.count += size


def decide(root, goal, budget, arm):
    started = time.monotonic()
    meter, structure, batches = Meter(budget), StructureMeter(
        {"prediction_samples": 42, "svd": 2, "linear_solve": 2}), []
    phase = ["unassigned"]

    def predict(actions):
        actions = np.asarray(actions, dtype=float)

        def raw(batch):
            position = exact.rollout(batch)[:, :2]
            return position, np.linalg.norm(position - goal, axis=1)

        positions, scores = meter.predict(actions, raw)
        batches.append({"phase": phase[0], "actions": actions.tolist(),
                        "positions": positions.tolist(), "scores": scores.tolist(),
                        "tokens": len(actions) * len(actions[0]) // 5})
        return positions, scores

    seed = 2026113000 + 10 * root
    bank, generated, prefix_scores, allocation, candidates = [], [], [], [], []
    if arm == "cem":
        phase[0] = "full_cem"
        count = (budget - meter.token_samples) // 3
        selected_chain, score, _ = optimize([], 15, count, np.random.default_rng(seed),
                                            lambda x: predict(x)[1])
        selected = {"name": "direct_cem", "chain": selected_chain.tolist(), "score": score}
        incumbent = None
    else:
        phase[0] = "generate"
        bank, generated = policy.generator.generate(
            lambda x: predict(np.asarray(x)[None])[0][0], goal, policy.NOMINALS, structure, mode="svd")
        phase[0] = "prefix_score"
        for item in bank:
            prefix_scores.append(float(predict(np.asarray(item["prefix"])[None])[1][0]))
        affordable = (budget - meter.token_samples) // 3
        suffix_total = 3 * affordable // 4
        states = [CEMState(item["prefix"], seed) for item in bank]

        def step(index, size):
            phase[0] = f"suffix:{index}"
            states[index].step(size, predict)
            allocation.append({"candidate": index, "samples": size})

        if arm == "greedy":
            order = [int(np.argmin(prefix_scores))]
            remaining = suffix_total
            while remaining:
                size = min(12, remaining)
                step(order[0], size)
                remaining -= size
        elif arm == "uniform":
            quotient, remainder = divmod(suffix_total, len(bank))
            for index in range(len(bank)):
                remaining = quotient + (index < remainder)
                while remaining:
                    size = min(12, remaining)
                    step(index, size)
                    remaining -= size
        elif arm == "adaptive":
            if suffix_total < 12 * len(bank):
                raise ValueError("insufficient budget for adaptive warmup")
            for index in range(len(bank)):
                step(index, 12)
            remaining = suffix_total - 12 * len(bank)
            while remaining:
                index = min(range(len(states)), key=lambda i: (states[i].score, i))
                size = min(12, remaining)
                step(index, size)
                remaining -= size
        else:
            raise ValueError("unknown allocation")
        for item, state in zip(bank, states, strict=True):
            if state.count:
                candidates.append({"name": item["name"], "count": state.count, "score": state.score,
                                   "chain": state.best.tolist()})
        incumbent = min(candidates, key=lambda item: item["score"])
        phase[0] = "joint_refine"
        count = (budget - meter.token_samples) // 3
        refined, score = policy.refine(incumbent["chain"], incumbent["score"], count,
                                       np.random.default_rng(seed + 1), lambda x: predict(x)[1])
        selected = {"name": "joint_refined" if score < incumbent["score"] else incumbent["name"],
                    "chain": refined.tolist(), "score": score}
    return {"root": root, "goal": goal.tolist(), "budget": budget, "arm": arm,
            "bank": bank, "generation": generated, "prefix_scores": prefix_scores,
            "allocation": allocation, "candidates": candidates, "incumbent": incumbent,
            "selected": selected, "counts": meter.as_dict(), "structure_counts": structure.counts,
            "structure_seconds": structure.seconds, "wall_seconds": time.monotonic() - started,
            "batches": batches}


def plan():
    if PLANS.exists() or AUDIT.exists():
        raise FileExistsError("sealed output exists")
    dependencies = hashes()
    rows = []
    for root in range(12):
        angle = 2 * np.pi * (root + .193) / 12
        goal = .9 * np.array([np.cos(angle), np.sin(angle)])
        for budget in BUDGETS:
            for arm in ARMS:
                rows.append(decide(root, goal, budget, arm))
        print(f"root {root}: both budgets and all allocation arms", flush=True)
    if dependencies != hashes():
        raise ValueError("dependency drift")
    with PLANS.open("x") as file:
        json.dump({"dependencies": dependencies, "rows": rows, "PC2_pass": False, "PC3_pass": False},
                  file, allow_nan=False, separators=(",", ":"))


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    source = json.loads(PLANS.read_text())
    if source["dependencies"] != hashes():
        raise ValueError("dependency mismatch")
    summary = []
    for row in source["rows"]:
        goal = np.asarray(row["goal"])
        tokens = 0
        selected_chain = np.asarray(row["selected"]["chain"])
        found_selected = False
        for batch in row["batches"]:
            actions = np.asarray(batch["actions"])
            if actions.shape[1:] == selected_chain.shape:
                found_selected |= bool(np.any(np.all(actions == selected_chain, axis=(1, 2))))
            positions = exact.rollout(actions)[:, :2]
            scores = np.linalg.norm(positions - goal, axis=1)
            np.testing.assert_allclose(positions, batch["positions"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(scores, batch["scores"], rtol=0, atol=1e-12)
            cost = len(actions) * actions.shape[1] // 5
            if cost != batch["tokens"]:
                raise ValueError("batch token mismatch")
            tokens += cost
        if not found_selected:
            raise ValueError("selected chain absent from prediction trace")
        chain = selected_chain
        terminal = np.linalg.norm(exact.rollout(chain[None])[0, :2] - goal)
        np.testing.assert_allclose(terminal, row["selected"]["score"], rtol=0, atol=1e-12)
        if tokens != row["counts"]["prediction_token_samples"] or tokens > row["budget"]:
            raise ValueError("total token mismatch")
        if row["incumbent"] and terminal > row["incumbent"]["score"] + 1e-12:
            raise ValueError("refinement lost incumbent")
        summary.append({"root": row["root"], "budget": row["budget"], "arm": row["arm"],
                        "tokens": tokens, "terminal": float(terminal)})
    with AUDIT.open("x") as file:
        json.dump({"plans_sha256": exact.sha(PLANS), "rows": summary,
                   "PC2_pass": False, "PC3_pass": False}, file, indent=2, allow_nan=False)
    print(f"audited {len(summary)} complete planning traces", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "audit"):
        raise SystemExit("choose plan or audit")
    (plan if sys.argv[1] == "plan" else audit)()
