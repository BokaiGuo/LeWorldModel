"""Predeclared wall development panel: response-initialized block programs."""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments import pc_multi_nominal_rgb_dev_20261103_codex as policy
from experiments import pc_successor_response_composition_e0_dev_20261103_codex as exact
from experiments.pc_budget_allocation_20261103_codex import allocate_samples, optimize

PROTOCOL = policy.ROOT.parent / "docs/PC_BLOCK_PROGRAM_WALL_PROSPECTIVE_DEV_PROTOCOL_20261103_CODEX.md"
PREFIX = policy.ROOT / "results/pc_block_program_wall_prospective_dev_20261103_codex"
PLANS = PREFIX.with_name(PREFIX.name + "_plans.json")
AUDIT = PREFIX.with_name(PREFIX.name + "_audit.json")
INTERVENTION = PREFIX.with_name(PREFIX.name + "_intervention.json")
FEASIBILITY = PREFIX.with_name(PREFIX.name + "_feasibility.json")
BUDGETS = (384, 768)
ARMS = ("svd_response", "ridge_response", "random_response", "nominal", "full_cem")
NOMINALS = np.asarray([[0, 0, 0, 0, 0, 0], [0, .4, .4, 0, .4, 0],
                       [0, -.4, .4, 0, .4, 0]], dtype=float)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes():
    return {str(path): sha(path) for path in
            (PROTOCOL, Path(__file__), Path(policy.__file__), Path(exact.__file__),
             Path(exact.wall.__file__), Path(exact.wall.physics.__file__), Path(optimize.__code__.co_filename))}


def tasks():
    return [{"start": [-.5, y, theta], "goal": [.5, gy]}
            for y, theta, gy in itertools.product((-.45, -.15, .15, .45),
                                                  (-np.pi / 3, 0., np.pi / 3), (-.35, .35))]


def decode(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 6 or np.any(np.abs(points) > 1 + 1e-12):
        raise ValueError("invalid block parameters")
    return np.repeat(points.reshape(-1, 3, 2), 5, axis=1)


def block_optimize(start, count, rng, score_fn, initial_score):
    loc, scale = np.asarray(start).copy(), np.full(6, 1 / 3)
    best, best_score = decode(loc[None])[0], float(initial_score)
    for offset in range(0, count, 12):
        size = min(12, count - offset)
        samples = np.clip(loc + rng.normal(size=(size, 6)) * scale, -1, 1)
        samples[0] = loc
        actions = decode(samples)
        scores = np.asarray(score_fn(actions))
        index = int(np.argmin(scores))
        if scores[index] < best_score:
            best, best_score = actions[index].copy(), float(scores[index])
        if size > 1:
            elite = samples[np.argsort(scores, kind="stable")[:min(4, size)]]
            loc = elite.mean(0)
            scale = np.maximum(elite.std(0), 1e-5)
    return best, best_score


def run_one(task, root, budget, arm):
    started = time.monotonic()
    goal, trace, used, phase = np.asarray(task["goal"]), [], [0], ["unset"]
    structure = {"svd": 0, "solve": 0, "seconds": 0.}

    def predict(actions):
        actions = np.asarray(actions, dtype=float)
        if actions.shape[1:] != (15, 2) or used[0] + len(actions) > budget:
            raise ValueError("invalid or exhausted full-rollout budget")
        pos = exact.endpoints(task, actions)
        scores = np.linalg.norm(pos - goal, axis=1)
        used[0] += len(actions)
        trace.append({"phase": phase[0], "actions": actions.tolist(), "scores": scores.tolist()})
        return pos, scores

    seed = 2026120600 + root
    evidence, bank, prefix_scores, programs = [], [], [], []
    if arm == "full_cem":
        phase[0] = "full_cem"
        chain, score, _ = optimize([], 15, budget, np.random.default_rng(seed), lambda x: predict(x)[1])
        selected = {"name": "full_cem", "chain": chain.tolist(), "score": score}
        incumbent = None
    else:
        rng = np.random.default_rng(seed + 50)
        for index, nominal in enumerate(NOMINALS):
            if arm in ("svd_response", "ridge_response"):
                phase[0] = f"response:{index}"
                center = predict(decode(nominal[None]))[0][0]
                directions = np.eye(6)
                plus = predict(decode(nominal + .05 * directions))[0]
                minus = predict(decode(nominal - .05 * directions))[0]
                matrix = ((plus - minus) / .1).T
                residual = goal - center
                start_time = time.monotonic()
                if arm == "svd_response":
                    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
                    delta = vt.T @ ((singular / (singular ** 2 + .001)) * (u.T @ residual))
                    structure["svd"] += 1
                else:
                    delta = np.linalg.solve(matrix.T @ matrix + .001 * np.eye(6), matrix.T @ residual)
                    structure["solve"] += 1
                structure["seconds"] += time.monotonic() - start_time
                evidence.append({"nominal": nominal.tolist(), "jacobian": matrix.tolist(),
                                 "residual": residual.tolist(), "raw_delta": delta.tolist()})
            elif arm == "random_response":
                delta = rng.normal(size=6)
                delta *= .5 / max(float(np.max(np.abs(delta))), 1e-12)
            elif arm == "nominal":
                delta = np.zeros(6)
            else:
                raise ValueError("invalid arm")
            delta = delta * min(1., .5 / max(float(np.max(np.abs(delta))), 1e-12))
            point = np.clip(nominal + delta, -1, 1)
            bank.append({"name": f"candidate_{index}", "parameters": point.tolist(),
                         "chain": decode(point[None])[0].tolist()})
        phase[0] = "candidate_score"
        scores = predict(np.asarray([x["chain"] for x in bank]))[1]
        for item, score in zip(bank, scores, strict=True):
            item["score"] = float(score)
        phase[0] = "prefix_score"
        prefix_actions = np.zeros((3, 15, 2))
        prefix_actions[:, :5] = np.asarray([x["chain"][:5] for x in bank])
        prefix_scores = predict(prefix_actions)[1].tolist()
        suffix_total = 3 * (budget - used[0]) // 4
        counts = allocate_samples(suffix_total, 0, len(bank), 1)
        for index, (item, count) in enumerate(zip(bank, counts, strict=True)):
            phase[0] = f"block_cem:{index}"
            chain, score = block_optimize(item["parameters"], count, np.random.default_rng(seed),
                                          lambda x: predict(x)[1], item["score"])
            programs.append({"name": item["name"], "chain": chain.tolist(), "score": score, "samples": count})
        incumbent = min(programs, key=lambda x: x["score"])
        phase[0] = "joint_refine"
        chain, score = policy.refine(incumbent["chain"], incumbent["score"], budget - used[0],
                                     np.random.default_rng(seed + 1), lambda x: predict(x)[1])
        selected = {"name": "joint_refined" if score < incumbent["score"] else incumbent["name"],
                    "chain": chain.tolist(), "score": score}
    return {"root": root, "task": task, "arm": arm, "budget": budget, "used": used[0], "bank": bank,
            "evidence": evidence, "prefix_scores": prefix_scores, "programs": programs,
            "incumbent": incumbent, "selected": selected, "structure": structure,
            "wall_seconds": time.monotonic() - started, "trace": trace}


def plan():
    if any(path.exists() for path in (PLANS, AUDIT, INTERVENTION, FEASIBILITY)):
        raise FileExistsError("output exists")
    source = hashes()
    rows = []
    for root, task in enumerate(tasks()):
        for budget in BUDGETS:
            for arm in ARMS:
                rows.append(run_one(task, root, budget, arm))
        print(f"root {root}: all arms and budgets complete", flush=True)
    if source != hashes():
        raise ValueError("source drift")
    with PLANS.open("x") as stream:
        json.dump({"dependencies": source, "rows": rows, "PC2_pass": False, "PC3_pass": False},
                  stream, separators=(",", ":"), allow_nan=False)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != hashes():
        raise ValueError("source drift")
    rows = []
    for row in data["rows"]:
        goal, selected = np.asarray(row["task"]["goal"]), np.asarray(row["selected"]["chain"])
        count, found = 0, False
        for batch in row["trace"]:
            actions = np.asarray(batch["actions"])
            scores = np.linalg.norm(exact.endpoints(row["task"], actions) - goal, axis=1)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            matching = np.flatnonzero(np.all(actions == selected, axis=(1, 2)))
            found |= any(abs(scores[i] - row["selected"]["score"]) < 1e-12 for i in matching)
            count += len(actions)
        if count != row["used"] or count != row["budget"] or not found:
            raise ValueError("trace or budget mismatch")
        if row["incumbent"] and row["selected"]["score"] > row["incumbent"]["score"] + 1e-12:
            raise ValueError("lost incumbent")
        rows.append({"root": row["root"], "arm": row["arm"], "budget": row["budget"], "used": count})
    with AUDIT.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "rows": rows}, stream, indent=2, allow_nan=False)
    print(f"audited {len(rows)} complete traces", flush=True)


def intervene():
    if INTERVENTION.exists():
        raise FileExistsError(INTERVENTION)
    data, audited = json.loads(PLANS.read_text()), json.loads(AUDIT.read_text())
    if data["dependencies"] != hashes() or audited["plans_sha256"] != sha(PLANS):
        raise ValueError("hash mismatch")
    rows = []
    for row in data["rows"]:
        if row["arm"] != "svd_response":
            continue
        goal, chain = np.asarray(row["task"]["goal"]), np.asarray(row["selected"]["chain"])
        greedy_index = int(np.argmin(row["prefix_scores"]))
        prefixes = {"selected": chain[:5], "zero": np.zeros((5, 2)),
                    "greedy": np.asarray(row["bank"][greedy_index]["chain"][:5])}
        suffixes = [np.tile(x, (10, 1)) for x in ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
        suffixes.append(chain[5:])
        controls = {}
        for name, prefix in prefixes.items():
            prefix_distance = float(np.linalg.norm(exact.endpoints(row["task"], prefix[None])[0] - goal))
            full = np.stack([np.concatenate((prefix, tail)) for tail in suffixes])
            scores = np.linalg.norm(exact.endpoints(row["task"], full) - goal, axis=1)
            best, score, values = optimize(prefix, 15, 120, np.random.default_rng(2026120700 + row["root"]),
                                           lambda x: np.linalg.norm(exact.endpoints(row["task"], x) - goal, axis=1))
            controls[name] = {"prefix": prefix.tolist(), "prefix_distance": prefix_distance,
                              "common_scores": scores.tolist(), "active_changes": (scores - scores[0]).tolist(),
                              "own_score": score, "own_chain": best.tolist(), "own_scores": values}
        initial = np.linalg.norm(np.asarray(row["task"]["start"])[:2] - goal)
        c = controls["selected"]
        nonprogress = c["prefix_distance"] >= initial - 1e-10
        active = all(c["active_changes"][-1] < controls[x]["active_changes"][-1] - 1e-10
                     for x in ("zero", "greedy"))
        own = all(c["own_score"] < controls[x]["own_score"] - 1e-10 for x in ("zero", "greedy"))
        rows.append({"root": row["root"], "budget": row["budget"], "chain": chain.tolist(),
                     "controls": controls, "nonprogress": bool(nonprogress), "active_response": bool(active),
                     "own_plan_advantage": bool(own), "limited_witness": bool(nonprogress and active and own)})
    with INTERVENTION.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "audit_sha256": sha(AUDIT), "rows": rows,
                   "PC2_pass": False, "PC3_pass": False}, stream, indent=2, allow_nan=False)
    print(f"audited interventions for {len(rows)} selected chains", flush=True)


def feasibility():
    if FEASIBILITY.exists():
        raise FileExistsError(FEASIBILITY)
    data = json.loads(PLANS.read_text())
    if data["dependencies"] != hashes():
        raise ValueError("hash mismatch")
    rows = []
    for root, task in enumerate(tasks()):
        records = []
        for item in exact.wall.feasibility(task):
            if item["within_horizon"] and item["steps_needed"] <= 15:
                chain = np.asarray(item["actions"])[:15] / exact.wall.HIGH
                distance = float(np.linalg.norm(exact.endpoints(task, chain[None])[0] - task["goal"]))
                records.append({"side": item["height"], "steps": item["steps_needed"], "distance": distance})
        rows.append({"root": root, "witnesses": records})
    with FEASIBILITY.open("x") as stream:
        json.dump({"plans_sha256": sha(PLANS), "rows": rows, "scope": "privileged diagnostic only"},
                  stream, indent=2, allow_nan=False)
    print(f"feasibility audited for {len(rows)} tasks", flush=True)


if __name__ == "__main__":
    commands = {"plan": plan, "audit": audit, "intervene": intervene, "feasibility": feasibility}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit("choose plan, audit, intervene or feasibility")
    commands[sys.argv[1]]()
