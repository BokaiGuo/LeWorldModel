"""Development factorial: frozen retention against equal-sample suffix optimizers."""
import json
from pathlib import Path

import numpy as np

from . import pc_wall_retention_side_20261103 as prior

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_WALL_RETENTION_FACTOR_DEV_PROTOCOL_20261103_CODEX.md"
SEAL = ROOT / "results/pc_wall_retention_factor_dev_20261103_codex_retention.json"
OUTPUT = ROOT / "results/pc_wall_retention_factor_dev_20261103_codex.json"
ARMS = ("greedy", "random", "lookahead", "diverse")


def diverse(task, screened):
    bank = prior.prefix_bank()
    trace, _ = prior.base.rollout(task["start"], np.concatenate((bank, np.zeros((len(bank), 10, 2))), axis=1))
    states = trace[:, 6]
    features = np.column_stack((states[:, :2], .5 * np.cos(states[:, 2]), .5 * np.sin(states[:, 2])))
    selected = [screened["same_bank_greedy"]]
    while len(selected) < 8:
        dist = np.linalg.norm(features[:, None, :] - features[selected][None, :, :], axis=2).min(axis=1)
        dist[selected] = -1
        selected.append(int(np.argmax(dist)))
    return selected


def stratified(task, index, prefix, probe_costs, cap):
    probes = prior.probe_family()
    best_idx = int(np.argmin(probe_costs))
    best, best_cost = probes[best_idx].copy(), float(probe_costs[best_idx])
    rng = np.random.default_rng(2026111800 + index)
    trace = []
    for start, center in enumerate((-.75, -.25, .25, .75)):
        mean, std = np.array([center, 0.]), np.array([.25, .5])
        for iteration in range(4):
            parameters = np.clip(mean + std * rng.standard_normal((cap // 16, 2)), -1, 1)
            proposals = prior.suffixes(parameters)
            scores = prior.evaluate(task, prefix, proposals)
            order = np.argsort(scores, kind="stable")
            if scores[order[0]] < best_cost:
                best, best_cost = proposals[order[0]].copy(), float(scores[order[0]])
            elites = parameters[order[:max(2, len(order) // 8)]]
            mean, std = elites.mean(axis=0), np.maximum(elites.std(axis=0), .02)
            trace.append({"start": start, "round": iteration, "parameters": parameters.tolist(),
                          "scores": scores.tolist(), "best_so_far": best_cost})
    return {"suffix": best.tolist(), "terminal_distance": best_cost, "optimizer_calls": cap, "trace": trace}


def load_prior():
    saved = json.loads(prior.OUTPUT.read_text())
    if any(prior.base.digest(Path(path)) != digest for path, digest in saved["dependencies"].items()):
        raise ValueError("prior sealed dependency mismatch")
    if len(saved["rows"]) != 8:
        raise ValueError("prior panel mismatch")
    return saved


def seal():
    if SEAL.exists() or OUTPUT.exists():
        raise FileExistsError("existing result or seal")
    saved = load_prior()
    rows = []
    for row in saved["rows"]:
        if prior.screen(row["task"], row["index"]) != row["screen"]:
            raise ValueError("prior screening mismatch")
        retained = dict(row["screen"]["retained"])
        retained["diverse"] = diverse(row["task"], row["screen"])
        rows.append({"index": row["index"], "retained": retained})
    record = {"protocol_sha256": prior.base.digest(PROTOCOL), "runner_sha256": prior.base.digest(Path(__file__)),
              "prior_sha256": prior.base.digest(prior.OUTPUT), "rows": rows}
    with SEAL.open("x") as f:
        json.dump(record, f, indent=2, allow_nan=False)
    print("Retention sealed before new optimization.", flush=True)


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    saved, sealed = load_prior(), json.loads(SEAL.read_text())
    if (sealed["protocol_sha256"] != prior.base.digest(PROTOCOL)
            or sealed["runner_sha256"] != prior.base.digest(Path(__file__))
            or sealed["prior_sha256"] != prior.base.digest(prior.OUTPUT)):
        raise ValueError("seal mismatch")
    rows = []
    for oldrow, frozen in zip(saved["rows"], sealed["rows"], strict=True):
        idx, task, bank = oldrow["index"], oldrow["task"], prior.prefix_bank()
        if frozen["index"] != idx or frozen["retained"]["diverse"] != diverse(task, oldrow["screen"]):
            raise ValueError("retention mismatch")
        relevant = sorted({i for indices in frozen["retained"].values() for i in indices})
        results = []
        for oldbudget in oldrow["budgets"]:
            cap = oldbudget["cap"]
            new = {i: stratified(task, idx, bank[i], oldrow["screen"]["probe_costs"][i], cap) for i in relevant}
            arms = {}
            for name in ARMS:
                indices = frozen["retained"][name]
                for optimizer, plans in (("original", oldbudget["census"]), ("stratified", new)):
                    selected = min(indices, key=lambda i: (plans[i]["terminal_distance"], i))
                    plan = plans[selected]
                    actions = np.concatenate((bank[selected], plan["suffix"]))
                    value = float(prior.evaluate(task, bank[selected], np.asarray(plan["suffix"])[None])[0])
                    if not np.isclose(value, plan["terminal_distance"], atol=1e-12, rtol=0):
                        raise ValueError("replay mismatch")
                    arms[f"{name}_{optimizer}"] = {
                        "retained": indices, "selected_index": selected, "actions": actions.tolist(),
                        "terminal_distance": value, "success": value < .05,
                        "screening_calls": 33 + (33 * 11 if name == "lookahead" else 0),
                        "continuation_calls": 8 * cap + (0 if name == "lookahead" else 8 * 11)}
            results.append({"cap": cap, "stratified_plans": new, "arms": arms})
        rows.append({"index": idx, "budgets": results})
        print(f"completed paired task {idx}", flush=True)
    summary = {}
    for cap in prior.CAPS:
        summary[str(cap)] = {}
        for arm in (f"{retention}_{optimizer}" for retention in ARMS for optimizer in ("original", "stratified")):
            items = [b["arms"][arm] for row in rows for b in row["budgets"] if b["cap"] == cap]
            summary[str(cap)][arm] = {"successes": sum(x["success"] for x in items),
                                      "mean_distance": float(np.mean([x["terminal_distance"] for x in items]))}
    report = {"scope": "exposed exact-model development; not PC or P1 evidence",
              "seal_sha256": prior.base.digest(SEAL), "rows": rows, "summary": summary,
              "full_cost_certified": False, "P1_admission": False}
    with OUTPUT.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in ("seal", "run"):
        raise SystemExit("choose seal or run")
    (seal if sys.argv[1] == "seal" else run)()
