"""Independent implementation of root-20 intervention and CEM reconstruction."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_response_temporal_v3_tail_local_mechanism_20261104_codex as panel

OUTPUT = panel.PREFIX.with_name(panel.PREFIX.name + "_witness_independent_audit.json")


def score(task, actions):
    endpoints = panel.prior.base.exact.endpoints(task, np.asarray(actions, dtype=np.float32))
    return np.linalg.norm(endpoints - np.asarray(task["goal"]), axis=1)


def independent_cem(task, prefix, seed):
    rng = np.random.default_rng(seed)
    loc, scale = np.zeros((4, 2)), np.full((4, 2), .35)
    incumbent = np.zeros((4, 2))
    best = np.concatenate((prefix, np.zeros((10, 2))))
    best_score = float(score(task, best[None])[0])
    knot_times, times = np.linspace(0, 9, 4), np.arange(10)
    for start in range(0, 119, 12):
        n = min(12, 119 - start)
        params = loc + rng.normal(size=(n, 4, 2)) * scale
        params[0] = incumbent
        interpolated = np.stack([np.stack([np.interp(times, knot_times, p[:, j]) for j in range(2)], axis=-1)
                                 for p in params])
        suffix = np.clip(interpolated, -1, 1)
        chains = np.concatenate((np.broadcast_to(prefix, (n, 5, 2)), suffix), axis=1)
        values = score(task, chains)
        winner = int(np.argmin(values))
        if values[winner] < best_score:
            best, best_score, incumbent = chains[winner].copy(), float(values[winner]), params[winner].copy()
        if n > 1:
            elites = params[np.argsort(values, kind="stable")[:min(4, n)]]
            loc, scale = elites.mean(axis=0), np.maximum(elites.std(axis=0), .07)
    return best, best_score


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    panel.configure()
    prior = json.loads(panel.INTERVENTION.read_text())
    rebuild = json.loads(panel.REBUILD.read_text())
    plans, audited = json.loads(panel.PLANS.read_text()), json.loads(panel.AUDIT.read_text())
    if (prior["plans_sha256"] != panel.prior.base.sha(panel.PLANS)
            or audited["plans_sha256"] != panel.prior.base.sha(panel.PLANS)
            or rebuild["intervention_sha256"] != panel.prior.base.sha(panel.INTERVENTION)
            or prior["dependencies"] != panel.dependencies()):
        raise ValueError("source seal mismatch")
    witnesses = [row for row in prior["rows"] if row["tail_all_intersection"]]
    if len(witnesses) != 1 or witnesses[0]["root"] != 20:
        raise ValueError("unexpected witness set")
    witness, root = witnesses[0], 20
    entries = {(r["root"], r["arm"]): r for r in plans["rows"]}
    selected = entries[root, "response"]
    raw_chain = np.asarray(selected["result"]["selected"]["chain"], dtype=np.float32)
    if not np.array_equal(raw_chain, witness["selected_chain_float32"]):
        raise ValueError("selected chain differs from sealed plan")
    value = float(score(witness["task"], raw_chain[None])[0])
    if abs(value - selected["result"]["selected"]["terminal_score"]) > 1e-12:
        raise ValueError("selected score mismatch")
    first = selected["result"]
    greedy = int(np.argmin(first["prefix_scores"]))
    expected = {"selected": raw_chain[:5], "zero": np.zeros((5, 2)),
                "greedy": np.asarray(first["first_bank"][greedy]["prefix"]),
                "random_bridge": np.asarray(entries[root, "matched_random"]["result"]["selected"]["chain"][:5]),
                "lookahead": np.asarray(entries[root, "smooth_cem"]["result"]["selected"]["chain"][:5])}
    task = witness["task"]
    suffixes = [np.tile(action, (10, 1)) for action in
                ((0., 0.), (.5, 0.), (-.5, 0.), (.5, .5), (.5, -.5))]
    suffixes.append(raw_chain[5:])
    arms, matrices = {}, {}
    wall, axes = panel.prior.base.exact.wall, np.eye(10).reshape(10, 5, 2)
    nominal = raw_chain[5:].copy()
    nominal[:5] = np.clip(nominal[:5], -.9, .9)
    np.testing.assert_array_equal(nominal, witness["nominal_tail"])
    for name, prefix in expected.items():
        recorded = witness["controls"][name]
        np.testing.assert_array_equal(prefix, recorded["prefix"])
        common = score(task, np.stack([np.concatenate((prefix, tail)) for tail in suffixes]))
        np.testing.assert_allclose(common, recorded["common_distances"], rtol=0, atol=1e-12)
        own_chain, own_value = independent_cem(task, prefix, 2027111400 + root)
        np.testing.assert_array_equal(own_chain, recorded["own_chain"])
        np.testing.assert_allclose(own_value, recorded["own_distance"], rtol=0, atol=1e-12)
        matrices[name], stable = {}, {}
        for eps in (.01, .05):
            tails = np.broadcast_to(nominal, (21, 10, 2)).astype(float).copy()
            tails[1:11, :5] += eps * axes
            tails[11:21, :5] -= eps * axes
            actions = np.concatenate((np.broadcast_to(prefix, (21, 5, 2)), tails), axis=1).astype(np.float32)
            padded = np.pad(actions, ((0, 0), (0, 1), (0, 0))) * wall.HIGH
            trajectory, blocked = wall.rollout(task["start"], padded)
            blocked = np.asarray(blocked)[:, :15]
            matrices[name][str(eps)] = ((trajectory[1:11, 15, :2] - trajectory[11:21, 15, :2]) / (2 * eps)).T
            stable[str(eps)] = bool(np.all(blocked == blocked[0]))
            record = recorded["tail_local_response"][str(eps)]
            np.testing.assert_allclose(matrices[name][str(eps)], record["response"], rtol=0, atol=1e-12)
            np.testing.assert_array_equal(blocked, record["blocked"])
            if not stable[str(eps)]:
                raise ValueError("unstable blocker sequence")
        arms[name] = {"common_distances": common.tolist(), "own_distance": own_value, "stable": stable}
    initial = np.linalg.norm(np.asarray(task["start"])[:2] - task["goal"])
    if witness["controls"]["selected"]["prefix_distance"] < initial - 1e-10:
        raise ValueError("selected prefix progressed")
    gaps = {}
    for name in ("zero", "greedy", "random_bridge", "lookahead"):
        selected_common, other = arms["selected"]["common_distances"], arms[name]["common_distances"]
        if (selected_common[-1] - selected_common[0] >= other[-1] - other[0] - 1e-10
                or arms["selected"]["own_distance"] >= arms[name]["own_distance"] - 1e-10):
            raise ValueError("fixed or optimized suffix comparison failed")
        gaps[name] = {str(eps): float(np.linalg.norm(matrices["selected"][str(eps)] - matrices[name][str(eps)]))
                      for eps in (.01, .05)}
        if min(gaps[name].values()) <= 1e-6:
            raise ValueError("tail response gap too small")
    output = {"root": root, "plans_sha256": panel.prior.base.sha(panel.PLANS),
              "intervention_sha256": panel.prior.base.sha(panel.INTERVENTION),
              "audit_script_sha256": panel.prior.base.sha(Path(__file__)),
              "selected_score": value, "tails": arms, "response_gaps": gaps,
              "limitations": ["shares exact wall dynamics and numpy", "not independent data or model replication"]}
    panel.prior.base.write(OUTPUT, output)
    print("Independent root-20 common suffix and separately implemented CEM reconstruction passed.", flush=True)


if __name__ == "__main__":
    main()
