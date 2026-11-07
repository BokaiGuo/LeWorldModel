"""RGB common-tail retention: model-only choices followed by physical replay."""
import json
import sys

import numpy as np
import torch

from experiments import pc_tworoom_wide_offset_alignment_20261107_codex as wide
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

ROOT, PANEL = wide.ROOT, wide.PANEL
PROTOCOL = ROOT.parent / "docs/PC_TWOROOM_COMMON_TAIL_RETENTION_PROTOCOL_20261107_CODEX.md"
PLAN = ROOT / "results/pc_tworoom_common_tail_retention_plan_20261107_codex.json"
RESULT = ROOT / "results/pc_tworoom_common_tail_retention_result_20261107_codex.json"


def libraries(current, goal):
    prefixes = wide.candidate_pool(current, goal)[:, :10]
    start, target, _, door = wide.repair.v2.v1.geometry(current, goal)
    anchor = start + 5. * prefixes[0].astype(float).sum(axis=0)
    tails = []
    for offset in (-12., -4., 4., 12., -8., -2., 2., 8.):
        points = [[target[0], door + offset], target]
        action, _ = wide.repair.waypoint_actions(anchor, points, 1., 15)
        tails.append(action)
    return prefixes, np.stack(tails[:4] + [np.zeros((15, 2), np.float32)] + tails[4:])


def chains(prefixes, tails):
    prefixes, tails = np.asarray(prefixes), np.asarray(tails)
    if prefixes.shape != (13, 10, 2) or tails.shape != (9, 15, 2):
        raise ValueError("invalid library shapes")
    out = np.concatenate((np.broadcast_to(prefixes[:, None], (13, 9, 10, 2)),
                          np.broadcast_to(tails[None], (13, 9, 15, 2))), axis=2)
    if not np.isfinite(out).all() or np.abs(out).max() > 1:
        raise ValueError("invalid actions")
    return out


def retain(a_cost, order):
    a_cost, order = np.asarray(a_cost, dtype=float), np.asarray(order)
    if (a_cost.shape != (13, 5) or not np.isfinite(a_cost).all()
            or order.shape != (13,) or not np.array_equal(np.sort(order), np.arange(13))):
        raise ValueError("invalid A matrix or permutation")
    opportunity = a_cost[:, :4].min(axis=1)
    ranked = np.argsort(opportunity, kind="stable")
    anchor = int(ranked[0])
    vectors = a_cost[:, 4, None] - a_cost[:, :4]

    def diversity(v):
        distances = ((v - v[anchor]) ** 2).sum(axis=1)
        distances[anchor] = -1.
        return [anchor, int(np.argmax(distances))]

    return {"lookahead": ranked[:2].tolist(), "descriptor": diversity(vectors),
            "shuffled": diversity(vectors[order])}


def final_choice(b_cost, indices):
    b_cost = np.asarray(b_cost, dtype=float)
    if b_cost.shape != (13, 4) or not np.isfinite(b_cost).all() or len(set(indices)) != 2:
        raise ValueError("invalid B scores or retention")
    if any(not 0 <= int(i) < 13 for i in indices):
        raise ValueError("invalid prefix index")
    local = np.unravel_index(np.argmin(b_cost[indices]), (2, 4))
    return [int(indices[local[0]]), int(local[1])]


def plan():
    if PLAN.exists():
        raise FileExistsError(PLAN)
    ref = json.loads(PANEL.REFERENCE.read_text())
    if ref["public_sha256"] != file_hash(PANEL.PUBLIC):
        raise ValueError("public input drift")
    model, scaler, transform, normalizer, boundary = wide.setup()
    fingerprint = model_fingerprint(model)
    rows = []
    print("Score common A/B tails and seal retention choices without physical labels.", flush=True)
    with np.load(PANEL.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for i, seed in enumerate(public["seeds"]):
            prefixes, tails = libraries(public["current"][i], public["goal"][i])
            full = chains(prefixes, tails).reshape(117, 25, 2)
            scores = []
            for begin in range(0, 117, 32):
                part = full[begin:begin+32]
                normal = scaler.transform(part.reshape(-1, 2)).reshape(1, len(part), 5, 10)
                context = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                           "goal": transform(public["goal"][i])[None, None, None].cuda(),
                           "action": torch.empty((1, 1, 0, 10), device="cuda")}
                cost, _ = boundary.score(context, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                scores.extend(cost[0].cpu().tolist())
            cost = np.asarray(scores).reshape(13, 9)
            order = np.random.default_rng(20261107 + i).permutation(13)
            kept = retain(cost[:, :5], order)
            selected = {name: final_choice(cost[:, 5:], idx) for name, idx in kept.items()}
            rows.append({"seed": int(seed), "prefixes": prefixes.tolist(), "tails": tails.tolist(),
                         "costs": cost.tolist(), "permutation": order.tolist(),
                         "retained": kept, "selected": selected})
            print(f"root={i} retained={kept}", flush=True)
    if len(rows) != 40 or model_fingerprint(model) != fingerprint:
        raise ValueError("panel/model drift")
    with PLAN.open("x") as f:
        json.dump({"track": "primary_rgb", "rows": rows, "model_sha256": fingerprint,
                   "normalizer_sha256": normalizer, "public_sha256": file_hash(PANEL.PUBLIC),
                   "protocol_sha256": file_hash(PROTOCOL), "runner_sha256": file_hash(__file__),
                   "wide_sha256": file_hash(wide.__file__), "repair_sha256": file_hash(wide.repair.__file__)},
                  f, separators=(",", ":"), allow_nan=False)


def replay():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if RESULT.exists():
        raise FileExistsError(RESULT)
    sealed = json.loads(PLAN.read_text())
    for key, path in (("public_sha256", PANEL.PUBLIC), ("protocol_sha256", PROTOCOL),
                      ("runner_sha256", __file__), ("wide_sha256", wide.__file__),
                      ("repair_sha256", wide.repair.__file__)):
        if sealed[key] != file_hash(path):
            raise ValueError(f"plan drift: {key}")
    reference = json.loads(PANEL.REFERENCE.read_text())
    if reference["truth_sha256"] != file_hash(PANEL.TRUTH):
        raise ValueError("truth drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    print("Physical replay of all sealed common-tail chains; no labels influence selection.", flush=True)
    try:
        with np.load(PANEL.PUBLIC, allow_pickle=False) as public, np.load(PANEL.TRUTH, allow_pickle=False) as truth:
            for i, row in enumerate(sealed["rows"]):
                if row["seed"] != int(public["seeds"][i]):
                    raise ValueError("seed mismatch")
                pool = chains(np.asarray(row["prefixes"]), np.asarray(row["tails"]))
                physical = []
                for prefix_chains in pool:
                    row_traces = []
                    for chain in prefix_chains:
                        env.reset(seed=row["seed"], options={"state": truth["initial"][i], "target_state": truth["target"][i]})
                        actor = env.unwrapped
                        np.testing.assert_array_equal(actor.render(), public["current"][i])
                        trace = []
                        for action in chain:
                            _, _, term, trunc, info = actor.step(np.asarray(action, np.float32))
                            trace.append([float(info["distance_to_target"]), bool(term), bool(trunc)])
                        row_traces.append(trace)
                    physical.append(row_traces)
                success = np.asarray(physical, dtype=object)[:, 5:, :, 1].astype(bool).any(axis=2)
                prefix_success = np.asarray(physical, dtype=object)[:, 5:, :10, 1].astype(bool).any(axis=2)
                outcomes = {}
                for name, (prefix, tail) in row["selected"].items():
                    outcomes[name] = {"success": bool(success[prefix, tail]),
                                      "prefix_success": bool(prefix_success[prefix, tail]),
                                      "useful_prefixes_retained": int(success[row["retained"][name]].any(axis=1).sum())}
                rows.append({"seed": row["seed"], "physical": physical, "outcomes": outcomes,
                             "B_success_prefixes": int(success.any(axis=1).sum())})
                print(f"root={i} outcomes={outcomes}", flush=True)
    finally:
        env.close()
    if len(rows) != 40:
        raise ValueError("panel drift")
    with RESULT.open("x") as f:
        json.dump({"plan_sha256": file_hash(PLAN), "truth_sha256": file_hash(PANEL.TRUTH),
                   "scope": "exposed physical replay, not independent or online", "rows": rows},
                  f, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("plan", "replay"):
        raise SystemExit("expected plan|replay")
    {"plan": plan, "replay": replay}[sys.argv[1]]()
