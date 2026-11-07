"""Actually query only retained B tails in the sealed RGB retention panel."""
import json
import time

import numpy as np
import torch

from experiments import pc_tworoom_common_tail_retention_20261107_codex as base
from experiments.pc_input_permissions import file_hash
from experiments.pc_prefix_novel_actual_encoding_20261105_codex import model_fingerprint

PROTOCOL = base.ROOT.parent / "docs/PC_TWOROOM_SPARSE_RETENTION_QUERY_PROTOCOL_20261107_CODEX.md"
OUTPUT = base.ROOT / "results/pc_tworoom_sparse_retention_query_20261107_codex.json"
ARMS = ("lookahead", "descriptor", "shuffled")


def sparse_search(prefixes, tails, order, score):
    """score receives only the explicit list of (prefix, tail) action pairs."""
    pool = base.chains(prefixes, tails)
    a_indices = [(p, t) for p in range(13) for t in range(5)]
    a = np.asarray(score(a_indices, np.stack([pool[p, t] for p, t in a_indices])), float)
    if a.shape != (65,) or not np.isfinite(a).all():
        raise ValueError("A score shape or values invalid")
    a = a.reshape(13, 5)
    retained = base.retain(a, order)
    out = {}
    for arm in ARMS:
        kept = retained[arm]
        indices = [(p, t) for p in kept for t in range(5, 9)]
        b = np.asarray(score(indices, np.stack([pool[p, t] for p, t in indices])), float)
        if b.shape != (8,) or not np.isfinite(b).all():
            raise ValueError("B score shape or values invalid")
        local = np.unravel_index(np.argmin(b), (2, 4))
        out[arm] = {"retained": kept, "b_indices": indices, "b_scores": b.tolist(),
                    "selected": [kept[local[0]], int(local[1])]}
    return a, out


def run():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sealed = json.loads(base.PLAN.read_text())
    for key, path in (("public_sha256", base.PANEL.PUBLIC), ("protocol_sha256", base.PROTOCOL),
                      ("runner_sha256", base.__file__), ("wide_sha256", base.wide.__file__),
                      ("repair_sha256", base.wide.repair.__file__)):
        if sealed[key] != file_hash(path):
            raise ValueError(f"source drift: {key}")
    model, scaler, transform, normalizer, boundary = base.wide.setup()
    fingerprint = model_fingerprint(model)
    if fingerprint != sealed["model_sha256"] or normalizer != sealed["normalizer_sha256"]:
        raise ValueError("model drift")
    rows = []
    print("Query A first, then only retained B for each arm; compare to sealed matrix afterwards.", flush=True)
    with np.load(base.PANEL.PUBLIC, allow_pickle=False) as public, torch.inference_mode():
        for i, old in enumerate(sealed["rows"]):
            if old["seed"] != int(public["seeds"][i]):
                raise ValueError("root mismatch")
            prefixes, tails = base.libraries(public["current"][i], public["goal"][i])
            np.testing.assert_array_equal(prefixes, np.asarray(old["prefixes"], np.float32))
            np.testing.assert_array_equal(tails, np.asarray(old["tails"], np.float32))
            batches = []

            def score(indices, actions):
                values = []
                for begin in range(0, len(actions), 32):
                    part = actions[begin:begin + 32]
                    normal = scaler.transform(part.reshape(-1, 2)).reshape(1, len(part), 5, 10)
                    context = {"pixels": transform(public["current"][i])[None, None, None].cuda(),
                               "goal": transform(public["goal"][i])[None, None, None].cuda(),
                               "action": torch.empty((1, 1, 0, 10), device="cuda")}
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    cost, _ = boundary.score(context, torch.as_tensor(normal, dtype=torch.float32, device="cuda"))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    values.extend(cost[0].cpu().tolist())
                    batches.append({"indices": indices[begin:begin + len(part)], "chains": len(part),
                                    "packets": 5 * len(part), "score_seconds": elapsed})
                return values

            a, arms = sparse_search(prefixes, tails, np.asarray(old["permutation"]), score)
            old_cost = np.asarray(old["costs"], float)
            for arm in ARMS:
                record = arms[arm]
                reference_b = [old_cost[p, t] for p, t in record["b_indices"]]
                record.update({"retained_matches": record["retained"] == old["retained"][arm],
                               "selected_matches": record["selected"] == old["selected"][arm],
                               "max_b_cost_delta": float(np.max(np.abs(np.asarray(record["b_scores"]) - reference_b)))})
            rows.append({"seed": int(old["seed"]), "a_scores": a.tolist(), "arms": arms,
                         "max_a_cost_delta": float(np.max(np.abs(a - old_cost[:, :5]))), "batches": batches})
            print(f"root={i}: 65 A plus 8 B per arm", flush=True)
    if len(rows) != 40 or model_fingerprint(model) != fingerprint:
        raise ValueError("model/panel drift")
    physical = json.loads(base.RESULT.read_text())
    if physical["plan_sha256"] != file_hash(base.PLAN) or physical["truth_sha256"] != file_hash(base.PANEL.TRUTH):
        raise ValueError("physical artifact drift")
    for row, observed in zip(rows, physical["rows"], strict=True):
        if row["seed"] != observed["seed"]:
            raise ValueError("physical root mismatch")
        trace = np.asarray(observed["physical"], dtype=object)
        for arm, info in row["arms"].items():
            p, t = info["selected"]
            info["success_from_prior_physical_replay"] = bool(np.asarray(trace[p, t + 5, :, 1], bool).any())
    output = {"track": "primary_rgb", "protocol_sha256": file_hash(PROTOCOL),
              "runner_sha256": file_hash(__file__), "plan_sha256": file_hash(base.PLAN),
              "physical_sha256": file_hash(base.RESULT), "model_sha256": fingerprint,
              "normalizer_sha256": normalizer, "rows": rows,
              "scope": "exposed task and candidate pool; prior physical replay; shared A timing"}
    with OUTPUT.open("x") as f:
        json.dump(output, f, allow_nan=False)


if __name__ == "__main__":
    run()
