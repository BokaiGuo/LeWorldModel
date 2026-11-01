"""CPU reconstruction of sealed E1 arbitrary targets; not a historical event receipt."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "results/agent_experiment1_common_controller_20261028_v1_prepared_MAIN01.json"
NEW = ROOT / "results/agent_e1_six_slot_common_prepare_20261029_v1_prepare_MAIN01.json"
SOURCE = ROOT / "experiments/spectral_reachability/agent_experiment1_common_controller_20261028_v1.py"
OLD_SHA = "8c472485d44c40ce3d95f8043d2c58f9ec2694fef74786a840d1699d5a1a86bd"
NEW_SHA = "cd96f6e28b25d3b7e3b6649c760b2f111747aec5c0ce9c8a41741318209df8a0"
SOURCE_SHA = "c34a303c5e2df56a2002f195118cc2272a7b4d10afb79f8ed63d730af1a732a9"
TASKS = (
    ("task_00.json.gz", "a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b"),
    ("task_01.json.gz", "101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74"),
)
TASK_DIR = ROOT / "results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01"
OUTPUT = ROOT / "results/codex_e1_arbitrary_reconstruction_20261101_exclusive.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_bound(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f"SHA mismatch: {path}")
    return raw


def reconstruct(task, root):
    z = np.asarray(task["root_context"]["latent_history"][-1], dtype=np.float64)
    goal = np.asarray(task["goal"], dtype=np.float64)
    if any(x.shape != (192,) or not np.isfinite(x).all() or
           not np.array_equal(x, x.astype(np.float32)) for x in (z, goal)):
        raise ValueError("invalid input latent")
    seed = (2028102801, 2028102802)[root]
    draws = np.random.Generator(np.random.PCG64(seed)).normal(size=(20, 192))
    scale = float(np.linalg.norm(goal - z) / np.sqrt(192))
    points = (z + scale * draws).astype(np.float32)
    costs = np.mean((points.astype(float) - goal) ** 2, axis=1)
    winner = min(range(20), key=lambda i: (float(costs[i]), i))
    return dict(seed=seed, draws=draws.tolist(), scale=scale,
                candidates=points.astype(float).tolist(), costs=costs.tolist(),
                winner_index=winner, target=points[winner].astype(float).tolist(),
                model_calls=0, learned=False)


def audit(old, new, tasks):
    rows = []
    if len(old["slots"]) != 2 or len(old["targets"]) != 2 or len(new["rows"]) != 6 or len(tasks) != 2:
        raise ValueError("denominator")
    for root, task in enumerate(tasks):
        slot, summary, target = old["slots"][root], old["targets"][root], new["rows"][3 * root]
        if (slot["source_slot"] != root or summary["source_slot"] != root or
            task["source_slot"] != root or task["task_id"] != slot["task_id"] or
            target["root_slot"] != root or target["arm"] != "arbitrary_latent_waypoint" or
            target["task_sha256"] != slot["task_sha256"] or
            target["task_id"] != slot["task_id"] or target["blind_sha256"] != slot["blind_sha256"]):
            raise ValueError("source identity")
        result = reconstruct(task["task"], root)
        digest = sha(json.dumps(result, sort_keys=True, allow_nan=False).encode())
        raw = np.asarray(result["target"], dtype="<f4").tobytes()
        if (digest != summary["arbitrary_sha256"] or
            result["target"] != summary["targets"]["arbitrary_latent_waypoint"] or
            raw != base64.b64decode(target["target_f32_le_base64"], validate=True) or
            sha(raw) != target["target_data_sha256"]):
            raise ValueError("reconstruction or target mismatch")
        rows.append(dict(root=root, winner_index=result["winner_index"], candidate_count=20,
                         task_sha256=slot["task_sha256"], reconstruction_sha256=digest,
                         target_sha256=sha(raw)))
    return dict(schema="codex.e1.arbitrary_reconstruction.v1", status="partial",
                rows=rows, producer_source_sha256=SOURCE_SHA,
                interpretation="deterministic CPU reconstruction, not historical execution journal",
                generation_model_calls_in_source=0, historical_receipt=None,
                blockers=["historical_attempt_failure_and_retry_events_unavailable",
                          "inherited_root_and_goal_generation_cost_not_allocated",
                          "shared_cost_and_cross_stage_overlap_not_certified"],
                equal_budget_certified=False, ranking_authorized=False, scientific_admission=False)


def load():
    old = json.loads(read_bound(OLD, OLD_SHA))
    new = json.loads(read_bound(NEW, NEW_SHA))
    read_bound(SOURCE, SOURCE_SHA)
    if old["bindings_sha256"].get(str(SOURCE)) != SOURCE_SHA:
        raise ValueError("producer source binding")
    tasks = []
    for root, (name, digest) in enumerate(TASKS):
        path = TASK_DIR / name
        if old["slots"][root]["task_export"] != str(path) or old["slots"][root]["task_sha256"] != digest:
            raise ValueError("task binding")
        tasks.append(json.loads(gzip.decompress(read_bound(path, digest))))
    return old, new, tasks


def main():
    result = audit(*load())
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"output": str(OUTPUT), "sha256": sha(raw), "status": result["status"]}))


if __name__ == "__main__":
    main()
