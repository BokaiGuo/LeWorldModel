"""Independent reconstruction and same-simulator audit of exposed fixed-pool selection."""
import json
from pathlib import Path

import h5py
import numpy as np

from experiments.pc_input_permissions import file_hash
from experiments import pc_tworoom_nonlinear_selection_20261107_codex as run

OUTPUT = run.source.ROOT / "results/pc_tworoom_nonlinear_selection_audit_20261107_codex.json"


def reconstruct_choice(embeddings, forecasts, offsets, scores):
    """Use the distance identity and stable ordered comparisons, not the runner's nearest."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    forecasts = np.asarray(forecasts, dtype=np.float64)
    offsets = np.asarray(offsets)
    scores = np.asarray(scores, dtype=np.float64)
    if (embeddings.ndim != 2 or forecasts.ndim != 2
            or embeddings.shape[1] != forecasts.shape[1]
            or offsets.shape != (len(embeddings), 2)
            or scores.shape != (len(forecasts),)
            or not len(embeddings) or not len(forecasts)
            or not all(np.isfinite(x).all() for x in (embeddings, forecasts, offsets, scores))):
        raise ValueError("invalid frozen inputs")
    distances = ((forecasts * forecasts).sum(axis=1)[:, None]
                 + (embeddings * embeddings).sum(axis=1)[None, :]
                 - 2 * forecasts @ embeddings.T)
    nearest = np.argmin(distances, axis=1)
    decoded = offsets[nearest]
    rank = sorted(range(len(scores)), key=lambda i: (float(np.square(decoded[i]).sum()), scores[i], i))
    return rank[0], decoded


def image_difference(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(f"image shape mismatch: {left.shape}, {right.shape}")
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {"mae": float(delta.mean()), "max": int(delta.max()),
            "differing_values": int(np.count_nonzero(delta))}


def audit():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    plan, physical, summary, previous = (
        json.loads(path.read_text()) for path in (run.PLAN, run.PHYSICAL, run.RESULT, run.source.PLAN)
    )
    if (physical["plan_sha256"] != file_hash(run.PLAN)
            or summary["plan_sha256"] != file_hash(run.PLAN)
            or summary["physical_sha256"] != file_hash(run.PHYSICAL)
            or plan["runner_sha256"] != file_hash(run.__file__)
            or plan["source_sha256"] != file_hash(run.source.PLAN)
            or plan["features_sha256"] != file_hash(run.comparison.FEATURES)
            or plan["chart_dictionary_sha256"] != file_hash(run.chart.DICTIONARY)):
        raise ValueError("source hash mismatch")
    scaler, normalizer = run.action_scaler()
    if normalizer != previous["normalizer_action_sha256"]:
        raise ValueError("normalizer drift")
    env = gym.make("swm/TwoRoom-v1", render_mode="rgb_array")
    rows = []
    try:
        with (np.load(run.chart.DICTIONARY, allow_pickle=False) as dictionaries,
              np.load(run.comparison.FEATURES, allow_pickle=False) as features,
              np.load(run.source.inputs.INPUT, allow_pickle=False) as inputs,
              h5py.File(run.source.common.DATA, "r") as data):
            for root, (selected, replay, source_row) in enumerate(
                    zip(plan["rows"], physical["rows"], previous["rows"], strict=True)):
                if (selected["root"] != root or selected["row"] != replay["row"]
                        or selected["row"] != source_row["row"]
                        or selected["row"] != int(inputs["rows"][root])):
                    raise ValueError("root identity mismatch")
                scores = features[f"root_{root}_scores"][0]
                if selected["valid"]:
                    index, decoded = reconstruct_choice(
                        dictionaries[f"root_{root}_embedding"], features[f"root_{root}_predicted"],
                        dictionaries[f"root_{root}_offsets"], scores)
                    if decoded[index].tolist() != selected["selected_offset"]:
                        raise ValueError("selected decoded offset mismatch")
                else:
                    index = int(np.argmin(scores))
                if index != selected["candidate_index"]:
                    raise ValueError("independent candidate selection mismatch")
                chains = [item[1] for item in run.candidates_for(source_row)]
                for ref in source_row["refinements"]:
                    chains.extend(ref["actions"])
                if len(chains) != 656:
                    raise ValueError("candidate count mismatch")
                actions = run.source.to_physical(np.asarray(chains[index], dtype=np.float32), scaler)
                np.testing.assert_array_equal(actions, np.asarray(selected["actions"], dtype=np.float32))
                env.reset(seed=42)
                actor = env.unwrapped
                actor._set_state(np.asarray(data["proprio"][selected["row"]]))
                actor._set_goal_state(np.asarray(data["proprio"][selected["row"] + 25]))
                current_diff = image_difference(inputs["current"][root], actor.render())
                goal_diff = image_difference(inputs["goal"][root], actor._target_img.cpu().numpy().transpose(1, 2, 0))
                max_error = 0.0
                for action, trace in zip(actions, replay["trace"], strict=True):
                    _, _, terminated, truncated, info = actor.step(action.copy())
                    max_error = max(max_error, abs(float(info["distance_to_target"]) - trace["distance"]))
                    if (bool(terminated), bool(truncated)) != (trace["terminated"], trace["truncated"]):
                        raise ValueError("termination mismatch")
                    if bool(terminated) != (float(info["distance_to_target"]) < 16.0):
                        raise ValueError("environment success predicate changed")
                if max_error > 1e-6:
                    raise ValueError("physical replay mismatch")
                rows.append({"root": root, "row": selected["row"], "index": index,
                             "current_rgb_difference": current_diff, "goal_rgb_difference": goal_diff,
                             "max_replay_distance_error": max_error})
    finally:
        env.close()
    with OUTPUT.open("x") as stream:
        json.dump({"rows": rows, "plan_sha256": file_hash(run.PLAN),
                   "physical_sha256": file_hash(run.PHYSICAL), "result_sha256": file_hash(run.RESULT),
                   "runner_sha256": file_hash(Path(__file__)),
                   "scope": "exposed roots and same simulator; pixel differences are not hidden-state certification"},
                  stream, indent=2, allow_nan=False)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    audit()
