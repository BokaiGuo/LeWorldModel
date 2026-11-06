"""Post-hoc first-layer linearity audit of saved v25 E0 candidate actions."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v25_fixed_bank_suffix_e0_dev_20261104_codex as panel


def candidate_error(base_action, base_feature, matrix, action, feature):
    base_action, action = np.asarray(base_action), np.asarray(action)
    base_feature, feature = np.asarray(base_feature), np.asarray(feature)
    matrix = np.asarray(matrix)
    predicted = base_feature + matrix @ (action - base_action).reshape(-1)
    observed_delta = feature - base_feature
    error = predicted - feature
    position_norm = float(np.linalg.norm(observed_delta[:2]))
    return {"predicted": predicted.tolist(), "observed": feature.tolist(),
            "position_absolute_error": float(np.linalg.norm(error[:2])),
            "position_relative_error": float(np.linalg.norm(error[:2]) / position_norm)
                if position_norm > 1e-8 else None,
            "observed_position_change": position_norm,
            "score_absolute_error": float(abs(error[2])),
            "actual_action_linf": float(np.max(np.abs(action - base_action)))}


def pairwise(records):
    concordant = discordant = tied = 0
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            predicted = left["predicted"][2] - right["predicted"][2]
            observed = left["observed"][2] - right["observed"][2]
            if abs(predicted) <= 1e-10 or abs(observed) <= 1e-10:
                tied += 1
            elif predicted * observed > 0:
                concordant += 1
            else:
                discordant += 1
    return {"concordant": concordant, "discordant": discordant, "tied": tied}


def main():
    output = panel.OUT / "candidate_linearity_posthoc.json"
    if output.exists():
        raise FileExistsError(output)
    manifest_path = panel.OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    panel.seal.verify(manifest["dependencies"])
    results = []
    for root, task in enumerate(panel.tasks()):
        row_path = panel.OUT / f"root{root:02d}_result.json"
        row = json.loads(row_path.read_text())
        if row["status"] != "completed":
            results.append({"root": root, "status": "failed"})
            continue
        path = panel.OUT / row["selection"]
        if panel.seal.digest(path) != row["selection_sha256"]:
            raise ValueError("selection hash mismatch")
        layer = json.loads(path.read_text())["layer"]
        calls = list(panel.read_calls(panel.OUT / f"root{root:02d}_generation"))
        per_nominal = []
        for response in layer["responses"]:
            nominal = response["nominal"]
            group = [(i, c) for i, c in enumerate(layer["candidates"])
                     if c["path"][0]["nominal"] == nominal]
            base = [c for _, c in group if c["path"][0]["mode"] is None]
            if response["parent"] != 0 or len(base) != 1:
                raise ValueError("unexpected first-layer nominal structure")
            base = base[0]
            base_action = np.asarray(base["chain"], np.float32)
            probe = [(actions, features, scores) for actions, features, scores in calls
                     if actions.shape == (21, 5, 2) and np.array_equal(actions[0], base_action)]
            if len(probe) != 1:
                raise ValueError("nominal finite-difference call is not unique")
            actions, features, scores = probe[0]
            axis = np.eye(10).reshape(10, 5, 2)
            expected = np.concatenate((base_action[None], base_action[None] + .05 * axis,
                                       base_action[None] - .05 * axis)).astype(np.float32)
            np.testing.assert_array_equal(actions, expected)
            augmented = np.column_stack((features, scores))
            matrix = (augmented[1:11] - augmented[11:21]).T / .1
            np.testing.assert_allclose(matrix, response["matrix"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(augmented[0], base["feature"], atol=1e-12, rtol=0)
            errors = []
            for index, candidate in group:
                if candidate["path"][0]["mode"] is None:
                    continue
                entry = candidate_error(base_action, base["feature"], matrix,
                                        candidate["chain"], candidate["feature"])
                entry.update(index=index, mode=candidate["path"][0]["mode"],
                             requested_scale=candidate["path"][0]["scale"])
                errors.append(entry)
            per_nominal.append({"nominal": nominal, "candidates": errors,
                                "score_pairs_within_nominal": pairwise(errors)})
        results.append({"root": root, "status": "completed", "nominals": per_nominal})
    panel.seal.verify(manifest["dependencies"])
    panel.OLD.writer(output, {"scope": "post-hoc, first layer, same exact dynamics; not PC1",
        "manifest_sha256": panel.seal.digest(manifest_path),
        "analysis_sha256": panel.seal.digest(Path(__file__)), "rows": results})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
