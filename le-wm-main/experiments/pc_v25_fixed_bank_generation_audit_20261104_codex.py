"""Post-run audit binding frozen first-layer choices to saved generation calls."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v25_fixed_bank_suffix_e0_dev_20261104_codex as panel
from experiments.spectral_reachability import pc_recursive_reachability_v25 as core


def audit():
    output = panel.OUT / "generation_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((panel.OUT / "manifest.json").read_text())
    panel.seal.verify(manifest["dependencies"])
    results = []
    for root, task in enumerate(panel.tasks()):
        row = json.loads((panel.OUT / f"root{root:02d}_result.json").read_text())
        if row["status"] != "completed":
            results.append({"root": root, "status": "failed"})
            continue
        path = panel.OUT / row["selection"]
        if panel.seal.digest(path) != row["selection_sha256"]:
            raise ValueError("selection hash mismatch")
        selection = json.loads(path.read_text())
        layer = selection["layer"]
        candidates = layer["candidates"]
        bank = np.asarray([item["chain"] for item in candidates], np.float32)
        calls = list(panel.read_calls(panel.OUT / f"root{root:02d}_generation"))
        matching = [(features, scores) for actions, features, scores in calls
                    if actions.shape == bank.shape and np.array_equal(actions, bank)]
        if len(matching) != 1:
            raise ValueError("candidate bank not uniquely bound to generation call")
        features, scores = matching[0]
        expected_features = np.column_stack((features, scores))
        np.testing.assert_allclose(expected_features, [x["feature"] for x in candidates],
                                   atol=1e-12, rtol=0)
        np.testing.assert_allclose(scores, [x["score"] for x in candidates], atol=1e-12, rtol=0)
        for probe in layer["reachability"]:
            index = probe["candidate"]
            matches = [(actions, features, scores) for actions, features, scores in calls
                       if actions.shape == (9, 15, 2)
                       and np.all(actions[:, :5] == bank[index])
                       and np.all(actions[0, 5:] == 0)]
            if len(matches) != 1:
                raise ValueError("successor probe not uniquely bound")
            actions, features, scores = matches[0]
            # Verify the exact packet-constant centered finite difference coordinates.
            axis = []
            for offset in (0, 5):
                for action in range(2):
                    direction = np.zeros((10, 2))
                    direction[offset:offset + 5, action] = 1 / np.sqrt(5)
                    axis.append(direction)
            expected = np.concatenate((np.zeros((1, 10, 2)), .05 * np.array(axis),
                                       -.05 * np.array(axis))).astype(np.float32)
            np.testing.assert_array_equal(actions[:, 5:], expected)
            augmented = np.column_stack((features, scores))
            matrix = (augmented[1:5] - augmented[5:9]).T / .1
            np.testing.assert_allclose(matrix, probe["matrix"], atol=1e-12, rtol=0)
            residual = np.r_[task["goal"], 0.] - augmented[0]
            solution = matrix.T @ np.linalg.solve(matrix @ matrix.T + .1 * np.eye(3), residual)
            error = residual - matrix @ solution
            value = error @ error + .1 * (solution @ solution)
            np.testing.assert_allclose(value, probe["value"], atol=1e-12, rtol=0)
        alternatives = core.retention_choices(scores=[item["score"] for item in candidates],
            features=[item["feature"] for item in candidates], reachability=layer["reachability"],
            count=4, greedy_slots=2, seed=2030401000 + root)
        if alternatives != layer["retention_alternatives"]:
            raise ValueError("retention recomputation mismatch")
        results.append({"root": root, "status": "passed", "candidates": len(bank),
                        "response_probes": len(layer["reachability"])})
    panel.seal.verify(manifest["dependencies"])
    panel.OLD.writer(output, {"status": "passed", "rows": results,
        "scope": "generation call and selector binding; same model and environment"})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    audit()
