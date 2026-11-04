"""Exposed E0 comparison of prediction-only masked v9 against v7 and CEM."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v7_e0_exposed_panel_dev_20261104_codex as panel
from experiments import pc_v7_e0_nonzero_cem_dev_20261104_codex as baseline
from experiments.pc_equal_resource_dev_20261103_codex import Meter
from experiments.spectral_reachability import pc_fd_curvature_mask_v9 as method
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = panel.source.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V9_FD_CURVATURE_MASK_E0_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v9_fd_curvature_mask_e0_dev_20261104_codex"
MANIFEST, RESULT, AUDIT = [PREFIX.with_name(PREFIX.name + s) for s in
                           ("_manifest.json", "_result.json", "_audit.json")]


def main():
    if any(p.exists() for p in (MANIFEST, RESULT, AUDIT)):
        raise FileExistsError("existing or partial run")
    sha = panel.source.sha
    if (json.loads(panel.AUDIT.read_text())["plans_sha256"] != sha(panel.OUT)
            or json.loads(baseline.AUDIT.read_text())["result_sha256"] != sha(baseline.OUT)):
        raise ValueError("source audit mismatch")
    plans, prior = (json.loads(p.read_text()) for p in (panel.OUT, baseline.OUT))
    if prior["v7_plans_sha256"] != sha(panel.OUT) or len(plans["rows"]) != 18:
        raise ValueError("source panel mismatch")
    dependencies = seal.capture([Path(__file__), PROTOCOL, Path(method.__file__), Path(method.v6.__file__), panel.OUT, panel.AUDIT,
                                 baseline.OUT, baseline.AUDIT, Path(panel.v6.__file__), Path(panel.v7.__file__),
                                 Path(panel.v6.v4.__file__), Path(panel.source.exact.__file__),
                                 Path(panel.source.exact.wall.__file__),
                                 *map(Path, plans["dependencies"]["files"]),
                                 *map(Path, prior["dependencies"]["files"])])
    panel.source.write(MANIFEST, {"dependencies": dependencies, "method": method.METHOD,
                                  "task_count": 18, "scope": "already exposed exact development"})
    print("Running v9 prediction-only FD curvature mask on the fixed exposed E0 panel.", flush=True)
    rows = []
    for original in plans["rows"]:
        task, root = original["task"], original["root"]
        goal, meter, trace = np.asarray(task["goal"]), Meter(1920), []
        def predict(actions):
            requested = np.asarray(actions, dtype=float)
            raw = requested.astype(np.float32)
            def evaluate(batch):
                pos = panel.source.exact.endpoints(task, batch)
                return pos, np.linalg.norm(pos-goal, axis=1)
            pos, scores = meter.predict(raw, evaluate)
            trace.append({"requested": requested.tolist(), "raw": raw.tolist(), "positions": pos.tolist(),
                          "scores": scores.tolist(), "tokens": len(raw)*raw.shape[1]//5})
            return pos, scores
        result = method.decide("response", "response", root, 0, goal, predict, meter, panel.core.NOMINALS)
        rows.append({"root": root, "task": task, "result": result, "counts": meter.as_dict(), "trace": trace})
        print(f"root={root}: masked {result['structure']['masked_coordinates']} columns", flush=True)
    seal.verify(dependencies)
    panel.source.write(RESULT, {"manifest_sha256": sha(MANIFEST), "rows": rows})
    samples = 0
    for original, row in zip(plans["rows"], rows, strict=True):
        if original["root"] != row["root"] or original["task"] != row["task"]:
            raise ValueError("task ordering changed")
        result, counts = row["result"], row["counts"]
        all_full, tokens, n = [], 0, 0
        for batch in row["trace"]:
            requested, raw = np.asarray(batch["requested"]), np.asarray(batch["raw"], np.float32)
            np.testing.assert_array_equal(requested.astype(np.float32), raw)
            pos = panel.source.exact.endpoints(row["task"], raw)
            scores = np.linalg.norm(pos-np.asarray(row["task"]["goal"]), axis=1)
            np.testing.assert_allclose(pos, batch["positions"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(scores, batch["scores"], atol=1e-12, rtol=0)
            cost = len(raw)*len(raw[0])//5
            if cost != batch["tokens"]:
                raise ValueError("batch token mismatch")
            tokens, n = tokens+cost, n+len(raw)
            if raw.shape[1] == 15:
                all_full.extend(scores.tolist())
        final = row["trace"][-1]
        choice = int(result["abstained"])
        np.testing.assert_array_equal(np.asarray(result["selected"]["chain"]), final["requested"][choice])
        responses = result["first_response"] + result["successor_responses"]
        masked_total = 0
        for index, response in enumerate(responses):
            batch = row["trace"][index + int(index >= 2)]
            positions = np.asarray(batch["positions"])
            difference = positions[1:11] - positions[11:21]
            raw_matrix = (difference / .1).T
            proxy = np.linalg.norm(positions[1:11]+positions[11:21]-2*positions[0], axis=1) / (
                np.linalg.norm(difference, axis=1)+1e-8)
            masked = proxy > .5
            matrix = raw_matrix.copy()
            matrix[:, masked] = 0.
            np.testing.assert_allclose(raw_matrix, response["raw_matrix"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(proxy, response["curvature_proxy"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(matrix, response["matrix"], rtol=0, atol=1e-12)
            if np.flatnonzero(masked).tolist() != response["masked_coordinates"]:
                raise ValueError("mask mismatch")
            masked_total += int(masked.sum())
        if (masked_total != result["structure"]["masked_coordinates"]
                or result["selected"]["terminal_score"] != final["scores"][choice]
                or result["selected"]["terminal_score"] > min(all_full) + 1e-12
                or tokens != counts["prediction_token_samples"] or tokens > 1920
                or n != counts["trajectory_samples"] or len(row["trace"]) != counts["prediction_calls"]
                or choice != int(result["paired_comparison"]["same_actual_float32_actions"]
                                 or final["scores"][1] <= final["scores"][0])):
            raise ValueError("selection, trace or budget mismatch")
        samples += n
    seal.verify(dependencies)
    panel.source.write(AUDIT, {"result_sha256": sha(RESULT), "verified_roots": len(rows),
                               "trajectory_samples": samples, "scope": "same exact backend replay"})
    print(f"Audited {len(rows)} roots and {samples} trajectories.", flush=True)


if __name__ == "__main__":
    main()
