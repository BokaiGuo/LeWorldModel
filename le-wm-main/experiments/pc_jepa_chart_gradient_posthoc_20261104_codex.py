"""Compare saved chart-distance and full-score finite difference gradients."""
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc_jepa_native_v12_contact_feedback_dev_20261104_codex"
PROTOCOL = ROOT.parent / "docs/PC_JEPA_CHART_GRADIENT_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUT = ROOT / "results/pc_jepa_chart_gradient_posthoc_20261104_codex.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    audit_path, summary_path = SOURCE / "audit.json", SOURCE / "summary.json"
    response_path = SOURCE / "response_result.json"
    audit, summary, response = (json.loads(p.read_text()) for p in (audit_path, summary_path, response_path))
    if (audit["status"] != "saved_data_audit_passed" or audit["summary_sha256"] != seal.digest(summary_path)
            or summary["results"]["response"]["status"] != "completed" or response["status"] != "completed"):
        raise ValueError("source audit mismatch")
    decisions = [SOURCE / f"response_{step}/output/decision.json" for step in range(3)]
    deps = seal.capture([Path(__file__), PROTOCOL, audit_path, summary_path, response_path, *decisions])
    rows = []
    axis = np.eye(10).reshape(10, 5, 2)
    for step, path in enumerate(decisions):
        source_record = response["decisions"][step]
        if source_record["step"] != step or source_record["decision_sha256"] != seal.digest(path):
            raise ValueError("decision hash mismatch")
        data = json.loads(path.read_text())
        compiler = data["decision"]["compiler"]
        target = np.asarray(data["chart_goal"], np.float64)
        layer = compiler["layers"][0]
        nominals = np.asarray(compiler["config"]["nominals"], np.float64)
        if target.shape != (2,) or nominals.shape != (2, 5, 2) or len(layer["responses"]) != 2:
            raise ValueError("unexpected response structure")
        for index in range(2):
            batch = data["evaluations"][index + 1]
            actions, features, scores = (np.asarray(batch[key], np.float64)
                                         for key in ("actions", "positions", "scores"))
            expected = np.concatenate((nominals[index][None], nominals[index][None] + .05 * axis,
                                       nominals[index][None] - .05 * axis)).astype(np.float32)
            if actions.shape != (21, 5, 2) or features.shape != (21, 2) or scores.shape != (21,):
                raise ValueError("probe shapes")
            np.testing.assert_array_equal(actions.astype(np.float32), expected)
            matrix = ((features[1:11] - features[11:21]) / .1).T
            np.testing.assert_allclose(matrix, layer["responses"][index]["matrix"], atol=1e-12, rtol=0)
            chart_cost = ((features - target) ** 2).sum(axis=1)
            chart_grad = (chart_cost[1:11] - chart_cost[11:21]) / .1
            score_grad = (scores[1:11] - scores[11:21]) / .1
            norm_chart, norm_score = np.linalg.norm(chart_grad), np.linalg.norm(score_grad)
            valid = (np.abs(chart_grad) > 1e-6) & (np.abs(score_grad) > 1e-6)
            rows.append({"step": step, "nominal": index, "chart_gradient": chart_grad.tolist(),
                "score_gradient": score_grad.tolist(), "chart_norm": float(norm_chart),
                "score_norm": float(norm_score),
                "cosine": float(chart_grad @ score_grad / (norm_chart * norm_score))
                          if norm_chart > 1e-6 and norm_score > 1e-6 else None,
                "comparable_coordinates": int(valid.sum()),
                "same_sign_coordinates": int(np.sum(chart_grad[valid] * score_grad[valid] > 0)),
                "selected_modes": layer["responses"][index]["selected_modes"]})
    seal.verify(deps)
    with OUT.open("x") as stream:
        json.dump({"status": "complete", "dependencies": deps, "rows": rows,
                   "scope": "post-hoc saved finite differences; one exposed goal"},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps([{k: row[k] for k in ("step", "nominal", "chart_norm", "score_norm", "cosine",
                                            "comparable_coordinates", "same_sign_coordinates")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
