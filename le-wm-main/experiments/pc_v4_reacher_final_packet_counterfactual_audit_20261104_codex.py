"""Audit the saved counterfactual branches against v4 trace and physics records."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v4_reacher_final_packet_counterfactual_20261104_codex as run
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = run.PREFIX.with_name(run.PREFIX.name + "_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    sha = run.live.host.bridge.sha_file
    manifest, result = json.loads(run.MANIFEST.read_text()), json.loads(run.RESULT.read_text())
    seal.verify(manifest["seal"])
    if result["manifest_sha256"] != sha(run.MANIFEST):
        raise ValueError("manifest mismatch")
    decision, execution = json.loads(run.DECISION.read_text()), json.loads(run.EXEC.read_text())
    expected = run.actions_and_scores(decision)
    if len(result["rows"]) != 5:
        raise ValueError("incomplete branches")
    before = result["rows"][0]["before"]
    summaries = []
    for row, ref in zip(result["rows"], expected, strict=True):
        if any(row[k] != ref[k] for k in ("name", "raw", "model_scores")) or len(row["trajectory"]) != 5:
            raise ValueError("action or score mismatch")
        for field in ("qpos", "qvel", "target_qpos"):
            np.testing.assert_allclose(row["before"][field], before[field], rtol=0, atol=1e-10)
        for state in [row["before"], *row["trajectory"]]:
            value = float(run.live.p0._qpos_distance(np.asarray(state["qpos"]), np.asarray(state["target_qpos"])))
            if abs(value - state["distance"]) > 1e-12:
                raise ValueError("physical distance mismatch")
        summaries.append({"name": row["name"], "model_score": float(np.mean(row["model_scores"])),
                          "physical_distance": row["trajectory"][-1]["distance"]})
    for field in ("qpos", "target_qpos", "distance"):
        np.testing.assert_allclose(result["rows"][0]["trajectory"][-1][field],
                                   execution["initial_and_block_outcomes"][3][field], rtol=0, atol=1e-8)
        np.testing.assert_allclose(before[field], execution["initial_and_block_outcomes"][2][field],
                                   rtol=0, atol=1e-8)
    pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            a, b = summaries[i], summaries[j]
            model = np.sign(a["model_score"] - b["model_score"])
            physical = np.sign(a["physical_distance"] - b["physical_distance"])
            pairs.append({"a": a["name"], "b": b["name"], "model_difference": a["model_score"] - b["model_score"],
                          "physical_difference": a["physical_distance"] - b["physical_distance"],
                          "opposite": bool(model * physical < 0)})
    seal.verify(manifest["seal"])
    run.live.host.write_new(OUT, {"manifest_sha256": sha(run.MANIFEST), "result_sha256": sha(run.RESULT),
                                 "script_sha256": sha(Path(__file__)), "summaries": summaries, "pairs": pairs,
                                 "opposite_pairs": sum(p["opposite"] for p in pairs),
                                 "scope": "record audit; not a second physical replay"})
    print(f"Audited five action branches; {sum(p['opposite'] for p in pairs)}/10 pairs disagree.")


if __name__ == "__main__":
    main()
