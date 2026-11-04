"""Extract and recount saved PointMaze feedback costs without rerunning planning."""
import json
from pathlib import Path

from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOL = ROOT.parent / "docs/PC_V7_POINTMAZE_SAVED_COST_POSTHOC_PROTOCOL_20261104_CODEX.md"
OUT = RESULTS / "pc_v7_pointmaze_saved_cost_posthoc_20261104_codex.json"
STEMS = {
    "v7": "pc_v7_pointmaze_rgb_live_dev_20261104_codex",
    "warm_cem": "pc_v7_pointmaze_matched_warm_cem_dev_20261104_codex",
}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    pair_path = RESULTS / (STEMS["warm_cem"] + "_pair_binding.json")
    pair = json.loads(pair_path.read_text())
    seal.verify(pair["dependencies"])
    if pair["initial_frames_equal"] != 16 or not pair["initial_fields_equal"]:
        raise ValueError("initial pair binding not verified")
    sha = seal.digest
    all_paths = [Path(__file__), PROTOCOL, pair_path]
    rows = []
    for arm, stem in STEMS.items():
        manifest = RESULTS / (stem + "_manifest.json")
        execution = RESULTS / (stem + "_response_execution.json")
        audit = RESULTS / (stem + "_audit.json")
        decisions = [RESULTS / (stem + f"_response_decision{i}.json") for i in range(3)]
        receipt, checked = json.loads(execution.read_text()), json.loads(audit.read_text())
        all_paths.extend([manifest, execution, audit, *decisions])
        if (checked["execution_sha256"] != sha(execution) or checked["manifest_sha256"] != sha(manifest)
                or receipt["manifest_sha256"] != sha(manifest) or len(receipt["rows"]) != 3):
            raise ValueError("execution/audit binding mismatch")
        for i, path in enumerate(decisions):
            data = json.loads(path.read_text())
            if receipt["rows"][i]["decision_sha256"] != sha(path) or data["binding"]["step"] != i:
                raise ValueError("decision binding mismatch")
            trace = data["trace"]
            calls = len(trace)
            samples = sum(len(batch["raw"]) for batch in trace)
            tokens = sum(len(batch["raw"])*len(batch["raw"][0])//5 for batch in trace)
            counts = data["counts"]
            if (calls, samples, tokens) != (counts["prediction_calls"], counts["trajectory_samples"],
                                            counts["prediction_token_samples"]):
                raise ValueError("trace accounting mismatch")
            structure = data["result"]["structure"]
            rows.append({"arm": arm, "step": i, "decision_sha256": sha(path),
                         "prediction_calls": calls, "trajectory_samples": samples, "tokens": tokens,
                         "prediction_seconds": counts["prediction_seconds"],
                         "encoding_seconds": data["encoding_seconds"], "planning_seconds": data["planning_seconds"],
                         "svd_calls": structure.get("svd"), "svd_seconds": structure.get("svd_seconds"),
                         "fd_action_samples": structure.get("fd_action_samples")})
    summary = {arm: {key: sum(row[key] for row in rows if row["arm"] == arm)
                     for key in ("prediction_calls", "trajectory_samples", "tokens", "prediction_seconds",
                                 "encoding_seconds", "planning_seconds")} for arm in STEMS}
    dependencies = seal.capture(all_paths)
    seal.verify(dependencies)
    with OUT.open("x") as stream:
        json.dump({"dependencies": dependencies, "rows": rows, "summary": summary,
                   "scope": "posthoc one-root saved timings; prediction includes synchronization",
                   "unmeasured": ["shared goal encoding", "model load", "environment render and execution",
                                  "peak memory", "pure GPU kernel time", "full cost certification"]},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
