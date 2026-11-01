"""Fixed-roster descriptive comparison; no significance or mechanism promotion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .p1_common_h10_analysis_20261101_main import metrics, delta

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = (("first_feedback", "p1_first_feedback_carry_20261101_main", "MAIN02"),
            ("response_diverse", "p1_response_diverse_carry_20261101_main", "MAIN02"),
            ("prefix_protected", "p1_prefix_protected_carry_20261101_main", "MAIN01"))


def summarize():
    rows, sources = [], {}
    for variant, name, tag in VARIANTS:
        for root in (0, 1):
            path = ROOT / f"results/{name}_root{root}_RUN_{tag}/report.json"
            data = path.read_bytes()
            sources[str(path)] = hashlib.sha256(data).hexdigest()
            report = json.loads(data)
            if report["status"] != "recorded_known_root_not_admitted":
                raise ValueError("incomplete fixed roster")
            for row in report["rows"]:
                cp = {c["step"]: c for c in row["checkpoints"]}
                change = delta(metrics(cp[5]["state"], cp[0]["goal"]),
                               metrics(cp[0]["state"], cp[0]["goal"])) if 5 in cp else None
                node = report["nodes"][row["episode_id"]][0]
                trace_path = Path(node["receipt"]["trace"])
                trace = trace_path.read_bytes()
                if hashlib.sha256(trace).hexdigest() != node["receipt"]["trace_sha256"]:
                    raise ValueError("trace identity")
                sources[str(trace_path)] = node["receipt"]["trace_sha256"]
                payload = {e["kind"]: e["payload"] for e in map(json.loads, trace.splitlines())}
                initial = np.asarray(payload["initial_candidates"]["raw"], np.float32)
                optimized = np.asarray(payload["optimized_candidates"]["raw"], np.float32)
                prefix_preserved = np.array_equal(initial[:, 0], optimized[:, 0])
                if variant == "prefix_protected":
                    if not prefix_preserved:
                        raise ValueError("prefix protection failed")
                    refinement = payload.get("winner_refinement")
                    if refinement and not np.array_equal(
                            np.asarray(refinement["before_raw"])[0],
                            np.asarray(refinement["after_raw"])[0]):
                        raise ValueError("refinement moved prefix")
                rows.append(dict(variant=variant, root=root, arm=row["arm"],
                                 success=row["native_success"], status=row["status"],
                                 steps=row["control_returned"], immediate_delta=change,
                                 initial_bank_singular_values=np.linalg.svd(
                                     initial.reshape(8, -1).astype(np.float64), compute_uv=False).tolist(),
                                 initial_prefix_preserved=prefix_preserved,
                                 initial_selected_index=payload["final_selection"]["index"]))
    return dict(rows=rows, sources=sources, independent_roots=0, known_development_roots=2,
                P1_admission=False, statistical_superiority_established=False,
                warning="Correlated development trajectories; do not treat arms or variants as independent roots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(rows=len(result["rows"]), output=str(args.output))))
