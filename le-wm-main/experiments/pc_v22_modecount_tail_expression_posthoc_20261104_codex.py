"""Project all sealed selected tails into the intervention's knot family."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v22_modecount_same_chain_posthoc_20261104_codex as source
from experiments.spectral_reachability import pc_response_temporal_search_v1 as knots
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

OUT = source.OUT / "tail_expression_posthoc.json"


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    audit = json.loads((source.OUT / "predicate_audit.json").read_text())
    seal.verify(manifest["dependencies"])
    if audit["result_sha256"] != seal.digest(source.OUT / "interventions.json"):
        raise ValueError("predicate audit binding mismatch")
    basis = np.stack([knots.interpolate(np.eye(4)[i].reshape(1, 4, 1).repeat(2, axis=2), 10)[0, :, 0]
                      for i in range(4)], axis=1)
    rows = []
    for original in source.source_rows():
        row = source.adapt(original)
        if row["status"] != "completed":
            rows.append({"root": row["root"], "status": "planner_failed"})
            continue
        selected = np.asarray(row["result"]["selected"], np.float32)
        params = np.linalg.lstsq(basis, selected[5:].astype(np.float64), rcond=None)[0]
        projected = np.clip(knots.interpolate(params[None], 10)[0], -1, 1).astype(np.float32)
        chains = np.stack((selected, np.concatenate((selected[:5], projected))), axis=0)
        positions = source.mechanism.exact.endpoints(row["task"], chains)
        scores = np.linalg.norm(positions - row["task"]["goal"], axis=1)
        rows.append({"root": row["root"], "status": "completed",
                     "selected_score": float(scores[0]), "projected_score": float(scores[1]),
                     "tail_rms_error": float(np.sqrt(np.mean((projected-selected[5:])**2))),
                     "knots": params.tolist(), "projected_tail": projected.tolist()})
    seal.verify(manifest["dependencies"])
    source.mechanism.writer(OUT, {"rows": rows, "intervention_sha256": seal.digest(source.OUT / "interventions.json"),
        "predicate_audit_sha256": seal.digest(source.OUT / "predicate_audit.json"),
        "script_sha256": seal.digest(Path(__file__)),
        "scope": "post-hoc least-squares expression diagnostic; not best possible knot fit or planner evidence"})
    print(json.dumps([{"root": r["root"], "selected": r.get("selected_score"),
                       "projected": r.get("projected_score")} for r in rows], indent=2))


if __name__ == "__main__":
    main()
