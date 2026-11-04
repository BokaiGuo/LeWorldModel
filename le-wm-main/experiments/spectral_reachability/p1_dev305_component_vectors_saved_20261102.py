"""Recompute all component scores from saved CUDA latent differences."""
import argparse
from pathlib import Path

import numpy as np

from . import p1_dev305_component_tangent_vectors_20261102 as source

base, need, ROOT = source.base, source.need, source.ROOT
PREP_SHA = "37f245aa3ea2d5b73f6de82d931d3ca3acd6268ab3b4fd7336c1bfc70cf3bebb"
REPORT_SHA = "b87b33fea647f7121c555d1521f2a4a57347e26189b0f0bbf0d9908d5b84c8fa"
OLD_SHA = "41459053e799137e7bbc2c81b0b4ae1f2767771933266f52f038a1fb729fa055"


def analyze():
    spec = base.read(ROOT / "results" / f"{source.NAME}_PREP_DEV01.json", PREP_SHA)
    report = base.read(ROOT / "results" / f"{source.NAME}_RUN_DEV01/report.json", REPORT_SHA)
    old = base.read(ROOT / "results/p1_dev305_component_tangent_bank_20261102_RUN_DEV01/report.json", OLD_SHA)
    need(spec == source.protocol() and report["contract_sha256"] == PREP_SHA
         and report["model_before"] == report["model_after"] == spec["model_hash"], "contract")
    rows = []
    for role, entry, before in zip(spec["roles"], report["rows"], old["rows"], strict=True):
        need(role["arm"] == entry["arm"] == before["arm"], "role")
        delta = np.asarray(entry["terminal_goal_deltas"], dtype=np.float64)
        jac = np.asarray(entry["jacobian"], dtype=np.float64)
        need(delta.shape == (17, 17, 192) and np.isfinite(delta).all(), "delta")
        scores, ridge, singular = source.component_scores(delta, jac)
        need(np.isclose(ridge, entry["ridge"]) and np.allclose(singular, entry["singular_values"]), "ridge")
        differences = {}
        for name, matrix in scores.items():
            saved = np.asarray(entry["scores"][name])
            previous = np.asarray(before["scores"][name])
            need(np.allclose(matrix, saved, atol=1e-12, rtol=1e-12), "saved scores")
            need(np.allclose(saved, previous, atol=1e-12, rtol=1e-12), "previous scores")
            pair = list(map(int, np.unravel_index(np.argmin(matrix), matrix.shape)))
            need(pair == entry["selected_pair"][name] == before["selected_pair"][name], "argmin")
            differences[name] = float(np.max(np.abs(saved - previous)))
        rows.append({"arm": role["arm"], "score_max_abs_difference_from_previous": differences,
                     "selected_pair": entry["selected_pair"],
                     "baseline_max_abs_difference": entry["baseline_max_abs_difference"]})
    return {"schema": "p1_dev305_component_vectors_saved_20261102.v1",
            "source_report_sha256": REPORT_SHA, "rows": rows,
            "scope": "saved-vector arithmetic and repeatability; no new physical or independent evidence",
            "independent_confirmation": False, "p1_admission": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(base.write_new(parser.parse_args().output, analyze()))
