"""Independent least-squares recomputation of goal sensitivity retention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.analyze_pusht_d2p_physical_calibration import digest


def residual_energy(goals, vector):
    goals, vector = np.asarray(goals, float), np.asarray(vector, float)
    if goals.shape != (5, 192) or vector.shape != (192,) or not np.isfinite(goals).all() or not np.isfinite(vector).all():
        raise ValueError("finite five-goal/192-coordinate contract required")
    directions, delta = (goals[1:]-goals[0]).T, vector-goals[0]
    solution, _, rank, _ = np.linalg.lstsq(directions, delta, rcond=None)
    residual = delta-directions@solution
    total, remaining = float(delta@delta), float(residual@residual)
    return {"original_energy": total, "retained_energy": remaining,
            "retained_fraction": remaining/total if total > 0 else None}, rank


def analyze(path):
    report = json.loads(path.read_text())
    protocol = json.loads((path.parent/"protocol.json").read_text())
    gates = {"cuda_only", "all44_forwards", "parameters_unchanged", "buffers_unchanged", "all_eval",
             "source_unchanged", "weights_unchanged", "config_unchanged", "intervention_unchanged"}
    if (report["schema"] != "pusht_d2p_nuisance_sensitivity.v1" or report["status"] != "completed"
            or report["protocol"] != protocol or set(report["gates"]) != gates
            or any(v is not True for v in report["gates"].values()) or report["encoder_forwards"] != 44
            or report["new_predict_samples"] != 0 or report["new_physical_future_branches"] != 0
            or report["P1_admission"] is not False):
        raise ValueError("complete frozen CUDA goal-sensitivity report required")
    reference_path = Path(protocol["intervention_path"])
    if digest(reference_path) != protocol["intervention_sha256"]:
        raise ValueError("original goal intervention changed")
    reference = json.loads(reference_path.read_text())
    for name, sha in protocol["implementation_sha256"].items():
        if digest(path.parent/"implementation"/(name+".py")) != sha:
            raise ValueError("archived sensitivity implementation changed")
    if [r["root_id"] for r in report["rows"]] != list(range(4)):
        raise ValueError("all four unique root identities required")
    expected_names = ["original"]+[p[0] for p in protocol["perturbations"]]
    values = {"object": [], "agent": []}
    for row, old in zip(report["rows"], reference["rows"], strict=True):
        if row["root_sha256"] != old["root_sha256"] or [p["name"] for p in row["probes"]] != expected_names:
            raise ValueError("root/probe identity changed")
        goals = np.asarray([v["goal_vector"] for v in old["variants"]], float)
        q = np.asarray(row["projection"], float)
        np.testing.assert_allclose(q.T, q, atol=1e-12)
        np.testing.assert_allclose(q@q, q, atol=1e-12)
        np.testing.assert_allclose((goals-goals[0])@q, 0., atol=1e-12)
        for probe in row["probes"]:
            computed, rank = residual_energy(goals, probe["goal_vector"])
            if rank != row["metric_receipt"]["rank"]:
                raise ValueError("independent nuisance rank differs")
            for key, value in computed.items():
                if value is None:
                    if probe[key] is not None: raise ValueError("zero-energy probe is not a defined fraction")
                else: np.testing.assert_allclose(value, probe[key], rtol=1e-9, atol=1e-12)
            if probe["name"] != "original" and computed["retained_fraction"] is not None:
                values[probe["name"].split("_")[0]].append(computed["retained_fraction"])
    if len(values["object"]) != 24 or len(values["agent"]) != 16:
        raise ValueError("all object/agent perturbations must have nonzero signal")
    summary = {k: {"min": float(np.min(v)), "median": float(np.median(v)), "max": float(np.max(v))} for k, v in values.items()}
    for kind in values:
        for name, value in summary[kind].items():
            np.testing.assert_allclose(value, report["summary"][kind][name], rtol=1e-9, atol=1e-12)
    return {"schema": "pusht_d2p_nuisance_sensitivity_analysis.v1", "report_sha256": digest(path),
            "independent_lstsq_fits": 44, "summary": summary, "P1_admission": False,
            "scope": "Goal-space finite-perturbation fidelity only, not preservation of controllability or actual task benefit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("immutable independent audit required")
    result = analyze(args.report)
    result["analyzer_sha256"] = digest(Path(__file__))
    with args.output.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["summary"]))


if __name__ == "__main__": main()
