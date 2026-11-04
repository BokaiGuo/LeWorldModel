"""Privileged post-hoc least-squares projection of known suffixes."""
import json
import sys
from pathlib import Path

import numpy as np

from experiments import pc_v7_e0_oracle_prefix_suffix_dev_20261104_codex as prior
from experiments.spectral_reachability import pc_v3_dependency_seal as seal

ROOT = prior.ROOT
PROTOCOL = ROOT.parent / "docs/PC_V7_E0_SUFFIX_KNOT_PROJECTION_DEV_PROTOCOL_20261104_CODEX.md"
PREFIX = ROOT / "results/pc_v7_e0_suffix_knot_projection_dev_20261104_codex"
RESULT, AUDIT = [PREFIX.with_name(PREFIX.name + x + ".json") for x in ("_result", "_audit")]


def design():
    return np.stack([np.interp(np.arange(10), np.linspace(0, 9, 4), np.eye(4)[i])
                     for i in range(4)], axis=1)


def calculate():
    previous = json.loads(prior.RESULT.read_text())
    audit = json.loads(prior.AUDIT.read_text())
    if audit["result_sha256"] != prior.sha(prior.RESULT):
        raise ValueError("prior diagnostic audit mismatch")
    matrix = design()
    rows = []
    for root, (task, full, _, _, _, _) in enumerate(prior.inputs()):
        knot = np.clip(np.linalg.lstsq(matrix, full[5:].astype(float), rcond=None)[0], -1, 1)
        projected = np.concatenate((full[:5], matrix @ knot)).astype(np.float32)
        zero = np.concatenate((full[:5], np.zeros((10, 2), np.float32)))
        chains = np.stack((full, projected, zero))
        positions = prior.exact.endpoints(task, chains)
        distances = np.linalg.norm(positions - np.asarray(task["goal"]), axis=1)
        rows.append({"root": root, "task": task, "knots": knot.tolist(), "chains": chains.tolist(),
                     "positions": positions.tolist(), "distances": distances.tolist(),
                     "suffix_action_rmse": float(np.sqrt(np.mean((projected[5:].astype(float)-full[5:])**2)))})
    d = np.asarray([r["distances"] for r in rows])
    return matrix, rows, {"full_success": int((d[:, 0] < .05).sum()),
                          "projection_success": int((d[:, 1] < .05).sum()),
                          "zero_success": int((d[:, 2] < .05).sum()),
                          "mean_distances": d.mean(axis=0).tolist()}


def run():
    if RESULT.exists() or AUDIT.exists():
        raise FileExistsError("existing or partial diagnostic")
    deps = seal.capture([Path(__file__), PROTOCOL, prior.RESULT, prior.AUDIT, prior.MANIFEST,
                         Path(prior.__file__), Path(prior.exact.__file__), Path(prior.exact.wall.__file__)])
    matrix, rows, summary = calculate()
    seal.verify(deps)
    prior.write(RESULT, {"seal": deps, "design": matrix.tolist(), "rows": rows, "summary": summary})
    print(json.dumps(summary), flush=True)


def audit():
    if AUDIT.exists():
        raise FileExistsError(AUDIT)
    result = json.loads(RESULT.read_text())
    seal.verify(result["seal"])
    matrix = np.asarray(result["design"])
    np.testing.assert_allclose(matrix, design(), rtol=0, atol=0)
    data = prior.inputs()
    for row, (task, full, _, _, _, _) in zip(result["rows"], data, strict=True):
        knots = np.clip(np.linalg.lstsq(matrix, full[5:].astype(float), rcond=None)[0], -1, 1)
        np.testing.assert_allclose(row["knots"], knots, atol=1e-14, rtol=0)
        chains = np.asarray(row["chains"], np.float32)
        np.testing.assert_array_equal(chains[0], full)
        np.testing.assert_array_equal(chains[1], np.concatenate((full[:5], matrix @ knots)).astype(np.float32))
        np.testing.assert_array_equal(chains[2], np.concatenate((full[:5], np.zeros((10, 2), np.float32))))
        pos = prior.exact.endpoints(task, chains)
        np.testing.assert_allclose(pos, row["positions"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.linalg.norm(pos - task["goal"], axis=1), row["distances"], atol=1e-12, rtol=0)
    distances = np.asarray([r["distances"] for r in result["rows"]])
    assert result["summary"]["mean_distances"] == distances.mean(axis=0).tolist()
    assert [result["summary"][k] for k in ("full_success", "projection_success", "zero_success")] == (distances < .05).sum(axis=0).tolist()
    prior.write(AUDIT, {"result_sha256": prior.sha(RESULT), "verified_roots": len(data),
                        "scope": "same-backend numerical reconstruction"})
    print(f"Audited {len(data)} roots.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("run", "audit"):
        raise SystemExit("choose run or audit")
    {"run": run, "audit": audit}[sys.argv[1]]()
