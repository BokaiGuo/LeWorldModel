"""Audit sealed direction bindings and recompute the reported comparisons."""
import json
from pathlib import Path

import numpy as np

from experiments import pc1_rgb_spectral_direction_adfd_20261103_codex as run

OUTPUT = run.OUTPUT.with_name(run.OUTPUT.stem + "_audit.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    sha = run.old.prior.sha
    data = json.loads(run.OUTPUT.read_text())
    if data["protocol_sha256"] != sha(run.PROTOCOL) or data["runner_sha256"] != sha(Path(run.__file__)):
        raise ValueError("protocol or runner changed")
    for path, digest in data["source_decisions_sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError("source changed")
    if len(data["rows"]) != 24 or len({(r["root"], r["branch"], r["direction_index"]) for r in data["rows"]}) != 24:
        raise ValueError("row identity mismatch")
    grouped = {scale: [] for scale in ("small", "medium", "candidate")}
    for row in data["rows"]:
        path = next(p for p in data["source_decisions_sha256"] if f'_root{row["root"]}_' in p)
        result = json.loads(Path(path).read_text())["result"]
        if row["branch"] == "first":
            evidence, parent = result["response"], None
        else:
            successor = result["successor_response"][int(row["branch"][-1])]
            evidence = successor["response"]
            parent = result["first_bank"][successor["parent_index"]]["prefix"]
        np.testing.assert_allclose(row["direction"], np.asarray(evidence["right_directions"][row["direction_index"]]).reshape(5, 2), atol=1e-7, rtol=0)
        np.testing.assert_allclose(row["nominal"], evidence["nominal"], atol=1e-7, rtol=0)
        if row["parent"] != parent:
            raise ValueError("parent mismatch")
        np.testing.assert_allclose(row["saved_jv"], np.asarray(evidence["jacobian"]) @ np.asarray(row["direction"]).reshape(-1), atol=1e-12, rtol=0)
        for item in row["measurements"]:
            if "skipped" in item:
                continue
            fd = (np.asarray(item["plus"]) - item["minus"]) / (2 * item["epsilon"])
            np.testing.assert_allclose(item["fd"], fd, atol=2e-7, rtol=1e-6)
            expected = run.comparison(row["ad"], item["fd"])
            for key in expected:
                if expected[key] is None:
                    if item["comparison"][key] is not None:
                        raise ValueError("near-zero metric mismatch")
                elif not np.isclose(expected[key], item["comparison"][key], atol=1e-12, rtol=1e-10):
                    raise ValueError("comparison mismatch")
            grouped[item["scale"]].append(item["comparison"])
    summary = {}
    for scale, items in grouped.items():
        summary[scale] = {"count": len(items), "skipped": 24 - len(items)}
        for field in ("absolute_error", "relative_error", "cosine"):
            values = [x[field] for x in items if x[field] is not None]
            summary[scale][field] = {"count": len(values), "median": float(np.median(values)) if values else None,
                                     "min": float(np.min(values)) if values else None,
                                     "max": float(np.max(values)) if values else None}
    with OUTPUT.open("x") as stream:
        json.dump({"result_sha256": sha(run.OUTPUT), "audit_sha256": sha(Path(__file__)),
                   "binding_and_arithmetic": "passed", "summary": summary}, stream, indent=2, allow_nan=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
