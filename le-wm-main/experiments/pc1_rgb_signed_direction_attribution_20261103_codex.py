"""Exploratory attribution of sealed RGB direction predictions, without model calls."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc1_rgb_spectral_direction_adfd_20261103_codex.json"
AUDIT = ROOT / "results/pc1_rgb_spectral_direction_adfd_20261103_codex_audit.json"
PROTOCOL = ROOT.parent / "docs/PC1_RGB_SIGNED_DIRECTION_ATTRIBUTION_PROTOCOL_20261103_CODEX.md"
OUTPUT = ROOT / "results/pc1_rgb_signed_direction_attribution_20261103_codex.json"
SOURCE_HASH = "1cbb77af20896aec68844d3239d1a57b8962a59ffb5fec4be2da30797d4e038e"
AUDIT_HASH = "119598944100bb126c92312ae3a08666ed03f5762716022d3d8db53d45827b25"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = [float(x) for x in values if x is not None]
    return {"n": len(values), "median": float(np.median(values)) if values else None,
            "max": max(values) if values else None}


def analyze(data):
    output = []
    for row in data["rows"]:
        center, ad, jv = (np.asarray(row[key], dtype=float) for key in ("center", "ad", "saved_jv"))
        for item in row["measurements"]:
            if "skipped" in item:
                raise ValueError("expected fully legal sealed panel")
            epsilon = float(item["epsilon"])
            plus, minus = np.asarray(item["plus"]), np.asarray(item["minus"])
            even = (plus + minus - 2 * center) / 2
            odd = (plus - minus) / 2
            deltas = {1: plus - center, -1: minus - center}
            if not np.allclose(deltas[1], even + odd, atol=1e-12, rtol=0) or not np.allclose(
                    deltas[-1], even - odd, atol=1e-12, rtol=0):
                raise ValueError("even/odd identity failed")
            signs = []
            for sign, delta in deltas.items():
                norm = float(np.linalg.norm(delta))
                errors = {key: float(np.linalg.norm(delta - sign * epsilon * vector))
                          for key, vector in (("ad", ad), ("saved_fd", jv))}
                signs.append({"sign": sign, "delta": delta.tolist(), "delta_norm": norm,
                              "absolute_error": errors,
                              "relative_error": {key: val / norm if norm >= 1e-6 else None
                                                 for key, val in errors.items()},
                              "even_over_delta": float(np.linalg.norm(even)) / norm if norm >= 1e-6 else None})
            output.append({"root": row["root"], "branch": row["branch"], "direction_index": row["direction_index"],
                           "scale": item["scale"], "epsilon": epsilon, "even": even.tolist(), "odd": odd.tolist(),
                           "even_norm": float(np.linalg.norm(even)), "odd_norm": float(np.linalg.norm(odd)),
                           "signs": signs})
    if len(output) != 72:
        raise ValueError("expected 24 directions x 3 scales")
    return output


def summarize(rows):
    result = {}
    for scale in ("small", "medium", "candidate"):
        selected = [r for r in rows if r["scale"] == scale]
        signs = [s for r in selected for s in r["signs"]]
        result[scale] = {
            "directions": len(selected), "signed": len(signs),
            "even_norm": stats(r["even_norm"] for r in selected),
            "odd_norm": stats(r["odd_norm"] for r in selected),
            "even_over_delta": stats(s["even_over_delta"] for s in signs),
            "ad_absolute": stats(s["absolute_error"]["ad"] for s in signs),
            "fd_absolute": stats(s["absolute_error"]["saved_fd"] for s in signs),
            "ad_relative": stats(s["relative_error"]["ad"] for s in signs),
            "fd_relative": stats(s["relative_error"]["saved_fd"] for s in signs),
            "ad_lower_count": sum(s["absolute_error"]["ad"] < s["absolute_error"]["saved_fd"] for s in signs),
            "per_root": {str(root): {
                "signed": sum(len(r["signs"]) for r in selected if r["root"] == root),
                "ad_relative": stats(s["relative_error"]["ad"] for r in selected if r["root"] == root for s in r["signs"]),
                "fd_relative": stats(s["relative_error"]["saved_fd"] for r in selected if r["root"] == root for s in r["signs"])
            } for root in range(4)}
        }
    return result


def main():
    if sha(SOURCE) != SOURCE_HASH or sha(AUDIT) != AUDIT_HASH:
        raise ValueError("sealed source or audit hash mismatch")
    audit = json.loads(AUDIT.read_text())
    if audit["result_sha256"] != SOURCE_HASH or audit["binding_and_arithmetic"] != "passed":
        raise ValueError("source audit mismatch")
    data = json.loads(SOURCE.read_text())
    rows = analyze(data)
    payload = {"scope": "exploratory analysis of exposed development roots",
               "dependencies": {str(p): sha(p) for p in (SOURCE, AUDIT, PROTOCOL, Path(__file__))},
               "summary": summarize(rows), "rows": rows}
    with OUTPUT.open("x") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
