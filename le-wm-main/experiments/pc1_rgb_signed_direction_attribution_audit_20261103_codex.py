"""Independent scalar arithmetic check of the signed directional attribution."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/pc1_rgb_spectral_direction_adfd_20261103_codex.json"
RESULT = ROOT / "results/pc1_rgb_signed_direction_attribution_20261103_codex.json"
OUTPUT = ROOT / "results/pc1_rgb_signed_direction_attribution_20261103_codex_audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-10)


def main():
    result, source = (json.loads(path.read_text()) for path in (RESULT, SOURCE))
    for name, digest in result["dependencies"].items():
        if sha(Path(name)) != digest:
            raise ValueError(f"dependency mismatch: {name}")
    records = {}
    for row in source["rows"]:
        for item in row["measurements"]:
            records[(row["root"], row["branch"], row["direction_index"], item["scale"])] = (row, item)
    by_scale = {scale: [] for scale in ("small", "medium", "candidate")}
    if len(records) != 72 or len(result["rows"]) != 72:
        raise ValueError("source or result cardinality mismatch")
    for entry in result["rows"]:
        key = (entry["root"], entry["branch"], entry["direction_index"], entry["scale"])
        row, item = records.pop(key)
        eps = item["epsilon"]
        if not close(entry["epsilon"], eps):
            raise ValueError("epsilon mismatch")
        for direction in entry["signs"]:
            sign = direction["sign"]
            predicted = item["plus" if sign == 1 else "minus"]
            delta = [a - b for a, b in zip(predicted, row["center"], strict=True)]
            norm = math.dist(delta, [0.0, 0.0])
            if any(not close(a, b) for a, b in zip(delta, direction["delta"], strict=True)):
                raise ValueError("delta mismatch")
            for label, vector in (("ad", row["ad"]), ("saved_fd", row["saved_jv"])):
                error = math.dist(delta, [sign * eps * x for x in vector])
                if not close(error, direction["absolute_error"][label]):
                    raise ValueError("error mismatch")
                relative = direction["relative_error"][label]
                if (norm < 1e-6 and relative is not None) or (norm >= 1e-6 and not close(relative, error / norm)):
                    raise ValueError("relative error mismatch")
            by_scale[entry["scale"]].append(direction)
        even = [(p + m - 2 * c) / 2 for p, m, c in zip(item["plus"], item["minus"], row["center"], strict=True)]
        odd = [(p - m) / 2 for p, m in zip(item["plus"], item["minus"], strict=True)]
        if any(not close(a, b) for a, b in zip(even, entry["even"], strict=True)) or any(
                not close(a, b) for a, b in zip(odd, entry["odd"], strict=True)):
            raise ValueError("even/odd mismatch")
    if records:
        raise ValueError("missing source records")
    for scale, signs in by_scale.items():
        summary = result["summary"][scale]
        for kind in ("ad", "saved_fd"):
            key = "ad_relative" if kind == "ad" else "fd_relative"
            values = [s["relative_error"][kind] for s in signs if s["relative_error"][kind] is not None]
            if summary[key]["n"] != len(values) or not close(summary[key]["median"], statistics.median(values)):
                raise ValueError("summary mismatch")
        if summary["ad_lower_count"] != sum(s["absolute_error"]["ad"] < s["absolute_error"]["saved_fd"] for s in signs):
            raise ValueError("count mismatch")
    with OUTPUT.open("x") as stream:
        json.dump({"result_sha256": sha(RESULT), "source_sha256": sha(SOURCE), "audit_sha256": sha(Path(__file__)),
                   "rows_checked": 72, "signed_items_checked": 144, "status": "passed"}, stream, indent=2)
    print("passed: 72 rows, 144 signed predictions, bindings and summary")


if __name__ == "__main__":
    main()
