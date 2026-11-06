"""Post-hoc search-tree coverage against saved comparison actions."""
import numpy as np


def coverage(layers, reference, segment_length):
    ref = np.asarray(reference, dtype=np.float32)
    if ref.ndim != 2 or segment_length <= 0 or not np.isfinite(ref).all():
        raise ValueError("invalid reference or segment length")
    result = []
    for depth, layer in enumerate(layers):
        length = (depth + 1) * segment_length
        if layer["depth"] != depth or length > len(ref):
            raise ValueError("invalid search depth")
        candidates = np.asarray([candidate["chain"] for candidate in layer["candidates"]], dtype=np.float32)
        retained = np.asarray(layer["retained"], dtype=int)
        if (candidates.ndim != 3 or candidates.shape[1:] != ref[:length].shape or not len(retained)
                or np.any(retained < 0) or np.any(retained >= len(candidates))
                or len(np.unique(retained)) != len(retained) or not np.isfinite(candidates).all()):
            raise ValueError("invalid candidates or retention")
        distances = np.sqrt(np.mean((candidates.astype(np.float64) - ref[None, :length])**2, axis=(1, 2)))
        nearest = int(np.argmin(distances))
        nearest_retained = int(retained[np.argmin(distances[retained])])
        result.append({"depth": depth, "prefix_steps": length, "candidate_count": len(candidates),
                       "retained_count": len(retained), "nearest_index": nearest,
                       "nearest_was_retained": bool(nearest in retained),
                       "nearest_rms": float(distances[nearest]),
                       "nearest_retained_index": nearest_retained,
                       "nearest_retained_rms": float(distances[nearest_retained]),
                       "nearest_score": float(layer["candidates"][nearest]["score"]),
                       "nearest_retained_score": float(layer["candidates"][nearest_retained]["score"])})
    return result


def main():
    import hashlib
    import json
    import sys
    from pathlib import Path

    from experiments.spectral_reachability import pc_v3_dependency_seal as seal

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m experiments.pc_saved_prefix_coverage RESULT_DIR")
    directory = Path(sys.argv[1])
    plan_path = directory / "plans.json"
    plans = json.loads(plan_path.read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if (audit["status"] != "passed" or audit["plans_sha256"] != digest(plan_path)
            or plans["manifest_sha256"] != digest(directory / "manifest.json")):
        raise ValueError("source hash or audit mismatch")
    seal.verify(manifest["dependencies"])
    rows, result = plans["rows"], []
    roots = sorted({row["root"] for row in rows})
    for root in roots:
        arms = {row["arm"]: row for row in rows if row["root"] == root}
        if set(arms) != set(manifest["arms"]) or any(row["status"] != "completed" for row in arms.values()):
            raise ValueError("missing or failed source arm")
        method, control = arms["v17"], arms["multistart_cem"]
        if not (control["result"]["selected_score"] < .05 <= method["result"]["selected_score"]):
            continue
        ref = np.asarray(control["result"]["selected"], dtype=np.float32)
        trace = directory / control["trace"]
        if digest(trace) != control["trace_sha256"]:
            raise ValueError("reference trace hash mismatch")
        found = False
        with np.load(trace, allow_pickle=False) as data:
            for i in range(control["batches"]):
                actions, scores = data[f"actions_{i}"], data[f"scores_{i}"]
                if actions.shape[1:] == ref.shape:
                    found |= bool(np.any(np.all(actions == ref, axis=(1, 2))
                                        & np.isclose(scores, control["result"]["selected_score"], atol=1e-12, rtol=0)))
        if not found:
            raise ValueError("reference was not scored")
        selected = np.asarray(method["result"]["selected"], dtype=np.float32)
        if selected.shape != ref.shape:
            raise ValueError("selected chain shape mismatch")
        result.append({"root": root, "task": control["task"],
                       "reference_score": control["result"]["selected_score"],
                       "method_score": method["result"]["selected_score"],
                       "selected_rms": float(np.sqrt(np.mean((selected.astype(float)-ref)**2))),
                       "layers": coverage(method["result"]["child"]["layers"], ref,
                                          method["result"]["config"]["segment_length"])})
    record = {"source_plans_sha256": digest(plan_path), "source_audit_sha256": digest(directory / "audit.json"),
              "selection": "control success and method failure, post-hoc",
              "scope": "saved action distances only; no causal claim", "rows": result}
    output = directory / "saved_prefix_coverage_posthoc.json"
    with output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
