"""Audit saved multistart CEM allocation and incumbents without rerunning it."""
import numpy as np


def audit_multistart(result, batches):
    """Check scored bank, each generation, and the final paired choice.

    Batches are the original (actions, scores) pairs in call order.  This
    checks the recorded search, not whether its optimization was sufficient.
    """
    if not batches:
        raise ValueError("missing prediction batches")
    bank, scores = batches[0]
    bank, scores = np.asarray(bank), np.asarray(scores, dtype=float)
    starts = result["starts"]
    if bank.ndim != 3 or scores.shape != (len(bank),) or not np.isfinite(scores).all():
        raise ValueError("invalid bank")
    indices = np.argsort(scores, kind="stable")[:len(starts)].tolist()
    if not starts or result["bank_indices"] != indices:
        raise ValueError("incorrect start indices")
    np.testing.assert_allclose(result["bank_scores"], scores, rtol=0, atol=1e-12)
    counts = [item["samples"] for item in starts]
    if min(counts) < 1 or max(counts) - min(counts) > 1 or counts != sorted(counts, reverse=True):
        raise ValueError("unequal allocation")
    cursor = 1
    best = float("inf")
    for item, index in zip(starts, indices, strict=True):
        if item["index"] != index:
            raise ValueError("start index mismatch")
        incumbent = float(scores[index])
        seen = 0
        for generation in item["generations"]:
            if cursor >= len(batches) - 1:
                raise ValueError("missing generation")
            actions, values = batches[cursor]
            actions, values = np.asarray(actions), np.asarray(values, dtype=float)
            size = generation["size"]
            if (actions.shape != (size, *bank.shape[1:]) or values.shape != (size,)
                    or not np.isfinite(actions).all() or not np.isfinite(values).all()
                    or np.any(np.abs(actions) > 1)):
                raise ValueError("invalid generation")
            incumbent = min(incumbent, float(values.min()))
            np.testing.assert_allclose(generation["best_score"], incumbent, rtol=0, atol=1e-12)
            seen += size
            cursor += 1
        if seen != item["samples"]:
            raise ValueError("sample count mismatch")
        np.testing.assert_allclose(item["score"], incumbent, rtol=0, atol=1e-12)
        best = min(best, incumbent)
    if cursor != len(batches) - 1:
        raise ValueError("unexpected generation batches")
    pair, pair_scores = batches[-1]
    pair, pair_scores = np.asarray(pair), np.asarray(pair_scores, dtype=float)
    if pair.shape != (2, *bank.shape[1:]) or pair_scores.shape != (2,) or not np.isfinite(pair_scores).all():
        raise ValueError("invalid final pair")
    np.testing.assert_allclose(pair_scores[0], best, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result["pair_scores"], pair_scores, rtol=0, atol=1e-12)
    choice = int(np.argmin(pair_scores))
    if result["pair_choice"] != choice:
        raise ValueError("incorrect final choice")
    np.testing.assert_array_equal(np.asarray(result["selected"], dtype=pair.dtype), pair[choice])
    np.testing.assert_allclose(result["selected_score"], pair_scores[choice], rtol=0, atol=1e-12)
    return {"starts": len(starts), "samples": sum(counts), "best_before_pair": best}


def main():
    import hashlib
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m experiments.pc_multistart_trace_audit RESULT_DIR")
    directory = Path(sys.argv[1])
    plans_path = directory / "plans.json"
    plans = json.loads(plans_path.read_text())
    rows = [row for row in plans["rows"] if row["arm"] == "multistart_cem"]
    if not rows or any(row["status"] != "completed" for row in rows):
        raise ValueError("missing or failed multistart rows")
    summaries = []
    for row in rows:
        trace = directory / row["trace"]
        if hashlib.sha256(trace.read_bytes()).hexdigest() != row["trace_sha256"]:
            raise ValueError("trace hash mismatch")
        with np.load(trace, allow_pickle=False) as data:
            batches = [(data[f"actions_{i}"], data[f"scores_{i}"]) for i in range(row["batches"])]
            summary = audit_multistart(row["result"], batches)
        summaries.append({"root": row["root"], **summary})
    output = directory / "multistart_allocation_audit.json"
    record = {"status": "passed", "plans_sha256": hashlib.sha256(plans_path.read_bytes()).hexdigest(),
              "rows": summaries, "scope": "saved traces; no independent replication"}
    with output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
