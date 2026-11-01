"""Saved arithmetic validation; does not rerun the frozen model or attest CUDA."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import p1_bounded_backtrack_host_20261030 as producer


def verify(result, report, prepared):
    one = producer.base.one
    if (result["schema"] != producer.NAME or result["source_report_sha256"] != one.source.REPORT_SHA
            or result["source_receipt_sha256"] != one.source.RECEIPT_SHA
            or result["calls"] != dict(predict=13120, action_encoder=13120, reverse=1280)
            or result["steps"] != list(producer.STEPS)
            or result["model_before"] != result["model_after"]
            or result["model_before"] != report["model_before"]
            or result["code_sha256"] != hashlib.sha256(Path(producer.__file__).read_bytes()).hexdigest()
            or result["training"] is not False or result["P1_admission"] is not False
            or result["physical_release"] is not False):
        raise ValueError("report contract")
    rows = result["rows"]
    if len(rows) != 160:
        raise ValueError("candidate denominator")
    counts = [0] * 5
    for n, row in enumerate(rows):
        arm = report["rows"][n // 20]
        candidate = arm["result"]["candidates"][n % 20]
        if (row["slot"], row["arm"], row["start"], row["iteration"], row["stage_uphill"]) != (
                arm["slot"], arm["arm"], candidate["start"], candidate["iteration"], candidate["stage_uphill"]):
            raise ValueError("candidate identity")
        goal = np.asarray(prepared["requests"][(n // 20) // 2]["task"]["goal"], np.float64)
        x = np.stack((np.asarray(candidate["suffix_solve"]["params"], np.float64), np.zeros(10)))

        def evaluation(e, expected, gradient):
            actual = np.asarray(e["logits"], np.float64)
            raw = np.tanh(expected.astype(np.float32)).reshape(2, 5, 2)
            endpoint = np.asarray(e["endpoint"], np.float64)
            first = np.asarray(e["first_endpoint"], np.float64)
            if (not np.array_equal(actual, expected) or not np.array_equal(e["raw"], raw)
                    or endpoint.shape != (192,) or first.shape != (192,)
                    or not np.isfinite(endpoint).all() or not np.isfinite(first).all()):
                raise ValueError("saved tensor mismatch")
            q = float(np.mean((endpoint - goal) ** 2))
            score = q + one.saved.DAMPING * float(np.sum(expected * expected)) / 192
            if not (np.isclose(q, e["terminal_q"], atol=1e-12, rtol=1e-12)
                    and np.isclose(score, e["regularized"], atol=1e-12, rtol=1e-12)):
                raise ValueError("saved score mismatch")
            grad = e["raw_gradient"]
            if gradient:
                if np.asarray(grad).shape != (2, 5, 2) or not np.isfinite(grad).all():
                    raise ValueError("invalid gradient")
            elif grad is not None:
                raise ValueError("unexpected gradient")

        if len(row["trajectory"]) != 8:
            raise ValueError("iteration count")
        previous = None
        for i, entry in enumerate(row["trajectory"]):
            if entry["iteration"] != i or len(entry["trials"]) != 4:
                raise ValueError("trial count")
            current = entry["current"]
            evaluation(current, x, True)
            if i == 0 and not np.array_equal(current["first_endpoint"], candidate["end"]["latent_history"][-1]):
                raise ValueError("incumbent mismatch")
            if previous is not None and current["regularized"] != previous:
                raise ValueError("current replay mismatch")
            for step, trial in zip(producer.STEPS, entry["trials"]):
                expected = producer.trial_logits(x, current["raw_gradient"], current["raw"], step)
                evaluation(trial, expected, False)
            selected = producer.choose(current["regularized"], [t["regularized"] for t in entry["trials"]])
            if entry["selected"] != selected:
                raise ValueError("selected trial mismatch")
            counts[4 if selected is None else selected] += 1
            if selected is not None:
                x = np.asarray(entry["trials"][selected]["logits"], np.float64)
                previous = entry["trials"][selected]["regularized"]
            else:
                previous = current["regularized"]
        evaluation(row["final"], x, False)
        if row["final"]["regularized"] != previous or row["best_regularized"] != previous:
            raise ValueError("final mismatch")
    copied = deepcopy(rows)
    summary = producer.base.summarize(report, copied)
    if any(result["summary"][k] != v for k, v in summary.items()) or copied != rows:
        raise ValueError("summary mismatch")
    winners = []
    for i in range(0, 160, 20):
        group = rows[i:i + 20]
        index = min(range(20), key=lambda j: (group[j]["best_regularized"], j))
        winners.append(dict(slot=group[0]["slot"], arm=group[0]["arm"], winner_index=index,
                            stage_uphill=group[index]["stage_uphill"]))
    if result["summary"]["arm_winners"] != winners:
        raise ValueError("winner mismatch")
    return dict(saved_arithmetic=True, candidates=len(rows), selected_counts=counts,
                gradient_recomputed=False, hardware_attested=False, P1_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    one = producer.base.one
    result = one.saved.read_verified(args.result, args.sha256)
    report = one.saved.read_verified(one.source.REPORT, one.source.REPORT_SHA)
    receipt = one.saved.read_verified(one.source.RECEIPT, one.source.RECEIPT_SHA)
    if receipt["report_sha256"] != one.source.REPORT_SHA:
        raise ValueError("receipt binding")
    prepared = one.saved.read_verified(report["prepared"], report["prepare_sha256"])
    one.saved.analyze(report, receipt, prepared)
    print(json.dumps(verify(result, report, prepared), sort_keys=True))


if __name__ == "__main__":
    main()
