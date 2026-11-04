"""Recompute retrospective cost comparison from source receipts."""
import json

from experiments.a1_pointmaze_proprio_spectral_extra_random_compare_20261103 import (
    EXTRA, OUTPUT, PROTOCOL, SPECTRAL, sha256_file,
)


def test_comparison_is_bound_to_saved_receipts():
    report, extra, spectral = (json.loads(path.read_text()) for path in (OUTPUT, EXTRA, SPECTRAL))
    assert sha256_file(PROTOCOL) == report["protocol_sha256"]
    assert report["new_model_inference"] is False
    assert report["new_physical_executions"] == 0
    for i, row in enumerate(report["rows"]):
        r = next(b for b in extra["rows"][i]["branches"] if b["arm"] == "proprio")
        s = next(b for b in spectral["rows"][i]["branches"] if b["arm"] == "spectral")
        random, structured = row["conditions"]
        assert row["root"] == extra["rows"][i]["root"] == spectral["rows"][i]["root"]
        assert random["initial_samples"] == 64 and structured["initial_samples"] == 48
        assert random["tail_samples"] == structured["tail_samples"] == 48
        assert (structured["jvp_calls"], structured["svd_calls"]) == (2, 1)
        assert random["distance_eval_only"] == r["distance_eval_only"]
        assert structured["distance_eval_only"] == s["distance_eval_only"]
        assert random["initial_encode_and_plan_seconds"] == r["initial_encode_and_plan_seconds"]
        assert structured["initial_encode_and_plan_seconds"] == (
            s["initial_encoding_seconds"] + s["initial_plan_seconds"])
        assert structured["structure_seconds"] == s["structure_seconds"]
        assert random["feedback_encode_and_plan_seconds"] == r["feedback_encode_and_plan_seconds"]
        assert structured["feedback_encode_and_plan_seconds"] == s["feedback_encode_and_plan_seconds"]
