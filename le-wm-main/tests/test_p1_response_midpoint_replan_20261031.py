import hashlib
import json

import numpy as np

from experiments.spectral_reachability import p1_response_midpoint_replan_20261031 as run


def test_real_cuda_output_binds_every_complete_saved_trajectory():
    path = run.ROOT / "results/p1_response_midpoint_replan_20261031_RUN01.json"
    assert run.sha(path) == "619d6ba332127b47a506c735857d7be7db3340fb406e680761087ded62ee0473"
    value = json.loads(path.read_text())
    entries, excluded, sources = run.source.collect()
    assert value["host"]["device"] == "cuda:0"
    assert value["model_before"] == value["model_after"]
    assert value["fixed_rows"] == 16
    assert value["excluded"] == excluded
    assert len(entries) == len(value["rows"]) == 13
    assert all(run.sha(path) == digest for path, digest in value["sources"].items())
    assert all(value["sources"][path] == digest for path, digest in sources.items())
    for entry, row in zip(entries, value["rows"], strict=True):
        assert row["index"] == entry["index"]
        assert row["first_packet_hex"] == entry["raw"][0].tobytes().hex()
        assert row["original_second_packet_hex"] == entry["raw"][1].tobytes().hex()
        assert row["original_action_sha256"] == entry["action_sha256"]
        assert row["midpoint_pixel_sha256"] == hashlib.sha256(entry["mid"].tobytes()).hexdigest()
        chosen = np.frombuffer(bytes.fromhex(row["second_packet_hex"]), dtype="<f4")
        assert chosen.shape == (10,)
        assert np.isfinite(chosen).all() and (np.abs(chosen) <= 1).all()
        assert row["chosen"] == ("optimized" if row["optimized_score"] < row["baseline_score"] else "original")
        if row["chosen"] == "original":
            assert row["second_packet_hex"] == row["original_second_packet_hex"]
    assert value["P1_admission"] is False
    assert value["equal_budget_certified"] is False
