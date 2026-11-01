from experiments.spectral_reachability import (
    p1_dev302_mixed_bank_cuda_v8_20261101_codex as subject,
)


def test_v8_large_variants_are_bounded_and_feasible():
    rows = subject.protocol()["prefix_bank"]["candidates"]
    assert len(rows) == 17
    for row in rows[13:]:
        assert max(abs(x) for action in row["actions"] for x in action) <= 1
        assert row["contact_feasible_prior"] is True
