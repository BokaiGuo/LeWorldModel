from experiments.spectral_reachability import (
    p1_dev303_blind_cuda_20261101_codex as subject,
)


def test_dev303_blind_cuda_contract_is_development_only():
    value = subject.build()
    assert value["seed"] == 301000003
    assert value["classification"] == "dev303_development_only"
    assert value["independent_confirmation"] is False
    assert value["P1_admission"] is False
