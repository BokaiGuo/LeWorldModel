import torch

from experiments.pc_jepa_multitask_unroll_parity_20261106_codex import digest


def test_digest_tracks_shape_dtype_and_values():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert digest(source) == digest(source.clone())
    assert digest(source)["sha256"] != digest(source + 1)["sha256"]
    assert digest(source)["shape"] != digest(source.reshape(3, 2))["shape"]
    assert digest(source)["dtype"] != digest(source.double())["dtype"]
