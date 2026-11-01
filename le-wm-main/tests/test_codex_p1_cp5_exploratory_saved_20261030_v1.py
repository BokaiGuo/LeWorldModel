from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_cp5_exploratory_saved_20261030_v1 as saved


def test_actual_saved_24_rows():
    directory = saved.run.ROOT/"results/codex_p1_cp5_exploratory_cuda_20261030_v1_RUN01"
    result = saved.consume(directory,
        "fd54ebcdbf5847cc662bd019e69be34222bbb739be05651302fb11fa2adeca3b",
        "7b75ffb04404b4f6c3e4f22884eafcef7b38fb6829f9acf723996007ba01f09e")
    assert len(result["rows"]) == 24 and not result["P1_admission"]
    with pytest.raises(ValueError, match="digest"):
        saved.consume(directory, "0"*64, "7b75ffb04404b4f6c3e4f22884eafcef7b38fb6829f9acf723996007ba01f09e")
