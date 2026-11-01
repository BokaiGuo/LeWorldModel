import json
from pathlib import Path

import pytest

from experiments.spectral_reachability import codex_p1_cp5_exploratory_cuda_20261030_v1 as m


def test_fixed_contract_and_leaf_caps():
    row = json.loads(m.BUNDLE.read_bytes())["rows"][0]
    sources = {str(Path(m.bridge.cp5.optimizer.__file__).resolve()): "a"*64,
               str(Path(m.durable.__file__).resolve()): "b"*64}
    policy, contract = m.make_contract(row, sources)
    m.spec.preflight(contract, m.spec.digest(contract))
    caps = contract["levels"][0]["caps"]
    assert caps["predict_calls"] == caps["action_encode_calls"] == 6
    assert caps["backward_calls"] == 2 and caps["replay_calls"] == 1
    assert caps["physical_steps"] == 0 and policy["horizon"] == 2


def test_invalid_output_rejected_before_gpu(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        m.run(tmp_path/"wrong")


def test_existing_attempt_cannot_be_overwritten():
    with pytest.raises(ValueError, match="exclusive"):
        m.run(m.BUNDLE)
