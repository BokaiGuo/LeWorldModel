"""CPU-only contract tests; no environment construction or model inference."""
import hashlib
import json

import numpy as np
import pytest
import torch

from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core
from experiments.spectral_reachability import codex_four_arm_full_cost_20261030 as spec
from experiments.spectral_reachability import codex_four_arm_host_cuda_20261030 as host
from experiments.spectral_reachability import pusht_d2p_packet_contract as packets


def test_existing_prepare_binds_domain_and_normalizer():
    path = host.ROOT / "results/codex_four_arm_host_cuda_20261030_PREP01.json"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == "b8044f55943b34c5a56a0b5db99cfc5105c5180cdf0f2139926fbb9c82d2b5ec"
    prepared = json.loads(data)
    task, policy = prepared["task"], prepared["policy"]
    assert prepared["known_request_slot"] == 0
    assert task["binding"]["action_domain_sha256"] == core.digest(core.action_domain())
    assert task["binding"]["normalizer_sha256"] == core.digest([task["action_mean"], task["action_std"]])
    assert policy["mean"] == task["action_mean"] and policy["std"] == task["action_std"]
    assert prepared["contract"]["identity"]["action_domain"] == task["binding"]["action_domain_sha256"]
    assert prepared["contract"]["phase_policy_sha256"] == spec.digest(policy)


def test_fixed_task_domain_digest_and_packet_shape():
    expected = "a6136cdc5e6a7330ec9879c1a117c323db5af30682cf7463f5e6d8298096c055"
    assert core.digest(core.action_domain()) == expected
    assert core.action_domain() == {
        "raw_bounds": [-1, 1], "packet_steps": 5, "action_dim": 2,
        "normalization": "original saved float32 subtract/divide; no /100 or clipping",
    }
    raw = np.array([[-1, 1], [0.1, -0.2], [0, 0], [0.4, 0.5], [1, -1]], np.float32)
    assert np.array_equal(packets.packet_commands(raw), raw)
    raw[0, 0] = np.nextafter(np.float32(-1), np.float32(-2))
    with pytest.raises(ValueError, match="bounds"):
        packets.packet_commands(raw)


def test_four_arm_normalization_matches_packet_primitive_cpu():
    raw = torch.tensor([[[-1., 1.], [0.1, -0.2], [0., 0.], [0.4, 0.5], [1., -1.]]])
    mean = torch.tensor([-0.0087, 0.0068])
    std = torch.tensor([0.2019, 0.2002])
    entry = ((raw - mean) / std).reshape(1, 10)
    assert torch.equal(entry, packets.normalize_live_packet(raw, mean=mean, std=std))
    assert not torch.equal(entry, ((raw / 100 - mean) / std).reshape(1, 10))


def test_fixed_resource_upper_bound_counts():
    caps = host.resource_caps(dim=192, horizon=3, population=3, iterations=2)
    # Every arm evaluates 2 generations plus one final replay. Structured
    # adds 3 predictions; immediate screening adds 3 one-packet predictions.
    assert caps["predict_calls"] == caps["action_encode_calls"] == 3 * 3 * 3 + 3
    assert caps["predict_samples"] == 30
    assert caps["selection_evaluations"] == caps["replay_calls"] == 3 * 3 + 3
    assert caps["backward_calls"] == caps["vjp_rows"] == 192
    assert caps["jacobian_calls"] == caps["solve_calls"] == 1
    assert caps["physical_steps"] == caps["encode_calls"] == caps["svd_calls"] == 0
    assert set(caps) == set(spec.AXES)
