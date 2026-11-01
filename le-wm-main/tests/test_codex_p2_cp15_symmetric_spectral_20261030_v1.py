from copy import deepcopy
import numpy as np
import pytest

from experiments.spectral_reachability import codex_p2_cp15_symmetric_spectral_20261030_v1 as p


def samples():
    context = {"latent_history": np.zeros((3, 192)).tolist(),
               "action_history": np.zeros((2, 10)).tolist()}
    jobs, pairs = [], []
    for i in range(4):
        arm = ("two_program_transfer", "joint_two_round")[i % 2]
        jobs.append(dict(base_index=i, policy="F", root_slot=i//2, arm=arm, task_id=str(i),
                         task_sha256=str(i), input_context=deepcopy(context), goal=[0.]*192,
                         prefix_raw=np.zeros((15, 2)).tolist()))
        pairs.append(dict(base=i, root=i//2, source_arm=arm, task_id=str(i), task_sha256=str(i),
                          feedback_context_sha256=p.source.source.digest(context),
                          inherited_model_resources={}))
    return jobs, pairs


def test_four_base_join_and_context_tamper():
    jobs, pairs = samples()
    assert len(p.bind(jobs, pairs)) == 4
    jobs[0]["input_context"]["latent_history"][0][0] = 1
    with pytest.raises(ValueError, match="feedback identity"):
        p.bind(jobs, pairs)


def test_missing_or_out_of_domain_rejected():
    jobs, pairs = samples()
    with pytest.raises(ValueError):
        p.bind(jobs[:-1], pairs)
    jobs[0]["prefix_raw"][0][0] = 1.1
    with pytest.raises(ValueError):
        p.bind(jobs, pairs)


def test_runtime_refuses_context_drift_before_backend_dispatch():
    jobs, pairs = samples()
    case = p.bind(jobs, pairs)[0]
    case["context"]["action_history"][0][0] = 1
    with pytest.raises(ValueError, match="context drift"):
        p.execute_arm(None, case, "joint", None)
