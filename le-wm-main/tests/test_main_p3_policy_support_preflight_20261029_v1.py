"""Synthetic policy routing only, no model or physics."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import main_p3_policy_support_preflight_20261029_v1 as m
from test_agent_p3_effective_schedule_20261029_v1 import request


def fixture(high=False):
    _, req = request()
    task = req['task']
    goal = np.asarray(task['goal'], np.float32)
    nominal = goal.copy()
    nominal[0] -= np.float32(1)
    responses = []
    for h in (1, 3):
        jac = np.zeros((192, 10*h), np.float32)
        jac[1 if high else 0, 0] = 1
        responses.append(dict(binding_sha256=m.owner.digest(task['binding']),
            parent_context_sha256=m.owner.digest(task['root_context']), horizon=h,
            nominal=nominal.tolist(), jacobian=jac.tolist(),
            producer_sha256=m.owner.sha(m.owner.signal.__file__)))
    return task, responses, m.owner.signal.spectra(task, responses)


def test_progress_retains_original_decisions():
    task, responses, diagnostic = fixture()
    x = m.route(task, responses, diagnostic)
    assert all(r['horizon_panel_supported'] for r in x['routes'].values())
    assert not x['routes']['adaptive']['original_depth_executed']
    assert not x['routes']['adaptive']['original_candidate_quota_executed']


def test_reconfiguration_not_silently_relabelled_progress():
    x = m.route(*fixture(True))
    assert x['routes']['fixed']['horizon_panel_supported']
    assert not x['routes']['adaptive']['horizon_panel_supported']
    assert x['routes']['adaptive']['original_decision']['extension_type'] == 'reconfiguration'


def test_tampered_diagnostic_rejected():
    task, responses, diagnostic = fixture()
    diagnostic = deepcopy(diagnostic)
    diagnostic['scheduling_gain'] += .01
    with pytest.raises(ValueError):
        m.route(task, responses, diagnostic)


def test_missing_history_denominator_rejected():
    report = dict(schema=m.owner.NAME+'.report', status='known_fullcommit_recorded', errors=[], nodes=[])
    with pytest.raises(ValueError):
        m.scan(report)
