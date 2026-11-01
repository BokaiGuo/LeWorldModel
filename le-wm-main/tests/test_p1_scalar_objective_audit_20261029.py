from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability.p1_scalar_objective_audit_20261029 import audit_arm


def make_data():
    goal = np.linspace(-1, 1, 192, dtype=np.float32)
    endpoint = np.linspace(0, 2, 192, dtype=np.float32)
    loss = float(np.mean((endpoint - goal) ** 2, dtype=np.float32))
    entry = {'context': {'latent_history': [endpoint.tolist()]}, 'native_float32_loss': loss}
    return {'goal': goal.tolist()}, {'gradients': [
        {'iteration': i, 'stage': deepcopy(entry), 'suffix': deepcopy(entry)} for i in range(1, 9)]}


def test_complete_scalar_objective_audit():
    task, result = make_data()
    assert audit_arm(task, result)['checked_scalar_losses'] == 16
    assert not audit_arm(task, result)['P1_admission']


@pytest.mark.parametrize('mutation', ['wrong_goal', 'wrong_loss', 'missing', 'order', 'nan'])
def test_rejects_wrong_or_incomplete_objective(mutation):
    task, result = make_data()
    if mutation == 'wrong_goal': task['goal'][0] += 10
    if mutation == 'wrong_loss': result['gradients'][2]['stage']['native_float32_loss'] += 0.1
    if mutation == 'missing': result['gradients'].pop()
    if mutation == 'order': result['gradients'][1]['iteration'] = 3
    if mutation == 'nan': result['gradients'][4]['suffix']['native_float32_loss'] = float('nan')
    with pytest.raises(ValueError): audit_arm(task, result)
