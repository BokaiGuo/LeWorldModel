"""Pure saved-plan validator fixtures, not physical or neural evidence."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability.d2p_composition_witness_v2 import digest
from experiments.spectral_reachability.run_p2_matched_physical_20261027 import selected_actions


def inputs():
    protocol = {'fixture': 'non-neural test only'}
    report = dict(status='calibration_completed', gates={'fixture_gate': True},
                  protocol_sha256=digest(protocol), rows=[])
    compilation = {'rows': []}
    for i in range(4):
        key = f'root_{i}'
        receipts = [dict(program_live_commands=[[j/16., 0.]]) for j in range(16)]
        bank = [np.repeat(np.array(r['program_live_commands'][0])[None], 5, axis=0).tolist() for r in receipts]
        compiled = dict(receipts=receipts, goal_vectors=[[0.]*192],
                        scoring={'metric': {'projection': np.eye(192).tolist()}},
                        input_context={'emb': [[[0.]*192 for _ in range(3)]],
                                       'actions': [[[0.]*10 for _ in range(3)]]})
        compilation['rows'].append(dict(root_sha256=key, compiled=compiled))
        root = {'root_id': key, 'context': dict(layout='states_with_between_state_actions_v1',
            context_length=3, latent_dim=192, action_dim=10,
            latent_history=compiled['input_context']['emb'][0],
            action_history=compiled['input_context']['actions'][0][1:])}
        declaration = dict(root=root, bank=bank)
        arms = []
        for arm in ('composition', 'flat_search'):
            receipt = dict(root=root, actions=bank[0]+bank[1], terminal_context={'latent_history': [[0.]*192]})
            receipt['receipt_sha256'] = digest(receipt)
            arms.append(dict(arm=arm, selected_path=[0,1], replay_error=None, selected_model_cost=0.,
                witness=dict(receipt=receipt, independent_full_replay_exact=True)))
        report['rows'].append(dict(root_sha256=key, declaration=declaration,
            declaration_sha256=digest(declaration), fixture_only=False,
            prediction_budget_matched=True, rows=arms))
    return report, protocol, compilation


def test_all_roots_arms_and_exact_raw_actions():
    args = inputs()
    saved = deepcopy(args)
    result = selected_actions(*args)
    assert len(result) == 8 and all(a.shape == (10,2) for a in result.values())
    assert args == saved


@pytest.mark.parametrize('change', ['duplicate_root', 'missing_arm', 'fixture', 'unmatched',
    'raw_action', 'receipt_digest', 'bank', 'no_replay', 'nonfinite_cost', 'changed_cost',
    'changed_protocol', 'failed_gate', 'source_context'])
def test_invalid_saved_plans_rejected_before_environment_or_cuda(change):
    report, protocol, compilation = inputs()
    row, arm = report['rows'][0], report['rows'][0]['rows'][0]
    if change == 'duplicate_root': report['rows'][1] = deepcopy(row)
    elif change == 'missing_arm': row['rows'].pop()
    elif change == 'fixture': row['fixture_only'] = True
    elif change == 'unmatched': row['prediction_budget_matched'] = False
    elif change == 'raw_action':
        receipt = arm['witness']['receipt']
        receipt['actions'][0][0] = .5
        receipt['receipt_sha256'] = digest({k:v for k,v in receipt.items() if k != 'receipt_sha256'})
    elif change == 'receipt_digest': arm['witness']['receipt']['receipt_sha256'] = 'x'*64
    elif change == 'bank':
        row['declaration']['bank'][0][0][0] = .4
        row['declaration_sha256'] = digest(row['declaration'])
    elif change == 'no_replay': arm['witness']['independent_full_replay_exact'] = False
    elif change == 'nonfinite_cost': arm['selected_model_cost'] = float('nan')
    elif change == 'changed_cost': arm['selected_model_cost'] = .5
    elif change == 'changed_protocol': protocol['changed'] = True
    elif change == 'failed_gate': report['gates']['fixture_gate'] = False
    elif change == 'source_context':
        compilation = deepcopy(compilation)
        compilation['rows'][0]['compiled']['input_context']['emb'][0][0][0] = 1.
    with pytest.raises(ValueError):
        selected_actions(report, protocol, compilation)
