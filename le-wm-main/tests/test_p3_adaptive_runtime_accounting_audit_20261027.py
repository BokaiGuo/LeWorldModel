"""Saved-event accounting tests; no model or physical execution."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.spectral_reachability.p3_adaptive_runtime_accounting_audit_20261027 import audit


def inputs():
    root=Path(__file__).resolve().parents[1]/'results/P3_ADAPTIVE_RUNTIME_20261027_cuda_v1'
    return (json.loads((root/'P3_ADAPTIVE_RUNTIME_report_20261027.json').read_text()),
            json.loads((root/'P3_ADAPTIVE_RUNTIME_protocol_20261027.json').read_text()))


def test_saved_event_correction_no_changes_to_counts_or_inputs():
    args=inputs();saved=deepcopy(args)
    result=audit(*args)
    assert args==saved
    assert result['corrected_rows']==2 and result['actual_predict_samples']==1171
    assert result['paired']['all_pairs_within_budget']
    changed=[r for r in result['corrections'] if r['old_horizon'] != r['actual_horizon']]
    assert all(r['actual_horizon']==2 and r['old_horizon']==3 for r in changed)
    assert sum(p['equal_resources'] for p in result['paired']['pairs'])==2
    assert not result['P3_admission'] and not result['efficacy_claim']


@pytest.mark.parametrize('change',['duplicate','missing','calls','events','failed','context_tokens','witness','binding'])
def test_corrupt_saved_receipts_are_not_repaired_by_inventing_counts(change):
    r,p=inputs();row=r['rows'][0]
    if change=='duplicate': r['rows'][1]=deepcopy(row)
    elif change=='missing': r['rows'].pop()
    elif change=='calls': row['prediction_receipt']['phases'][0]['predict_calls']+=1
    elif change=='events': row['search']['layers'][0]['attempted_expansions']+=1
    elif change=='failed': row['failure']={'message':'failure must remain upstream'}
    elif change=='context_tokens': row['processed_action_context_tokens']-=1
    elif change=='witness': row['search']['best_plan']['actions'].pop()
    elif change=='binding': r['protocol_id']='changed'
    with pytest.raises(ValueError):audit(r,p)
