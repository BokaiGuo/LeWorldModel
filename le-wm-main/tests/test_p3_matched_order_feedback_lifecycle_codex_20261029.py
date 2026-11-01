"""Synthetic lifecycle fault injections; no model, GPU or physical environment."""
from experiments.spectral_reachability import p3_matched_order_feedback_lifecycle_codex_20261029 as lifecycle
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as storage


def test_four_fixed_rows_and_cleanup_after_failures(tmp_path):
    store = storage.Store(tmp_path/'run'); closed=[]
    def create(slot,root,policy):
        if slot == 1: raise RuntimeError('setup failure')
        return object(),None,lambda:closed.append(slot)
    def engine(root,policy,raw,session,factory,sink,permit_ram=False):
        if policy == 'signal_order': raise ValueError('planning failure')
        return dict(policy=policy,status='callback_feedback_complete_not_certified')
    result = lifecycle.execute([{},{}],store,create,None,engine=engine,permit_ram=True)
    assert [r['slot'] for r in result['rows']] == [0,1,2,3]
    assert [r['status'] for r in result['rows']] == ['callback_feedback_complete_not_certified',
        'engineering_failed','callback_feedback_complete_not_certified','engineering_failed']
    assert closed == [0,2,3] and result['fixed_denominator'] == 4
    assert not result['physical_certified'] and not result['scientific_pass']


def test_cleanup_failure_is_not_success(tmp_path):
    store = storage.Store(tmp_path/'run')
    def create(slot,root,policy):
        def close(): raise RuntimeError('close failed')
        return object(),None,close
    def engine(root,policy,raw,session,factory,sink,permit_ram=False):
        return dict(policy=policy,status='callback_feedback_complete_not_certified')
    result = lifecycle.execute([{},{}],store,create,None,engine=engine,permit_ram=True)
    assert all(r['status']=='engineering_failed' and r['cleanup']=='failed' for r in result['rows'])
