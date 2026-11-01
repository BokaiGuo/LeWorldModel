"""Shared-input native search integration. RAM callbacks only, never CUDA evidence."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v2 as comp
from experiments.spectral_reachability import p2_no_composition_sra_20261027_v2 as sra
from experiments.spectral_reachability import p2_full20_sampling_20261027_v2 as sampling
from experiments.spectral_reachability import d2p_composition_witness_v2 as witness
from experiments.spectral_reachability import chat_p2_20261027_candidate_census as census
from test_p2_composition_predict_frontier_20261027_v1 import base, legacy_inputs

KINDS = ('joint80','composition40','sra40','cem40','random40')


def shared(kind, *, replay_transform=None, metric_transform=None, suffix_transform=None):
    """Same root/history/metric/domain for all five original native search calls."""
    _, args, packet_calls, suffix_calls = legacy_inputs()
    root_before = deepcopy(args['root'])
    if replay_transform:
        args['replay'] = replay_transform(args['replay'])
    if metric_transform:
        args['target_metric'] = metric_transform(args['target_metric'])
    if suffix_transform:
        args['suffix'] = suffix_transform(args['suffix'])
    if kind in ('joint80','composition40'):
        args['bank'] = comp.candidate_bank(base(),comp.ROOTS[0])
        result = comp.run_arm(arm=comp.ARMS[int(kind=='composition40')],**args)
    elif kind == 'sra40':
        kwargs={k:v for k,v in args.items() if k != 'suffix'}
        result=sra.run_bank(**kwargs)
    else:
        kwargs={k:v for k,v in args.items() if k not in ('suffix','bank','fixture')}
        result=sampling.run_fixture_arm(arm='cem' if kind=='cem40' else 'random',seed=20261027,**kwargs)
    assert args['root'] == root_before
    return result,args,packet_calls,suffix_calls


def complete(result):
    return result.get('complete',result.get('candidate_and_replay_complete',result.get('prediction_budget_matched',False)))


@pytest.mark.parametrize('kind',KINDS)
def test_same_input_every_complete_candidate_has_action_replay_and_history(kind):
    r,args,calls,_=shared(kind)
    assert complete(r) and r['fixture_only'] is True
    assert len(r['attempts'])==(80 if kind=='joint80' else 40)
    search_calls=len(calls)
    for candidate in r['attempts']:
        assert candidate['completed']
        actions=candidate['actions']
        assert np.asarray(actions).shape==(10,2)
        assert np.max(np.abs(actions))<=1
        initial=deepcopy(args['root']['context'])
        middle=args['replay'](deepcopy(initial),deepcopy(actions[:5]))
        terminal=args['replay'](deepcopy(middle),deepcopy(actions[5:]))
        whole=args['replay'](deepcopy(initial),deepcopy(actions))
        assert terminal==whole
        assert terminal['action_history']==[np.asarray(a).reshape(10).tolist() for a in (actions[:5],actions[5:])]
        assert middle['latent_history'][:2]==initial['latent_history'][1:]
        if 'witness' in candidate:
            w=candidate['witness']
            assert w['receipt']['terminal_context']==terminal
            assert w['receipt']['actions']==actions
            assert len(w['edges'])==(2 if kind=='composition40' else 1)
            if kind=='composition40':
                assert w['edges'][0]['output_context']==w['edges'][1]['input_context']==middle
                assert candidate['suffix_compilation']['input_context_sha256']==comp.digest(middle)
    # Search-call count is captured BEFORE independent verification, never hiding
    # the extra RAM replay work as a free part of the native search budget.
    expected=164 if kind in ('joint80','composition40') else 84
    assert search_calls==expected
    assert len(calls)==search_calls+4*len(r['attempts'])
    count=80 if kind=='joint80' else 40
    summary=census.census(r,arm=r['arm'],expected_count=count)
    assert summary['denominator']==summary['completed_count']==count
    assert summary['deployable_engineering_candidate'] and summary['P2_admission'] is False
    assert summary['measured_world_model_api_calls'] is None
    assert census.verify_census(summary,r,arm=r['arm'],expected_count=count)


@pytest.mark.parametrize('kind',KINDS)
def test_failure_never_refunds_first_packet_or_deploys_a_complete_run(kind):
    invocations=[]
    def fail_second_packet(original):
        def replay(c,a):
            invocations.append(deepcopy(a))
            if len(invocations)==2:
                raise RuntimeError('RAM packet2 failure')
            return original(c,a)
        return replay
    r,_,_,_=shared(kind,replay_transform=fail_second_packet)
    assert not complete(r)
    assert r['ledger'][0]['completed'] and not r['ledger'][1]['completed']
    assert r['ledger'][1]['predict_calls']==r['ledger'][1]['predict_samples']==1
    assert r['attempts'][0]['completed'] is False
    expected=80 if kind=='joint80' else 40
    # Native composition stops on failure; the other producers retain all slots.
    # The full preregistered denominator is NOT inferred from successful attempts.
    assert len(r['attempts'])<=expected
    if kind in ('sra40','cem40','random40'):
        assert len(r['attempts'])==expected
    else:
        assert r['winner'] is None and len(r['attempts'])==1
    summary=census.census(r,arm=r['arm'],expected_count=expected)
    assert summary['denominator']==expected and len(summary['slots'])==expected
    assert summary['dispatch_ledger_totals']['failed_packet_attempts']==1
    assert summary['dispatch_ledger_totals']['predict_samples']>=2
    assert not summary['deployable_engineering_candidate']
    if kind in ('joint80','composition40'):
        assert summary['not_attempted_count']==expected-1


@pytest.mark.parametrize('kind',KINDS)
def test_equal_cost_keeps_original_first_slot_not_posthoc_favorite(kind):
    r,_,_,_=shared(kind,metric_transform=lambda _:lambda c:1.)
    assert complete(r)
    selected=r['winner']['candidate_id'] if 'winner' in r else r['selected_candidate_id']
    assert selected==r['attempts'][0]['candidate_id']


@pytest.mark.parametrize('damage',['context','goal','action_domain','raw_actions'])
def test_boundary_goal_domain_and_action_tamper_rejected_by_original_witness(damage):
    r,args,_,_=shared('composition40')
    original=deepcopy(r['attempts'][0]['witness'])
    w=deepcopy(original);root=deepcopy(args['root']);domain=deepcopy(args['action_domain'])
    if damage=='context':
        w['edges'][1]['input_context']['action_history'][0][0]=.1
    elif damage=='goal':
        root['goal_rgb_sha256']='e'*64
    elif damage=='action_domain':
        domain['normalization_sha256']='e'*64
    else:
        w['edges'][0]['actions'][0][0]=.1
    if damage=='goal':
        checked=witness.compose(root=root,provenance=args['provenance'],action_domain=domain,
            edges=w['edges'],replay=args['replay'],max_edges=2,max_actions=10)
        assert checked!=original['receipt']  # A new goal creates a different receipt.
    else:
        with pytest.raises(ValueError):
            witness.compose(root=root,provenance=args['provenance'],action_domain=domain,
                edges=w['edges'],replay=args['replay'],max_edges=2,max_actions=10)


def test_suffix_cannot_claim_a_different_predicted_history():
    def wrong_context(original):
        def suffix(c,p,i):
            out=original(c,p,i);out['input_context_sha256']='e'*64;return out
        return suffix
    r,_,_,_=shared('composition40',suffix_transform=wrong_context)
    assert not complete(r) and r['winner'] is None
    assert len(r['attempts'])==1 and r['error'] is not None


def test_census_cannot_shrink_denominator_reorder_or_forge_completion():
    result,_,_,_=shared('sra40')
    with pytest.raises(ValueError,match='exceed'):
        census.census(result,arm=result['arm'],expected_count=39)
    reordered=deepcopy(result);reordered['attempts'][:2]=reordered['attempts'][:2][::-1]
    with pytest.raises(ValueError,match='order'):
        census.census(reordered,arm=result['arm'],expected_count=40)
    summary=census.census(result,arm=result['arm'],expected_count=40)
    summary['completed_count']=39
    with pytest.raises(ValueError,match='drift'):
        census.verify_census(summary,result,arm=result['arm'],expected_count=40)
    missing=deepcopy(result);missing['ledger']=[]
    assert not census.census(missing,arm=result['arm'],expected_count=40)['deployable_engineering_candidate']
