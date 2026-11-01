"""Existing supported-axis guard around native RAM search callbacks.

No model dispatch, derivatives, physics or CUDA. Forward reservations describe
only the explicit mock callback scope, not total pipeline or measured compute.
"""
from copy import deepcopy

import pytest

from experiments.spectral_reachability import p2_supported_resource_guard_20261027_v1 as guard
from experiments.spectral_reachability import chat_p2_20261027_candidate_census as census
from chat_p2_20261027_test_native_search_integration import shared, KINDS, complete


EXPECTED_CALLBACKS=dict(joint80=164,composition40=164,sra40=84,cem40=84,random40=84)


def guarded_search(kind, cap, fail_packet=None, fail_suffix=False):
    bindings={k:'a'*64 for k in ('root_sha256','task_sha256','model_sha256','protocol_sha256')}
    ledger=guard.BudgetLedger(guard.old.forward(cap),bindings)
    events=[];serial=[0];entered_packets=[0]

    def reserve(packets,purpose,callback):
        serial[0]+=1
        name=f'RAM_{serial[0]:03d}'
        work=guard.old.forward(packets)
        event=dict(work_id=name,packets=packets,callback_entered=False,returned=False,error=None)
        events.append(event)
        # Original supported guard performs the atomic arithmetic mutation.
        # A cap rejection must occur BEFORE any mock callback/packet entry.
        try:
            ledger.reserve(name,work,purpose)
            event['callback_entered']=True
            entered_packets[0]+=packets
            result=callback()
            ledger.finish(name,work)
            event['returned']=True
            return result
        except Exception as exc:
            event['error']=dict(type=type(exc).__name__,message=str(exc))
            # Failed or unknown work remains reserved/pending, not finish(0).
            raise

    def replay_transform(original):
        def replay(c,a):
            def call():
                if fail_packet==entered_packets[0]:
                    raise RuntimeError('RAM underlying packet failure')
                return original(c,a)
            return reserve(1,'candidate_evaluation',call)
        return replay

    def suffix_transform(original):
        def suffix(c,p,i):
            # Existing legacy mock suffix explicitly calls nominal+ADforward
            # twice. Reserve its two RAM callbacks, not a fictional Jacobian.
            def call():
                if fail_suffix:
                    raise RuntimeError('RAM suffix failure; completion unknown')
                return original(c,p,i)
            return reserve(2,'candidate_generation',call)
        return suffix

    result,args,packet_calls,suffix_calls=shared(kind,replay_transform=replay_transform,
        suffix_transform=suffix_transform)
    return result,ledger.receipt(),events,entered_packets[0],len(packet_calls),len(suffix_calls)


@pytest.mark.parametrize('kind',KINDS)
def test_exact_callback_cap_completes_native_search_without_padding(kind):
    expected=EXPECTED_CALLBACKS[kind]
    result,receipt,events,entered,observed,suffix=guarded_search(kind,expected)
    assert complete(result) and result['fixture_only']
    assert entered==observed==expected
    assert receipt['reserved']['predict_calls']==receipt['completed']['predict_calls']==expected
    assert receipt['reserved']['predict_samples']==receipt['completed']['predict_samples']==expected
    assert not receipt['pending'] and all(e['returned'] for e in events)
    assert suffix==(80 if kind=='composition40' else 0)
    assert not receipt['actual_dispatch_verified'] and not receipt['full_resource_equal']
    assert all(receipt['reserved'][k] is None for k in guard.old.UNKNOWN)


@pytest.mark.parametrize('kind',KINDS)
def test_cap_minus_one_stops_before_next_callback_and_blocks_release(kind):
    cap=EXPECTED_CALLBACKS[kind]-1
    result,receipt,events,entered,observed,_=guarded_search(kind,cap)
    assert not complete(result)
    assert entered==observed==cap
    assert receipt['reserved']['predict_calls']==cap
    blocked=[e for e in events if not e['callback_entered']]
    assert blocked and all('predispatch resource cap exceeded' in e['error']['message'] for e in blocked)
    summary=census.census(result,arm=result['arm'],expected_count=80 if kind=='joint80' else 40)
    assert not summary['deployable_engineering_candidate']
    # Blocked dispatches in the native attempt ledger are NOT actual entries.
    assert summary['measured_world_model_api_calls'] is None


@pytest.mark.parametrize('kind',KINDS)
def test_callback_failure_keeps_reservation_and_full_candidate_census(kind):
    cap=EXPECTED_CALLBACKS[kind]
    result,receipt,events,entered,_,_=guarded_search(kind,cap,fail_packet=1)
    assert not complete(result) and entered<=cap
    failed=[e for e in events if e['callback_entered'] and not e['returned']]
    assert len(failed)==1 and failed[0]['error']['type']=='RuntimeError'
    assert failed[0]['work_id'] in receipt['pending']
    assert receipt['reserved']['predict_calls']>receipt['completed']['predict_calls']
    count=80 if kind=='joint80' else 40
    summary=census.census(result,arm=result['arm'],expected_count=count)
    assert summary['denominator']==len(summary['slots'])==count
    assert not summary['failed_attempts_refunded'] and not summary['deployable_engineering_candidate']
    before=deepcopy(receipt)
    receipt['reserved']['predict_calls']=0
    assert before['reserved']['predict_calls']>0  # Receipt data is owned, not live state.


def test_suffix_two_packet_reservation_blocks_before_both_mock_forwards():
    result,receipt,events,entered,observed,suffix=guarded_search('composition40',2)
    assert not complete(result) and entered==observed==1 and suffix==0
    assert receipt['reserved']['predict_calls']==1 and not receipt['pending']
    assert events[-1]['packets']==2 and not events[-1]['callback_entered']
    summary=census.census(result,arm=result['arm'],expected_count=40)
    assert summary['not_attempted_count']==39


def test_suffix_failure_retains_full_group_reservation_not_fictional_completion():
    result,receipt,events,entered,observed,suffix=guarded_search('composition40',164,fail_suffix=True)
    assert not complete(result) and entered==3 and observed==1 and suffix==0
    assert receipt['reserved']['predict_calls']==3
    assert receipt['completed']['predict_calls']==1 and len(receipt['pending'])==1
    assert events[-1]['error']['message']=='RAM suffix failure; completion unknown'
