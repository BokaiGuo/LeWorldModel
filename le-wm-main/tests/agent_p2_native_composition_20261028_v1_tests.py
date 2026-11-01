"""RAM callback wiring and saved node identity checks; no model or physics."""
from copy import deepcopy
import hashlib

import pytest
from experiments.spectral_reachability import agent_p2_native_composition_20261028_v1 as m
from test_p2_composition_predict_frontier_20261027_v1 import base, legacy_inputs


def args_and_observer():
    _,args,calls,suffix_calls=legacy_inputs()
    args['bank']=m.candidate_bank(base(),m.native.ROOTS[0])
    original=args['suffix'];observed=[]
    def suffix(context,prefix,index):
        value=original(context,prefix,index)
        observed.append((deepcopy(context),prefix.tolist(),index,deepcopy(value)))
        return value
    args['suffix']=suffix
    return args,calls,suffix_calls,observed


def test_reuse_is_exact_identity_not_another_runner():
    assert m.execute_known is m.runtime.execute_known
    assert m.prepare_known is m.runtime.prepare and m.verify_known is m.runtime.verify
    assert m.run_known_root is m.runtime.run_known_root
    assert m.run_native_arm is m.native.run_arm
    assert m.compile_root_cuda is m.native.compile_cuda
    assert m.compile_suffix_cuda is m.native.suffix_cuda


def test_RAM_native_two_program_dispatch_at_own_full_context_no_pruning():
    args,calls,suffix_calls,observed=args_and_observer();root=deepcopy(args['root'])
    result=m.run_native_arm(arm=m.native.ARMS[1],**args)
    assert result['complete'] and result['fixture_only'] and args['root']==root
    assert len(result['attempts'])==len(observed)==40
    assert len(calls)==164 and len(suffix_calls)==80  # 40 nominal + 40 ADforward RAM callbacks.
    assert len(result['ledger'])==84
    for event,(context,prefix,index,local) in zip(result['attempts'],observed,strict=True):
        first,second=event['witness']['edges']
        assert index==event['index'] and prefix==args['bank']['logits'][index][:10]
        assert context==first['output_context']==second['input_context']
        assert first['input_context']==root['context']
        assert first['actions']==args['bank']['commands'][index][0]
        assert len(first['actions'])==len(second['actions'])==5
        assert first['actions']+second['actions']==event['actions']
        assert local==event['suffix_compilation']
        assert local['input_context_sha256']==m.native.digest(context)
        assert event['witness']['receipt']['terminal_context']==second['output_context']
    assert result['winner']['independent_full_replay_exact'] is True


def test_RAM_native_no_composition_uses_original_joint_proposals_not_suffix():
    args,calls,_,_=args_and_observer()
    args['suffix']=lambda *a:pytest.fail('control must not compile a suffix')
    result=m.run_native_arm(arm=m.native.ARMS[0],**args)
    assert result['complete'] and len(result['attempts'])==80
    assert len(calls)==164 and len(result['ledger'])==164
    for index,event in enumerate(result['attempts']):
        edges=event['witness']['edges']
        assert len(edges)==1 and len(edges[0]['actions'])==10
        assert event['actions']==sum(args['bank']['commands'][index],[])
        assert event['logits']==args['bank']['logits'][index]
        assert 'suffix_compilation' not in event


@pytest.mark.parametrize('damage',['disconnected_context','suffix_failure','metric_failure'])
def test_RAM_failure_has_no_deployable_winner_or_replacement_root(damage):
    args,_,_,_=args_and_observer();original=args['suffix']
    def suffix(c,p,i):
        if damage=='suffix_failure':raise RuntimeError('fixed failure, no retry')
        value=original(c,p,i);value['input_context_sha256']='a'*64;return value
    if damage=='metric_failure':args['target_metric']=lambda c:float('nan')
    else:args['suffix']=suffix
    result=m.run_native_arm(arm=m.native.ARMS[1],**args)
    assert not result['complete'] and result['winner'] is None
    assert len(result['attempts'])==1 and result['attempts'][0]['index']==0
    assert result['faults']['primary'] is not None
    # Remaining fixed slots are preserved by runtime.DispatchGuard/report census,
    # not by pretending this partial native attempts list contains 40 evaluations.


def test_confirmatory_rejected_before_read_or_model_load():
    with pytest.raises(ValueError,match='independent P1/freshness'):
        m.execute_known(None,None,None,main_review_sha256=None,confirmatory=True)


def test_complete_resource_contract_not_equal_FLOPs():
    cost=m.runtime.C;allocation=m.resource_allocation();actual=[]
    for arm in m.native.ARMS:
        r=allocation['arms'][arm]
        assert r['candidate_count']==m.runtime.contract.COUNTS[arm]
        assert cost.add(*r['edge_evaluation_upper'])==r['per_candidate_upper']
        a=cost.add(allocation['graph_base_shared_upper'],allocation['winner_replay_upper'],
            *[r['per_candidate_upper'] for _ in range(r['candidate_count'])])
        assert all(a[k]<=allocation['common_standalone_cap'][k] for k in cost.KNOWN)
        assert all(a[k] is None for k in cost.UNKNOWN)
        actual.append(a)  # Planned upper only, not a measured run.
    comparison=cost.compare(*actual)
    assert comparison['predict_frontier_equal'] and not comparison['full_resource_equal']
    assert 'flops' in comparison['unknown_axes']
    assert allocation['full_resource_equal'] is False and not allocation['padding']


def test_saved_all_four_roots_have_true_two_edge_node_seams_not_latent_waypoints():
    root=m.runtime.ROOT/'results/agent_p2_runtime_guard_20261028_v1_known_MAIN01'
    report=root/'report.json'
    assert hashlib.sha256(report.read_bytes()).hexdigest()=='0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0'
    counts=[0,0]
    for slot in range(4):
        row=m.native.read(root/f'root_{slot}.json');bank=m.native.read(root/f'root_{slot}_bank.json')['expanded']
        for a,result in enumerate(row['results']):
            assert result['arm']==m.native.ARMS[a]
            assert len(result['attempts'])==(80 if a==0 else 40)
            for index,event in enumerate(result['attempts']):
                edges=event['witness']['edges'];receipt=event['witness']['receipt']
                assert len(edges)==(1 if a==0 else 2)
                assert sum((e['actions'] for e in edges),[])==event['actions']==receipt['actions']
                assert edges[-1]['output_context']==receipt['terminal_context']
                assert event['actions'][:5]==bank['commands'][index][0]
                if a==1:
                    assert edges[0]['output_context']==edges[1]['input_context']
                    assert event['suffix_compilation']['input_context_sha256']==m.native.digest(edges[1]['input_context'])
                    assert len(edges[0]['actions'])==len(edges[1]['actions'])==5
                counts[a]+=1
    assert counts==[320,160]  # saved known roots, NOT independent samples.


def test_no_scientific_or_library_composition_claim():
    assert len(m.UNMET_CONFIRMATORY_DEPENDENCIES)==4
    assert 'independently_precompiled_reusable_program_library_composition' in m.UNIMPLEMENTED_GENERALIZATIONS
