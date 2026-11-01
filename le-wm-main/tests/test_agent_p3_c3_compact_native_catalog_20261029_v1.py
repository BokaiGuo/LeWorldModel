"""Only injected RAM callbacks; no production authority discovery/fullstrict."""
from copy import deepcopy
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_c3_compact_native_catalog_20261029_v1 as m


@pytest.fixture(scope='module')
def fixtures():
    return runpy.run_path(str(Path(__file__).parent/'test_agent_p3_c3_common_cap_cuda_20261029_v1.py'))


def sample(fixtures, mode='adaptive', K=17, cls=None, task=None):
    task=fixtures['task']() if task is None else task
    store=m.RamExecution()
    cls=fixtures['Nonlinear'] if cls is None else cls
    sink=type('Sink',(),{'event':lambda self,e:store.event(e)})()
    r=m.original.run_tree(task,lambda t,s:cls(t,s),sink,mode=mode,K=K,
        permit_ram=True,kind='RAM_FIXTURE',capsule=lambda cid,c:store.capsule(0,cid,c))
    return task,r,store


@pytest.fixture(scope='module')
def actual_R(fixtures): return sample(fixtures)


def install(t,r,events,slots=()):
    # Explicit pre-seal fixture discovery is NOT a production source closure.
    static_values=[]
    for value in [r]+list(slots):
        for group in value['R_groups']:
            b=group['bank']
            static_values.extend([b['rule'],b['authority_manifest']['sources'],
                                  b['authority_manifest']['numeric_environment']])
    extra=dict(fields=sorted(m.codec.keys_of([r,events,list(slots)])),bank_literals=static_values)
    seal=m.routing.seal_static([t,deepcopy(t)],{},extra)
    store,cat,budget=m.routing.install_static(seal)
    return m.NativeCatalog(store,cat),budget


@pytest.fixture(scope='module')
def packed_R(actual_R):
    t,r,s=actual_R; n,budget=install(t,r,s.events)
    info=n.ingest(0,t,r,s.events)
    return n,budget,info


def test_complete_actual_R_native_journal_catalog_and_original_reconstruction(actual_R,packed_R):
    t,r,s=actual_R; n,budget,info=packed_R
    assert info['original_events']==len(s.events)
    assert info['projected_events']<len(s.events)
    assert all(x['complete_native_closure'] for x in info['calls'])
    assert info['context_counts']['nominal_boundary']>0
    assert n.verify_projection(0,s.events)
    with pytest.raises(ValueError,match='complete slot transport'):
        n.reconstruct(0,t,r,s.events,permit_fixture=True)
    finished=n.finish_slot(0,r,s.events)
    summary=n.reconstruct(0,t,r,s.events,permit_fixture=True)
    assert summary['plan_available'] and summary['R_groups']>0
    assert len(r['jobs'])==144 and len(r['resource_receipt']['caps'])==27
    ref=finished['result']
    assert m.canonical(n.store.get(bytes.fromhex(ref['object_sha256'])))==m.canonical(r)
    # Native targets derive from actual W views, not generic RAM transcript Ls.
    for call in r['native_calls']:
        key=n.catalog.get('transcript',call['transcript'])
        assert n.store.objects[key][:1]==b'W'
        p=n.capsule(0,call['call_id'],call['transcript'])
        assert '__virtual__' in p['path'] and not Path(p['path']).exists()
    print(m.canonical(dict(label='RAM_NATIVE_CATALOG_NOT_PRODUCTION',static=budget,
        projection={k:info[k] for k in ('original_events','projected_events','context_counts')},
        native_calls=len(info['calls']),objects_and_indices=n.store.nodes_bytes,
        journal_bytes=104*len(n.store.journal),routing=finished['routing'])).decode())
    assert n.verify_full_journal()['complete_original_order']
    assert [m.canonical(n.store.event(i)) for i in range(len(s.events))]==[
        m.canonical(dict(kind='arm_event',slot=0,payload=e)) for e in s.events]


def test_actual_child_contexts_have_native_boundary_origins(actual_R,packed_R):
    t,r,s=actual_R; n,_,_=packed_R
    root=m.semantic(t['root_context'])
    children=[c['transcript'][0]['task']['root_context'] for c in r['native_calls']
              if m.semantic(c['transcript'][0]['task']['root_context'])!=root]
    assert children
    for context in children:
        assert n.catalog.contains('context',context)
        origins=n.origins[m.provenance_key('context',context)]
        assert any('replay_boundary' in o or 'boundary_index' in o for o in origins)


def test_no_arbitrary_same_shape_J_context_or_transcript_source(actual_R,packed_R):
    t,r,s=actual_R; n,_,_=packed_R
    response=r['native_calls'][0]['transcript'][-1]['result']
    changed=deepcopy(response['jacobian']); changed[0][0]+=.125
    assert not n.catalog.contains('native_literal',changed)
    changed=deepcopy(t['root_context']); changed['latent_history'][0][0]+=.125
    assert not n.catalog.contains('context',changed)
    call=deepcopy(r['native_calls'][0]['transcript']);call[-1]['result']['nominal'][0]+=.125
    with pytest.raises(ValueError): n.capsule(0,0,call)


@pytest.mark.parametrize('damage',['extra_buffer','J','taskgoal','gap','boolcall','orphan','hiddenkind','recorddrop'])
def test_actual_native_mutations_rejected_before_authority(actual_R,damage):
    t,rr,ss=actual_R;r=deepcopy(rr);es=deepcopy(ss.events)
    cid=0; positions=[i for i,e in enumerate(es) if e['call']==cid]
    if damage=='extra_buffer':
        i=next(i for i in positions if es[i]['payload']['kind']=='jacobian')
        es[i]['payload']['arrays']['unexpected']=deepcopy(es[i]['payload']['arrays']['value'])
    elif damage=='J':es[positions[-1]]['payload']['result']['jacobian'][0][0]+=.125
    elif damage=='taskgoal':es[positions[0]]['payload']['task']['goal'][0]+=.125
    elif damage=='gap':es.insert(positions[3],dict(kind='io',call=None,payload=dict(kind='R_attempt',parent_context_sha256='0'*64,bank_sha256='0'*64)))
    elif damage=='boolcall':es[positions[0]]['call']=False
    elif damage=='orphan':es[positions[1]]['call']=None
    elif damage=='hiddenkind':es[positions[1]]['payload']['kind']='unexpected'
    else:r['native_calls'].pop()
    # Coherently mutate both transcript and journal where relevant: validator
    # must check original native arrays/math, not merely equality of two lists.
    if damage in ('J','taskgoal','extra_buffer'):
        r['native_calls'][0]['transcript']=[deepcopy(e['payload']) for e in es if e['call']==0]
    n,_=install(t,r,es)
    with pytest.raises((ValueError,KeyError,AssertionError)):n.ingest(0,t,r,es)


def test_failure_prefix_preserved_but_never_bank_source(fixtures):
    class Fail(fixtures['Native']):
        def __init__(self,t,s):super().__init__(t,s,fail=True)
    t,r,s=sample(fixtures,cls=Fail)
    n,_=install(t,r,s.events);row=n.ingest(0,t,r,s.events)
    assert len(row['calls'])==1 and not row['calls'][0]['complete_native_closure']
    assert row['calls'][0]['status']=='failure_prefix_only'
    assert len(r['jobs'])==144 and r['model_plan'] is None
    call=r['native_calls'][0]['transcript']
    assert n.catalog.contains('transcript',call)
    with pytest.raises(ValueError,match='complete validated'): n.capsule(0,0,call)
    n.finish_slot(0,r,s.events)
    summary=n.reconstruct(0,t,r,s.events,permit_fixture=True)
    assert not summary['plan_available']
    assert n.verify_full_journal()['events']==len(s.events)


def test_actual_progress_bank_K1_cap_full_saved_reconstruction(fixtures):
    t=fixtures['f']['fixtures']['original']['task']()
    t,r,s=sample(fixtures,mode='fixedH1',K=1,cls=fixtures['Native'],task=t)
    n,_=install(t,r,s.events);n.ingest(0,t,r,s.events)
    banks=[e['payload']['bank'] for e in s.events if e['payload']['kind']=='progress_bank']
    ordinals=[i for i,q in enumerate(r['queues']) if q['status']=='queued' and q['decision']['extension_type']=='progress']
    n.finish_slot(0,r,s.events)
    assert n.verify_full_journal()['events']==len(s.events)
    summary=n.reconstruct(0,t,r,s.events,permit_fixture=True)
    assert summary['status'].startswith('resource_cap') and not summary['plan_available']
    assert len(banks)==len(ordinals)>0


def test_projection_corruption_rejects_entire_real_event(actual_R,packed_R):
    t,r,s=actual_R;n,_,_=packed_R;events=deepcopy(s.events)
    index=next(i for i,e in enumerate(events) if e['payload']['kind']=='direction')
    events[index]['payload']['direction']=191
    with pytest.raises(ValueError,match='exact bytes'):n.verify_projection(0,events)


def test_whole_saved_consumer_reads_auxiliary_and_result_not_caller_copies(actual_R,packed_R):
    t,r,s=actual_R;n,_,_=packed_R
    events=deepcopy(s.events)
    index=next(i for i,e in enumerate(events) if e['payload']['kind']=='queue')
    events[index]['payload']['queue']['decision']['horizon']=3
    if events[index]==s.events[index]:events[index]['payload']['queue']['decision']['horizon']=1
    with pytest.raises(ValueError,match='whole saved auxiliary'):
        n.reconstruct(0,t,r,events,permit_fixture=True)
    changed=deepcopy(r);changed['full_C3_demonstrated']=True
    with pytest.raises(ValueError,match='whole saved result'):
        n.reconstruct(0,t,changed,s.events,permit_fixture=True)


def test_no_new_production_command_or_fit_claim():
    p=m.coverage();assert p['slots']==12 and p['request_denominator']==1728
    assert p['reserve_bytes']==512*1024**2 and p['whole_output_bytes']==4*1024**3
    assert not p['production_ready'] and not p['complete_transport_fit_proved'] and p['missing']
    r=subprocess.run([sys.executable,'-m',m.__name__,'coverage'],capture_output=True,text=True,check=True)
    assert 'online auxiliary' in r.stdout
    r=subprocess.run([sys.executable,'-m',m.__name__,'run'],capture_output=True,text=True)
    assert r.returncode!=0


def test_twelve_slot_1728_actual_controlflow_failure_blocks_no_padding(fixtures):
    class Fail(fixtures['Native']):
        def __init__(self,t,s):super().__init__(t,s,fail=True)
    t=fixtures['task']();s=m.RamExecution()
    p,rows=s.run([t,deepcopy(t)],lambda t,s:Fail(t,s))
    assert len(rows)==12 and len(s.results)==1
    assert len(s.results[0]['jobs'])==144
    assert rows[0]['status']=='failed_or_budget_exhausted'
    assert all(r['status']=='blocked_after_runtime_failure' for r in rows[1:])
    summary=m.original.validate_rows(p,rows,s.events,s.result_reader,
        lambda slot,cid,call:s.capsule(slot,cid,call),permit_fixture=True)
    assert len(summary)==1 and not summary[0]['plan_available']
    assert len(m.original.roster())*144==1728


def test_twelve_complete_actual_RAM_original_saved_flow(fixtures):
    t=fixtures['task']();u=deepcopy(t)
    u['goal']=[0.]*191+[1.];u['binding']['goal_sha256']=m.original.digest(u['goal'])
    s=m.RamExecution();p,rows=s.run([t,u],lambda t,s:fixtures['Nonlinear'](t,s))
    assert len(rows)==len(s.results)==12
    assert sum(len(r['jobs']) for r in s.results.values())==1728
    summary=m.original.validate_rows(p,rows,s.events,s.result_reader,
        lambda slot,cid,call:s.capsule(slot,cid,call),permit_fixture=True)
    assert len(summary)==12
    assert any(r['status']=='no_witness' for r in rows)
    assert any(r['status'].startswith('resource_cap') for r in rows)
    assert any(r['status']=='tree_complete_not_admitted' for r in rows)
    assert all(len(r['resources']['caps'])==27 for r in summary)
