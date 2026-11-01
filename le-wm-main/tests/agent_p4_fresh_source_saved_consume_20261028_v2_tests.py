"""RAM sourcepack fixtures ONLY. No reserved root is physically collected."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p4_fresh_source_saved_consume_20261028_v2 as c
import agent_p4_fresh_source_20261028_v2_tests as h


class RAM(h.RAM):
    def frame(self,name,value):
        stream=io.BytesIO();Image.fromarray(value).save(stream,format='PNG')
        self.files[name]=stream.getvalue();return c.digest(self.files[name])
    def file_sha(self,name):
        self.files[name]=b''.join(h.m.raw(e) for e in self.events)
        return c.digest(self.files[name])


@pytest.fixture(scope='module')
def fixture():
    # Execute only the actual producer's orchestration over RAM fake methods.
    # No gym/environment module, GPU, reset of real roots or physical RNG.
    with pytest.MonkeyPatch.context() as patch:
        p=h.m.build('calibration_only');r,s,_=h.run(patch,p=p,store=RAM())
    assert r['sourcepack_complete'] and r['reachable_count']==4
    folder=Path('/RAM/source');prepared=Path('/RAM/prepared.json')
    files={str(folder/k):v for k,v in s.files.items()}
    files[str(prepared)]=h.m.raw(p)
    files[str(folder/'execution_binding.json')]=h.m.raw(dict(prepared=str(prepared),prepare_sha256=c.digest(files[str(prepared)])))
    return folder,prepared,files


def use(fixture,mutation=None):
    folder,prepared,files=fixture;files=dict(files)
    if mutation:mutation(folder,prepared,files)
    def provider(path):return files[path] if path in files else Path(path).read_bytes()
    return c.consume(folder,c.digest(files[str(folder/'sourcepack.json')]),prepared,c.digest(files[str(prepared)]),reader=c.Reader(provider))


def rechain(folder,files,events,pack):
    previous=None
    for i,e in enumerate(events):
        e['index']=i;e['previous_sha256']=previous
        e['event_sha256']=c.semantic({k:v for k,v in e.items() if k!='event_sha256'});previous=e['event_sha256']
    payload=b''.join(h.m.raw(e) for e in events);files[str(folder/'journal.jsonl')]=payload
    pack['journal_sha256']=c.digest(payload);pack['journal_chain_tail']=previous
    files[str(folder/'sourcepack.json')]=h.m.raw(pack)


def test_all_four_source_phases_full_arrays_and_actual_operation_tree(fixture):
    r=use(fixture)
    assert r['fixed_denominator']==4 and r['phase_denominator']==12 and r['reachable_count']==4
    assert r['verified_observations']>200 and r['verified_native_returns']>80
    assert r['status']=='saved_structure_and_native_telemetry_verified_NOT_fresh'
    assert r['actual_witness_raw_per_step'] is True and r['strict_complete'] is True
    assert r['freshness'] is r['confirmed'] is r['physical_release'] is False
    assert r['historical_certificate']=='unknown'


@pytest.mark.parametrize('mutation',['state','body_bits','pixel','warm_bits','witness','identity','goal','blind','counts','drop_phase','slot','binding','chain'])
def test_saved_mutations_refused_even_rehashed_when_applicable(fixture,mutation):
    def mutate(folder,prepared,files):
        pack=json.loads(files[str(folder/'sourcepack.json')]);events=[json.loads(x) for x in files[str(folder/'journal.jsonl')].splitlines()]
        if mutation=='state':next(e for e in events if e['kind']=='fresh_observation')['state'][0]+=1.
        elif mutation=='body_bits':next(e for e in events if e['kind']=='fresh_observation')['snapshot']['bodies']['agent']['float64_bits']='00'*72
        elif mutation=='pixel':next(e for e in events if e['kind']=='fresh_observation')['pixel_sha256']='0'*64
        elif mutation=='warm_bits':next(e for e in events if e['kind']=='warmup_return')['raw_float32_bits']='00'*8
        elif mutation=='witness':
            path=str(folder/'s0_construction_commit.json');v=json.loads(files[path]);v['commands'][0][0]+=.125;files[path]=h.m.raw(v)
        elif mutation=='identity':
            e=next(e for e in events if e['kind']=='actual_root_pre_witness_check');e['identity']['physical_pose'][0]+=1.
        elif mutation=='goal':
            e=next(e for e in events if e['kind']=='native_return' and e['phase']=='replay0');e['reward']+=1.
        elif mutation=='blind':
            row=pack['rows'][0];path=str(folder/row['blind_payload'])
            with np.load(io.BytesIO(files[path]),allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            arrays['raw_action_history'][:]=0.;stream=io.BytesIO();np.savez_compressed(stream,**arrays)
            files[path]=stream.getvalue();row['blind_payload_sha256']=c.digest(files[path])
            next(e for e in events if e['kind']=='slot_finished' and e['receipt']['slot']['index']==0)['receipt']=deepcopy(row)
            events[-1]['rows']=deepcopy(pack['rows'])
        elif mutation=='counts':pack['counts']['public_step']['completed']-=1
        elif mutation=='drop_phase':pack['rows'][0]['phases'].pop()
        elif mutation=='slot':pack['rows'].pop()
        elif mutation=='binding':files[str(folder/'execution_binding.json')]=h.m.raw({})
        elif mutation=='chain':events[1]['previous_sha256']='bad';files[str(folder/'journal.jsonl')]=b''.join(h.m.raw(e) for e in events);pack['journal_sha256']=c.digest(files[str(folder/'journal.jsonl')]);files[str(folder/'sourcepack.json')]=h.m.raw(pack);return
        rechain(folder,files,events,pack)
    with pytest.raises((ValueError,KeyError)):use(fixture,mutate)


def test_partial_failures_keep_all_denominators_never_promoted(fixture):
    def mutate(folder,prepared,files):
        pack=json.loads(files[str(folder/'sourcepack.json')]);pack.update(status='engineering_failed',sourcepack_complete=False,errors=[dict(stage='RAM partial')])
        events=[json.loads(x) for x in files[str(folder/'journal.jsonl')].splitlines()];events[-1]['status']=pack['status']
        rechain(folder,files,events,pack)
    r=use(fixture,mutate)
    assert r['status']=='partial_preserved_NOT_complete' and len(r['rows'])==4 and r['phase_denominator']==12


def test_operation_tree_rejects_physics_moved_outside_public_call(fixture):
    folder,prepared,files=fixture
    pack=json.loads(files[str(folder/'sourcepack.json')]);p=json.loads(files[str(prepared)])
    _,_,_,ops=c.journal(files[str(folder/'journal.jsonl')],pack,p)
    i=next(i for i,o in ops.items() if o['entry']['resource']=='physics_step' and ops[o['parent']]['entry']['resource']=='public_step')
    ops[i]['parent']=None
    with pytest.raises(ValueError):c.operation_tree(ops)


@pytest.mark.parametrize('s',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'])
def test_strict_json(s):
    with pytest.raises(ValueError):c.decode(s)


def test_reader_byte_hash_cap_and_path_bounds():
    r=c.Reader(lambda path:b'abc')
    with pytest.raises(ValueError):r.read('/RAM/a','0'*64)
    with pytest.raises(ValueError):r.read('/RAM/a',cap=2)
    with pytest.raises(ValueError):r.read('../bad')


def test_native_angle_formula_not_silently_replaced_by_object_only_or_wrapped_metric():
    # Protocol preserves native's known noncanonical negative angle term.
    g=np.array([0.,0.,10.,10.,-10.,0.,0.]);s=np.array([100.,100.,10.,10.,0.,0.,0.])
    a=abs(g[4]-s[4]);native=bool(np.linalg.norm(g[:4]-s[:4])<20 and min(a,2*np.pi-a)<np.pi/9)
    assert not native  # object XY equality alone cannot certify whole native task.


def test_rejected_duplicates_preserve_four_without_blind_or_witness_dispatch():
    with pytest.MonkeyPatch.context() as patch:
        p=h.m.build('calibration_only');r,s,_=h.run(patch,p=p,store=RAM(),duplicate=True)
    folder=Path('/RAM/duplicates');prepared=Path('/RAM/duplicate_prepare.json')
    files={str(folder/k):v for k,v in s.files.items()};files[str(prepared)]=h.m.raw(p)
    files[str(folder/'execution_binding.json')]=h.m.raw(dict(prepared=str(prepared),prepare_sha256=c.digest(files[str(prepared)])))
    result=use((folder,prepared,files))
    assert result['reachable_count']==1 and result['fixed_denominator']==4 and len(result['rows'])==4
    assert result['strict_complete'] is True


def test_old_producer_three_frozen_sha_not_changed():
    assert all(c.digest((c.ROOT/path).read_bytes())==hsh for path,hsh in c.PINS.items())



@pytest.mark.parametrize('mutation',['missing_action','actual_bits','actual_value','native_bits','wrong_link','pose_entry','pose_return','pose_observation','missing_pose','extra_action'])
def test_actual_telemetry_rehashed_mutations_rejected(fixture,mutation):
    def mutate(folder,prepared,files):
        pack=json.loads(files[str(folder/'sourcepack.json')])
        events=[json.loads(x) for x in files[str(folder/'journal.jsonl')].splitlines()]
        a=next(e for e in events if e['kind']=='native_action_return')
        n=next(e for e in events if e['kind']=='native_return')
        if mutation=='missing_action':events.remove(a)
        elif mutation=='actual_bits':a['raw_float32_bits']='00'*8
        elif mutation=='actual_value':a['raw_action'][0]+=.125
        elif mutation=='native_bits':n['raw_float32_bits']='00'*8
        elif mutation=='wrong_link':n['action_operation_id']=a['operation_id']
        elif mutation in ('pose_entry','pose_return','pose_observation'):
            kind=dict(pose_entry='goal_pose_entry',pose_return='goal_pose_return',pose_observation='fresh_observation')[mutation]
            e=next(e for e in events if e['kind']==kind);pose=e['goal_pose'] if kind=='fresh_observation' else e['pose']
            pose['values'][0]+=1.;pose['float64_bits']=np.asarray(pose['values'],dtype='<f8').tobytes().hex()
        elif mutation=='missing_pose':events.remove(next(e for e in events if e['kind']=='goal_pose_return'))
        elif mutation=='extra_action':events.insert(a['index']+1,deepcopy(a))
        rechain(folder,files,events,pack)
    with pytest.raises((ValueError,KeyError)):use(fixture,mutate)


def test_saved_telemetry_does_not_certify_freshness(fixture):
    r=use(fixture)
    assert r['strict_complete'] and r['verified_action_returns']==r['counts']['public_step']['completed']
    assert r['verified_goal_pose_operations']>r['verified_action_returns']
    assert r['historical_certificate']=='unknown'
    assert r['freshness'] is r['confirmed'] is r['P1_admission'] is r['P4_admission'] is r['physical_release'] is False


def test_native_delayed_pose_initialization_saved_consume():
    class NativeLifecycle(h.Env):
        def __init__(self):
            super().__init__();del self.goal_pose
        def reset(self,*args,**kwargs):
            self.goal_pose=np.array([256.,256.,np.pi/4])
            return super().reset(*args,**kwargs)
    with pytest.MonkeyPatch.context() as patch:
        p=h.m.build('calibration_only');r,s,_=h.run(patch,p=p,base=NativeLifecycle,store=RAM())
    assert r['sourcepack_complete'],r['errors']
    folder=Path('/RAM/native_lifecycle');prepared=Path('/RAM/native_lifecycle_prepare.json')
    files={str(folder/k):v for k,v in s.files.items()};files[str(prepared)]=h.m.raw(p)
    files[str(folder/'execution_binding.json')]=h.m.raw(dict(prepared=str(prepared),prepare_sha256=c.digest(files[str(prepared)])))
    assert use((folder,prepared,files))['strict_complete']


def test_consistently_forged_outline_pose_rejected(fixture):
    def mutate(folder,prepared,files):
        pack=json.loads(files[str(folder/'sourcepack.json')])
        events=[json.loads(x) for x in files[str(folder/'journal.jsonl')].splitlines()]
        for e in events:
            field='pose' if e['kind'] in ('goal_pose_initial','goal_pose_entry','goal_pose_return') else 'goal_pose'
            if field not in e or e[field] is None:continue
            e[field]['values'][0]+=1.
            e[field]['float64_bits']=np.asarray(e[field]['values'],dtype='<f8').tobytes().hex()
        rechain(folder,files,events,pack)
    with pytest.raises(ValueError):use(fixture,mutate)

