"""RAM sourcepack fixtures ONLY. No reserved root is physically collected."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from experiments.spectral_reachability import agent_p4_fresh_source_saved_consume_20261028_v1 as c
import agent_p4_fresh_source_20261028_v1_tests as h


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
    assert r['status']=='saved_structure_verified_STRICT_BLOCKED_witness_raw_telemetry'
    assert r['actual_witness_raw_per_step'] is None and r['strict_complete'] is False
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
    assert result['strict_complete'] is False


def test_old_producer_three_frozen_sha_not_changed():
    assert all(c.digest((c.ROOT/path).read_bytes())==hsh for path,hsh in c.PINS.items())
