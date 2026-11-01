"""Explicit authorized saved inputs + RAM-only negative fixtures; no temp files."""
from copy import deepcopy
import io
from pathlib import Path
import sys
import numpy as np
import pytest
from PIL.PngImagePlugin import PngInfo
from experiments.spectral_reachability import p1_constructive_feedback_prefix_replay_audit_20261027_v1 as a

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'results/p1_constructive_plan_physical_20261027_v1_main01/report.json'
OLD_SHA='d205850e95f639b649fba9b5c6ee8059f2507431753a538b1ce30eccbd234300'
NEW=ROOT/'results/p1_constructive_observed_suffix_physical_20261027_v1_main01/report.json'
NEW_SHA='f864cc80a88eb289b578d271679d0bc895043a3b7bc798ae141097797a555b5e'


@pytest.fixture(scope='module')
def old():
    run=a.load_run(OLD,OLD_SHA);cache={}
    for e in run['events']:
        if e['kind']=='fresh_observation':
            r=e['receipt']
            if (r['phase']=='plan' and r['step']<=5) or (r['phase']=='warmup' and r['step'] in (0,5,10)):cache[r['png']]=run['png'](r)
    run['png']=lambda receipt:cache[receipt['png']]
    return run


def test_actual_old_new_prefix_exact_and_distinct_reports():
    r=a.audit(OLD,OLD_SHA,NEW,NEW_SHA)
    assert r['status']=='exact_saved_prefix_replay_verified' and r['distinct_report_inputs']
    assert len(r['rows'])==4 and sum(x['prefix_PNG_byte_pairs'] for x in r['rows'])==24
    assert sum(x['warmup_checkpoint_byte_pairs'] for x in r['rows'])==12
    assert all(x['prefix_state_steps']==list(range(6)) and x['own_command_steps']==list(range(1,6)) and x['all_exact'] for x in r['rows'])
    assert not r['P1_admission'] and r['new_physics']==r['new_model_calls']==0


def observation(run,step=3,phase='plan'):
    return next(e for e in run['events'] if e['kind']=='fresh_observation' and e['branch']==0 and e['receipt']['phase']==phase and e['receipt']['step']==step)
def transition(run,step=3):
    return next(e for e in run['events'] if e['kind']=='returned_transition' and e['branch']==0 and e['phase']=='plan' and e['step']==step)


def test_same_report_control_is_explicitly_not_distinct(old):
    result=a.compare(old,deepcopy(old))
    assert not result['distinct_report_inputs']


@pytest.mark.parametrize('mutation',['source','root','arm','partial','missingframe','warmup','command','returned_command','missingdispatch','missingreturn','boolslot','goalrgb','snapshot','statebit','terminal','reset','placement'])
def test_any_prefix_or_identity_mismatch_fails_closed(old,mutation):
    new=deepcopy(old);ev=new['events'];obs=observation(new)
    if mutation=='source':new['manifest']['source_prepare_sha256']='f'*64
    elif mutation=='root':new['report']['rows'][0]['root_state'][0]+=1
    elif mutation=='arm':new['report']['rows'][0]['arm']='other'
    elif mutation=='partial':new['report']['status']='partial_or_failed'
    elif mutation=='missingframe':ev.remove(obs)
    elif mutation=='warmup':ev.remove(observation(new,5,'warmup'))
    elif mutation=='command':next(e for e in ev if e['kind']=='operation_attempted' and e['resource']=='public_step' and e.get('phase')=='plan')['raw_action'][0]+=0.01
    elif mutation=='returned_command':transition(new)['command'][0]+=0.01
    elif mutation=='missingdispatch':ev.remove(next(e for e in ev if e['kind']=='operation_completed' and e['resource']=='public_step' and e.get('phase')=='plan'))
    elif mutation=='missingreturn':ev.remove(transition(new))
    elif mutation=='boolslot':new['report']['rows'][0]['root_slot']=False
    elif mutation=='goalrgb':next(e for e in ev if e['kind']=='goal_attached')['goal_rgb_sha256']='f'*64
    elif mutation=='snapshot':obs['receipt']['snapshot']['block_velocity'][0]=np.nextafter(obs['receipt']['snapshot']['block_velocity'][0],np.inf)
    elif mutation=='statebit':
        value=np.nextafter(obs['receipt']['state'][0],np.inf)
        for k in ('state','observation_state'):obs['receipt'][k][0]=value
        obs['receipt']['snapshot']['state'][0]=value;transition(new)['state'][0]=value
    elif mutation=='terminal':transition(new)['terminated']=True
    elif mutation=='reset':next(e for e in ev if e['kind']=='operation_attempted' and e['resource']=='reset')['seed']+=1
    elif mutation=='placement':next(e for e in ev if e['kind']=='placement_request')['requested'][0]+=1
    with pytest.raises(ValueError):a.compare(old,new)


def test_signed_zero_state_bits_are_not_hidden_by_numeric_equality(old):
    left,right=deepcopy(old),deepcopy(old)
    for run,value in ((left,0.),(right,-0.)):
        r=observation(run)['receipt']
        r['state'][5]=r['observation_state'][5]=r['snapshot']['state'][5]=value;transition(run)['state'][5]=value
    with pytest.raises(ValueError,match='float64 bits'):a.compare(left,right)


def test_same_pixels_but_different_PNG_file_bytes_are_rejected(old):
    new=deepcopy(old);obs=observation(new);r=obs['receipt'];original=new['png'];data=original(r)
    with a.Image.open(io.BytesIO(data)) as im:
        pixels=np.asarray(im).copy();info=PngInfo();info.add_text('RAM-only','identical pixels, different file bytes');out=io.BytesIO();im.save(out,format='PNG',pnginfo=info)
    changed=out.getvalue();assert changed!=data
    with a.Image.open(io.BytesIO(changed)) as im:assert np.array_equal(np.asarray(im),pixels)
    r['file_sha256']=a.sha_bytes(changed)
    guard=next(e for e in new['events'] if e['kind']=='guard_receipt' and e['branch']==0 and e['phase']=='plan' and e['step']==3)
    guard['file_sha256']=r['file_sha256'];new['png']=lambda receipt:changed if receipt['png']==r['png'] else original(receipt)
    with pytest.raises(ValueError,match='PNG FILE BYTES'):a.compare(old,new)


def test_external_SHA_and_hashchain_fail_closed(old):
    with pytest.raises(ValueError,match='SHA mismatch'):a.load_run(OLD,'0'*64)
    ev=deepcopy(old['events']);ev[1]['kind']='corrupted'
    with pytest.raises(ValueError,match='hashchain'):a.chain(b''.join((a.json.dumps(e)+'\n').encode() for e in ev))
    with pytest.raises(ValueError):a.decode(b'{"a":1,"a":2}')


def test_no_neural_physics_or_producer_import():
    assert not any(n in sys.modules for n in ('torch','gymnasium','pymunk','stable_worldmodel'))
    import ast
    tree=ast.parse(Path(a.__file__).read_text())
    assert not any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('experiments') for n in ast.walk(tree))
