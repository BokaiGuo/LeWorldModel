"""Explicit saved RGBv2 + RAM mutations only; no producer or neural imports/tmp."""
from copy import deepcopy
import io
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
from experiments.spectral_reachability import p1_constructive_selected_rgb_saved_audit_20261027_v2 as a

ROOT=Path(__file__).resolve().parents[1]
CASES=[('p1_constructive_observed_suffix_rgb_20261027_v2_cuda_main01','2cc16ea94dc76640a09ddb6d3d3cf8c70f171d9423b3896702715d48132306b1',8),
       ('p1_constructive_matched_ridge_gn_rgb_20261027_v2_cuda_main01','f6dbf29177769e8ec0be9e5ea38b928d201812a5ea44c07fbcedd0ebd3ebabbe',4)]


@pytest.fixture(scope='module')
def saved():
    path=ROOT/'results'/CASES[0][0]/'report.json';r=a.read(path,CASES[0][1]);binding=a.read(path.parent/'binding.json')
    p=a.read(binding['prepared'],binding['prepare_sha256']);args=p['arguments']
    tasks=[a.load_task(f,h,i) for i,(f,h) in enumerate(zip(args['task_exports'],args['task_shas']))]
    _,images=a.load_parent(args,tasks);events=[a.decode(l) for l in a.data(path.parent/'events.jsonl').splitlines()]
    zs={row['latent']['file']:a.npy(a.data(path.parent/row['latent']['file'],row['latent']['sha256'])) for row in r['rows']}
    return r,p,events,images,zs,tasks


@pytest.mark.parametrize('stem,sha,n',CASES)
def test_both_actual_reports_independently_verified(stem,sha,n):
    result=a.audit(ROOT/'results'/stem/'report.json',sha)
    assert result['status']=='saved_rgb_v2_bindings_metrics_verified' and result['images']==n and result['findings']==[]
    assert result['resources']=={k:dict(attempted=n,completed=n) for k in ('encoder','projector')}
    assert result['parent_physical']['native_successes']==0 and result['parent_physical']['counts']['physics_step']['completed']==102*n
    assert all(x['metrics']['predicted_Q'] is None and x['metrics']['Q_improvement_sign_reversed'] is None and x['metrics']['prediction_status']=='unknown_not_recomputed' for x in result['rows'])
    assert not result['P1_admission'] and not result['P2_admission'] and result['new_model_calls']==result['new_physics']==0


def inspect(value):
    r,p,events,images,zs,_=value;return a.inspect(r,p,events,images,lambda receipt:zs[receipt['file']])


@pytest.mark.parametrize('change',['Q','rootQ','mse','delta','prediction0','signfalse','statusknown','itempred','missingrow','reorder','resource','dedup','partial','NPYindex','moduleindex','modulereturn','eventreceipt','provenance','taskslot','rootlatent','imageQ'])
def test_RAM_mutations_fail_closed(saved,change):
    r,p,ev,images,zs,tasks=deepcopy(saved);row=r['rows'][0];item=images[0]
    if change=='Q':row['metrics']['actual_Q']+=1e-12
    elif change=='rootQ':row['metrics']['root_Q']+=1e-12
    elif change=='mse':row['metrics']['actual_vs_root_mse']+=1e-12
    elif change=='delta':row['metrics']['actual_Q_change']=0
    elif change=='prediction0':row['metrics']['predicted_Q']=0
    elif change=='signfalse':row['metrics']['Q_improvement_sign_reversed']=False
    elif change=='statusknown':row['metrics']['prediction_status']='known'
    elif change=='itempred':item['predicted']=[0.]*192
    elif change=='missingrow':r['rows'].pop()
    elif change=='reorder':r['rows'].reverse()
    elif change=='resource':r['resources']['encoder']['completed']-=1
    elif change=='dedup':del ev[6:12]
    elif change=='partial':r['status']='partial_or_failed'
    elif change=='NPYindex':row['latent']['file']='image_01.native192.npy'
    elif change=='moduleindex':ev[1]['entry']['image']=1
    elif change=='modulereturn':ev[2]['entry']['completed']=False
    elif change=='eventreceipt':ev[5]['receipt']['status']='partial'
    elif change=='provenance':row['provenance']['arm']='other'
    elif change=='taskslot':row['source_slot']=1
    elif change=='rootlatent':item['root'][0]+=1e-9
    elif change=='imageQ':item['Q']='fitted'
    with pytest.raises(ValueError):inspect((r,p,ev,images,zs,tasks))


@pytest.mark.parametrize('change',['physicalSHA','parentcounts','parentcommands','parentroot','parentgoal','missingcheckpoint','earlystop','guard','source_mapping'])
def test_parent_bindings_semantically_checked(saved,change):
    args=deepcopy(saved[1]['arguments']);tasks=deepcopy(saved[-1]);original=a.read
    if change=='physicalSHA':
        args['physical_sha']='0'*64
        with pytest.raises(ValueError):a.load_parent(args,tasks)
        return
    def changed(path,*pos,**kw):
        v=original(path,*pos,**kw)
        if str(path)==args['physical_report']:
            if change=='parentcounts':v['counts']['physics_step']['completed']-=1
            elif change=='parentroot':v['rows'][0]['root_state'][0]+=1
            elif change=='parentgoal':v['rows'][0]['goal'][0]+=1
            elif change=='missingcheckpoint':v['rows'][0]['checkpoints'].pop()
            elif change=='earlystop':v['rows'][0]['full_horizon']=False
            elif change=='parentcommands':v['rows'][0]['selection_digest']='0'*64
        if Path(path).name=='journal.jsonl':raise AssertionError('journal is bytes, not generic JSON')
        if Path(path).name=='binding.json':raise AssertionError('parent uses execution_binding')
        return v
    if change in ('guard','source_mapping'):
        # Mutate the parsed parent journal/prepare metadata in RAM after SHA read.
        if change=='guard':
            oldchain=a.chain
            def badchain(b):
                ev=oldchain(b);next(e for e in ev if e['kind']=='guard_receipt' and e['phase']=='plan' and e['step']==5)['accepted']=False;return ev
            with patch.object(a,'chain',badchain),pytest.raises(ValueError):a.load_parent(args,tasks)
        else:
            def badread(path,*pos,**kw):
                v=original(path,*pos,**kw)
                if v.get('schema')==a.PHYSICAL+'.prepare':v['source_slots'][0]['root_sha256']='0'*64
                return v
            with patch.object(a,'read',badread),pytest.raises(ValueError):a.load_parent(args,tasks)
    else:
        with patch.object(a,'read',changed),pytest.raises(ValueError):a.load_parent(args,tasks)


def test_nativeNPY_is_lossless_not_float64_or_trailing(saved):
    z=next(iter(saved[4].values()));out=io.BytesIO();np.save(out,z,allow_pickle=False)
    assert np.array_equal(a.npy(out.getvalue()),z)
    with pytest.raises(ValueError):a.npy(out.getvalue()+b'x')
    out=io.BytesIO();np.save(out,z.astype(np.float64),allow_pickle=False)
    with pytest.raises(ValueError):a.npy(out.getvalue())
    bad=z.copy();bad[0]=np.nan
    with pytest.raises(ValueError):a.native192(bad)


def test_task_semantic_hash_Q_and_history_are_not_guessed(saved):
    path=saved[1]['arguments']['task_exports'][0];sha=saved[1]['arguments']['task_shas'][0]
    original=a.read
    for change in ('Q','slot','goal','contract','tokens'):
        e=original(path,sha);t=e['task']
        if change=='Q':t['projection'][0][0]=2
        elif change=='slot':e['source_slot']=False
        elif change=='goal':t['goal'][0]+=1
        elif change=='contract':e['task_contract']['metric']='new_fit'
        elif change=='tokens':t['root_context']['action_history'][0][0]+=1
        with patch.object(a,'read',return_value=e),pytest.raises(ValueError):a.load_task(path,sha,0)


def test_external_report_SHA_and_input_json_are_strict():
    with pytest.raises(ValueError):a.audit(ROOT/'results'/CASES[0][0]/'report.json','0'*64)
    with pytest.raises(ValueError):a.decode(b'{"x":1,"x":2}')
    with pytest.raises(ValueError):a.flat('/READ_ONLY','../escape.npy')


def test_no_neural_physics_or_producer_import():
    assert not any(n in sys.modules for n in ('torch','gymnasium','pymunk','stable_worldmodel'))
    import ast
    tree=ast.parse(Path(a.__file__).read_text())
    assert not any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('experiments') for n in ast.walk(tree))
