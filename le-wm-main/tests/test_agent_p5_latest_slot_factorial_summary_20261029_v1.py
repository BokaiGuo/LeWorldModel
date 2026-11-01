"""Nonneural fixtures only; synthetic receipts are not production certificates."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_latest_slot_factorial_summary_20261029_v1 as m


def write_json(path,value):
    path.write_bytes(m.raw(value));return m.sha(path)


def array_ref(tmp_path,name,a):
    path=tmp_path/(name+'.npy');np.save(path,a,allow_pickle=False)
    return dict(directory=str(tmp_path),array=dict(file=path.name,sha256=m.sha(path),
        file_bytes=path.stat().st_size,nbytes=a.nbytes,dtype=str(a.dtype),shape=list(a.shape)))


def small(i):
    # Exactly-zero factual response for the FIRST pair only, not assumed24.
    label=np.array([0.,0.]) if i in (1,2) else np.array([i/100.,i/50.])
    cells={c:np.array([i/50.+j*.2,i/100.-j*.1]) for j,c in enumerate(m.CELLS)}
    # Include an exact-zero predicted response independently of factual zero.
    if i in (3,4):cells['AP']=np.array([0.,0.])
    score=m.cell_scores(cells,label,dict(bitwise_equal=True,maxabs=0.))
    return cells,label,score


def fixture():
    requests=[];rows=[]
    for i in range(164):
        root,cond=m.owner.bridge.factor.ORDER[i//41]
        req=dict(index=i,root_slot=root,condition=cond,task_id='fixedtask',selection_digest='literal_plan')
        requests.append(req);rows.append(dict(req,predictions={a:None for a in m.owner.ARMS},score=small(i)[2]))
    pairs=[]
    for s in m.protocol()['pair_schedule']:
        a,la,sa=small(s['minus_index']);b,lb,sb=small(s['plus_index'])
        pairs.append(m.pair_record(s,a,b,la,lb,sa,sb))
    return dict(requests=requests),dict(rows=rows,pairs=pairs)


def fake_branch(req,row,directory,used):
    assert all(req[k]==row[k] for k in ('index','root_slot','condition'))
    value=small(req['index'])
    m.require(value[2]==row['score'],'independent saved fixture score')
    return value


def test_effect_formula_and_mse_not_feature_interaction():
    values=dict(AA=np.array([0.,0.]),AP=np.array([1.,0.]),PA=np.array([0.,1.]),PP=np.array([1.,1.]))
    e=m.effects(values)
    np.testing.assert_array_equal(e['visual'],[0.,1.]);np.testing.assert_array_equal(e['proprio'],[1.,0.])
    np.testing.assert_array_equal(e['interaction'],[0.,0.])
    s=m.cell_scores(values,np.array([0.,0.]),{})
    assert s['common_label_cell_MSE']==dict(AA=0.,AP=.5,PA=.5,PP=1.)
    assert s['MSE_effects']==dict(visual=.5,proprio=.5,interaction=0.)


@pytest.mark.parametrize('bad',('missing','shape','nan'))
def test_four_corner_rejection(bad):
    x={c:np.zeros(2) for c in m.CELLS}
    if bad=='missing':x.pop('PP')
    if bad=='shape':x['PP']=np.zeros(3)
    if bad=='nan':x['PP'][0]=np.nan
    with pytest.raises(ValueError):m.effects(x)


def test_pair_zero_null_sign_and_full_span_not_fd():
    a={c:np.zeros(2) for c in m.CELLS};b={c:np.ones(2) for c in m.CELLS}
    la=np.zeros(2);lb=np.zeros(2)
    out=m.pair_record({},a,b,la,lb,m.cell_scores(a,la,{}),m.cell_scores(b,lb,{}))
    assert out['observed_response_L2']==0
    assert all(v['response_cosine'] is None for v in out['cells'].values())
    b['AP']=np.zeros(2);lb=np.ones(2)
    out=m.pair_record({},a,b,la,lb,m.cell_scores(a,la,{}),m.cell_scores(b,lb,{}))
    assert out['cells']['AP']['response_cosine'] is None
    assert out['cells']['AA']['predicted_response_L2']==float(np.sqrt(2))
    assert m.sign(-0.)=='zero' and m.sign(None)=='null'


def test_all164_80_baselines_strata_zero_identity(monkeypatch):
    p,r=fixture();monkeypatch.setattr(m,'branch',fake_branch)
    v=m.summarize(p,r,Path('/fixture'),{})
    assert len(v['branches'])==164 and len(v['pairs'])==80 and len(v['tables'])==9
    assert [v['index'] for v in v['branches'] if v['baseline']]==[0,41,82,123]
    assert v['zero_sets']['observed_pair_indices']==[0]
    assert v['zero_sets']['predicted_pair_indices']['AP']==[1]
    # Equal cardinality is explicitly NOT equal membership.
    assert v['zero_sets']['predicted_equals_observed']['AP'] is False
    alltable=v['tables'][0]['summary'];assert alltable['branches']==164 and alltable['pairs']==80
    assert all(sum(t.values())==80 for t in alltable['transitions'].values())
    assert [x['summary']['pairs'] for x in v['tables'] if x['factor']=='root_condition']==[20]*4


@pytest.mark.parametrize('bad',('omitbranch','omitpair','reorder','pairidentity','pairscore','rowscore','condition','boolindex'))
def test_full_fixed_denominator_and_arithmetic_mutations(monkeypatch,bad):
    p,r=fixture();monkeypatch.setattr(m,'branch',fake_branch)
    if bad=='omitbranch':r['rows'].pop()
    elif bad=='omitpair':r['pairs'].pop()
    elif bad=='reorder':r['rows'][1],r['rows'][2]=r['rows'][2],r['rows'][1]
    elif bad=='pairidentity':r['pairs'][0]['coord']=9
    elif bad=='pairscore':r['pairs'][0]['observed_response_L2']=1.
    elif bad=='rowscore':r['rows'][0]['score']['MSE_effects']['visual']+=1
    elif bad=='condition':p['requests'][0]['condition']='on'
    else:p['requests'][1]['index']=True
    with pytest.raises((ValueError,AssertionError)):m.summarize(p,r,Path('/fixture'),{})


def full_branch_fixture(tmp_path):
    req=dict(index=0,root_slot=0,condition='off')
    a=np.zeros((1,3,256,384),np.float32);a[0,:2]=99
    req['AA']=array_ref(tmp_path,'aa',a)
    # Poison provenance-only H2 output: it MUST NOT be read as PP.
    req['H2_second_output_provenance_only']={'directory':'/nonexistent','array':{}}
    req['common_label']=array_ref(tmp_path,'label',np.ones((1,256,384),np.float32))
    req['H2_label']=array_ref(tmp_path,'other',np.full((2,256,384),2.,np.float32))
    row=dict(index=0,root_slot=0,condition='off',predictions={})
    cells={'AA':a[0,-1].astype(np.float64)}
    for i,(arm,cell) in enumerate(zip(m.owner.ARMS,('AP','PA','PP'),strict=True)):
        b=np.full(a.shape,np.float32(i+1));b[0,:2]=-99
        ref=array_ref(tmp_path,cell,b);row['predictions'][arm]=ref['array'];cells[cell]=b[0,-1].astype(np.float64)
    row['score']=m.cell_scores(cells,np.ones((256,384)),dict(bitwise_equal=False,maxabs=1.))
    return req,row


def test_real_saved_arrays_terminal_semantics_newPP_only(tmp_path):
    q,r=full_branch_fixture(tmp_path);used={}
    cells,target,s=m.branch(q,r,tmp_path,used)
    assert set(s['common_label_cell_MSE'])==set(m.CELLS)
    assert s['common_label_cell_MSE']==dict(AA=1.,AP=0.,PA=1.,PP=4.)
    assert np.all(cells['PP']==3) and np.all(target==1) and len(used)==6


@pytest.mark.parametrize('bad',('filebytes','sha','trailing','dtype','shape','nonfinite','score','arms','identity'))
def test_real_saved_array_and_score_rejection(tmp_path,bad):
    q,r=full_branch_fixture(tmp_path)
    if bad=='filebytes':q['AA']['array']['file_bytes']+=1
    elif bad=='sha':q['AA']['array']['sha256']='0'*64
    elif bad=='trailing':
        ref=q['AA'];path=Path(ref['directory'])/ref['array']['file']
        with path.open('ab') as f:f.write(b'x')
        ref['array'].update(sha256=m.sha(path),file_bytes=path.stat().st_size)
    elif bad in ('dtype','shape','nonfinite'):
        a=np.zeros((1,3,256,384),np.float32)
        if bad=='dtype':a=a.astype(np.float64)
        if bad=='shape':a=a[:,0]
        if bad=='nonfinite':a.flat[0]=np.nan
        q['AA']=array_ref(tmp_path,'broken',a)
    elif bad=='score':r['score']['common_label_cell_MSE']['PP']=0.
    elif bad=='arms':r['predictions'].pop('latest_PP')
    else:r['index']=True
    with pytest.raises(ValueError):m.branch(q,r,tmp_path,{})


def authority_fixture(tmp_path,monkeypatch):
    p,r=fixture();p.update(schema=m.owner.NAME+'.prepare',protocol=m.owner.protocol(),bindings={})
    prepared=tmp_path/'prepare.json';ph=write_json(prepared,p)
    r.update(schema=m.owner.NAME+'.report',protocol=m.owner.protocol(),artifacts={},counts={},inherited_costs={},
        execution_binding=dict(prepared=str(prepared),prepare_sha256=ph))
    report=tmp_path/'report.json';rh=write_json(report,r)
    receipt=dict(schema=m.owner.NAME+'.consume',status='all164_latest_slot_factorial_and80_pairs_verified',report_sha256=rh,
        branches=164,pairs=80,new_predictor_calls=492,actual_native_AA_prefix_bitwise_verified=True,P1_admission=False,P5_admission=False)
    strict=tmp_path/'strict.json';sh=write_json(strict,receipt);calls=[]
    def consume(output,expected):calls.append((output,expected));return deepcopy(receipt)
    monkeypatch.setattr(m.owner,'consume',consume);monkeypatch.setattr(m,'branch',fake_branch)
    return dict(output=str(tmp_path),report_sha=rh,prepared=str(prepared),prepare_sha=ph,
        strict_receipt=str(strict),strict_sha=sh),receipt,calls


def test_full_strict_bridge_and_deterministic_verify(tmp_path,monkeypatch):
    args,receipt,calls=authority_fixture(tmp_path,monkeypatch)
    v=m.analyze(**args);assert len(calls)==1 and v['authority']['full_source_strict_rebuilt_here'] is True
    assert v['whole_history_comparison'] is None and v['new_neural_calls']==v['new_physics_calls']==0
    out=tmp_path/'summary.json';h=write_json(out,v)
    assert m.verify(out,h)['verified'] is True and len(calls)==2


@pytest.mark.parametrize('bad',('statusonly','wrongreport','prepare','wholehistory','externalsha','ownerdrift','bounddrift','missingprefix'))
def test_authority_mutations_failclosed(tmp_path,monkeypatch,bad):
    args,receipt,calls=authority_fixture(tmp_path,monkeypatch)
    if bad=='statusonly':args['strict_sha']=write_json(Path(args['strict_receipt']),dict(status=receipt['status']))
    elif bad=='wrongreport':receipt['report_sha256']='0'*64
    elif bad=='prepare':args['prepare_sha']='0'*64
    elif bad=='wholehistory':
        r=json.loads((tmp_path/'report.json').read_text());r['schema']='agent_p5_history_factorial_cuda_20261029_v1.report'
        args['report_sha']=write_json(tmp_path/'report.json',r)
    elif bad=='externalsha':args['report_sha']='0'*64
    elif bad=='ownerdrift':monkeypatch.setattr(m,'OWNER_SHA','0'*64)
    elif bad=='bounddrift':
        p=json.loads(Path(args['prepared']).read_text());p['bindings'][str(tmp_path/'report.json')]='0'*64
        args['prepare_sha']=write_json(Path(args['prepared']),p)
        r=json.loads((tmp_path/'report.json').read_text());r['execution_binding']['prepare_sha256']=args['prepare_sha']
        args['report_sha']=write_json(tmp_path/'report.json',r)
        receipt['report_sha256']=args['report_sha'];args['strict_sha']=write_json(Path(args['strict_receipt']),receipt)
    else:receipt['actual_native_AA_prefix_bitwise_verified']=False;args['strict_sha']=write_json(Path(args['strict_receipt']),receipt)
    with pytest.raises(ValueError):m.analyze(**args)
    if bad in ('prepare','wholehistory','externalsha','ownerdrift'):assert calls==[]


def test_describe_empty_signs_no_epsilon():
    assert m.describe([None,None])==dict(total=2,available=0,null=2,mean=None,min_median_max=None)
    assert m.sign(1e-300)=='positive' and m.sign(-1e-300)=='negative'
    with pytest.raises(ValueError):m.describe([np.nan])


def test_cold_import_no_neural_or_physics():
    code='import sys; from experiments.spectral_reachability import '+m.NAME+'; assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,check=True)
