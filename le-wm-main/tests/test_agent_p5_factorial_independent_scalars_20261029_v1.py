"""Independent synthetic RAM/files only. Never import producer or run actual strict."""
import ast
from copy import deepcopy
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_factorial_independent_scalars_20261029_v1 as m


def save(directory,name,a):
    p=directory/name;b=io.BytesIO();np.save(b,a,allow_pickle=False);p.write_bytes(b.getvalue())
    return dict(file=name,sha256=m.sha(p),file_bytes=p.stat().st_size,shape=list(a.shape),dtype=str(a.dtype),nbytes=a.nbytes)


@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path/'inner';(root/'results').mkdir(parents=True);(root/'tests').mkdir();(root/'innerdocs').mkdir()
    (root/'tests'/('test_'+m.NAME+'.py')).write_text('synthetic test binding')
    (root/'innerdocs'/(m.NAME+'.md')).write_text('synthetic doc binding')
    d=root/'results'/'synthetic-factorial';d.mkdir();inv={};rows=[];calls=[];slotrows=[]
    # First complete signed pair + nominal, real native shapes, no NN.
    # Only NEW module fixture N is reduced; production CLI enforces164.
    monkeypatch.setattr(m,'N',3);monkeypatch.setattr(m,'ROOT',root)
    full=m.schedule();monkeypatch.setattr(m,'schedule',lambda:deepcopy(full[:1]))
    for i in range(3):
        predictions={};features={}
        for j,cell in enumerate(m.CELLS):
            v=np.full((1,3,256,384),999,np.float32);v[:,-1]=np.float32(j+i/16.)
            p=np.full((1,3,256,16),j+100,np.float32)
            vr=save(d,f'v{i}_{cell}.npy',v);pr=save(d,f'p{i}_{cell}.npy',p)
            for rec in (vr,pr):inv[rec['file']]={k:rec[k] for k in ('sha256','file_bytes')}
            predictions[cell]={'$tuple':[{'$array':vr},None,{'$array':pr}]};features[cell]=v[:,-1].copy()
            calls.append(dict(index=4*i+j,T=3,status='returned',units=dict(api=1,batch_samples=1,frame_samples=3),outputs=predictions[cell]))
        label=np.full((2,256,384),-999,np.float32);label[1]=np.float32(i/32.)
        lr=save(d,f'label{i}.npy',label);inv[lr['file']]={k:lr[k] for k in ('sha256','file_bytes')}
        # Independent expected arithmetic on constants, not producer methods.
        ms={c:(j+i/32.)**2 for j,c in enumerate(m.CELLS)}
        ef=dict(visual=((ms['RA']-ms['AA'])+(ms['RR']-ms['AR']))/2,
            proprio=((ms['AR']-ms['AA'])+(ms['RR']-ms['RA']))/2,
            interaction=ms['RR']-ms['RA']-ms['AR']+ms['AA'],joint=ms['RR']-ms['AA'])
        norm=lambda v:float(np.sqrt(256*384*v*v))
        row=dict(index=i,root_slot=0,condition='off',task_id='synthetic-task',selection_digest='synthetic-selection',
            predictions=predictions,RR_verified=dict(inputs=[True]*3,full3_outputs=[True]*3),label=lr,
            score=dict(MSE=ms,MSE_effects=ef,feature_effect_L2=dict(visual=norm(2),proprio=norm(1),interaction=0.,joint=norm(3))))
        rows.append(row);slotrows.append((features,label[1:2]))
    pairs=[m.pair_scalars(s,slotrows[s['minus_index']],slotrows[s['plus_index']]) for s in m.schedule()]
    u=dict(api=12,batch_samples=12,frame_samples=36)
    r=dict(schema=m.OWNER+'.report',status='all164_early_history_factorial_recorded',rows=rows,calls=calls,pairs=pairs,
        summaries=m.summarize(rows,pairs),protocol=dict(branches=3,cells=list(m.CELLS),budget_mode='all_four_NEW_656',frames=[5,10,15],target=20,pairs=m.schedule(),predictor=u),
        errors=[],source_errors=[],counts=dict(attempted=u,returned=u,validated=u),model_before={'fixture':'weights'},
        model_after_predictions={'fixture':'weights'},model_after={'fixture':'weights'},artifacts=inv,elapsed_seconds=1.,P1_admission=False,P5_admission=False)
    report=d/'report.json';report.write_bytes(m.raw(r))
    c=dict(schema=m.OWNER+'.consume',status='all164_early_history_four_NEW_cells_and80_pairs_verified',report_sha256=m.sha(report),
        branches=3,pairs=1,predictor=u,RR_all_native_IO_bitwise_verified=True,prior_fullstrict_rebuilt=False,P1_admission=False,P5_admission=False)
    receipt=root/'results'/'synthetic-strict.json';receipt.write_bytes(m.raw(c))
    return NS(root=root,d=d,r=r,c=c,report=report,receipt=receipt,
        args=dict(report=str(report),report_sha=m.sha(report),receipt=str(receipt),receipt_sha=m.sha(receipt)))


def test_independent_native_shaped_whole_group_and_verify(case):
    a=m.analyze(**case.args)
    assert a['status']=='all164_fourcell_and80_pair_independent_scalars_verified'
    assert len(a['rows'])==3 and len(a['pairs'])==1 and len(a['all_slot_evidence'])==3
    assert a['rows'][0]['score']['MSE']==dict(AA=0.,AR=1.,RA=4.,RR=9.)
    assert a['summaries'][0]['responses']['RR']['signs']==dict(positive=1,negative=0,zero=0,null=0)
    assert a['verification_boundary']['all_full3_outputs_decoded'] and not a['verification_boundary']['prior_fullstrict_rebuilt']
    output=case.root/'results'/(m.NAME+'_fixture.json');meta=m.write_new(output,a)
    assert m.verify(output,meta['sha256'])['status']=='independent_factorial_scalar_analysis_reproduced'
    with pytest.raises(ValueError):m.write_new(output,a)


@pytest.mark.parametrize('fault',('receipt_sha','report_sha','partial','incomplete','RR','gates','wrongreport'))
def test_complete_explicit_authority_required_before_array_read(case,monkeypatch,fault):
    args=case.args.copy();c=deepcopy(case.c)
    if fault=='receipt_sha':args['receipt_sha']='0'*64
    elif fault=='report_sha':args['report_sha']='0'*64
    elif fault=='partial':c['status']='blocked_partial_factorial'
    elif fault=='incomplete':c['branches']=40
    elif fault=='RR':c['RR_all_native_IO_bitwise_verified']=False
    elif fault=='gates':c['P5_admission']=True
    else:c['report_sha256']='0'*64
    case.receipt.write_bytes(m.raw(c))
    if fault!='receipt_sha':args['receipt_sha']=m.sha(case.receipt)
    monkeypatch.setattr(m,'array',lambda *a:pytest.fail('arrays must not open before authoritative strict'))
    with pytest.raises(ValueError):m.analyze(**args)


@pytest.mark.parametrize('fault',('row','cell','alias','call','proprio','label','MSE','effect','pair','summary','counts','protocol'))
def test_tamper_rejects_all_cells_slots_and_independent_saved_scalars(case,fault):
    r=deepcopy(case.r)
    if fault=='row':r['rows'][-1]['condition']='on'
    elif fault=='cell':r['rows'][-1]['predictions'].pop('AR')
    elif fault=='alias':r['rows'][-1]['predictions']['AR']=r['rows'][-1]['predictions']['AA']
    elif fault=='call':r['calls'][-1]['index']=0
    elif fault=='proprio':
        rec=r['rows'][-1]['predictions']['RR']['$tuple'][2]['$array'];(case.d/rec['file']).write_bytes(b'broken')
    elif fault=='label':r['rows'][-1]['label']['shape']=[1,256,384]
    elif fault=='MSE':r['rows'][-1]['score']['MSE']['AR']+=1
    elif fault=='effect':r['rows'][-1]['score']['MSE_effects']['interaction']+=1
    elif fault=='pair':r['pairs'][-1]['cells']['RA']['cosine']=-1.
    elif fault=='summary':r['summaries'][0]['sign_transitions']['AA__to__RR']['positive__to__positive']=0
    elif fault=='counts':r['counts']['validated']['api']-=1
    else:r['protocol']['frames']=[0,5,10]
    case.report.write_bytes(m.raw(r));args=case.args|dict(report_sha=m.sha(case.report));c=deepcopy(case.c);c['report_sha256']=args['report_sha']
    case.receipt.write_bytes(m.raw(c));args['receipt_sha']=m.sha(case.receipt)
    with pytest.raises((ValueError,KeyError)):m.analyze(**args)


def test_all_full3_slots_decoded_finite_not_just_scored_last(case):
    r=deepcopy(case.r);rec=r['rows'][0]['predictions']['AR']['$tuple'][0]['$array']
    b=io.BytesIO((case.d/rec['file']).read_bytes());a=np.load(b,allow_pickle=False);a[0,0,0,0]=np.nan
    rec.update(save(case.d,rec['file'],a));r['artifacts'][rec['file']]={k:rec[k] for k in ('sha256','file_bytes')}
    # calls share original tree aliases through deepcopy, so failure is finiteness not missing link.
    case.report.write_bytes(m.raw(r));args=case.args|dict(report_sha=m.sha(case.report));c=case.c|dict(report_sha256=args['report_sha'])
    case.receipt.write_bytes(m.raw(c));args['receipt_sha']=m.sha(case.receipt)
    with pytest.raises(ValueError):m.analyze(**args)


def test_math_hand_values_zero_negative_cosine_and_all80_schedule():
    assert m.N==164 and len(m.schedule())==80
    assert m.effects(dict(AA=1.,AR=3.,RA=4.,RR=10.))==dict(visual=5.,proprio=4.,interaction=4.,joint=9.)
    z=np.zeros((1,256,384),np.float32);v=z.copy();v[0,0,0]=1
    a={c:z for c in m.CELLS};b={c:-v for c in m.CELLS}
    for spec in m.schedule():
        x=m.pair_scalars(spec,(a,z),(b,v))
        assert all(c['cosine']==-1. and c['cosine_sign']=='negative' for c in x['cells'].values())
        assert x['cells']['RR']['central_predicted_L2']==1/(2*spec['h'])
        x=m.pair_scalars(spec,(a,z),(b,z))
        assert all(c['cosine'] is None and c['observed_zero'] for c in x['cells'].values())
    assert [m.sign(x) for x in (None,0.,1.,-1.)]==['null','zero','positive','negative']
    perpendicular=z.copy();perpendicular[0,0,1]=1
    orthogonal={c:perpendicular for c in m.CELLS}
    p=m.pair_scalars(m.schedule()[0],(a,z),(orthogonal,v))
    assert all(x['cosine']==0. and x['cosine_sign']=='zero' and not x['predicted_zero'] and not x['observed_zero'] for x in p['cells'].values())


def test_independent_all164_80_fourstrata_zero_kept():
    z=np.zeros((1,256,384),np.float32);cs={c:z for c in m.CELLS}
    score=m.row_scalars(cs,z);rows=[dict(root_slot=m.ORDER[i//41][0],condition=m.ORDER[i//41][1],score=score) for i in range(164)]
    ps=[m.pair_scalars(s,(cs,z),(cs,z)) for s in m.schedule()];groups=m.summarize(rows,ps)
    assert [(x['branches'],x['pairs']) for x in groups]==[(164,80)]+[(41,20)]*4
    assert groups[0]['responses']['AA']['signs']==dict(positive=0,negative=0,zero=0,null=80)
    assert all(t['null__to__null']==80 for t in groups[0]['sign_transitions'].values())


@pytest.mark.parametrize('fault',('trailer','dtype','shape','symlink'))
def test_npy_strict_byte_contract(tmp_path,fault):
    a=np.zeros((1,3,256,384),np.float32);rec=save(tmp_path,'x.npy',a);inv={rec['file']:{k:rec[k] for k in ('sha256','file_bytes')}}
    if fault=='trailer':(tmp_path/'x.npy').write_bytes((tmp_path/'x.npy').read_bytes()+b'x')
    elif fault=='dtype':rec=save(tmp_path,'x.npy',a.astype(np.float64))
    elif fault=='shape':rec['shape']=[1,1,256,384]
    else:(tmp_path/'link.npy').symlink_to(tmp_path/'x.npy');rec['file']='link.npy'
    if fault=='trailer':rec.update(sha256=m.sha(tmp_path/'x.npy'),file_bytes=(tmp_path/'x.npy').stat().st_size)
    inv={rec['file']:{k:rec[k] for k in ('sha256','file_bytes')}}
    with pytest.raises(ValueError):m.array(tmp_path,rec,(1,3,256,384),inv,{})


def test_import_independence_cli_and_no_neural_or_producer(tmp_path):
    tree=ast.parse(Path(m.__file__).read_text());modules=[]
    for x in ast.walk(tree):
        if isinstance(x,ast.Import):modules.extend(a.name for a in x.names)
        elif isinstance(x,ast.ImportFrom):modules.append(x.module or '')
    assert all(not any(k in s for k in ('torch','experiments','producer','pymunk')) for s in modules)
    r=subprocess.run([sys.executable,'-m',m.__name__,'--help'],capture_output=True,text=True)
    assert r.returncode==0 and 'analyze' in r.stdout and 'verify' in r.stdout
    r=subprocess.run([sys.executable,'-c',f'import sys; import {m.__name__}; assert "torch" not in sys.modules; assert not any("early_history_factorial" in k for k in sys.modules)'],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
