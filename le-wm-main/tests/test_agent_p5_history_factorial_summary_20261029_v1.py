"""Synthetic saved-only fixtures, no production strict or neural/physics calls."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_history_factorial_summary_20261029_v1 as m


def zero_join():
    rows=[dict(s,missing=[],observed_response_zero=i<24) for i,s in enumerate(m.protocol()['pair_schedule'])]
    return dict(schema=m.ZERO_SCHEMA,status='all80_exact_identity_joined',rows=rows,P1_admission=False,P5_admission=False,
        historical_callback_membership=None,sufficient_original_cause_certified=False,bindings={},
        authority=dict(jepa_report_sha256=m.owner.PINS['H1'][1],jepa_receipt_sha256=m.owner.PINS['H1_strict'][1]))


def test_no_neural_environment_import_and_fixed_protocol():
    assert not any(k in sys.modules for k in ('torch','pygame','pymunk','cv2','stable_worldmodel'))
    p=m.protocol();assert p['branches']==164 and p['pairs']==80 and p['original_nonzero_pairs']==56 and p['original_zero_pairs']==24
    assert p['exploratory_post_outcome'] and not p['P5_admission'] and not p['physically_reachable_mixed_context']


def test_original_membership_and_baseline_not_double_counted():
    cohorts,mapping=m.original_cohorts(zero_join())
    assert len(cohorts)==80 and cohorts.count('fixed56')==56 and cohorts.count('fixed24')==24
    assert len(mapping)==164 and sum(v['cohort']=='baseline4' for v in mapping.values())==4
    assert sum(v['cohort']=='fixed56_endpoints112' for v in mapping.values())==112
    assert sum(v['cohort']=='fixed24_endpoints48' for v in mapping.values())==48


@pytest.mark.parametrize('change',('missing','order','unknown','wrongcount','ambiguous','admission'))
def test_cohort_mutations_rejected(change):
    z=zero_join()
    if change=='missing':z['rows'].pop()
    elif change=='order':z['rows'][0],z['rows'][1]=z['rows'][1],z['rows'][0]
    elif change=='unknown':z['rows'][0]['observed_response_zero']=None
    elif change=='wrongcount':z['rows'][0]['observed_response_zero']=False
    elif change=='ambiguous':z['rows'][0]['missing']=['unmapped']
    else:z['P5_admission']=True
    with pytest.raises(ValueError):m.original_cohorts(z)


def test_exact_signs_no_cutoff_and_null():
    assert [m.sign(x) for x in (-1e-300,1e-300,-0.,None)]==['negative','positive','zero','null']
    assert m.describe([None,0,2])==dict(total=3,available=2,null=1,mean=1.,min_median_max=[0.,1.,2.])
    assert m.describe([])==dict(total=0,available=0,null=0,mean=None,min_median_max=None)
    with pytest.raises(ValueError):m.describe([float('nan')])
    with pytest.raises(ValueError):m.sign(float('inf'))


def test_independent_effect_algebra_scalar_vector_and_mse_not_same():
    f=m.effects(dict(AA=1.,AP=3.,PA=5.,PP=11.))
    assert f==dict(visual=6.,proprio=4.,interaction=4.)
    x=np.arange(6).reshape(2,3);v={c:x+offset for c,offset in zip(m.CELLS,(1,3,5,11),strict=True)}
    assert np.array_equal(m.effects(v)['interaction'],np.full((2,3),4.))
    with pytest.raises(ValueError):m.effects({'AA':1})


def test_all12_transitions_all16_bins_including_null_and_zero():
    rows=[]
    for x,y in ((-1,1),(0,None),(None,-1),(1,0)):
        rows.append(dict(observed_response_L2=1.,cells={c:dict(response_cosine=v,predicted_response_L2=1.,plus_minus_MSE=0.)
            for c,v in zip(m.CELLS,(x,y,y,x),strict=True)},response_effect_L2=dict(visual=0.,proprio=0.,interaction=0.)))
    s=m.pair_summary(rows)
    assert s['pairs']==4 and len(s['transitions'])==12
    assert all(len(v)==16 and sum(v.values())==4 for v in s['transitions'].values())
    assert s['transitions']['AA_to_AP']['negative_to_positive']==1
    assert s['transitions']['AA_to_AP']['zero_to_null']==1
    assert s['transitions']['AA_to_AP']['null_to_negative']==1
    assert s['transitions']['AA_to_AP']['positive_to_zero']==1


def test_independent_pair_cosine_all_four_classes_and_fixed_zero_cohort(monkeypatch):
    specs=m.protocol()['pair_schedule'];cohorts,mapping=m.original_cohorts(zero_join())
    vecs=dict(AA=np.array([-1.,0.]),AP=np.array([1.,0.]),PA=np.array([0.,1.]),PP=np.array([0.,0.]))
    info={};branches=[];requests=[dict(index=i) for i in range(164)]
    for i in range(164):
        plus=mapping[i]['sign']=='plus';pair=mapping[i]['pair_index']
        cells={k:(v.copy() if plus else np.zeros(2)) for k,v in vecs.items()}
        label=np.array([1.,0.]) if plus and cohorts[pair]=='fixed56' else np.zeros(2)
        scores={c:float(np.mean((v-label)**2)) for c,v in cells.items()}
        branches.append(dict(common_label_cell_MSE=scores));info[i]=(cells,label,{})
    expected=[]
    for j,s in enumerate(specs):
        lo,hi=s['minus_index'],s['plus_index'];on=1. if cohorts[j]=='fixed56' else 0.
        cs={c:dict(predicted_response_L2=float(np.linalg.norm(v)),response_cosine=None if on==0 or c=='PP' else dict(AA=-1.,AP=1.,PA=0.)[c],
            plus_minus_MSE=branches[hi]['common_label_cell_MSE'][c]-branches[lo]['common_label_cell_MSE'][c]) for c,v in vecs.items()}
        expected.append(dict(s,observed_response_L2=on,cells=cs,response_effect_L2=dict(visual=.5,proprio=float(np.sqrt(1.25)),interaction=float(np.sqrt(5.)))))
    monkeypatch.setattr(m,'independent_branch',lambda req,*args:info[req['index']])
    rows=m.independent_pairs(dict(rows=requests,pairs=expected),dict(requests=requests),Path('/unused'),cohorts,branches,{},lambda:None)
    assert len(rows)==80 and [m.sign(rows[-1]['cells'][c]['response_cosine']) for c in m.CELLS]==['negative','positive','zero','null']
    assert all(rows[j]['cells'][c]['response_cosine'] is None for j in range(24) for c in m.CELLS)
    bad=deepcopy(expected);bad[-1]['cells']['AA']['response_cosine']=0.
    with pytest.raises(ValueError):m.independent_pairs(dict(rows=requests,pairs=bad),dict(requests=requests),Path('/unused'),cohorts,branches,{},lambda:None)


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);results=tmp_path/'results';results.mkdir()
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=2*1024**3))
    directory=results/(m.owner.NAME+'_synthetic');directory.mkdir()
    arrays=tmp_path/'arrays';arrays.mkdir();records={}
    def npy(key,shape,value):
        a=np.full(shape,value,np.float32);b=io.BytesIO();np.save(b,a,allow_pickle=False);p=arrays/(key+'.npy');p.write_bytes(b.getvalue())
        records[key]=dict(directory=str(arrays),array=dict(file=p.name,sha256=m.sha(p),file_bytes=p.stat().st_size,nbytes=a.nbytes,dtype='float32',shape=list(shape)))
        return records[key]
    for c,value in zip(m.CELLS,(1.,2.,3.,4.),strict=True):npy(c,(1,3,256,384),value)
    for j in (0,1):npy('label'+str(j),(1,256,384),j);npy('other'+str(j),(2,256,384),j+.125)
    for c in ('AP','PA'):(directory/(c+'.npy')).write_bytes((arrays/(c+'.npy')).read_bytes())
    z=zero_join();cohorts,mapping=m.original_cohorts(z);plus={s['plus_index'] for i,s in enumerate(m.protocol()['pair_schedule']) if cohorts[i]=='fixed56'}
    reqs=[];rows=[]
    for i in range(164):
        cond=m.owner.bridge.factor.ORDER[i//41];n=int(i in plus)
        req=dict(index=i,root_slot=cond[0],condition=cond[1],AA=records['AA'],PP=records['PP'],common_label=records['label'+str(n)],H2_label=records['other'+str(n)])
        row=dict(index=i,root_slot=cond[0],condition=cond[1],predictions={},score=None)
        for arm,cell in zip(m.owner.ARMS,('AP','PA'),strict=True):
            row['predictions'][arm]=deepcopy(records[cell]['array'])
        # Fixture arithmetic is written explicitly; production uses separate owner functions.
        scores={c:(v-n)**2 for c,v in zip(m.CELLS,(1.,2.,3.,4.),strict=True)}
        row['score']=dict(common_label_cell_MSE=scores,MSE_effects={k:float(v) for k,v in m.effects(scores).items()},
            feature_effect_L2=dict(visual=2*np.sqrt(256*384),proprio=np.sqrt(256*384),interaction=0.),
            label_H1_H2_numeric_seam=dict(bitwise_equal=False,maxabs=.125))
        reqs.append(req);rows.append(row)
    report=dict(schema=m.owner.NAME+'.report',status='all164_history_factorial_recorded',protocol=m.owner.protocol(),rows=rows,
        errors=[],source_errors=[],pairs=[],artifacts={c+'.npy':{k:records[c]['array'][k] for k in ('sha256','file_bytes')} for c in ('AP','PA')})
    branch_rows=[dict(rows[i],**mapping[i],**rows[i]['score']) for i in range(164)]
    for i,s in enumerate(m.protocol()['pair_schedule']):
        lo,hi=s['minus_index'],s['plus_index'];on=float(np.sqrt(256*384)) if cohorts[i]=='fixed56' else 0.
        report['pairs'].append(dict(s,observed_response_L2=on,cells={c:dict(predicted_response_L2=0.,response_cosine=None,
            plus_minus_MSE=branch_rows[hi]['common_label_cell_MSE'][c]-branch_rows[lo]['common_label_cell_MSE'][c]) for c in m.CELLS},
            response_effect_L2=dict(visual=0.,proprio=0.,interaction=0.)))
    prepared=dict(schema=m.owner.NAME+'.prepare',protocol=m.owner.protocol(),requests=reqs,
        bindings={str(Path(m.owner.__file__).resolve()):m.OWNER_SHA})
    def save(path,v):path.write_bytes(m.raw(v));return m.sha(path)
    pp=results/'synthetic_prepare.json';ph=save(pp,prepared)
    report['execution_binding']=dict(prepared=str(pp),prepare_sha256=ph)
    rp=directory/'report.json';rh=save(rp,report)
    strict=dict(schema=m.owner.NAME+'.consume',status='all164_history_factorial_and80_pairs_verified',report_sha256=rh,
        branches=164,pairs=80,new_predictor_calls=328,P1_admission=False,P5_admission=False)
    sp=results/'synthetic_strict.json';sh=save(sp,strict)
    zp=results/'synthetic_zero.json';zh=save(zp,z);monkeypatch.setattr(m,'ZERO_SHA',zh)
    calls=[]
    def consume(path,h):calls.append((path,h));assert h==rh;return deepcopy(strict)
    monkeypatch.setattr(m.owner,'consume',consume)
    for p in (tmp_path/'tests'/('test_'+m.NAME+'.py'),tmp_path/'innerdocs'/(m.NAME+'.md')):p.parent.mkdir(exist_ok=True);p.write_text('fixture binding\n')
    return NS(directory=directory,report=report,strict=strict,strict_path=sp,zero=z,zero_path=zp,records=records,calls=calls,
        args=dict(output=str(directory),report_sha=rh,strict_receipt=str(sp),strict_sha=sh,zero_analysis=str(zp),zero_sha=zh),results=results)


def test_independent4cell_scalar_and_batch_seam_exact(bundle):
    b=bundle;rp=m.owner.bridge.read(b.directory/'report.json');p=m.owner.bridge.read(rp['execution_binding']['prepared'])
    _,label,score=m.independent_branch(p['requests'][0],rp['rows'][0],b.directory,{})
    assert not label.any() and score['common_label_cell_MSE']==dict(AA=1.,AP=4.,PA=9.,PP=16.)
    assert score['label_H1_H2_numeric_seam']==dict(bitwise_equal=False,maxabs=.125)
    bad=deepcopy(rp['rows'][0]);bad['score']['common_label_cell_MSE']['AA']+=1e-12
    with pytest.raises(ValueError):m.independent_branch(p['requests'][0],bad,b.directory,{})


def test_full_synthetic_analyze_writer_verify_and_no_extra_production(bundle):
    b=bundle;r=m.analyze(**b.args)
    assert len(b.calls)==1 and r['source_strict_receipt_rebuilt_here'] and len(r['branches'])==164 and len(r['pairs'])==80
    assert len(r['tables'])==9
    table=r['tables'][0]
    assert {c:v['branches'] for c,v in table['branches'].items()}==dict(all164=164,baseline4=4,fixed56_endpoints112=112,fixed24_endpoints48=48)
    assert {c:v['pairs'] for c,v in table['pairs'].items()}==dict(all80=80,fixed56=56,fixed24=24)
    assert table['branches']['all164']['label_batch_seam']['bitwise_different']==164
    assert all(sum(v.values())==80 for v in table['pairs']['all80']['transitions'].values())
    target=b.results/(m.NAME+'_fixture.json');receipt=m.write_new(r,target)
    assert m.verify(target,receipt['sha256'])['status']=='saved_factorial_summary_recomputed'
    assert len(b.calls)==2
    with pytest.raises(ValueError):m.write_new(r,target)


def test_strict_receipt_must_be_rebuilt_not_status_trusted(bundle):
    b=bundle;bad=deepcopy(b.strict);bad['extra_unverified']=True;b.strict_path.write_bytes(m.raw(bad))
    with pytest.raises(ValueError,match='exactly rebuilt'):m.analyze(**dict(b.args,strict_sha=m.sha(b.strict_path)))
    assert len(b.calls)==1


def test_wrong_zero_authority_or_report_sha_rejected(bundle):
    with pytest.raises(ValueError):m.analyze(**dict(bundle.args,zero_sha='0'*64))
    assert not bundle.calls
    with pytest.raises((AssertionError,ValueError)):m.analyze(**dict(bundle.args,report_sha='0'*64))


def test_npy_trailing_data_and_scalar_tamper_rejected(bundle):
    r=deepcopy(bundle.records['AA']);p=Path(r['directory'])/r['array']['file'];p.write_bytes(p.read_bytes()+b'trailing')
    r['array']['sha256']=m.sha(p);r['array']['file_bytes']=p.stat().st_size
    with pytest.raises(ValueError):m.load_ref(r,{})


def test_disk_cap_no_overwrite_and_cli_parsing(bundle,monkeypatch,capsys):
    p=bundle.results/(m.NAME+'_writer_fixture.json')
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:NS(free=m.FLOOR))
    with pytest.raises(ValueError):m.write_new({'status':'fixture'},p)
    monkeypatch.setattr(sys,'argv',[m.NAME,'verify','--analysis','fake','--analysis-sha','0'*64])
    seen=[]
    monkeypatch.setattr(m,'verify',lambda *args:seen.append(args) or dict(status='fixture_recomputed'))
    m.main();assert seen==[('fake','0'*64)] and json.loads(capsys.readouterr().out)['status']=='fixture_recomputed'
