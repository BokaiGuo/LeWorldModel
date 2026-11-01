"""RAM + temporary saved files only; no producer import or fullstrict."""
from copy import deepcopy
import gzip
from io import BytesIO
import json

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_same_action_error_decomposition_20261029_v1 as m


def v(x): return np.full(192,x,np.float32).tolist()


def fixture():
    p=dict(schema=m.ENC+'.prepare',expected_model_hash='a'*64,
           contract=dict(fixed_denominator=16,plan_denominator=4,checkpoints=[5,10,15,20]),
           rows=[],plans=[],physical_report_sha256='b'*64,physical_receipt_sha256='c'*64)
    r=dict(schema=m.ENC+'.report',status='terminal_encoded_science_blocked',contract=deepcopy(p['contract']),
           errors=[],blocked=[],model_unchanged=True,source_verified=True,model_hash_before='a'*64,
           model_hash_after='a'*64,counts={},rows=[],artifact_sha256={})
    receipt=dict(schema=m.ENC+'.consume',status='all16_H3_witness_RGB_verified_NOT_admitted',
                 fixed_denominator=16,plan_denominator=4,known_roots=2,fresh_roots=0,rows=[],counts={},
                 physical_report_sha256='b'*64,physical_receipt_sha256='c'*64)
    for obj in (r,receipt):
        obj.update(P1_admission=False,P3_admission=False,C3=False,physical_release=False)
    arrays={}
    for b in range(4):
        cmds=np.arange(40,dtype=np.float32).reshape(20,2)/64
        raw10=np.concatenate((np.zeros((5,2),np.float32),cmds[:5]))
        task=dict(goal=v(0),binding=dict(model_sha256='a'*64))
        witness=dict(task=task,witness_sha256=str(b)*64,request=dict(raw10=raw10.tolist()),
                     winner=dict(actions=cmds[5:].tolist(),whole=[dict(latent_history=[v(.5+j*.125)]) for j in range(3)]))
        plan=dict(commands=cmds.tolist(),task_id=f't{b}',selection_digest=str(b+1)*64,source_witness=witness)
        p['plans'].append(plan)
        for j,t in enumerate((5,10,15,20)):
            i=4*b+j; pred=None if j==0 else v(.5+(j-1)*.125); actual=v(.25 if j==0 else .75)
            history=np.concatenate((raw10,cmds[5:t]))[-10:]
            spec=dict(index=i,plan_index=b,root_slot=b//2,condition='off',arm=('endpoints','midpoints')[b%2],
                task_id=f't{b}',checkpoint=t,selection_digest=plan['selection_digest'],
                causal_raw10_bits=history.tobytes().hex(),physical_status='horizon_completed',
                causal_raw10=history.tolist(),model_hash='a'*64,witness_sha256=witness['witness_sha256'],
                task_binding=task['binding'],full20=cmds.tolist(),raw_float32_bits=cmds.tobytes().hex(),
                goal192=v(0),predicted192=pred,observed_anchor192=v(.25) if j==0 else None,
                model_Q=None if j==0 else m.q(m.vector(pred),m.vector(v(0))),
                native_metrics=dict(value=1),response=dict(step=t),source=dict(png=f'{i}.png'),status='not_started',latent=None)
            p['rows'].append(spec)
            row=deepcopy(spec);row.update(status='encoded',latent={'id':i});r['rows'].append(row);arrays[i]=actual
            saved={k:deepcopy(spec[k]) for k in m.IDENTITY}
            saved.update(role='observed_anchor_reencode' if j==0 else 'same_action_H3_prediction',
                predicted192=pred,actual192=actual,goal192=v(0),model_Q=spec['model_Q'],
                actual_encoded_Q=m.q(m.vector(actual),m.vector(v(0))),latent_missing=False,
                source=spec['source'],native_metrics=spec['native_metrics'],response=spec['response'],
                observed_anchor_reencode_MSE=0. if j==0 else None)
            vals=None if j==0 else m.decompose_point(pred,actual,v(0))
            for key in ('model_minus_actual_Q','absolute_Q_error'):
                saved[key]=None if vals is None else vals[key]
            saved['prediction_vs_encoded_MSE']=None if vals is None else vals['error_mse']
            receipt['rows'].append(saved)
    return p,r,receipt,arrays


def analyze(f):
    p,r,s,a=f
    return m.analyze_documents(p,r,s,lambda d:a[d['id']])


def test_full_roster_and_temporal():
    out=analyze(fixture())
    assert [len(out[k]) for k in ('points','anchors','temporal_pairs')]==[12,4,12]
    assert all(abs(x['identity_residual'])<1e-14 for x in out['temporal_pairs'])
    assert [x['checkpoint'] for x in out['points']]==[10,15,20]*4


def test_cancellation_nonzero_error():
    out=m.decompose_point(v(-1),v(1),v(0))
    assert out['error_mse']==4 and out['actual_residual_cross_term']==-4
    assert out['model_minus_actual_Q']==0
    assert out['alignment_cosine']==pytest.approx(-1,abs=1e-15)


def test_orthogonal_error():
    a=v(0);p=v(0);a[0]=p[0]=1.;p[1]=2.
    out=m.decompose_point(p,a,v(0))
    assert out['actual_residual_cross_term']==0 and out['alignment_cosine']==0
    assert out['error_mse']==pytest.approx(4/192)


def test_fixed_random_arithmetic():
    rng=np.random.default_rng(42)
    for _ in range(8):
        xs=[rng.normal(size=192).astype(np.float32).tolist() for _ in range(3)]
        out=m.decompose_point(*xs)
        assert abs(out['identity_residual'])<=out['identity_roundoff_bound']


def test_missing_anchor_stays_anchor():
    f=fixture();p,r,s,_=f
    p['rows'][0]['source']=r['rows'][0]['source']=s['rows'][0]['source']=None
    r['rows'][0]['latent']=None;r['rows'][0]['status']='missing_physical_checkpoint'
    s['rows'][0].update(actual192=None,latent_missing=True,actual_encoded_Q=None,observed_anchor_reencode_MSE=None)
    out=analyze(f)
    assert out['coverage']['anchors']['missing']==1
    assert out['coverage']['forecasts']['available']==12


@pytest.mark.parametrize('p,a,g',[(1,1,0),(1,0,0),(2,1,0),(-2,1,0)])
def test_identities(p,a,g):
    x=m.decompose_point(v(p),v(a),v(g))
    assert x['identity_residual']==x['prediction_form_identity_residual']==0
    if p==a or a==g: assert x['alignment_cosine'] is None


def test_missing_keeps_all_denominators():
    f=fixture();p,r,s,_=f;i=6
    p['rows'][i]['source']=r['rows'][i]['source']=s['rows'][i]['source']=None
    r['rows'][i]['latent']=None;r['rows'][i]['status']='missing_physical_checkpoint'
    sr=s['rows'][i];sr.update(actual192=None,latent_missing=True)
    for key in ('actual_encoded_Q','model_minus_actual_Q','absolute_Q_error','prediction_vs_encoded_MSE'):
        sr[key]=None
    out=analyze(f)
    assert out['coverage']['forecasts']==dict(fixed=12,available=11,missing=1)
    assert out['coverage']['temporal_pairs']['missing']==2
    assert out['points'][4]['model_Q'] is not None and out['points'][4]['error_mse'] is None


@pytest.mark.parametrize('bad',[True,float('nan'),float('inf'),'0',None])
def test_bad_vector_values(bad):
    a=v(0);a[0]=bad
    with pytest.raises(ValueError):m.decompose_point(a,v(0),v(0))


@pytest.mark.parametrize('which',['schema','identity','goal','Q','boolQ','rawbits','witness','prediction','gate','missing'])
def test_mutations(which):
    f=fixture();p,r,s,a=f
    if which=='schema':s['schema']='old.consume'
    elif which=='identity':s['rows'][1]['checkpoint']=15
    elif which=='goal':s['rows'][1]['goal192']=v(2)
    elif which=='Q':s['rows'][1]['model_minus_actual_Q']+=1
    elif which=='boolQ':s['rows'][1]['model_Q']=True
    elif which=='rawbits':p['rows'][1]['raw_float32_bits']='00'
    elif which=='witness':p['plans'][0]['source_witness']['winner']['actions'][0][0]=.9
    elif which=='prediction':p['plans'][0]['source_witness']['winner']['whole'][0]['latent_history'][-1]=v(2)
    elif which=='gate':s['P3_admission']=True
    elif which=='missing':s['rows'][1]['latent_missing']=True
    with pytest.raises((ValueError,KeyError)):analyze(f)


def save_bundle(tmp_path):
    p,r,s,a=fixture();out=tmp_path/'encoder';out.mkdir()
    for row in r['rows']:
        i=row['index'];b=BytesIO();np.save(b,np.asarray(a[i],np.float32),allow_pickle=False)
        payload=gzip.compress(b.getvalue(),mtime=0);name=f'{i}.npy.gz';(out/name).write_bytes(payload)
        d=dict(file=name,sha256=m.sha(payload),array_sha256=m.sha(np.asarray(a[i],np.float32).tobytes()),shape=[192],dtype='float32')
        row['latent']=d;r['artifact_sha256'][name]=d['sha256']
    prep=tmp_path/'prepare.json';prep.write_bytes(m.canonical(p));ph=m.sha(prep.read_bytes())
    r.update(prepared=str(prep),prepare_sha256=ph)
    rp=out/'report.json';rp.write_bytes(m.canonical(r));rh=m.sha(rp.read_bytes())
    s['report_sha256']=rh;receipt=tmp_path/'receipt.json';receipt.write_bytes(m.canonical(s))
    return dict(prepared_path=str(prep),prepare_sha256=ph,output_dir=str(out),report_sha256=rh,
                receipt_path=str(receipt),receipt_sha256=m.sha(receipt.read_bytes()))


def test_real_saved_bundle_and_exclusive_writer(tmp_path):
    args=save_bundle(tmp_path);result=m.build_saved_decomposition(**args)
    assert result['evidence_scope']['full_upstream_rebuilt_here'] is False
    target=tmp_path/(m.NAME+'_fixture.json')
    m.write(target,result)
    with pytest.raises(FileExistsError):m.write(target,result)


@pytest.mark.parametrize('field',['prepare_sha256','report_sha256','receipt_sha256'])
def test_external_sha_mutation(tmp_path,field):
    args=save_bundle(tmp_path);args[field]='0'*64
    with pytest.raises(ValueError):m.build_saved_decomposition(**args)


def test_coherent_receipt_schema_mutation(tmp_path):
    args=save_bundle(tmp_path);path=tmp_path/'receipt.json';s=json.loads(path.read_bytes())
    s['schema']='other.consume';path.write_bytes(m.canonical(s));args['receipt_sha256']=m.sha(path.read_bytes())
    with pytest.raises(ValueError):m.build_saved_decomposition(**args)


def test_array_mutation(tmp_path):
    args=save_bundle(tmp_path);(tmp_path/'encoder/1.npy.gz').write_bytes(b'bad')
    with pytest.raises(ValueError):m.build_saved_decomposition(**args)


@pytest.mark.parametrize('data',[b'{"a":1,"a":2}',b'{"x":NaN}'])
def test_json_rejects(data):
    with pytest.raises(ValueError):m.decode(data)
