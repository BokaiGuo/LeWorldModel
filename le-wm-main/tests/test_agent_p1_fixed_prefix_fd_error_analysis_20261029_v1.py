"""Synthetic saved receipt and tamper fixtures; no producer full strict."""
from copy import deepcopy
import pytest
from experiments.spectral_reachability import agent_p1_fixed_prefix_fd_error_analysis_20261029_v1 as m


@pytest.fixture
def receipt():
    rows=[];audits=[];invalid=[]
    for i in range(64):
        pairs=[];nulls=[]
        for space in m.SPACES:
            for scale in m.STEPS:
                for j in range(10):
                    k=len(pairs);valid=not (space=='chart_logit' and j==0)
                    ad=0.0 if j==1 else -1.0 if j==2 else 1.0
                    fd=2.0 if j==2 else 1.0
                    p=dict(space=space,requested_step=scale,coordinate=j,valid=valid,saved_AD_direction=ad,
                        plus_step=scale,minus_step=scale,denominator=2*scale,
                        plus_Q=fd*scale if valid else None,minus_Q=-fd*scale if valid else None,
                        finite_difference=fd if valid else None,error_vs_saved_AD=fd-ad if valid else None)
                    pairs.append(p)
                    if not valid:nulls.append(dict(pair=k,space=space,coordinate=j,requested_step=scale,reasons=['chart_ball_boundary']))
        n=sum(x['valid'] for x in pairs)
        rows.append(dict(index=i,status='sealed',result=dict(index=i,pairs=pairs,pair_denominator=40,valid_pairs=n,predict_calls=2*n)))
        audits.append(dict(index=i,pairs=40,valid_pairs=n,native_forwards=2*n));invalid.append(nulls)
    return dict(schema=m.host.NAME+'.consume',status='verified_model_only',complete=True,report_sha256='a'*64,
        group_denominator=64,pair_denominator=2560,rows=rows,audits=audits,invalid_causes=invalid,
        P1_admission=False,freshness=False,physical_release=False)


def test_full_denominators_null_and_zero_relative(receipt):
    x=m.analyze(receipt,'a'*64)
    assert len(x['rows'])==2560 and len(x['cross_scale'])==1280
    assert [b['denominator'] for b in x['by_space_scale']]==[640]*4
    assert [b['available'] for b in x['by_space_scale']]==[640,640,576,576]
    assert x['rows'][1]['relative_denominator']==0 and x['rows'][1]['relative_error'] is None
    assert x['rows'][2]['sign_relation']=='opposite' and x['rows'][2]['absolute_error']==3
    assert x['cross_scale_summary'][1]['available']==576
    assert not x['gradient_correctness_established'] and x['thresholds'] is None


@pytest.mark.parametrize('case',['partial','missing','coordinate','fd','err','cause','adscale','audit'])
def test_mutated_saved_data_rejected(receipt,case):
    r=deepcopy(receipt); p=r['rows'][0]['result']['pairs'][0]
    if case=='partial':r['complete']=False
    elif case=='missing':r['rows'].pop()
    elif case=='coordinate':p['coordinate']=4
    elif case=='fd':p['finite_difference']+=1
    elif case=='err':p['error_vs_saved_AD']+=1
    elif case=='cause':r['invalid_causes'][0][0]['reasons']=[]
    elif case=='adscale':r['rows'][0]['result']['pairs'][10]['saved_AD_direction']=2
    else:r['audits'][0]['native_forwards']+=2
    with pytest.raises(ValueError):m.analyze(r,'a'*64)


def test_external_bytes_must_equal_rebuilt(tmp_path,monkeypatch,receipt):
    out=tmp_path/'output';out.mkdir(); prepared=tmp_path/'prepared.json'
    prepared.write_bytes(m.raw(dict(bindings={})))
    report=dict(prepared=str(prepared),prepare_sha256=m.sha(prepared),artifact_sha256={})
    (out/'report.json').write_bytes(m.raw(report));report_sha=m.sha(out/'report.json')
    receipt['report_sha256']=report_sha
    persisted=tmp_path/'receipt.json';persisted.write_bytes(m.raw(receipt))
    monkeypatch.setattr(m,'rebuild',lambda *_:deepcopy(receipt))
    result=m.build(str(out),report_sha,str(persisted),m.sha(persisted))
    assert result['analysis']['pair_denominator']==2560
    persisted.write_bytes(m.raw(receipt)+b' ')
    with pytest.raises(ValueError,match='persisted receipt bytes'):
        m.build(str(out),report_sha,str(persisted),m.sha(persisted))
