"""Pure RAM float32/AST/metadata regressions; no actual controller/physics."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_delivered_mean_raw_domain_20261029_v1 as m


def stats():return np.asarray([-.0087,.0068],np.float32),np.asarray([.2019,.2002],np.float32)


def test_full_time_major_layout_no_padding_no_action_selection():
    mean,std=stats();z=np.arange(30,dtype=np.float32).reshape(1,3,10)/32
    r=m.check_mean(z,mean,std)
    raw=np.frombuffer(bytes.fromhex(r['array_bits']['raw']),dtype=np.float32).reshape(15,2)
    expected=np.add(np.multiply(z.reshape(15,2),std,dtype=np.float32),mean,dtype=np.float32)
    assert m.same(raw,expected) and len(r['coordinates'])==30
    assert [(x['packet'],x['within_packet'],x['axis']) for x in r['coordinates']]==[
        (h,s,a) for h in range(3) for s in range(5) for a in range(2)]
    assert [x['raw_step'] for x in r['coordinates']]==[i for i in range(1,16) for _ in range(2)]
    assert all(not r[k] for k in ('raw_deployment_authorized','original_prediction_cost_reuse_authorized','CUDA_token_identity_certified'))


def test_domain_reports_every_coordinate_and_never_clips():
    mean,std=stats();z=np.full((1,3,10),100,np.float32);z[0,0,0]=-100
    r=m.check_mean(z,mean,std)
    assert r['status']=='blocked_native_domain' and len(r['out_of_domain'])==30
    assert r['raw_candidate'][0][0]<-1 and r['raw_candidate'][-1][-1]>1
    assert [x['flat_index'] for x in r['out_of_domain']]==list(range(30))


def test_inclusive_native_endpoints_bitexact_arithmetic_fixture():
    mean=np.zeros(2,np.float32);std=np.ones(2,np.float32)
    z=np.tile(np.asarray([-1.,1.],np.float32),15).reshape(1,3,10)
    r=m.check_mean(z,mean,std)
    assert r['domain_ok'] and r['roundtrip_bitexact']
    assert r['status']=='bitexact_in_domain_checks_only_not_released'
    assert not r['raw_deployment_authorized']


def test_one_ulp_outside_rejected_no_tolerance():
    z=np.zeros((1,3,10),np.float32);z.flat[0]=np.nextafter(np.float32(1),np.float32(np.inf));z.flat[29]=np.nextafter(np.float32(-1),np.float32(-np.inf))
    r=m.check_mean(z,np.zeros(2,np.float32),np.ones(2,np.float32))
    assert [x['flat_index'] for x in r['out_of_domain']]==[0,29]
    assert r['roundtrip_bitexact'] and not r['domain_ok']


def test_public_stats_roundtrip_nonexact_not_hidden_by_numeric_equality():
    mean,std=stats();z=np.full((1,3,10),np.float32(.001),np.float32)
    r=m.check_mean(z,mean,std)
    assert r['domain_ok'] and not r['roundtrip_bitexact'] and r['needs_paid_actual_token_replay']
    assert r['status']=='needs_paid_actual_token_replay'
    assert len(r['roundtrip_differences'])==30
    assert all(x['normalized_bits']!=x['roundtrip_bits'] and x['ulp_distance']>0 for x in r['roundtrip_differences'])
    assert not r['original_prediction_cost_reuse_authorized']


def test_signedzero_distinct_bits_even_with_zero_ulp_distance():
    mean,std=stats();z=np.zeros((1,3,10),np.float32);z.flat[0]=np.float32(-0.)
    r=m.check_mean(z,mean,std)
    assert not r['roundtrip_bitexact'] and r['needs_paid_actual_token_replay']
    assert [x['flat_index'] for x in r['signed_zero_differences']]==[0]
    c=r['coordinates'][0];assert c['normalized_bits']=='80000000' and c['roundtrip_bits']=='00000000'
    assert c['ulp_distance']==0 and c['zero_signs']['normalized']=='negative' and c['zero_signs']['roundtrip']=='positive'
    assert len(r['signed_zero_occurrences'])==30


@pytest.mark.parametrize('x,y,d',[(-1.,-1.,0),(0.,-0.,0),(-1.,1.,2130706432)])
def test_ordered_ulp_definition(x,y,d):
    a,b=np.array([x,y],np.float32).view(np.uint32)
    assert abs(m.ordered(a)-m.ordered(b))==d


def test_adjacent_negative_and_positive_ulps():
    for x in (-1.,1.):
        a=np.array([x,np.nextafter(np.float32(x),np.float32(np.inf))],np.float32).view(np.uint32)
        assert abs(m.ordered(a[0])-m.ordered(a[1]))==1


def test_nonfinite_derived_values_serializable_and_blocked():
    z=np.full((1,3,10),np.finfo(np.float32).max,np.float32)
    r=m.check_mean(z,np.zeros(2,np.float32),np.full(2,2,np.float32))
    assert len(r['out_of_domain'])==30 and not r['roundtrip_bitexact']
    assert r['raw_candidate'][0]==[None,None] and r['coordinates'][0]['ulp_distance'] is None
    json.loads(m.encode(r));assert r['coordinates'][0]['raw_bits']=='7f800000'


@pytest.mark.parametrize('fault',['shape','f64','nan','std_zero','std_negative','stats_shape'])
def test_input_abi_no_coerce_or_padding(fault):
    mean,std=stats();z=np.zeros((1,3,10),np.float32)
    if fault=='shape':z=z[:,:2]
    elif fault=='f64':z=z.astype(np.float64)
    elif fault=='nan':z.flat[0]=np.nan
    elif fault=='std_zero':std[0]=0
    elif fault=='std_negative':std[1]=-1
    else:mean=mean[:1]
    with pytest.raises(ValueError):m.check_mean(z,mean,std)


def test_literal_stats_ast_only_and_native_source_identity():
    mean,std=stats()
    parsed=m.literal_stats((m.ROOT/'experiments/run_pusht_counterfactual.py').read_text())
    assert m.same(parsed[0],mean) and m.same(parsed[1],std)
    env=m.ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py'
    contract=m.native_contract(env.read_text());assert contract['low']==[-1.,-1.] and contract['relative'] and contract['action_scale']==100
    assert not contract['extra_div100']


@pytest.mark.parametrize('old,new',[('low=-1.0','low=-2.0'),('dtype=np.float32','dtype=np.float64'),('relative=True','relative=False'),('self.action_scale = 100','self.action_scale = 1')])
def test_native_contract_mutations(old,new):
    text=(m.ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py').read_text()
    assert old in text
    with pytest.raises(ValueError):m.native_contract(text.replace(old,new))


def test_real_saved_metadata_normalizer_and_task_mapping_only():
    # Narrow metadata/AST checks, no saved actual-mean analysis or full consume.
    refs=m.authorities();docs={k:m.read(v['path'],v['sha256']) for k,v in refs.items()}
    cm,physical=docs['prepare'],docs['physical_prepare'];sources={}
    mean,std,n=m.source_authority(cm,physical,sources)
    assert len(sources)==len(m.SOURCE_PINS) and not n['original_Hi_scaler_certified']
    p=cm['delivery_artifact']['authorities']['program_receipt'];d=m.read(p['path'],p['sha256'])
    for i in (1,3):
        plan=d['plans'][i];assert physical['plans'][i]['planner_plan']==plan
        assert plan['normalization']==dict(mean=mean.tolist(),std=std.tolist())
        warm=np.asarray(plan['root_warm_raw10'],np.float32)
        _,norm=m.forward(warm,mean,std)
        assert m.same(norm,np.asarray(plan['root_context']['action_history'],np.float32).reshape(10,2))
        payload=cm['delivery_artifact']['rows'][3*(i//2)]['controller_payload']
        assert m.same(norm.reshape(1,2,10),np.asarray(payload['a'],np.float32))
        assert m.sha(plan['task_file'])==plan['task_sha256']
    assert docs['receipt']['report_sha256']==refs['report']['sha256']


def test_fixture_prepare_verify_and_no_overwrite(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    mean,std=stats();a=dict(schema=m.TAG+'.explicit_RAM_fixture',checks=m.check_mean(np.zeros((1,3,10),np.float32),mean,std))
    monkeypatch.setattr(m,'build',lambda:deepcopy(a));p=tmp_path/'results'/(m.TAG+'_fixture.json')
    out=m.prepare(p);assert not m.verify(p,out['sha256'])['raw_deployment_authorized']
    m.main(['verify','--manifest',str(p),'--manifest-sha256',out['sha256']]);assert json.loads(capsys.readouterr().out)['verified']
    with pytest.raises(FileExistsError):m.prepare(p)
    a['checks']['raw_deployment_authorized']=True
    with pytest.raises(ValueError):m.verify(p,out['sha256'])


def test_output_path_and_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    with pytest.raises(ValueError):m.output_path(tmp_path/'outside.json')
    with pytest.raises(ValueError):m.output_path(tmp_path/'results'/'wrong.json')
    monkeypatch.setattr(m,'CAP',1);monkeypatch.setattr(m,'build',lambda:dict(fixture=True))
    with pytest.raises(ValueError):m.prepare(tmp_path/'results'/(m.TAG+'_fixture.json'))


def test_cold_cli_no_torch_or_physics():
    script='import sys; from experiments.spectral_reachability import '+m.TAG+' as m; m.main(["authorities"]); assert "torch" not in sys.modules; assert "pymunk" not in sys.modules'
    p=subprocess.run([sys.executable,'-B','-c',script],cwd=m.ROOT,capture_output=True,text=True,check=True)
    assert set(json.loads(p.stdout))==set(m.PINS)


def build_fixture(monkeypatch):
    """Synthetic metadata/array read callbacks, NOT an authenticated HOST run."""
    h='a'*64;mean,std=stats();protocol=m.controller.protocol()
    refs={k:dict(path='/fixture/'+k+'.json',sha256=h) for k in m.PINS}
    refs['report']['path']='/fixture/output/report.json'
    sources={'fixture_helper':h};program=dict(plans=[None]*4)
    physical=dict(plans=[None]*4);delivery_rows=[{} for _ in range(6)]
    for root,i in ((0,1),(1,3)):
        warm=np.zeros((10,2),np.float32);_,a=m.forward(warm,mean,std)
        p=dict(normalization=dict(mean=mean.tolist(),std=std.tolist()),task_id='task'+str(root),task_sha256=h,
               blind_sha256='blind'+str(root),task_file='/fixture/task'+str(root)+'.gz',
               root_warm_raw10=warm.tolist(),root_context=dict(action_history=a.reshape(2,10).tolist()))
        program['plans'][i]=p;physical['plans'][i]=dict(planner_plan=p)
        delivery_rows[3*root]=dict(task_id=p['task_id'],task_sha256=h,blind_sha256=p['blind_sha256'],controller_payload=dict(a=a.reshape(1,2,10).tolist()))
    cm=dict(sources=sources,protocol=protocol,expected_state={'explicit_RAM_metadata':'not_a_model'},
        delivery_artifact=dict(authorities=dict(program_receipt=dict(path='/fixture/program.json',sha256=h)),rows=delivery_rows))
    receipt=dict(denominator=dict(total=6,returned=4,missing=2),status='all4_target_controller_saved_verified_2_arbitrary_missing',
                 predict_API=48,predictor_samples=216,manifest_sha256=h,report_sha256=h,local_checks=[])
    report=dict(manifest=refs['prepare']['path'],manifest_sha256=h,completed=True,errors=[],protocol=protocol,
                model_before=cm['expected_state'],model_after=cm['expected_state'],rows=[],files={})
    docs={refs['prepare']['path']:cm,refs['report']['path']:report,refs['receipt']['path']:receipt,
          refs['physical_prepare']['path']:physical,'/fixture/program.json':program}
    arrays={}
    for i in range(6):
        row=dict(slot=i,root=i//3,arm=m.controller.ARMS[i%3],status='missing' if i in (2,5) else 'returned',
                 debits=dict(known=True,predict_API=0,predictor_samples=0))
        if i in m.controller.ACTIVE:
            td=dict(file=f'slot{i}_trace.json',file_bytes=1,file_sha256=h,root=i//3)
            row.update(trace=td,arrays={},local_check={'explicit_RAM_fixture':True});report['files'][td['file']]=td
            receipt['local_checks'].append(dict(slot=i,root=i//3,arm=row['arm'],local_check=row['local_check']))
            trace=dict(completed=True,errors=[],spec=protocol['spec'],backend='host_cuda_installed_CEM',host_tensor_seam_checked=True,
                       selected=dict(mean='array_0'),native_return=dict(actions='array_1',mean='array_2'))
            docs['/fixture/output/'+td['file']]=trace
            for j in (0,1,2):
                desc=dict(file=f'slot{i}_array_{j}.npy',file_bytes=1,file_sha256=h,root=i//3,
                          shape=[1,3,10],dtype='<f4',data_sha256=h)
                row['arrays'][f'array_{j}']=desc
                report['files'][desc['file']]={k:desc[k] for k in ('file','file_bytes','file_sha256','root')}
                arrays[desc['file']]=np.full((1,3,10),i/32,np.float32)
        report['rows'].append(row)
    monkeypatch.setattr(m,'authorities',lambda:deepcopy(refs))
    monkeypatch.setattr(m,'read',lambda p,h:deepcopy(docs[str(p)]))
    monkeypatch.setattr(m,'sha',lambda p:h)
    monkeypatch.setattr(m.controller,'closure',lambda:dict(sources))
    monkeypatch.setattr(m,'source_authority',lambda cm,pp,sources:(mean,std,{'explicit_RAM_fixture':True}))
    monkeypatch.setattr(m.controller,'load_array',lambda folder,desc:arrays[desc['file']].copy())
    def forbidden(*args,**kwargs):raise AssertionError('never full strict/model in this tool')
    monkeypatch.setattr(m.controller,'consume',forbidden);monkeypatch.setattr(m.controller,'verify',forbidden)
    return docs,arrays,refs


def test_build_complete_six_slot_fixture_not_production(monkeypatch):
    _,_,_=build_fixture(monkeypatch);r=m.build()
    assert r['denominator']==dict(total=6,checked=4,missing=2)
    assert [x['slot'] for x in r['rows'] if x['checks'] is None]==[2,5]
    assert len(r['rows'])==6 and not r['raw_deployment_authorized'] and not r['original_prediction_cost_reuse_authorized']
    assert r['actual_new_calls']==dict(NN=0,physics=0,full_strict=0)
    assert all(len(x['checks']['coordinates'])==30 for x in r['rows'] if x['checks'])


@pytest.mark.parametrize('fault',['array_mean','receipt_check','status','slot','normalizer','task','warmup','missing_started'])
def test_build_metadata_and_array_mutations(monkeypatch,fault):
    docs,arrays,refs=build_fixture(monkeypatch);r=docs[refs['report']['path']];c=docs[refs['receipt']['path']]
    p=docs['/fixture/program.json']['plans'][1]
    if fault=='array_mean':arrays['slot0_array_1.npy'].flat[0]+=1
    elif fault=='receipt_check':c['local_checks'][0]['local_check']={'tampered':True}
    elif fault=='status':r['rows'][0]['status']='partial'
    elif fault=='slot':r['rows'][0]['slot']=1
    elif fault=='normalizer':p['normalization']['mean'][0]+=1
    elif fault=='task':p['task_id']='different'
    elif fault=='warmup':p['root_warm_raw10'][0][0]=.5
    else:r['rows'][2]['status']='returned'
    with pytest.raises(ValueError):m.build()
