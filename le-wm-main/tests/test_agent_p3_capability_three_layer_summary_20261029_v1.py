"""Synthetic saved ABI only. No real analysis, NN, physics or ancestor fullstrict."""
from copy import deepcopy
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p3_capability_three_layer_summary_20261029_v1 as m


def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(m.saved.encoded(v)+b'\n');return m.saved.ref(str(p))


def read(r):return json.loads(Path(r['path']).read_bytes())


def fixture(tmp_path,missing=(),conflict=False):
    """Full ten slots with own temporary NPY files and Pauli's published ABI."""
    out=tmp_path/'rgb';out.mkdir();stageplans=[];probes=[];specs=[];rows=[];tasks=[];original=[];originalrows=[];oldbranches=[]
    resources=dict(attempted=dict(predict_API=24,VJP_directions=1344),completed=dict(predict_API=24,VJP_directions=1344),FLOPs=None)
    counts={k:dict(attempted=n,completed=n) for k,n in m.saved.COUNTS.items()}
    physicalrefs={k:dict(path=str(tmp_path/(k+'.json')),sha256=hashlib.sha256(k.encode()).hexdigest()) for k in ('prepared','batch','receipt')}
    authorities={key:physicalrefs[a] for key,a in (('physical_prepare','prepared'),('physical_batch','batch'),('physical_receipt','receipt'))}
    artifacts={}
    for e,(root,role) in enumerate(m.ORDER):
        ctx=dict(latent_history=[[0.]*192,[1.]*192,[2.]*192],action_history=[[0.]*10,[0.]*10],
            action_dim=10,latent_dim=192,context_length=3,layout='states_with_between_state_actions_v1')
        t=dict(goal=[0.]*192,projection=np.eye(192).tolist(),action_mean=[0.,0.],action_std=[1.,1.],root_context=ctx,
            binding=dict(model_sha256='RAM_only_original192'))
        for k,v in (('goal_sha256',t['goal']),('projection_sha256',t['projection']),
                    ('normalizer_sha256',[t['action_mean'],t['action_std']]),('context_sha256',ctx)):
            t['binding'][k]=m.digest(v)
        raw5=[[.125,.125]]*5
        u=deepcopy(ctx);u['latent_history']=ctx['latent_history'][1:]+[[1.]*192];u['action_history']=[ctx['action_history'][-1],[.125]*10]
        z=deepcopy(ctx);z['latent_history']=ctx['latent_history'][1:]+[[2.]*192]
        plan=dict(index=e,root_slot=root,role=role,task=t,task_sha256=m.digest(t),endpoint_identity=dict(index=e,root_slot=root,role=role),
            raw5=raw5,causal_raw10=[[0.,0.]]*10,whole=[u],segmented=[u],nominal_context=z,
            signal=dict(nominal=[2.]*192,parent_context_sha256=m.digest(ctx),binding_sha256=m.digest(t['binding'])),Q=dict(whole=1.,nominal=4.))
        plan['witness_sha256']=m.digest(plan);stageplans.append(plan)
        probes.append(dict(index=e,root_slot=root,role=role,witness_sha256=plan['witness_sha256'],task_sha256=plan['task_sha256'],
            endpoint_identity=plan['endpoint_identity'],native_cp10=dict(goal7=[0.]*7),Q_original_numpy=plan['Q'],Q_fsum=plan['Q']))
        for arm in ('U','Z2'):
            i=len(rows);raw15=[[0.,0.]]*10+(raw5 if arm=='U' else [[0.,0.]]*5)
            incoming=np.asarray(raw15[-10:],dtype='<f4');tokens=incoming.reshape(2,10)
            state=[3.,4.,0.,0.,0.,12. if arm=='U' else 20.,0.]
            if conflict and root==1:
                state=[3.,4.,0.,0.,0.,20.,0.] if arm=='U' else [6.,8.,0.,0.,0.,0.,0.]
            cp=dict(phase='plan',step=15,overlay_index=0,condition='off',state=state,snapshot=dict(state=state,geometry='RAM_complete_snapshot'),
                png='cp15.png',file_sha256='1'*64,pixel_sha256='2'*64)
            metrics=m.native_math(state,[0.]*7)
            response=dict(step=15,state=state,reward=-metrics['native7_distance'],native_success=True,terminated=True,truncated=False)
            job=dict(selection_digest='RAM_only_selection',raw15=raw15)
            pr=dict(job=job,counts=counts,status='horizon_completed',endpoint15=cp,responses=[response])
            if i in missing:
                pr.update(status='native_stopped',endpoint15=None,responses=[])
            entry=dict(schema=m.saved.PHYSICAL+'.job_consume',output=str(tmp_path/f'job{i:02}'),row=pr)
            original.append(entry);originalrows.append(pr)
            source=None if i in missing else cp|dict(path=str(Path(entry['output'])/'cp15.png'))
            row=dict(index=i,endpoint=e,root_slot=root,role=role,probe=arm,checkpoint=15,absolute_step=25,history_absolute_steps=[15,20,25],
                source=source,physical_receipt_entry_sha256=m.digest(entry),physical_status=pr['status'],physical_counts=counts,
                physical_response=None if source is None else response,missing_reason='physical_cp15_missing' if source is None else None,
                raw15=raw15,raw15_bits=np.asarray(raw15,dtype='<f4').tobytes().hex(),causal_raw10=incoming.tolist(),causal_raw10_bits=incoming.tobytes().hex(),
                normalized_action_history=tokens.tolist(),normalized_float32_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
                cp10_task=t,cp10_identity=plan['endpoint_identity'],selection_digest=job['selection_digest'],unknown_semantics=dict(FLOPs=None),
                status='missing_physical_cp15' if source is None else 'not_started',latent=None)
            specs.append(deepcopy(row))
            oldbranches.append(dict(index=i,endpoint=e,root_slot=root,role=role,probe=arm,
                metrics=None if source is None else metrics,original_native_success=None if source is None else True,
                reported_D_original_reward=None if source is None else metrics['native7_distance'],
                D_original_reward=None if source is None else metrics['native7_distance'],eligible_same_time_endpoint=source is not None))
            if source is None:
                rows.append(row);tasks.append(None);continue
            latent=np.full(192,1.5 if arm=='U' else 2.5,dtype='<f4')
            buf=io.BytesIO();np.save(buf,latent,allow_pickle=False);f=f'r{i}_latent.npy.gz';(out/f).write_bytes(gzip.compress(buf.getvalue(),mtime=0))
            artifacts[f]=m.sha(str(out/f))
            descriptor=dict(file=f,sha256=artifacts[f],array_sha256=hashlib.sha256(latent.tobytes()).hexdigest(),shape=[192],dtype='float32')
            row.update(status='encoded',latent=descriptor,pixels=dict(RAM_not_loaded=True),preprocessed=dict(RAM_not_loaded=True),modules=[])
            identity=dict(namespace=m.RGB,index=i,root_slot=root,role=role,probe=arm,history_absolute_steps=[15,20,25],
                source=source,cp10_identity=plan['endpoint_identity'],raw10_sha256=hashlib.sha256(incoming.tobytes()).hexdigest(),
                encoded_latent=descriptor,cp10_task_sha256=m.digest(t))
            newctx=deepcopy(ctx);newctx['latent_history']=ctx['latent_history'][1:]+[latent.tolist()];newctx['action_history']=tokens.tolist()
            newt=deepcopy(t);newt['root_context']=newctx
            newt['binding'].update(context_sha256=m.digest(newctx),task_id=m.digest([m.RGB,identity]),
                task_sha256=m.digest([t,newctx,identity]),task_contract_sha256=m.digest(identity))
            tasks.append(dict(index=i,task=newt,task_sha256=m.digest(newt),identity=identity,**m.FLAGS));rows.append(row)
    stage=dict(schema=m.B+'.consume',status='saved_complete_NOT_admitted',plans=stageplans,resource_receipt=resources)
    bref=put(tmp_path/'B.json',stage)
    cap=dict(schema='agent_p3_capability_signal_summary_20261029_v1.summary',signals=[{}]*7,probes=probes,
        authorities=dict(stageB_receipt=bref),inherited_stageB_resource_receipt=resources,
        **{k:v for k,v in m.FLAGS.items() if k!='equal_total_compute'})
    capref=put(tmp_path/'cap.json',cap)
    pairs=[]
    for e in range(5):
        u,z=oldbranches[2*e:2*e+2]
        pairs.append(dict(G=m.difference(z['D_original_reward'],u['D_original_reward'])['value'],
            error_improvements={k:dict(A_minus_B=m.difference(None if z['metrics'] is None else z['metrics'][k],
                None if u['metrics'] is None else u['metrics'][k])['value']) for k in m.AXES},
            success_relation='null' if u['metrics'] is None or z['metrics'] is None else 'tie'))
    ph=dict(schema=m.saved.NAME+'.summary',fixture_only=True,original_receipts=original,original_rows=originalrows,
        mapped_rows=[dict(terminal_control_step=10 if i in missing else 15) for i in range(10)],
        known_roots=2,fresh_roots=0,upstream_fullstrict_reexecuted=False,request=physicalrefs|dict(capability=capref),
        inherited_stageB_resource_receipt=resources,native_resource_vectors=[counts]*10,
        arithmetic=dict(branches=oldbranches,U_Z2_pairs=pairs,
            C_control_pairs=[dict(index=2*ci+j,root_slot=root,C_arm=arm,control_role=role,original_C_no_witness=ci!=0)
                for ci,(root,arm) in enumerate(((0,'H1'),(0,'H3'),(1,'H1'),(1,'H3'))) for j,role in enumerate(('Z','P'))],
            C_dual_control_conditions=[dict(root_slot=root,C_arm=arm,original_immediate_cost_both_controls=False if i==0 else None)
                for i,(root,arm) in enumerate(((0,'H1'),(0,'H3'),(1,'H1'),(1,'H3')))]),**m.FLAGS)
    phref=put(tmp_path/'ph.json',ph)
    n=10-len(missing);counts={k:dict(attempted=n if k in ('encoder','projector') else 0,
        completed=n if k in ('encoder','projector') else 0) for k in ('encoder','projector','action_encoder','predict_API','predictor','pred_proj')}
    source=Path(m.__file__).parent/(m.RGB+'.py');sref=m.saved.ref(str(source))
    p=dict(schema=m.RGB+'.prepare',fixture_only=True,bindings={str(source):sref['sha256']},authorities=authorities,rows=specs,
        expected_calls={k:v['completed'] for k,v in counts.items()},expected_model_hash='RAM_frozen_192')
    pref=put(tmp_path/'prepare.json',p)
    tr=put(out/'cp15_tasks.json',tasks);artifacts['cp15_tasks.json']=tr['sha256']
    r=dict(schema=m.RGB+'.report',prepared=pref['path'],prepare_sha256=pref['sha256'],status='all10_cp15_slots_saved_RGB_bridge_NOT_admitted',
        errors=[],blocked=[],source_verified=True,model_hash_before=p['expected_model_hash'],model_hash_after=p['expected_model_hash'],
        model_inventory_before=[dict(RAM=True)],model_inventory_after=[dict(RAM=True)],counts=counts,rows=rows,tasks_sha256=tr['sha256'],elapsed_seconds=1.,
        **{k:v for k,v in m.FLAGS.items() if k!='P2_admission'})
    rr=put(out/'report.json',r);artifacts['report.json']=rr['sha256']
    c=dict(schema=m.RGB+'.completion',prepared=pref['path'],prepare_sha256=pref['sha256'],report_sha256=rr['sha256'],
        status='published_within_cap',inclusive_wall_seconds=2.,artifact_sha256=artifacts,**{k:v for k,v in m.FLAGS.items() if k!='P2_admission'})
    cr=put(out/'completion.json',c)
    s=dict(schema=m.RGB+'.consume',status=r['status'],report_sha256=rr['sha256'],completion_sha256=cr['sha256'],rows=rows,tasks=tasks,
        fixed_denominator=10,encoded=n,missing=10-n,known_roots=2,fresh_roots=0,counts=counts,source_authorities=authorities,
        consume_new_neural_calls=0,consume_new_physics=0,total_FLOPs=None,actual_module_IO_bytes=n*1024,
        **{k:v for k,v in m.FLAGS.items() if k!='P2_admission'})
    sr=put(tmp_path/'receipt.json',s)
    return dict(rgb_prepare=pref,rgb_report=rr,rgb_completion=cr,rgb_receipt=sr,rgb_source=sref,
        physical_summary=phref,capability_summary=capref,stageB_receipt=bref)


def edit(req,key,fn):
    value=read(req[key]);fn(value);req[key]=put(Path(req[key]['path']),value)


def edit_rows(req,fn):
    r=read(req['rgb_report']);s=read(req['rgb_receipt']);c=read(req['rgb_completion'])
    fn(r['rows'][0]);s['rows']=deepcopy(r['rows'])
    req['rgb_report']=put(Path(req['rgb_report']['path']),r)
    c['report_sha256']=c['artifact_sha256']['report.json']=req['rgb_report']['sha256']
    req['rgb_completion']=put(Path(req['rgb_completion']['path']),c)
    s['report_sha256']=req['rgb_report']['sha256'];s['completion_sha256']=req['rgb_completion']['sha256']
    req['rgb_receipt']=put(Path(req['rgb_receipt']['path']),s)


def test_full_tempdisk_ten_build_write_verify(tmp_path):
    req=fixture(tmp_path);v=m.build(req,fixture=True)
    assert len(v['branches'])==10 and len(v['pairs'])==5
    assert all(x['model_gain']['value']==3. and x['real_latent_gain']['value']==4. for x in v['pairs'])
    assert all(x['signs']['model_vs_real']=='same' for x in v['pairs'])
    assert v['cross_context_model_comparisons']==[]
    assert len(v['original_C_control_pairs'])==8 and len(v['original_C_dual_control_conditions'])==4
    assert v['original_C_dual_control_conditions'][0]['original_immediate_cost_both_controls'] is False
    assert all(v[k] is False for k in m.FLAGS)
    r=m.saved.write_new(str(tmp_path/'summary.json'),v)
    assert m.verify(r['path'],r['sha256'],fixture=True)['fixture_only']
    with pytest.raises(ValueError):m.build(req)


@pytest.mark.parametrize('missing',[(0,),(1,),(2,3),(9,),tuple(range(10))])
def test_fixed_missing_and_earlystop_no_padding(tmp_path,missing):
    out=m.build(fixture(tmp_path,missing),fixture=True)
    assert len(out['branches'])==10 and len(out['pairs'])==5
    for i in missing:
        assert out['branches'][i]['actual_latent192'] is None
        assert out['branches'][i]['latent']['actual_Q_fsum'] is None
        assert out['pairs'][i//2]['real_latent_gain']['sign']=='null'
        assert out['pairs'][i//2]['model_gain']['value']==3.


def test_independent_arithmetic_nonzero_cross_and_units():
    v=m.latent_math([1.]*192,[2.]*192,[.5]*192)
    assert v==dict(predicted_Q_fsum=.25,actual_Q_fsum=2.25,error_MSE=1.,cross_term=1.,
        actual_minus_predicted_Q=2.,decomposition_roundoff=0.)
    n=m.native_math([3.,4.,12.,0.,.5,8.,15.],[0.]*7)
    assert n['native7_squared']==458.25 and n['joint_xy_squared']==169. and n['velocity_squared']==289.
    assert n['object_xy_error']==12. and n['raw_angle_squared']==.25
    assert m.native_math([0.,0.,0.,0.,7.,0.,0.],[0.]*7)['native_angle_term']<0


def test_full_fixture_keeps_XY_velocity_native_conflict(tmp_path):
    out=m.build(fixture(tmp_path,conflict=True),fixture=True)
    for pair in out['pairs'][3:]:
        assert pair['model_gain']['sign']==pair['real_latent_gain']['sign']=='positive'
        assert pair['native_reward_distance_gain']['sign']=='negative'
        assert pair['native_axis_gains']['joint_xy_error']['value']==5.
        assert pair['native_axis_gains']['velocity_error']['value']==-20.
        assert pair['signs']['real_vs_native']=='reversal'
        assert pair['signs']['real_vs_native_axes']['joint_xy_error']=='same'
        assert pair['signs']['real_vs_native_axes']['velocity_error']=='reversal'
        assert pair['native_success']['U'] is pair['native_success']['Z2'] is True


def test_signs_no_product_underflow_and_ties_null():
    assert m.alignment(m.difference(5e-324,0.),m.difference(0.,5e-324))=='reversal'
    assert m.alignment(m.difference(0.,0.),m.difference(0.,0.))=='both_tie'
    assert m.alignment(m.difference(1.,0.),m.difference(0.,0.))=='one_tie'
    assert m.alignment(m.difference(None,0.),m.difference(1.,0.))=='null'


@pytest.mark.parametrize('key',list(m.PINS))
def test_full_external_sha_binding(tmp_path,key):
    req=fixture(tmp_path);req[key]['sha256']='0'*64
    with pytest.raises(ValueError):m.build(req,fixture=True)


@pytest.mark.parametrize('mutation',[
    lambda s:s['rows'].pop(),lambda s:s['rows'].reverse(),lambda s:s.update(fixed_denominator=9),
    lambda s:s.update(encoded=9),lambda s:s['source_authorities']['physical_receipt'].update(sha256='0'*64),
    lambda s:s.update(P3_admission=True),lambda s:s['counts']['predict_API'].update(attempted=1,completed=1),
    lambda s:s['tasks'][0]['task']['root_context']['latent_history'][0].__setitem__(0,9.)])
def test_rgb_receipt_mutations(tmp_path,mutation):
    req=fixture(tmp_path);edit(req,'rgb_receipt',mutation)
    with pytest.raises(ValueError):m.build(req,fixture=True)


@pytest.mark.parametrize('mutation',[
    lambda r:r['cp10_task']['root_context']['latent_history'][0].__setitem__(0,9.),
    lambda r:r['raw15'][10].__setitem__(0,.25),lambda r:r['source']['snapshot'].update(geometry='wrong'),
    lambda r:r['normalized_action_history'][0].__setitem__(0,99.),
    lambda r:r['physical_response'].update(native_success=False),lambda r:r['latent'].update(array_sha256='0'*64)])
def test_resealed_rows_semantic_mutations(tmp_path,mutation):
    req=fixture(tmp_path);edit_rows(req,mutation)
    with pytest.raises(ValueError):m.build(req,fixture=True)


def test_corrupt_npy_and_own_summary(tmp_path):
    req=fixture(tmp_path);r=read(req['rgb_report']);p=Path(req['rgb_report']['path']).parent/r['rows'][0]['latent']['file']
    p.write_bytes(p.read_bytes()+b'x')
    with pytest.raises(ValueError):m.build(req,fixture=True)


@pytest.mark.parametrize('fault',['shape','dtype','extra','array_hash','negative_zero_bits'])
def test_original_npy_bounded_reader(tmp_path,fault):
    a=np.zeros((192,),dtype='<f4');data=io.BytesIO()
    if fault=='shape':a=a.reshape(1,192)
    if fault=='dtype':a=a.astype('<f8')
    if fault=='negative_zero_bits':a[0]=-0.
    np.save(data,a,allow_pickle=False);body=data.getvalue()+(b'extra' if fault=='extra' else b'')
    path=tmp_path/'latent.npy.gz';path.write_bytes(gzip.compress(body,mtime=0));h=m.sha(str(path))
    e=dict(file=path.name,sha256=h,shape=[192],dtype='float32',array_sha256=hashlib.sha256(np.zeros(192,dtype='<f4').tobytes()).hexdigest())
    if fault=='array_hash':e['array_sha256']='0'*64
    with pytest.raises(ValueError):m.load_latent(tmp_path,e,dict(artifact_sha256={path.name:h}),{})


def test_verify_rejects_result_tamper(tmp_path):
    req=fixture(tmp_path);v=m.build(req,fixture=True);v['pairs'][0]['real_latent_gain']['value']=999.
    r=m.saved.write_new(str(tmp_path/'bad.json'),v)
    with pytest.raises(ValueError):m.verify(r['path'],r['sha256'],fixture=True)


def test_context_switch_rejected_even_if_valid_shape():
    p=dict(root_context=dict(latent_history=[[0.]*192,[1.]*192,[2.]*192],action_history=[[0.]*10,[0.]*10],layout='x'),
        action_mean=[0.,0.],action_std=[1.,1.])
    child=dict(latent_history=[[1.]*192,[2.]*192,[3.]*192],action_history=[[0.]*10,[0.]*10],layout='x')
    m.check_context(p,child,[[0.,0.]]*5)
    child['latent_history'][0]=[9.]*192
    with pytest.raises(ValueError):m.check_context(p,child,[[0.,0.]]*5)


@pytest.mark.parametrize('fault',['no_witness','logical_order','immediate_cost'])
def test_original_logical_C_cannot_be_dropped_or_promoted(tmp_path,fault):
    req=fixture(tmp_path)
    def change(v):
        if fault=='no_witness':v['arithmetic']['C_control_pairs'][2]['original_C_no_witness']=False
        elif fault=='logical_order':v['arithmetic']['C_dual_control_conditions'].reverse()
        else:v['arithmetic']['C_dual_control_conditions'][0]['original_immediate_cost_both_controls']=True
    edit(req,'physical_summary',change)
    with pytest.raises(ValueError):m.build(req,fixture=True)


def test_cli_and_no_neural_imports():
    result=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'build','--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--rgb-receipt-sha' in result.stdout
    result=subprocess.run([sys.executable,'-c','import sys; from experiments.spectral_reachability import '+m.NAME+
        '; assert not any(k.split(".")[0] in {"torch","PIL","stable_worldmodel"} for k in sys.modules)'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
