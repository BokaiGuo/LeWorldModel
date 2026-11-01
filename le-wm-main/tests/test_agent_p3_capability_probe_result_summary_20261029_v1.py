"""Synthetic metadata only; no real report analysis, native IO, or fullstrict."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.spectral_reachability import agent_p3_capability_probe_result_summary_20261029_v1 as m


def put(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(m.encoded(value)+b'\n')
    return m.ref(str(path))


def capability():
    k=m.math_contract
    out=dict(schema=k.CAPABILITY+'.summary',signals=[{} for _ in range(7)],probes=[],comparisons=[],
        C_dual_control_conditions=[],**k.FLAGS)
    for i,(root,role) in enumerate(k.ENDPOINTS):
        out['probes'].append(dict(index=i,root_slot=root,role=role,native_cp10=dict(goal7=[0.]*7)))
    for ci,(root,arm) in enumerate(k.LOGICAL):
        for j,role in enumerate(('Z','P')):
            out['comparisons'].append(dict(index=2*ci+j,root_slot=root,C_arm=arm,control_role=role,
                available=ci==0,immediate_cost_condition=None if ci else bool(j),
                R10_original_solver_C_vs_control=dict(relation='null' if ci else 'less')))
        out['C_dual_control_conditions'].append(dict(root_slot=root,C_arm=arm,
            C_immediate_cost_both_controls=None if ci else False))
    return out


def observation(step,state):
    return dict(phase='plan',step=step,overlay_index=0,condition='off',state=state,
        snapshot=dict(state=state,full_snapshot_witness='RAM_only_not_actual'),
        file_sha256=hashlib.sha256(str(step).encode()).hexdigest(),pixel_sha256=hashlib.sha256(bytes([step])).hexdigest(),
        png=f'cp{step}.png')


def make_row(job, *, last=15, status='horizon_completed', velocity=0., terminal_success=False):
    ops=[]
    for resource,n in m.COUNTS.items():
        if resource=='public_step':
            for i in range(10+last):
                phase,step=('warmup',i+1) if i<10 else ('plan',i-9)
                state=[float(step+20),0.,0.,0.,0.,velocity,0.]
                if terminal_success and last>10 and i==9+last:
                    state[0]=1.  # valid success geometry, large velocity remains separate
                result=dict(state=state,snapshot=observation(step,state)['snapshot'],
                    goal_state=[0.]*7,reward=-m.math_contract.native_math(state,[0.]*7)['native7_distance'],
                    terminated=terminal_success and i==9+last,truncated=status=='native_stopped' and i==9+last,
                    actual_bits='00'*8,actual_dtype='<f4')
                ops.append(dict(index=len(ops),resource=resource,status='returned',phase=phase,step=step,raw_bits='00'*8,result=result))
        else:
            for _ in range(n):
                ops.append(dict(index=len(ops),resource=resource,status='returned',result={}))
    plan=[x for x in ops if x['resource']=='public_step' and x['phase']=='plan']
    cps=[observation(step,[float(step+20),0.,0.,0.,0.,velocity,0.]) for step in (0,5,10,15) if step<=last]
    if terminal_success and last==15:
        cps[-1]['state'][0]=1.;cps[-1]['snapshot']['state'][0]=1.
    # Prefix geometry cannot depend on the synthetic suffix velocity diagnostic.
    for cp in cps:
        if cp['step']<=10:
            cp['state'][5]=0.;cp['snapshot']['state'][5]=0.
            if cp['step']:
                op=next(x for x in plan if x['step']==cp['step']);op['result']['state'][5]=0.
                op['result']['snapshot']['state'][5]=0.
                op['result']['reward']=-m.math_contract.native_math(op['result']['state'],[0.]*7)['native7_distance']
    responses=[dict(step=x['step'],state=x['result']['state'],reward=x['result']['reward'],
        native_success=x['result']['terminated'],terminated=x['result']['terminated'],truncated=x['result']['truncated'],
        native_metrics=dict(native_success=x['result']['terminated'],native_distance=-x['result']['reward'])) for x in plan]
    observations=[dict(receipt=cp) for cp in cps]
    # Only filler is synthetic: the inherited physical receipt, never this test,
    # is responsible for all warmup/overlay event validation.
    observations += [dict(receipt=dict(phase='warmup',step=0,overlay_index=1)) for _ in range(81-len(cps))]
    return dict(job=job,status=status,error='RAM injected failure' if status=='engineering_failed' else None,
        cleanup_errors=[],counts={k:dict(attempted=sum(x['resource']==k for x in ops),
            completed=sum(x['resource']==k for x in ops)) for k in m.COUNTS},operations=ops,
        pending_operation_indices=[],observations=observations,guards=[51,66] if last>10 else [51] if last>5 else [],
        responses=responses,checkpoints=cps,missing_checkpoints=[x for x in (0,5,10,15) if x>last],
        endpoint15=next((x for x in cps if x['step']==15),None),unknowns=dict(FLOPs=None),
        new_neural_calls=0,fixture_only=False,**m.math_contract.FLAGS)


def fixture(tmp_path, *, last=15, status='horizon_completed', velocity=0., terminal_success=False, transport=False):
    cap=capability(); refs=dict(stageB={},U=[]); zero=[[0.,0.] for _ in range(10)]
    for key in ('prepare','report','completion','receipt'):
        refs['stageB'][key]=put(tmp_path/f'B_{key}.json',dict(RAM_only=key))
    resources=dict(attempted={'predict_API':24,'VJP_directions':1344},completed={'predict_API':24,'VJP_directions':1344},unknown_FLOPs=None)
    cap['authorities']=dict(stageB_receipt=refs['stageB']['receipt'])
    cap['inherited_stageB_resource_receipt']=resources
    witness=dict(schema='agent_p3_actual_cp10_full_witness_20261029_v1.witness',rows=[])
    plans=[]
    for i,(root,role) in enumerate(m.math_contract.ENDPOINTS):
        identity=dict(index=i,root_slot=root,role=role,raw10_float32_sha256=hashlib.sha256(bytes(80)).hexdigest())
        source=dict(step=10,pixel_sha256='RAM')
        plan=dict(index=i,root_slot=root,role=role,raw5=zero[:5],raw_float32_sha256=hashlib.sha256(bytes(40)).hexdigest(),
            endpoint_identity=identity,source=source,causal_raw10=zero,task=dict(RAM_task=i),
            checkpoint=10,probe_checkpoint=15,whole=[dict(RAM_context=True)],segmented=[dict(RAM_context=True)])
        ba=refs['stageB']
        plan.update(raw_float32_bits=bytes(40).hex(),native_call_ids=dict(signal=i+2,whole=7+2*i,segmented=8+2*i),
            authority=dict(prepared=ba['prepare']['path'],prepare_sha256=ba['prepare']['sha256'],
                report=ba['report']['path'],report_sha256=ba['report']['sha256'],
                completion=ba['completion']['path'],completion_sha256=ba['completion']['sha256']))
        plan['witness_sha256']=m.digest(plan);plans.append(plan)
        cap['probes'][i].update({k:plan[k] for k in ('raw5','raw_float32_sha256','endpoint_identity','source','witness_sha256')})
        refs['U'].append(put(tmp_path/f'U{i}.json',dict(schema=m.STAGEB+'.witness',plan=plan,
            receipt=refs['stageB']['receipt']['path'],receipt_sha256=refs['stageB']['receipt']['sha256'],
            evidence_kind='MAIN_SHA_bound_saved_receipt')))
        witness['rows'].append(dict(index=i,root_slot=root,role=role,warm_raw10=zero,control_raw10=zero,
            source_row=dict(goal=[0.]*7),stageA_task=dict(identity=identity,task=plan['task']),stageA_row=dict(source=source),
            observations=[dict(receipt=observation(n,[float(n+20),0.,0.,0.,0.,0.,0.])) for n in (0,5,10)]))
    refs['full_witness']=put(tmp_path/'full_witness.json',witness)
    p=dict(schema=m.PHYSICAL+'.prepare',authority=refs,jobs=[],known_roots=2,fresh_roots=0,
        fixture_only=True,stageB_connected=True,inherited_stageB_resource_receipt=resources,**m.math_contract.FLAGS)
    src=Path(m.__file__).parent/(m.PHYSICAL+'.py')
    p['bindings']={str(src):m.sha(str(src))}
    for i,(endpoint,root,role,probe) in enumerate(m.math_contract.JOBS):
        job=dict(index=i,endpoint=endpoint,root_slot=root,source_role=role,probe=probe,warm10=zero,
            raw15=zero+zero[:5],source_witness_sha256=refs['full_witness']['sha256'],U_witness=refs['U'][endpoint])
        job['selection_digest']=m.digest(job,compact=False);p['jobs'].append(job)
    prepared=put(tmp_path/'prepare.json',p)
    b=dict(schema=m.PHYSICAL+'.batch',prepared=prepared['path'],prepare_sha256=prepared['sha256'],jobs=[],**m.math_contract.FLAGS)
    s=dict(schema=m.PHYSICAL+'.consume',status='saved_partial_transport_NOT_verified' if transport else 'all10_saved_native_jobs_reconstructed_NOT_admitted',
        prepared=prepared['path'],prepare_sha256=prepared['sha256'],fixed_denominator=10,jobs=[],source_authorities=refs,
        inherited_stageB_resource_receipt=resources,new_neural_calls=0,FLOPs=None,**m.math_contract.FLAGS)
    for i,job in enumerate(p['jobs']):
        output=tmp_path/'batch'/f'job{i:02}';output.mkdir(parents=True)
        binding=dict(schema=m.PHYSICAL+'.job_binding',prepared=prepared['path'],prepare_sha256=prepared['sha256'],
            job=job,authority=refs,fixture_only=False,
            guard_transport_fields=['index','previous_sha256','event_sha256','branch','arm','selection_digest','receipt/png'])
        br=put(output/'binding.json',binding)
        if transport and i==0:
            f=dict(schema=m.PHYSICAL+'.publication_failure',binding_sha256=m.digest(binding,compact=False),
                status='transport_failed_NOT_verified',unknown_final_callbacks=True,endpoint15=None,error='RAM lost publication')
            fr=put(output/'failure.json',f)
            b['jobs'].append(dict(output=str(output),status=f['status'],failure_sha256=fr['sha256'],report_sha256=None,completion_sha256=None))
            s['jobs'].append(dict(schema=m.PHYSICAL+'.failure_consume',output=str(output),failure_sha256=fr['sha256'],
                status=f['status'],declarations=f,counts=None,endpoint15=None,unknown_final_callbacks=True,**m.math_contract.FLAGS))
            continue
        row=make_row(job,last=last if i==0 else 15,status=status if i==0 else 'horizon_completed',
            velocity=velocity if i==0 else 0.,terminal_success=terminal_success if i==0 else False)
        report=dict(schema=m.PHYSICAL+'.report',binding=binding,row=row,
            journal_tail='RAM_only_no_actual_journal',journal_sha256='0'*64,elapsed_seconds=1.)
        rr=put(output/'report.json',report)
        cr=put(output/'completion.json',dict(schema=m.PHYSICAL+'.completion',report_sha256=rr['sha256'],
            artifact_sha256={'report.json':rr['sha256'],'binding.json':br['sha256']},
            bytes_before_completion=Path(rr['path']).stat().st_size+Path(br['path']).stat().st_size,
            inclusive_wall_seconds=2.,wall_scope='through report hash; completion self-write excluded'))
        seals=dict(output=str(output),report_sha256=rr['sha256'],completion_sha256=cr['sha256'])
        b['jobs'].append(seals)
        s['jobs'].append(dict(schema=m.PHYSICAL+'.job_consume',row=row,event_count=834,fixture_only=False,**seals,**m.math_contract.FLAGS))
    request=dict(prepared=prepared,batch=put(tmp_path/'batch'/'batch.json',b),receipt=put(tmp_path/'receipt.json',s),
        capability=put(tmp_path/'capability.json',cap),physical_source=m.ref(str(src)))
    return request


def read(r):
    return json.loads(Path(r['path']).read_bytes())


def edit(request, key, fn):
    obj=read(request[key]);fn(obj);request[key]=put(Path(request[key]['path']),obj)


def edit_row(request, fn):
    """Reseal outer metadata so rejection must come from semantic joins."""
    s=read(request['receipt']);b=read(request['batch']);r=s['jobs'][0];fn(r['row'])
    out=Path(r['output']);report=json.loads((out/'report.json').read_bytes());report['row']=r['row']
    rr=put(out/'report.json',report);completion=json.loads((out/'completion.json').read_bytes())
    completion['report_sha256']=completion['artifact_sha256']['report.json']=rr['sha256']
    cr=put(out/'completion.json',completion)
    r['report_sha256']=b['jobs'][0]['report_sha256']=rr['sha256']
    r['completion_sha256']=b['jobs'][0]['completion_sha256']=cr['sha256']
    request['receipt']=put(Path(request['receipt']['path']),s);request['batch']=put(Path(request['batch']['path']),b)


def test_full10_temp_disk_build_write_verify(tmp_path):
    req=fixture(tmp_path);out=m.build(req,fixture=True)
    assert len(out['original_rows'])==10 and all(out['original_rows'])
    assert len(out['arithmetic']['branches'])==10 and len(out['arithmetic']['U_Z2_pairs'])==5
    assert len(out['arithmetic']['C_control_pairs'])==8 and len(out['arithmetic']['C_dual_control_conditions'])==4
    assert out['arithmetic']['C_dual_control_conditions'][0]['full_chain_both_controls'] is False
    assert all(x['G']==0. for x in out['arithmetic']['U_Z2_pairs'])
    r=m.write_new(str(tmp_path/'analysis.json'),out)
    assert m.verify(r['path'],r['sha256'],fixture=True)['fixture_only']
    with pytest.raises(FileExistsError):m.write_new(r['path'],out)
    with pytest.raises(ValueError):m.verify(r['path'],r['sha256'])
    assert out['upstream_fullstrict_reexecuted'] is False and out['status']=='fixture_only_NOT_evidence'
    assert all(out[k] is False for k in m.math_contract.FLAGS)


@pytest.mark.parametrize('last',[5,10,14])
def test_early_native_stop_no_cp15_padding(tmp_path,last):
    req=fixture(tmp_path,last=last,status='native_stopped',velocity=256.,terminal_success=last>10)
    out=m.build(req,fixture=True);branch=out['arithmetic']['branches'][0]
    assert branch['metrics'] is None and branch['terminal_native_success'] is (last>10)
    assert branch['terminal_control_step']==last and out['arithmetic']['U_Z2_pairs'][0]['G'] is None


def test_native_success_does_not_mean_small_native7(tmp_path):
    req=fixture(tmp_path,velocity=256.,terminal_success=True)
    branch=m.build(req,fixture=True)['arithmetic']['branches'][0]
    assert branch['original_native_success'] is True and branch['D_original_reward']>256
    assert branch['metrics']['velocity_squared']==256**2


def test_engineering_failure_after_cp15_not_eligible(tmp_path):
    out=m.build(fixture(tmp_path,status='engineering_failed'),fixture=True)
    assert out['arithmetic']['branches'][0]['metrics'] is not None
    assert out['arithmetic']['U_Z2_pairs'][0]['G'] is None


def test_warmup_terminal_uses_actual_goal_before_attachment(tmp_path):
    req=fixture(tmp_path,last=0,status='native_stopped')
    def change(row):
        row['checkpoints']=[];row['endpoint15']=None;row['missing_checkpoints']=[0,5,10,15]
        row['observations']=[]
        last=next(x for x in reversed(row['operations']) if x['resource']=='public_step')
        last['result']['goal_state']=[1.]*7
    edit_row(req,change)
    branch=m.build(req,fixture=True)['arithmetic']['branches'][0]
    assert branch['terminal_control_step']==0 and branch['metrics'] is None
    assert branch['terminal_metrics']['goal7']==[1.]*7


def test_engineering_no_return_never_invents_terminal(tmp_path):
    req=fixture(tmp_path,status='engineering_failed')
    def change(row):
        row['operations']=[]
        row['counts']={k:dict(attempted=0,completed=0) for k in m.COUNTS}
        row['observations']=[];row['responses']=[];row['guards']=[];row['checkpoints']=[]
        row['missing_checkpoints']=[0,5,10,15];row['endpoint15']=None
    edit_row(req,change)
    out=m.build(req,fixture=True)
    assert out['arithmetic']['branches'][0]['terminal_metrics'] is None
    assert out['arithmetic']['U_Z2_pairs'][0]['G'] is None


def test_transport_failure_all_denominators_unknown_no_fake_row(tmp_path):
    out=m.build(fixture(tmp_path,transport=True),fixture=True)
    assert len(out['original_receipts'])==10 and out['original_rows'][0] is None
    assert out['native_resource_vectors'][0] is None
    assert out['arithmetic']['branches'][0]['terminal_metrics'] is None
    assert out['arithmetic']['U_Z2_pairs'][0]['G'] is None


@pytest.mark.parametrize('key',list(m.PINS))
def test_external_full_file_sha_not_status(tmp_path,key):
    req=fixture(tmp_path);req[key]['sha256']='0'*64
    with pytest.raises(ValueError):m.build(req,fixture=True)


@pytest.mark.parametrize('mutation',[
    lambda s:s.update(fixed_denominator=9),lambda s:s['jobs'].pop(),
    lambda s:s['jobs'].reverse(),lambda s:s.update(prepared='/missing'),
    lambda s:s['source_authorities']['stageB']['receipt'].update(sha256='0'*64),
    lambda s:s.update(full_C3_demonstrated=True),
    lambda s:s.update(inherited_stageB_resource_receipt={}),
    lambda s:s['jobs'][0]['row'].update(status='engineering_failed')])
def test_top_and_whole_original_row_mutation(tmp_path,mutation):
    req=fixture(tmp_path);edit(req,'receipt',mutation)
    with pytest.raises(ValueError):m.build(req,fixture=True)


@pytest.mark.parametrize('mutation',[
    lambda r:r['counts']['reset'].update(completed=0),
    lambda r:r['guards'].remove(66),lambda r:r['missing_checkpoints'].append(15),
    lambda r:r['checkpoints'][1]['snapshot'].update(hidden_shape='changed'),
    lambda r:r['checkpoints'][1].update(pixel_sha256='0'*64),
    lambda r:r['responses'][2].update(reward=-999.),
    lambda r:r['operations'][next(i for i,x in enumerate(r['operations']) if x['resource']=='public_step')].update(raw_bits='80'*8),
    lambda r:r['operations'][next(i for i,x in enumerate(r['operations']) if x['resource']=='public_step' and x['phase']=='plan')]['result'].update(goal_state=[1.]*7),
    lambda r:r.update(new_neural_calls=1)])
def test_semantic_mutation_resealed_report_still_rejected(tmp_path,mutation):
    req=fixture(tmp_path);edit_row(req,mutation)
    with pytest.raises(ValueError):m.build(req,fixture=True)


def test_spaced_and_compact_codec_and_positive_zero():
    assert m.digest([1,2])!=m.digest([1,2],compact=False)
    assert m.raw_bits([[0.,0.]],1)!=m.raw_bits([[-0.,0.]],1)
    for value in ([[.1,0.]],[[1.01,0.]],[[True,0.]]):
        with pytest.raises(ValueError):m.raw_bits(value,1)


def test_own_summary_tamper_and_no_real_authority_from_fixture(tmp_path):
    req=fixture(tmp_path);value=m.build(req,fixture=True)
    value['arithmetic']['U_Z2_pairs'][0]['G']=100
    r=m.write_new(str(tmp_path/'tampered.json'),value)
    with pytest.raises(ValueError):m.verify(r['path'],r['sha256'],fixture=True)
    with pytest.raises(ValueError):m.build(req)


def test_cli_help_no_native_imports():
    result=subprocess.run([sys.executable,'-m','experiments.spectral_reachability.'+m.NAME,'build','--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--batch-sha' in result.stdout and '--physical-source-sha' in result.stdout
    code=('import sys; from experiments.spectral_reachability import '+m.NAME+'; '
          'assert not any(k.split(".")[0] in {"torch","stable_worldmodel","numpy","PIL"} for k in sys.modules)')
    clean=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert clean.returncode==0,clean.stderr


def test_duplicate_and_nonfinite_json(tmp_path):
    for i,data in enumerate((b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}')):
        p=tmp_path/f'bad{i}.json';p.write_bytes(data)
        with pytest.raises(ValueError):m.load(m.ref(str(p)),{})
