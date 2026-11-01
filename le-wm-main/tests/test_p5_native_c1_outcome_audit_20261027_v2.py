"""Non-neural saved-file fixtures; never construct/import a native environment."""
import ast
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import p5_native_c1_outcome_audit_20261027_v2 as a


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(a.canonical(value))


def ledger(path,events,row):
    events=[*events,dict(phase='terminal',status=row['status'],summary_sha256=a.digest(row))]
    previous=None;lines=[]
    for i,event in enumerate(events):
        entry=dict(sequence=i,previous_sha256=previous,event=event)
        entry['entry_sha256']=a.digest(entry);previous=entry['entry_sha256']
        lines.append(a.canonical(entry))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b'\n'.join(lines)+b'\n')
    row['journal_sha256']=a.sha(path)
    return a.journal(path,row['arm'],row)


def close_row(arm='compiled',status='completed'):
    return dict(arm=arm,status=status,error=None,frames=0,prefix_gate_frames=[],physics_executed=False,
                native_call_receipts=dict(python_call_entries=dict(close=1),space_step_events=[]),
                close_receipt=dict(status='completed',call_attempted=True,call_completed=True,return_durable=True,error=None,original_error=None))


def close_evidence(row):
    return dict(events=[dict(phase='intent',operation='close'),dict(phase='returned',operation='close',counters_after=row['native_call_receipts'])],operations=[('close',{})],close_failures=[])


def notrun(arm):
    r=close_row(arm,'not_run');r['native_call_receipts']=None
    r['close_receipt'].update(status='not_constructed',call_attempted=False,call_completed=False,return_durable=False)
    return r


def public(frame):
    return dict(bodies=dict(agent=dict(position=[float(max(frame-10,0)),0.],velocity=[0.,0.]),
                            block=dict(position=[200.,200.],angle=0.)),
                env=dict(seed=a.SEED,goal_pose=dict(values=[256.,256.,np.pi/4]),coverage_arr=[.2]*frame,
                         n_contact_points=0,success_threshold=.95,latest_action=[0.,0.]))


def test_source_prepare_current_sealed_v2():
    before=set(sys.modules)
    m=a.source_prepare(a.ROOT)
    assert m['status']=='prepared_auditor_no_outcomes_opened'
    assert m['seals'][str(a.ROOT/f'experiments/spectral_reachability/{a.EXECUTOR}.py')]==a.EXECUTOR_SHA
    assert not any(n.split('.')[0] in ('torch','pymunk','gym','transformers') for n in set(sys.modules)-before)


def test_auditor_ast_no_neural_physics_import_or_execution():
    tree=ast.parse(Path(a.__file__).read_text())
    names=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): names.extend(x.name.split('.')[0] for x in node.names)
        if isinstance(node,ast.ImportFrom): names.append((node.module or '').split('.')[0])
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            assert node.func.attr not in ('forward','backward','step','reset','seed','load_state_dict','system','run')
    assert not set(names)&{'torch','pymunk','gym','pygame','transformers','pickle','subprocess'}


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}'])
def test_strict_json(raw):
    with pytest.raises(ValueError): a.strict_json(raw)


def test_symlink_rejected(tmp_path):
    f=tmp_path/'a';f.write_bytes(b'a');link=tmp_path/'b';link.symlink_to(f)
    with pytest.raises(ValueError): a.sha(link)


@pytest.mark.parametrize('rows',[[],[dict(arm='compiled')]*3,[dict(arm=x) for x in reversed(a.ARMS)]])
def test_all_arm_slots_ordered(rows):
    with pytest.raises(ValueError): a.arm_index(rows)


def test_agent_motion_is_not_object_response():
    m=a.suffix_metrics([public(i) for i in range(16)])
    assert m['agent_suffix_path_px']==5 and m['object_suffix_path_px']==0
    assert m['agent_only_suffix_motion'] and not m['C2_established']
    assert m['coverage_gain_since_decision']==0 and not m['C1_effectiveness_claim']


def test_angle_principal_branch_and_no_endpoint_padding():
    p=public(15);p['bodies']['block']['angle']=2*np.pi+np.pi/4
    assert a.pose(p)['object_angle_goal_error']<1e-14
    with pytest.raises(ValueError): a.suffix_metrics([p]*15)


def test_outcome_compares_every_control_without_scalar():
    rows=[]
    for i,arm in enumerate(a.ARMS):
        rows.append(dict(arm=arm,status='completed',metrics=dict(terminal=dict(object_xy_goal_error=10.+i,object_angle_goal_error=.5,native_final_coverage=.2))))
    result=a.compare_outcomes(rows)
    assert result['compiled_minus_control']['factual_reference']['object_xy_goal_error']==-2
    assert not result['C1_effectiveness_claim']
    rows[2]['status']='guard_abort'
    assert 'compiled_minus_control' not in a.compare_outcomes(rows)


def test_cleanup_completed():
    r=close_row();assert a.cleanup(r,close_evidence(r)) is False


@pytest.mark.parametrize('field',['call_attempted','call_completed','return_durable'])
def test_completed_cleanup_failclosed(field):
    r=close_row();r['close_receipt'][field]=False
    with pytest.raises(ValueError): a.cleanup(r,close_evidence(r))


def test_original_error_cannot_be_masked():
    r=close_row(status='guard_abort');r['error']='RuntimeError: close error'
    r['close_receipt']['original_error']='ValueError: budget exhausted'
    with pytest.raises(ValueError,match='masked'): a.cleanup(r,close_evidence(r))


def test_failed_cleanup_blocks_all_later_dispatch():
    rows=[notrun(x) for x in a.ARMS];r=rows[0]
    r.update(status='guard_abort',error='ValueError: budget exhausted',native_call_receipts=dict(python_call_entries=dict(close=1),space_step_events=[]))
    r['close_receipt'].update(status='failed',call_attempted=True,error='RuntimeError: close error',original_error=r['error'])
    failure=dict(phase='close_failed',arm='compiled',**r['close_receipt'])
    ev={x:dict(events=[],operations=[],close_failures=[]) for x in a.ARMS}
    ev['compiled']=dict(events=[dict(phase='intent',operation='close'),failure],operations=[('close',{})],close_failures=[failure])
    assert a.dispatch_census(rows,ev)[2]['subsequent_dispatch_blocked']
    ev['nominal']['events']=[dict(phase='preflight_intent')]
    with pytest.raises(ValueError,match='dispatch after'): a.dispatch_census(rows,ev)


def test_constructor_uncertainty_blocks_dispatch():
    rows=[notrun(x) for x in a.ARMS];rows[0]['status']='guard_abort';rows[0]['close_receipt']['status']='construction_attempted'
    ev={x:dict(events=[],operations=[],close_failures=[]) for x in a.ARMS}
    assert a.dispatch_census(rows,ev)[0]['cleanup_incomplete']


def test_terminal_only_notrun_journal(tmp_path):
    row=notrun('compiled');ev=ledger(tmp_path/'j',[],row)
    assert not ev['records'] and not ev['operations']


def test_suffix_intent_without_eleven_gates_rejected(tmp_path):
    row=notrun('compiled');row['status']='guard_abort'
    events=[dict(phase='intent',role='compiled',operation='step',details=dict(frame=i)) for i in range(1,12)]
    with pytest.raises(ValueError,match='ALL eleven'): ledger(tmp_path/'j',events,row)


def test_hashchain_tamper_rejected(tmp_path):
    row=notrun('compiled');path=tmp_path/'j';ledger(path,[],row)
    raw=json.loads(path.read_text());raw['sequence']=5;path.write_bytes(a.canonical(raw)+b'\n');row['journal_sha256']=a.sha(path)
    with pytest.raises(ValueError,match='chain'): a.journal(path,'compiled',row)


def test_operation_after_close_rejected(tmp_path):
    row=notrun('compiled');row['status']='guard_abort'
    events=[dict(phase='intent',role='compiled',operation=op,details={}) for op in ('close','construct')]
    with pytest.raises(ValueError,match='after cleanup'): ledger(tmp_path/'j',events,row)


def test_complete_saved_three_arm_integration(tmp_path,monkeypatch):
    """Audit arithmetic/receipts on fixtures, NOT a native physical run."""
    out=tmp_path/'out';out.mkdir();ready_path=tmp_path/'ready.json'
    branches=[dict(arm=x,prefix_env_commands=np.zeros((10,2)).tolist(),suffix_env_packet=np.zeros((5,2)).tolist()) for x in a.ARMS]
    ready=dict(branches=branches,contract={},selected_packet=dict(packet=branches[0]['suffix_env_packet']),native_constructor={},prefix_identity=[])
    write(ready_path,ready);ready_sha=a.sha(ready_path)
    rows=[];helper=a.numeric_helpers(a.ROOT)
    for arm in a.ARMS:
        d=out/arm;d.mkdir();np.save(d/'frozen_env_commands.npy',np.zeros((15,2),np.float64),allow_pickle=False)
        row=close_row(arm);row.update(frames=16,prefix_gate_frames=list(range(11)),source_hashes_unchanged=True,physics_executed=True)
        events=[dict(phase='preflight_intent',branch=branches[a.ARMS.index(arm)],ready_manifest_file_sha256=ready_sha)]
        def operation(op,detail):
            events.extend([dict(phase='intent',role=arm,operation=op,details=detail),dict(phase='returned',role=arm,operation=op)])
        operation('construct',dict(config={}));operation('seed',dict(seed=a.SEED));operation('reset',dict(seed=a.SEED))
        publics=[]
        for frame in range(16):
            p=public(frame);p['bodies']['agent']['position']=[0.,0.];publics.append(p)
            transition=dict(kind='reset') if frame==0 else dict(kind='step',done=False,reward=float(np.clip(.2/.95,0,1)),info=dict(final_coverage=.2,max_coverage=.2))
            if frame:
                cmd=dict(frame=frame,action_index=frame-1,packet_index=(frame-1)//5,step_within_packet=(frame-1)%5,raw_env_argument=[0.,0.],pixel_displacement=[0.,0.],agent_xy_before=[0.,0.],absolute_target=[0.,0.],expected_substeps=10,dt=.01,pre_public_sha256=a.sha(d/f'frame_{frame-1:02d}/full_public_state.json'))
                write(d/f'command_{frame:02d}.json',cmd);operation('step',cmd)
            f=d/f'frame_{frame:02d}';f.mkdir();state=np.array([0,0,200,200,0,0,0],np.float32)
            np.save(f/'state.npy',state,allow_pickle=False);np.save(f/'proprio.npy',state[[0,1,5,6]],allow_pickle=False)
            np.save(f/'rgb.npy',np.zeros((224,224,3),np.uint8),allow_pickle=False)
            write(f/'full_public_state.json',p);write(f/'transition.json',transition)
            rec=dict(frame=frame,files={n:dict(sha256=a.sha(f/n),bytes=(f/n).stat().st_size) for n in a.NAMES},state=state.tolist(),proprio=state[[0,1,5,6]].tolist())
            events.append(dict(phase='assets_durable',arm=arm,record=rec))
            if frame<=10:
                events.append(dict(phase='prefix_byte_gate_passed',frame=frame,arm=arm))
                if arm=='compiled': ready['prefix_identity'].append(dict(frame=frame,files={n:dict(path=str(f/n),sha256=a.sha(f/n)) for n in a.NAMES}))
        counts=dict(a.EXPECTED)
        steps=[dict(dt=.01,context=dict(role=arm,operation='reset',frame=None))]
        steps.extend(dict(dt=.01,context=dict(role=arm,operation='step',frame=i)) for i in range(1,16) for _ in range(10))
        row['native_call_receipts']=dict(python_call_entries=counts,space_step_events=steps)
        operation('close',{});events[-1]['counters_after']=row['native_call_receipts']
        row['endpoint']=a.pose(publics[-1]);ledger(d/'journal.jsonl',events,row)
        write(d/'report.json',row);rows.append(dict(**row,report_sha256=a.sha(d/'report.json')))
    # ready SHA needs to refer to the final prefix identity; update preflight chains.
    write(ready_path,ready);ready_sha=a.sha(ready_path)
    for row in rows:
        d=out/row['arm'];r=a.read_json(d/'report.json');events=[json.loads(l)['event'] for l in (d/'journal.jsonl').read_bytes().splitlines()][:-1]
        events[0]['ready_manifest_file_sha256']=ready_sha;r.pop('journal_sha256');ledger(d/'journal.jsonl',events,r)
        write(d/'report.json',r);row.update(r,report_sha256=a.sha(d/'report.json'))
    write(out/'all_branches.json',dict(branches=branches,ready_manifest_file_sha256=ready_sha));write(out/'selected_packet.json',ready['selected_packet'])
    write(out/'dispatch_intent.json',dict(rows=[dict(arm=x,status='not_run',frames=0) for x in a.ARMS],contract={},ready_manifest_file_sha256=ready_sha))
    write(out/'report.json',dict(schema=a.EXECUTOR,status='native_c1_three_branch_execution_recorded',rows=rows,ready_manifest_file_sha256=ready_sha,neural_executed=False,C1_effectiveness_claim=False,C2_established=False,P5_admission=False))
    frozen=dict(ready_manifest=str(ready_path),execution_dir=str(out),seals={str(p):a.sha(p) for p in out.rglob('*') if p.is_file()})
    monkeypatch.setattr(a,'bind',lambda *args:frozen);monkeypatch.setattr(a,'authority',lambda *args:dict(fixture_only=True))
    result=a.audit(a.ROOT,frozen)
    assert result['all_checks'] and result['counts']['space_step']==453
    assert len(result['rows'])==3 and not result['C1_effectiveness_claim']
    # True byte mismatch after rebinding manifests is still caught by prefix gate.
    f=out/'nominal/frame_00/rgb.npy';x=np.load(f,allow_pickle=False);x[0,0,0]=1;np.save(f,x,allow_pickle=False)
    frozen['seals'][str(f)]=a.sha(f)
    with pytest.raises(ValueError): a.audit(a.ROOT,frozen)


def test_authority_nested_native_compiler_input_schema(tmp_path):
    freeze=tmp_path/'freeze';freeze.mkdir()
    prepared=dict(schema=a.BRIDGE,status='ready_for_main_CUDA_compile',blockers=[],contract=dict(compiler=dict(expected_completed=dict(predictor=13))),
                  runner_sources=dict(executor=dict(path=str(a.ROOT/f'experiments/spectral_reachability/{a.EXECUTOR}.py'),sha256=a.EXECUTOR_SHA),compiler=dict(path='fixture.py',sha256='a'*64)),
                  audit_source_seal=dict(fixture=True),compiler_inputs_sha256='b'*64,compiler_inputs=dict(allowed_data_files={'fixture_rgb.npy':'c'*64}),
                  goal=dict(rgb_sha256='d'*64),model_identity={'fixture':'identity'},prefix_env_commands=np.zeros((10,2)).tolist(),
                  factual_reference_packet=np.full((5,2),.1).tolist(),root_identity_sha256='e'*64)
    roles=('response_columns','goal_features','nominal_successor','ridge_successor','selected_replay','gram','rhs','ridge_delta')
    receipt=dict(schema=a.BRIDGE+'.compiler',status='selected_packet_replayed',prepared_sha256=a.digest(prepared),compiler_inputs_sha256=prepared['compiler_inputs_sha256'],
                 used_observed_frame15=False,physical_calls=0,selected_replay_exact=True,attempted=dict(predictor=13),completed=dict(predictor=13),partial_work_unknown=False,
                 compiler_data_bindings=prepared['compiler_inputs']['allowed_data_files'],goal_rgb_sha256=prepared['goal']['rgb_sha256'],compiler_source=prepared['runner_sources']['compiler'],
                 backend='host_cuda_only',model_before=prepared['model_identity'],model_after=prepared['model_identity'],artifact_roles={k:k+'.npy' for k in roles},artifact_bindings={k+'.npy':'a'*64 for k in roles},
                 candidates=[dict(id='nominal',env_packet=np.zeros((5,2)).tolist(),independent_goal_mse=1.),dict(id='ridge',env_packet=np.full((5,2),.2).tolist(),independent_goal_mse=1.)],
                 selected_id='nominal',selected_env_packet=np.zeros((5,2)).tolist())
    selected=dict(packet=receipt['selected_env_packet'],selection_sha256=a.digest(receipt))
    branches=[dict(arm=arm,seed=a.SEED,prefix_env_commands=prepared['prefix_env_commands'],suffix_env_packet=prepared['factual_reference_packet'] if arm=='factual_reference' else selected['packet'],
                   compiler_receipt_sha256=a.digest(receipt),root_identity_sha256=prepared['root_identity_sha256']) for arm in a.ARMS]
    pp=tmp_path/'protocol.json';rp=tmp_path/'compiler.json';write(pp,prepared);write(rp,receipt)
    write(freeze/'selected_packet.json',selected)
    write(freeze/'all_branches.json',dict(prepared_path=str(pp),compiler_receipt_path=str(rp),prepared_file_sha256=a.sha(pp),compiler_receipt_file_sha256=a.sha(rp),branches=branches))
    write(freeze/'freeze_receipt.json',dict(schema=a.BRIDGE+'.freeze',status='all_three_branches_durable_not_executed',census={x:'planned_not_constructed' for x in a.ARMS},
          all_branches_sha256=a.sha(freeze/'all_branches.json'),selected_packet_sha256=a.sha(freeze/'selected_packet.json'),source_seal=prepared['audit_source_seal'],new_neural_calls=0,new_physics_steps=0))
    ready=dict(freeze_dir=str(freeze),contract=prepared['contract'],branches=branches,selected_packet=selected)
    assert a.authority(ready)['selected_id']=='nominal'  # exact tie: retain nominal-first
    ready['selected_packet']=dict(packet=np.ones((5,2)).tolist(),selection_sha256=a.digest(receipt))
    with pytest.raises(ValueError,match='packet binding'): a.authority(ready)
