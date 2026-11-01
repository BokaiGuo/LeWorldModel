"""RAM-only exact execution reuse; no production CLI, model or environment."""
from copy import deepcopy
import json
import runpy
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_model_library_selector_evaluation_20261028_v1 as m


def fixtures():
    # Reuse frozen selector's RAM fixture, not its production build/consume.
    candidates=runpy.run_path(str(m.ROOT/'tests/test_agent_p1_model_library_selector_20261028_v1.py'))['candidates']()
    rows=[];cases=[]
    for i,c in enumerate(candidates):
        p=c['original_plan'];prefix=m.selector.raw(p['prefix_commands'],5)
        incoming=np.concatenate((m.panel.base.physics.moving.warmup_commands()[5:],prefix))
        context=dict(action_history=m.teacher.kernel.normalized_history(incoming).tolist(),anchor=i//5)
        p['teacher_context_sha256']=m.digest(context)
        p['plan_digest']=m.digest({k:v for k,v in p.items() if k!='plan_digest'})
        goal192=[0.]*192;c['prediction_witness']['goal_sha256']=m.array_sha(np.asarray(goal192,np.float32))
        c['prediction_digest']=m.digest(c['prediction_witness'])
        job={k:deepcopy(p[k]) for k in ('index','task','root_slot','condition','role','library_index','library_label','anchor_id',
            'blind_sha256','task_id','plan_digest','commands','prefix_commands','suffix_commands','teacher_context_sha256')}
        job.update(anchor_index=i//5,parent_selection_digest='p'*64);job['job_digest']=m.digest(job)
        root=[0.]*7;start=[float(i//5),0.,0.,0.,0.,0.,0.];end=[float(i%5),0.,0.,0.,0.,0.,0.]
        goal=[10.,0.,0.,0.,.1,0.,0.];outline=[256.,256.,0.]
        anchor=dict(anchor_id=p['anchor_id'],task_id=p['task_id'],condition=p['condition'],role=p['role'],goal=goal192,
            png_sha256='f'*64,pixel_sha256='e'*64)
        blind=dict(context=context,goal=goal192,q=np.eye(192).tolist(),raw_history=incoming.astype(float).tolist())
        frame=dict(state=start,sha256='f'*64,pixel_sha256='e'*64,pre=dict(state=start),post=dict(state=start))
        physical=dict(**deepcopy(job),initial_exact=True,history_exact=True,start_verified=True,teacher_history_verified=True,
            root_state=root,start_state=start,last_state=end,goal=goal,goal_pose=outline,goal_pixel_sha256='d'*64,
            checkpoints=[dict(step=5,absolute_step=15,frame=frame)],status='failed' if i==12 else 'horizon_completed')
        metrics={k:float(i%5) for k in m.panel.base.METRICS}
        metrics.update(task_success=False,native_coverage=None,object_only_coverage=None)
        row={k:deepcopy(p[k]) for k in ('index','root_slot','condition','role','library_index','library_label','anchor_id',
            'blind_sha256','task_id','plan_digest')}
        row.update(job_digest=job['job_digest'],goal192=goal192,physical_goal=goal,outline_pose=outline,
            goal_binding=dict(state=goal,pose=outline,pixel_sha256='d'*64),array_receipts=dict(prediction=c['prediction_witness']['shifted_context']),
            suffix_raw_bits=c['suffix_raw_bits'],predicted_Q=p['model_Q'],actual_Q=None if i==12 else float(i%5),
            boundary_trace={'10':dict(state=root),'15':dict(state=start),'20':dict(state=end)},returned_state=end,
            actual_return_event_sha256='a'*64,physical_metrics=metrics,native_success=False,returned_reward=-float(i%5))
        commands=np.concatenate((m.panel.base.physics.moving.warmup_commands(),m.selector.raw(p['commands'],10)))
        phases=[('warmup',s) for s in range(1,11)]+[('prefix',s) for s in range(11,16)]+[('suffix',s) for s in range(1,6)]
        returns=[dict(phase=phase,step=step,raw_bits=a.tobytes().hex(),actual_bits=a.tobytes().hex(),actual_dtype='<f4',
            terminated=False,truncated=False,state=root if j==9 else start if j==14 else end,event_sha256='a'*64)
            for j,((phase,step),a) in enumerate(zip(phases,commands,strict=True))]
        histories=[dict(context_sha256=p['teacher_context_sha256'],teacher_png_sha256='f'*64,
            raw_history=incoming.astype(float).tolist(),tokens=context['action_history'])]
        cases.append((c,row,physical,job,anchor,blind,returns,histories));rows.append(row)
    sealed=m.selector.select40(candidates)
    return sealed,rows,cases


def joined():
    sealed,rows,cases=fixtures()
    return sealed,rows,[m.join_slot(*case) for case in cases]


def test_full40_eight_policy_pairs_unknown12_and_legacy_identity():
    sealed,rows,joins=joined();a=m.assemble(sealed,rows,joins)
    assert len(a['rows'])==40 and len(a['exact_reuse_witnesses'])==40 and len(a['groups'])==8
    assert a['known_roots']==2 and a['fresh_roots']==0
    assert a['rows'][12]['actual_Q'] is None and a['exact_reuse_witnesses'][12]['physical_source_status']=='failed'
    g=a['groups'][2]
    assert g['exploratory_model_selector']['original_index']==12 and g['exploratory_model_selector']['unknown_actual_latent']
    assert g['selector_minus_diagonal']['actual_latent_Q_gain'] is None
    assert g['exploratory_model_selector']['physical_metrics']['native_distance'] is not None
    assert not any(g['new_policy_actually_dispatched'] for g in a['groups'])


@pytest.mark.parametrize('change',['plan','job','prefix','suffix','context','goal192','goal7','outline','goalpixel','start','root',
    'png','snapshot','history','tokens','action_actual','return_missing','return_terminal','state20','prediction','initial_flag','job_digest'])
def test_exact_reuse_mutations_fail_closed(change):
    _,_,cases=fixtures();c,r,p,j,a,b,returns,h=deepcopy(cases[1])
    if change=='plan':r['plan_digest']='0'*64
    elif change=='job':r['job_digest']='0'*64
    elif change=='prefix':p['prefix_commands'][0][0]=.125
    elif change=='suffix':r['suffix_raw_bits']='00'
    elif change=='context':b['context']['anchor']=999
    elif change=='goal192':b['goal'][0]=1.
    elif change=='goal7':p['goal']=[999.]*7
    elif change=='outline':p['goal_pose']=[0.]*3
    elif change=='goalpixel':p['goal_pixel_sha256']='0'*64
    elif change=='start':p['start_state']=[999.]*7
    elif change=='root':p['root_state']=[999.]*7
    elif change=='png':a['png_sha256']='0'*64
    elif change=='snapshot':p['checkpoints'][0]['frame']['post']={'other':True}
    elif change=='history':h[0]['raw_history'][0][0]=.5
    elif change=='tokens':h[0]['tokens']=[[0.]*10]*2
    elif change=='action_actual':returns[15]['actual_bits']='00'
    elif change=='return_missing':returns.pop()
    elif change=='return_terminal':returns[-1]['terminated']=True
    elif change=='state20':returns[-1]['state']=[999.]*7
    elif change=='prediction':r['array_receipts']['prediction']={'file':'other'}
    elif change=='initial_flag':p['history_exact']=False
    else:j['job_digest']='0'*64
    with pytest.raises(ValueError):m.join_slot(c,r,p,j,a,b,returns,h)


@pytest.mark.parametrize('change',['drop','order','join_digest','prediction_digest','fill12','selection_digest','chosen_raw','group_start'])
def test_evaluation_full_denominator_and_sealed_choice_mutations(change):
    s,r,j=joined()
    if change=='drop':r.pop()
    elif change=='order':r[0],r[1]=r[1],r[0]
    elif change=='join_digest':j[1]['plan_digest']='0'*64
    elif change=='prediction_digest':j[1]['prediction_digest']='0'*64
    elif change=='fill12':r[12]['actual_Q']=0.
    elif change=='selection_digest':s['selections'][0]['selected_original_index']=4
    elif change=='chosen_raw':
        s['selections'][0]['commands_raw_bits']='00'
        s['selections'][0]['selection_digest']=m.digest({k:v for k,v in s['selections'][0].items() if k!='selection_digest'})
    else:j[1]['actual_start15']=[999.]*7
    with pytest.raises(ValueError):m.assemble(s,r,j)


def test_changing_physical_outcomes_does_not_change_sealed_selector():
    s,r,j=joined();before=deepcopy(s);a=m.assemble(s,r,j)
    for row in r:
        row['physical_metrics']['all_xy_error']=-999. if row['library_index']==4 else 999.
        row['returned_reward']=999. if row['library_index']==4 else -999.
    b=m.assemble(s,r,j)
    assert s==before
    assert [g['exploratory_model_selector']['original_index'] for g in a['groups']]==[g['exploratory_model_selector']['original_index'] for g in b['groups']]


def test_all_metrics_independent_zero_reference_and_ties_unknown():
    s,r,j=joined();a=m.assemble(s,r,j);g=a['groups'][0]
    assert g['selector_minus_diagonal']['physical_metrics_higher_is_better']['task_success']==0.
    assert g['selector_minus_diagonal']['physical_metrics_higher_is_better']['native_coverage'] is None
    assert g['exploratory_model_selector']['own_zero_comparison']['reference_index']==0
    assert g['original_generated_diagonal']['own_zero_comparison']['reference_index']==0
    assert set(g['selector_minus_diagonal']['physical_metrics_higher_is_better'])==set(m.panel.base.METRICS)


def test_whole_evaluation_json_roundtrip_tamper_verify(tmp_path):
    a=m.assemble(*joined());p=tmp_path/'evaluation.json';p.write_bytes(m.canonical(a))
    assert json.loads(p.read_text())==a and m.verify(p,m.sha(p),a)['verified']
    changed=deepcopy(a);changed['groups'][0]['exploratory_model_selector']['predicted_Q']=999.
    p.write_bytes(m.canonical(changed))
    with pytest.raises(ValueError):m.verify(p,m.sha(p),a)
    with pytest.raises(ValueError):m.verify(p,'0'*64,a)


def test_consume_full_rebuild_no_scientific_release(tmp_path,monkeypatch):
    a=m.assemble(*joined());p=tmp_path/'evaluation.json';p.write_bytes(m.canonical(a));calls=[]
    def build():calls.append(True);return a
    monkeypatch.setattr(m,'build',build);out=m.consume(p,m.sha(p))
    assert calls==[True] and out['fixed_slots']==40 and out['fixed_policy_pairs']==8
    assert out['P1_admission'] is out['fresh_confirmation'] is out['C2_claim'] is out['physical_release'] is False


def test_new_own_prefix_output_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir();p=tmp_path/'results'/(m.NAME+'_test.json')
    m.write(p,{})
    with pytest.raises(FileExistsError):m.write(p,{})
    with pytest.raises(ValueError):m.write(tmp_path/'results/MAIN.json',{})


@pytest.mark.parametrize('mutation',[None,'teacher_input','compiler_source'])
def test_full_real_producer_layout_RAM_build_and_seal_before_labels(monkeypatch,mutation):
    sealed,rows,cases=fixtures();sealed['bindings']={}
    authority=dict(rows=rows,bindings={},boundary_array_receipts={})
    physical=dict(rows=[c[2] for c in cases],journal=dict(file='journal.jsonl'),journal_tail='j'*64,
        execution_binding=dict(prepared='/ram/physical_prepare.json',prepare_sha256='p'*64,
            compiler_report_sha256=m.selector.REPORT_SHA,compiler_consume_sha256=m.selector.RECEIPT_SHA))
    prep=dict(jobs=[c[3] for c in cases],anchors=[cases[i*5][4] for i in range(8)],
        teacher_inputs=[deepcopy(cases[i*5][5]) for i in range(8)])
    model=dict(rows=[dict(blind_input=cases[i*5][5]) for i in range(8)])
    if mutation=='teacher_input':prep['teacher_inputs'][0]['context']['anchor']=999
    elif mutation=='compiler_source':physical['execution_binding']['compiler_report_sha256']='0'*64
    events=[]
    for c in cases:
        events.extend(dict(**e,kind='raw_return',job_digest=c[3]['job_digest']) for e in c[6])
        events.extend(dict(**e,kind='teacher_history_verified',job_digest=c[3]['job_digest']) for e in c[7])
    order=[]
    def read(path,*args):
        if path==m.SELECTOR:order.append('selector_read');return sealed
        if path==m.PANEL:order.append('panel_read');return authority
        if path==m.ROOT/m.rgb.INPUTS['physical'][0]:return physical
        if path==m.selector.REPORT:return model
        assert str(path)=='/ram/physical_prepare.json';return prep
    def model_build():order.append('model_build_no_labels');return sealed
    monkeypatch.setattr(m.teacher,'read',read)
    monkeypatch.setattr(m.selector,'build',model_build)
    monkeypatch.setattr(m.panel,'build',lambda:authority)
    monkeypatch.setattr(m.panel.base.physics,'read_events',lambda *args:events)
    monkeypatch.setattr(m.rgb,'verify_bindings',lambda bindings:None)
    if mutation:
        with pytest.raises(ValueError):m.build()
    else:
        result=m.build()
        assert len(result['rows'])==40 and len(result['groups'])==8
        assert result['selector_received_physical_labels'] is False and result['new_physics_calls']==0
    assert order[:3]==['selector_read','model_build_no_labels','panel_read']
