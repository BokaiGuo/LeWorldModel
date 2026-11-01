"""RAM arrays only; no model/simulator/prepare files or research results."""
from copy import deepcopy
from io import BytesIO
import numpy as np
import pytest
from experiments.spectral_reachability import agent_kinodynamic_matched_rgb_20261028_v1 as m


def pixels(slot,goal=False):return np.full((224,224,3),slot+4 if goal else slot,np.uint8)


def original(slot,goal=False):
    f=BytesIO()
    if goal:np.savez_compressed(f,rgb_history=np.stack([pixels(slot)]*3),goal_rgb=pixels(slot,True),raw_action_history=m.moving.fixed_raw())
    else:
        from PIL import Image
        Image.fromarray(pixels(slot)).save(f,format='PNG')
    return f.getvalue()


def context(v):
    return dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
        latent_history=np.full((3,192),v,np.float32).tolist(),action_history=np.zeros((2,10),np.float32).tolist())


def fixture():
    rows=[];plans=[];tasks=[];results=[];goals=[]
    for s in (0,1):
        tc=dict(pixel_sha256=[m.digest(pixels(s).tobytes())]*3+[m.digest(pixels(s,True).tobytes())],
            normalization=m.moving.normalization(),model_bindings=m.arrays.MODELS,with_target=False)
        plan=dict(root_slot=s,arm=m.ARM,task_export=f'/task{s}',task_sha256=str(s)*64,task_id=f'task{s}',blind_sha256=m.digest(original(s,True)),
            selection_digest='c'*64,commands=np.zeros((10,2),np.float32).tolist(),selected_path=[0,0],root_sha256='d'*64)
        plans.append(plan);cs=[context(s+1),context(s+2)]
        tasks.append(dict(source_slot=s,task_export=f'/task{s}',task_sha256=str(s)*64,task_id=f'task{s}',blind_sha256=plan['blind_sha256'],
            task_contract=tc,task=dict(goal=np.full(192,s+3,np.float32).tolist(),root_context=context(s)),prediction=deepcopy(cs[-1])))
        results.append(dict(source_slot=s,arm=m.ARM,task_id=f'task{s}',commands=deepcopy(plan['commands']),selected_path=[0,0],
            whole_contexts=deepcopy(cs),segmented_contexts=deepcopy(cs),prediction=deepcopy(cs[-1]),model_terminal_cost=1.))
        goals.append(dict(kind='goal',root_slot=s,path=f'/source/{s}.npz',file_sha256=plan['blind_sha256'],pixel_sha256=tc['pixel_sha256'][3],
            packet_metadata=dict(rgb_history_pixel_sha256=tc['pixel_sha256'][:3],raw_action_sha256=m.digest(m.moving.fixed_raw().tobytes()))))
        state=[float(s)]*7
        checkpoints=[dict(phase='plan',step=k,png=f'b{s}_{k}.png',file_sha256=m.digest(original(s)),pixel_sha256=tc['pixel_sha256'][2],
            state=deepcopy(state),observation_state=deepcopy(state),snapshot=dict(state=deepcopy(state))) for k in (0,5,10)]
        rows.append(dict(root_slot=s,arm=m.ARM,selected_plan=plan,initial_exact=True,full_horizon=True,own_returned=10,warmup_returned=10,
            first_error=None,secondary_errors=[],status='horizon_completed',root_state=state,final_state=state,checkpoints=checkpoints,
            initial_metrics={'native_distance':7.},final_metrics={'native_distance':8.},native_success=False))
    own=dict(physical_report_sha256=m.REPORT_SHA,checkpoint_complete=True,fixed_branch_denominator=2,
        prepared=dict(plans=plans),report=dict(errors=[],full_horizons=2,status='matched_domain_beam_physical_recorded_not_admitted',
            prepare_sha256=m.PREPARE_SHA,planner_report_sha256=m.physical.REPORT_SHA,rows=rows,counts={'fixture':dict(attempted=2,completed=2)}))
    return own,tasks,results,goals


def prepared():
    p=m.assemble(*fixture())
    p.update(shapes=m.arrays.config_shapes(dict(encoder=dict(image_size=224,patch_size=14,pretrained=False),projector=dict(input_dim=192,output_dim=192))),
        arguments={'physical_report':'/fixture/report.json','physical_sha':m.REPORT_SHA},expected_model_hash=m.arrays.MODEL_TENSOR_SHA)
    return p


class RAM(m.arrays.Artifacts):
    def __init__(self):self.files={};self.hashes={};self.order=[]
    def check(self,*a,**kw):return None
    def write(self,name,raw,final=False):
        assert name not in self.files
        self.files[name]=raw;self.hashes[name]=m.digest(raw);self.order.append(name);return self.hashes[name]
    def blob(self,name):return self.files[name]


def finish(**override):
    return dict(dict(model_hash_before=m.arrays.MODEL_TENSOR_SHA,model_hash_after=m.arrays.MODEL_TENSOR_SHA,
        cuda_verified=True,source_verified=True,MATH=True,TF32=False,autocast=False,blocked=[]),**override)


def evaluated(fail=None,final=None):
    p=prepared();store=RAM();entries=[]
    def loader(im):return pixels(im['root_slot'],im['kind']=='goal'),original(im['root_slot'],im['kind']=='goal')
    def encode(a,i):
        if i==fail:raise KeyboardInterrupt('fixture failure; no retry')
        z=np.full(192,i,np.float32);receipts=[]
        for j,kind in enumerate(('encoder','projector')):
            e=dict(image=i,resource=kind,completed=False,inputs=[store.array(f'i{i}_{kind}_in',np.full((1,3,224,224) if j==0 else (1,192),i,np.float32))])
            if j==0:e['preprocessed']=store.array(f'i{i}_pre',np.full((1,1,3,224,224),i,np.float32))
            ordinal=2*i+j+1;store.put(f'module_{ordinal:02d}_intent.json',deepcopy(e))
            e.update(completed=True,output=store.array(f'i{i}_{kind}_out',np.full((1,257,192) if j==0 else (1,192),i,np.float32)))
            store.put(f'module_{ordinal:02d}_returned.json',e);receipts.append(e);entries.append(e)
        return z,receipts
    store.put('execution_binding.json',{'fixture':True})
    report=m.evaluate(p,store,encode,loader,final or finish,entries)
    return p,store,report


def test_fixed8_images_real257_and_no_history_reencoding():
    p=prepared();assert len(p['images'])==8 and len(p['rows'])==2
    assert [x['role'] for x in p['images']]==['goal','root','prefix5','terminal10']*2
    assert p['shapes']['hidden']==[1,257,192]
    assert p['contract']['module_caps']['encoder']==dict(API=8,samples=8,frames=8)
    assert p['contract']['new_predict']==p['contract']['new_physics']==0


@pytest.mark.parametrize('mutation',range(11))
def test_join_failclosed(mutation):
    own,tasks,res,goals=fixture()
    if mutation==0:own['report']['rows'].pop()
    elif mutation==1:own['report']['rows'][0]['checkpoints'].pop(1)
    elif mutation==2:own['report']['rows'][0]['arm']='oldbeam'
    elif mutation==3:tasks[0]['task_contract']['normalization']['mean'][0]=0.
    elif mutation==4:res[0]['segmented_contexts'][0]['latent_history'][-1][0]=123.
    elif mutation==5:res[0]['commands'][0][0]=.1
    elif mutation==6:own['report']['rows'][0]['checkpoints'][0]['pixel_sha256']='f'*64
    elif mutation==7:own['report']['rows'][0]['full_horizon']=False
    elif mutation==8:res[0]['model_terminal_cost']=2.
    elif mutation==9:own['report']['prepare_sha256']='f'*64
    else:goals[0]['pixel_sha256']='f'*64
    with pytest.raises((ValueError,KeyError)):m.assemble(own,tasks,res,goals)


def test_no_crossroot_pixel_dedup():
    f=fixture();f[3][1]['pixel_sha256']=f[3][0]['pixel_sha256'];f[1][1]['task_contract']['pixel_sha256'][3]=f[3][0]['pixel_sha256']
    assert len(m.assemble(*f)['images'])==8  # equal pixels alone grant no alias


def test_success_seal_before_q_and_independent_consumer(monkeypatch):
    p,s,r=evaluated();assert r['status']=='encoded_sealed'
    assert s.order.index('encoding_seal.json')<s.order.index('metrics.json')
    assert r['counts']['projector']['frames']==dict(attempted=8,completed=8)
    def forbidden(*a,**kw):raise AssertionError('consumer may not call producer arithmetic')
    monkeypatch.setattr(m,'score_rows',forbidden)
    out=m.consume_report(r,p,s.blob);assert len(out['rows'])==2 and out['P1_admission'] is False
    assert r['rows'][0]['checkpoints'][0]['Q']==4.


@pytest.mark.parametrize('field',['Q','Q_encoded_goal','model_Q','prediction_MSE','prediction_maxabs'])
def test_saved_metric_mutation(field):
    p,s,r=evaluated();r['rows'][0]['checkpoints'][1][field]+=1
    with pytest.raises(ValueError):m.consume_report(r,p,s.blob)


def test_partial_interrupt_keeps_two_denominator_and_null_q():
    p,s,r=evaluated(fail=2)
    assert r['status']=='partial_or_failed' and r['first_error']['type']=='KeyboardInterrupt'
    assert len(r['rows'])==2 and len(r['images'])==8 and 'metrics.json' not in s.files
    assert all(cp['Q'] is None for row in r['rows'] for cp in row['checkpoints'])
    assert r['counts']['encoder']['API']==dict(attempted=2,completed=2)
    assert r['images'][2]['status']=='failed' and r['images'][3]['status']=='not_started'


@pytest.mark.parametrize('metadata',[{'source_verified':False},{'model_hash_after':'f'*64},{'cuda_verified':False},{'blocked':['predict']},{'MATH':False},{'TF32':True},{'autocast':True}])
def test_bad_final_metadata_never_scores(metadata):
    _,s,r=evaluated(final=lambda:finish(**metadata));assert r['status']=='partial_or_failed' and 'metrics.json' not in s.files


def test_actual_counter_failure_not_refunded():
    ops=[dict(resource='encoder',completed=True),dict(resource='projector',completed=False)]
    c=m.census(ops);assert c['projector']['API']==dict(attempted=1,completed=0)
    assert c['encoder']['samples']==c['encoder']['frames']==dict(attempted=1,completed=1)


def test_config_does_not_accept197():
    with pytest.raises(ValueError):m.arrays.config_shapes(dict(encoder=dict(image_size=224,patch_size=16,pretrained=False),projector=dict(input_dim=192,output_dim=192)))


def test_budget_caps_and_wall_in_ram():
    b=m.Budget(clock=lambda:0.,free=lambda:m.CAP+m.FLOOR);b.start_gate()
    b.used=m.CAP-m.FINAL
    with pytest.raises(ValueError):b.check(1)
    b.check(m.FINAL,final=True)
    b.clock=lambda:601.
    with pytest.raises(ValueError):b.check()


def test_no_cuda_authorization_no_build(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *a:pytest.fail('verify should not run'))
    with pytest.raises(ValueError):m.run('unused','f'*64,'unused')


def test_wrong_external_report_rejected_before_saved_read(monkeypatch):
    monkeypatch.setattr(m,'sha',lambda *a:pytest.fail('wrong authority must stop before reading'))
    with pytest.raises(ValueError):m.build('/fixture/report.json','f'*64)


def test_finalization_fault_preserves_all_image_receipts():
    def fail():raise RuntimeError('final source mismatch')
    _,s,r=evaluated(final=fail)
    assert len(r['images'])==8 and all(i['status']=='encoded' for i in r['images'])
    assert r['errors'][0]['stage']=='finalize' and 'metrics.json' not in s.files
    assert r['counts']['encoder']['API']==dict(attempted=8,completed=8)


def test_module_array_tamper_rejected():
    p,s,r=evaluated();receipt=r['images'][0]['modules'][0]['output'];s.files[receipt['file']]=b'corrupt'
    with pytest.raises(ValueError):m.consume_report(r,p,s.blob)


def test_artifact_census_rejects_foreign_file():
    p,s,r=evaluated();s.files['foreign']=b'unbound';r['artifact_sha256']['foreign']=m.digest(b'unbound')
    with pytest.raises(ValueError):m.consume_report(r,p,s.blob)
