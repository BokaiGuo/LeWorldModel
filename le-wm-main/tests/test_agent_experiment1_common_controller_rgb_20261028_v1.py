"""RAM arrays only; no model/simulator/prepare files or research results."""
from copy import deepcopy
from io import BytesIO
import numpy as np
import pytest
from experiments.spectral_reachability import agent_experiment1_common_controller_rgb_20261028_v1 as m


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
    rows=[];plans=[];tasks=[];science=[]
    for i,(slot,arm) in enumerate((s,a) for s in (0,1) for a in m.ARMS):
        tc=dict(pixel_sha256=[m.digest(pixels(slot).tobytes())]*3+['f'*64],
            normalization=m.moving.normalization(),model_bindings=m.arrays.MODELS,with_target=False)
        cs=[context(slot+1),context(slot+2)]
        plan=dict(root_slot=slot,arm=arm,task_export=f'/task{slot}',task_sha256=str(slot)*64,task_id=f'task{slot}',
            blind_sha256='f'*64,selection_digest=str(i)*64,commands=np.zeros((10,2),np.float32).tolist(),
            root_sha256='d'*64,target=np.full(192,slot+.123456789,np.float64).tolist(),
            predicted_contexts=cs,prefix_target_cost=(1.-.123456789)**2,model_terminal_cost=1.,
            tracking_cost=.5*((1.-.123456789)**2)+.5)
        plans.append(plan)
        tasks.append(dict(source_slot=slot,task_export=plan['task_export'],task_sha256=plan['task_sha256'],task_id=plan['task_id'],
            blind_sha256=plan['blind_sha256'],task_contract=tc,
            task=dict(goal=np.full(192,slot+3,np.float32).tolist(),root_context=context(slot))))
        # Match actual NumPy reduction exactly rather than scalar repeated-rounding.
        plan['prefix_target_cost']=float(np.mean((np.asarray(cs[0]['latent_history'][-1],float)-np.asarray(plan['target'],float))**2))
        plan['tracking_cost']=.5*plan['prefix_target_cost']+.5*plan['model_terminal_cost']
        state=[float(slot)]*7
        cps=[dict(phase='plan',step=k,png=f'b{i}_{k}.png',file_sha256=m.digest(original(slot)),pixel_sha256=tc['pixel_sha256'][2],
            state=deepcopy(state),observation_state=deepcopy(state),snapshot=dict(state=deepcopy(state))) for k in (0,5,10)]
        rows.append(dict(root_slot=slot,arm=arm,selected_plan=plan,initial_exact=True,full_horizon=True,own_returned=10,
            warmup_returned=10,first_error=None,secondary_errors=[],status='horizon_completed',root_state=state,
            final_state=state,checkpoints=cps,initial_metrics={'native_distance':7.},final_metrics={'native_distance':8.},native_success=False))
    for slot in (0,1):
        for arm in m.producer.ALL_ARMS:
            science.append(dict(root_slot=slot,arm=arm,status='unknown' if arm==m.producer.LEARNED else 'horizon_completed',
                native_success=None if arm==m.producer.LEARNED else False))
    return dict(physical_report_sha256=m.REPORT_SHA,checkpoint_complete=True,fixed_branch_denominator=4,
        scientific_slot_denominator=6,learned_unknown_slots=2,scientific_rows=science,prepared=dict(plans=plans),
        report=dict(errors=[],full_horizons=4,status='common_controller_physical_calibration_recorded',
            prepare_sha256=m.PREPARE_SHA,rows=rows,counts={'fixture':dict(attempted=4,completed=4)})),tasks


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


def test_fixed8_real257_inherits_root_goal_and_six_slots():
    p=prepared();assert len(p['images'])==8 and len(p['rows'])==4
    assert [x['role'] for x in p['images']]==['prefix5','terminal10']*4
    assert p['shapes']['hidden']==[1,257,192] and len(p['scientific_rows'])==6
    assert p['contract']['module_caps']['encoder']==dict(API=8,samples=8,frames=8)
    assert p['contract']['new_predict']==p['contract']['new_physics']==0
    assert p['tasks'][0]['target192'][0]==.123456789  # no float32 truncation of arbitrary target


@pytest.mark.parametrize('mutation',range(11))
def test_join_failclosed(mutation):
    own,tasks=fixture();r=own['report']
    if mutation==0:r['rows'].pop()
    elif mutation==1:r['rows'][0]['checkpoints'].pop(1)
    elif mutation==2:r['rows'][0]['arm']='wrong'
    elif mutation==3:tasks[0]['task_contract']['normalization']['mean'][0]=0.
    elif mutation==4:own['prepared']['plans'][0]['predicted_contexts'][0]['latent_history'][-1][0]=123.
    elif mutation==5:own['scientific_rows'].pop()
    elif mutation==6:r['rows'][0]['checkpoints'][0]['pixel_sha256']='e'*64
    elif mutation==7:r['rows'][0]['full_horizon']=False
    elif mutation==8:own['prepared']['plans'][0]['model_terminal_cost']=2.
    elif mutation==9:r['prepare_sha256']='f'*64
    else:tasks[0]['task_contract']['with_target']=True
    with pytest.raises((ValueError,KeyError)):m.assemble(own,tasks)


def test_equal_pixels_do_not_drop_fixed_endpoints():
    p=prepared();assert len(p['images'])==8
    assert len({x['pixel_sha256'] for x in p['images']})==2


def test_full_receipt_not_summary():
    own,_=fixture();m.receipt_equal(own,deepcopy(own))
    bad=deepcopy(own);bad['report']['rows'][0]['native_success']=True
    with pytest.raises(ValueError):m.receipt_equal(own,bad)


def test_success_seal_before_q_and_independent_consumer(monkeypatch):
    p,s,r=evaluated();assert r['status']=='encoded_sealed'
    assert s.order.index('encoding_seal.json')<s.order.index('metrics.json')
    assert r['counts']['projector']['frames']==dict(attempted=8,completed=8)
    def forbidden(*a,**kw):raise AssertionError('consumer may not call producer arithmetic')
    monkeypatch.setattr(m,'score_rows',forbidden)
    out=m.consume_report(r,p,s.blob);assert len(out['rows'])==4 and out['P1_admission'] is False
    assert r['rows'][0]['checkpoints'][0]['Q']==9.
    assert r['rows'][0]['checkpoints'][0]['actual_tracking_MSE']!=r['rows'][0]['checkpoints'][0]['Q']


@pytest.mark.parametrize('field',['Q','intermediate_target_MSE','actual_tracking_MSE','model_Q','model_tracking_MSE','prediction_MSE','prediction_maxabs'])
def test_saved_metric_mutation(field):
    p,s,r=evaluated();r['rows'][0]['checkpoints'][1][field]+=1
    with pytest.raises(ValueError):m.consume_report(r,p,s.blob)


def test_partial_interrupt_keeps_four_denominator_and_null_q():
    p,s,r=evaluated(fail=2)
    assert r['status']=='partial_or_failed' and r['first_error']['type']=='KeyboardInterrupt'
    assert len(r['rows'])==4 and len(r['images'])==8 and 'metrics.json' not in s.files
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


def test_tracking_cost_and_full_precision_target():
    p,s,r=evaluated()
    row=r['rows'][0];cp=row['checkpoints'][0]
    assert cp['intermediate_target_MSE']==float(np.mean(np.full(192,.123456789,np.float64)**2))
    assert row['actual_tracking_cost']==.5*cp['actual_tracking_MSE']+.5*row['checkpoints'][1]['Q']
    r['rows'][0]['actual_tracking_cost']+=1
    with pytest.raises(ValueError):m.consume_report(r,p,s.blob)


def test_unknown_scientific_slot_cannot_be_relabelled():
    p,s,r=evaluated();r['scientific_rows'][1]['status']='failed'
    with pytest.raises(ValueError):m.consume_report(r,p,s.blob)


def test_partial_metrics_explicit_null():
    _,_,r=evaluated(fail=7)
    assert len(r['rows'])==4 and len(r['scientific_rows'])==6
    assert all(row['actual_tracking_cost'] is None for row in r['rows'])
    assert all(cp[k] is None for row in r['rows'] for cp in row['checkpoints']
        for k in ('Q','intermediate_target_MSE','actual_tracking_MSE','model_Q','prediction_MSE'))

