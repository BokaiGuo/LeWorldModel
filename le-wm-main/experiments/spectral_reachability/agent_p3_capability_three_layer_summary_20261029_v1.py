"""Fixed five same-context U/Z2 pairs: saved model, real RGB, native axes only."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import hashlib
import io
import math
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_capability_probe_result_summary_20261029_v1 as saved

NAME='agent_p3_capability_three_layer_summary_20261029_v1'
RGB='agent_p3_same_time_cp15_rgb_bridge_20261029_v1'
B='agent_p3_same_time_capability_probe_cuda_20261029_v1'
ORDER=((0,'C'),(0,'Z'),(0,'P'),(1,'Z'),(1,'P'))
FLAGS=dict(P1_admission=False,P2_admission=False,P3_admission=False,full_C3_demonstrated=False,
    scientific_admission=False,physical_release=False,freshness=False,independent_confirmation=False,
    equal_FLOPs=False,equal_total_compute=False,training=False)
PINS=dict(rgb_prepare='25037ca2cbcef58df997cf9f3fd52c2c8119fc329745e45e9d4f25f20cb608e2',
    rgb_report='017a5fe3b47a418435bb7e86f4812558c43548da8a1a06fcc69e0ab2ddbc717d',
    rgb_completion='90960e63603353cc205c0e838a08cb2866e6d97beb788ed6a2a3a743271816ec',
    rgb_receipt='946ca6cc017fc66591920eca7b7d1d42bc07e8d9ebf169c0b67ecfb0b8c30cc9',
    rgb_source='aa502fd6650cb90318bcca748349cca301c42ad9c19ffad3073255bd35859067',
    physical_summary='e59eeadc6e6de711e60e50709162c4310ce0e69e9b3fdb8a5db838fd74797b19',
    capability_summary='88587a2d9f610df775c321eda6497bf3602c6ccd993b8779d7f36c6ad4744a4d',
    stageB_receipt='682a78366b10a60895c4530f2fc6cc401e6439b625f4ba0a8aa996dc2fed79a7')
IO_SOURCE_SHA='178c385b41830a685ab04c6a735725c92da42f0c8e604f226f58e0fe70ec0f66'
need,same,digest,sha=saved.need,saved.same,saved.digest,saved.sha
AXES=('native7_distance','native7_squared','joint_xy_error','joint_xy_squared','agent_xy_error','agent_xy_squared',
      'object_xy_error','object_xy_squared','velocity_error','velocity_squared','raw_angle_error','raw_angle_squared','wrapped_angle_error')


def array(value,shape,*,f32=False):
    def numeric(v):
        if type(v) is list:
            for x in v:numeric(x)
        else:need(type(v) in (int,float) and math.isfinite(v),'finite literal number, not bool')
    numeric(value)
    a=np.asarray(value,np.float64)
    need(a.shape==shape and np.isfinite(a).all(),'fixed full array shape')
    if f32:need(np.array_equal(a,a.astype(np.float32).astype(np.float64)),'already native float32')
    return a


def sq(values):return math.fsum(float(x)*float(x) for x in values)


def latent_math(pred,actual,goal):
    p=array(pred,(192,),f32=True);g=array(goal,(192,));r=p-g
    out=dict(predicted_Q_fsum=sq(r)/192,actual_Q_fsum=None,error_MSE=None,cross_term=None,
        actual_minus_predicted_Q=None,decomposition_roundoff=None)
    if actual is not None:
        a=array(actual,(192,),f32=True);e=a-p
        q=sq(a-g)/192;err=sq(e)/192;cross=2*math.fsum(float(x)*float(y) for x,y in zip(r,e,strict=True))/192
        out.update(actual_Q_fsum=q,error_MSE=err,cross_term=cross,actual_minus_predicted_Q=q-out['predicted_Q_fsum'],
            decomposition_roundoff=q-math.fsum((out['predicted_Q_fsum'],err,cross)))
    return out


def native_math(state,goal):
    s=array(state,(7,));g=array(goal,(7,));d=s-g;terms=[float(x)*float(x) for x in d]
    total=math.fsum(terms);xy=math.fsum(terms[:4]);vel=math.fsum(terms[5:]);angle=abs(float(d[4]))
    return dict(state7=state,goal7=goal,signed_residual7=d.tolist(),squared_residual7=terms,
        native7_squared=total,native7_distance=math.sqrt(total),joint_xy_squared=xy,joint_xy_error=math.sqrt(xy),
        agent_xy_squared=math.fsum(terms[:2]),agent_xy_error=math.sqrt(math.fsum(terms[:2])),
        object_xy_squared=math.fsum(terms[2:4]),object_xy_error=math.sqrt(math.fsum(terms[2:4])),
        velocity_squared=vel,velocity_error=math.sqrt(vel),raw_angle_squared=terms[4],raw_angle_error=angle,
        wrapped_angle_error=abs(math.atan2(math.sin(float(d[4])),math.cos(float(d[4])))),
        native_angle_term=min(angle,2*math.pi-angle),
        native7_group_roundoff=total-math.fsum((xy,terms[4],vel)))


def difference(a,b):
    if a is None or b is None:return dict(value=None,sign='null')
    need(math.isfinite(a) and math.isfinite(b),'finite pair operands')
    return dict(value=a-b,sign='positive' if a>b else 'negative' if a<b else 'tie')


def alignment(a,b):
    if 'null' in (a['sign'],b['sign']):return 'null'
    if 'tie' in (a['sign'],b['sign']):return 'both_tie' if a['sign']==b['sign'] else 'one_tie'
    return 'same' if a['sign']==b['sign'] else 'reversal'


def check_context(task,child,raw5):
    parent=task['root_context'];array(parent['latent_history'],(3,192),f32=True)
    array(child['latent_history'],(3,192),f32=True)
    need(set(child)==set(parent) and all(same(child[k],parent[k]) for k in parent if k not in ('latent_history','action_history')),
         'native full child metadata/layout')
    mean=array(task['action_mean'],(2,),f32=True).astype(np.float32)
    std=array(task['action_std'],(2,),f32=True).astype(np.float32);need(np.all(std>0),'positive original normalizer')
    raw=array(raw5,(5,2),f32=True).astype('<f4');saved.raw_bits(raw5,5)
    token=((raw-mean)/std).reshape(10)
    need(same(child['latent_history'][:-1],parent['latent_history'][1:]) and
         array(child['action_history'],(2,10),f32=True).astype('<f4').tobytes()==
         np.asarray([parent['action_history'][-1],token.tolist()],dtype='<f4').tobytes(),
         'same parent shift and outgoing once-normalized action token')


def load_latent(out,e,completion,seen):
    need(e['shape']==[192] and e['dtype']=='float32' and Path(e['file']).name==e['file'] and e['file'].endswith('.npy.gz'),
         'literal latent descriptor, no path escape')
    path=str(saved.absolute(str(out/e['file'])))
    need(completion['artifact_sha256'][e['file']]==e['sha256']==sha(path),'latent compressed full-byte SHA')
    need(Path(path).stat().st_size<=2048,'fixed 192-vector compressed bound')
    with gzip.open(path,'rb') as f:body=f.read(897)
    need(len(body)==896,'192 float32 NPY1 exact128 header, no extra members/bytes')
    stream=io.BytesIO(body);a=np.load(stream,allow_pickle=False)
    need(a.shape==(192,) and a.dtype==np.dtype('<f4') and a.flags.c_contiguous and np.isfinite(a).all() and
         stream.tell()==len(body) and hashlib.sha256(a.tobytes()).hexdigest()==e['array_sha256'],'original NPY literal float32 bits')
    seen[path]=e['sha256'];return a.tolist()


def closure():
    root=Path(__file__).resolve().parents[2]
    files=[Path(__file__).resolve(),root/'tests'/('test_'+NAME+'.py'),root/'innerdocs'/(NAME+'.md'),Path(saved.__file__).resolve(),
        Path(saved.math_contract.__file__).resolve()]
    need(sha(str(Path(saved.__file__).resolve()))==IO_SOURCE_SHA,'frozen IO helper source, not producer arithmetic')
    return {str(f):sha(str(f)) for f in files}


def build(request,*,fixture=False):
    need(set(request)==set(PINS),'eight explicit complete authority files')
    if not fixture:need(all(request[k]['sha256']==h for k,h in PINS.items()),'MAIN exact fixed actual authorities')
    sources=closure();seen={}
    p,r,c,s,ph,cap,b=[saved.load(request[k],seen) for k in
        ('rgb_prepare','rgb_report','rgb_completion','rgb_receipt','physical_summary','capability_summary','stageB_receipt')]
    source=request['rgb_source'];need(Path(source['path']).name==RGB+'.py' and sha(source['path'])==source['sha256'] and
        p['bindings'][source['path']]==source['sha256'],'frozen own RGB source authority');seen[source['path']]=source['sha256']
    need(p['schema']==RGB+'.prepare' and p['fixture_only'] is fixture and r['schema']==RGB+'.report' and
         c['schema']==RGB+'.completion' and s['schema']==RGB+'.consume','actual RGB four authority ABI')
    need(r['prepared']==c['prepared']==request['rgb_prepare']['path'] and
         r['prepare_sha256']==c['prepare_sha256']==request['rgb_prepare']['sha256'] and
         c['report_sha256']==s['report_sha256']==request['rgb_report']['sha256'] and
         s['completion_sha256']==request['rgb_completion']['sha256'],'all RGB prepare/report/completion/receipt byte joins')
    need(c['status']=='published_within_cap' and 0<=r['elapsed_seconds']<=c['inclusive_wall_seconds']<=600,
         'inclusive RGB publication wall status')
    need(s['status']==r['status']=='all10_cp15_slots_saved_RGB_bridge_NOT_admitted' and r['errors']==r['blocked']==[] and
         r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
         r['model_inventory_before']==r['model_inventory_after'],'complete frozen RGB saved authority')
    for v in (r,c,s):
        need(all(v[k] is False for k in FLAGS if k!='P2_admission'),'original scientific flags explicitly false')
    need(same(r['rows'],s['rows']) and len(s['rows'])==len(p['rows'])==len(s['tasks'])==10 and
         s['fixed_denominator']==10 and s['known_roots']==2 and s['fresh_roots']==0 and
         s['consume_new_neural_calls']==s['consume_new_physics']==0 and s['total_FLOPs'] is None,'complete ten RGB rows and known2')
    need(ph['schema']==saved.NAME+'.summary' and ph['fixture_only'] is fixture and len(ph['original_receipts'])==len(ph['original_rows'])==10 and
         ph['known_roots']==2 and ph['fresh_roots']==0 and ph['upstream_fullstrict_reexecuted'] is False,
         'entire inherited physical scalar authority, not status alone')
    need(cap['schema']=='agent_p3_capability_signal_summary_20261029_v1.summary' and len(cap['signals'])==7 and len(cap['probes'])==5 and
         b['schema']==B+'.consume' and b['status']=='saved_complete_NOT_admitted' and len(b['plans'])==5,
         'complete original capability/B authorities')
    need(same(cap['authorities']['stageB_receipt'],request['stageB_receipt']) and
         same(ph['request']['capability'],request['capability_summary']) and
         same(cap['inherited_stageB_resource_receipt'],b['resource_receipt']) and
         same(ph['inherited_stageB_resource_receipt'],b['resource_receipt']), 'full upstream resource/summary inheritance')
    expected_authority={k:ph['request'][a] for k,a in (('physical_prepare','prepared'),('physical_batch','batch'),('physical_receipt','receipt'))}
    need(same(s['source_authorities'],p['authorities']) and same(p['authorities'],expected_authority), 'RGB and scalar same original physical triplet')
    need(all(ph[k] is False for k in FLAGS) and
         all(cap[k] is False for k in FLAGS if k!='equal_total_compute'),'explicit inherited producer flags, no nonexistent cap field')
    need(len(ph['arithmetic']['branches'])==10 and len(ph['arithmetic']['U_Z2_pairs'])==5 and
         len(ph['arithmetic']['C_control_pairs'])==8 and len(ph['arithmetic']['C_dual_control_conditions'])==4,
         'all original logical C and native denominators')
    logical=((0,'H1'),(0,'H3'),(1,'H1'),(1,'H3'))
    for ci,(root,arm) in enumerate(logical):
        dual=ph['arithmetic']['C_dual_control_conditions'][ci]
        need((dual['root_slot'],dual['C_arm'])==(root,arm) and
             dual['original_immediate_cost_both_controls'] is (False if ci==0 else None),
             'fixed four logical C, known immediate false and three no_witness')
        for j,role in enumerate(('Z','P')):
            pair=ph['arithmetic']['C_control_pairs'][2*ci+j]
            need((pair['index'],pair['root_slot'],pair['C_arm'],pair['control_role'])==(2*ci+j,root,arm,role) and
                 pair['original_C_no_witness'] is (ci!=0), 'all eight original C/control slots')
    tasks=saved.load(dict(path=str(Path(request['rgb_report']['path']).parent/'cp15_tasks.json'),sha256=r['tasks_sha256']),seen)
    need(same(tasks,s['tasks']) and c['artifact_sha256']['cp15_tasks.json']==r['tasks_sha256'] and
         c['artifact_sha256']['report.json']==request['rgb_report']['sha256'], 'whole cp15 task publication exact bytes')
    branches=[];encoded=0
    for i,(spec,row,taskout) in enumerate(zip(p['rows'],s['rows'],s['tasks'],strict=True)):
        e=i//2;root,role=ORDER[e];arm='Z2' if i%2 else 'U';plan=b['plans'][e];prior=cap['probes'][e]
        need((row['index'],row['endpoint'],row['root_slot'],row['role'],row['probe'])==(i,e,root,role,arm), 'fixed ten slots/order')
        immutable=lambda x:{k:v for k,v in x.items() if k not in ('status','latent','pixels','preprocessed','modules')}
        need(same(immutable(row),immutable(spec)),'whole RGB immutable prepared slot')
        need(digest({k:v for k,v in plan.items() if k!='witness_sha256'})==plan['witness_sha256']==prior['witness_sha256'] and
             (plan['index'],plan['root_slot'],plan['role'])==(e,root,role),'original stageB whole plan content identity')
        t=plan['task'];ctx=t['root_context'];goal=t['goal'];array(goal,(192,))
        need(same(t,row['cp10_task']) and same(plan['endpoint_identity'],row['cp10_identity']) and
             same(prior['endpoint_identity'],row['cp10_identity']) and plan['task_sha256']==digest(t)==prior['task_sha256'],
             'same exact cp10 parent task/identity, never cross-context predictions')
        need(np.array_equal(array(t['projection'],(192,192)),np.eye(192)),'original identity192 Q only')
        for key,value in (('goal_sha256',goal),('projection_sha256',t['projection']),
                          ('normalizer_sha256',[t['action_mean'],t['action_std']]),('context_sha256',ctx)):
            need(t['binding'][key]==digest(value),'original task component digest')
        need(plan['signal']['parent_context_sha256']==digest(ctx) and
             plan['signal']['binding_sha256']==digest(t['binding']) and
             same(plan['nominal_context']['latent_history'][-1],plan['signal']['nominal']) and
             plan['whole']==plan['segmented'] and len(plan['whole'])==1,'same original zero forward and U dual replay')
        check_context(t,plan['whole'][0],plan['raw5']);check_context(t,plan['nominal_context'],[[0.,0.]]*5)
        child=plan['whole'][0] if arm=='U' else plan['nominal_context'];pred=child['latent_history'][-1]
        expected_tail=plan['raw5'] if arm=='U' else [[0.,0.]]*5
        need(saved.raw_bits(row['raw15'][:10],10)==saved.raw_bits(plan['causal_raw10'],10) and
             saved.raw_bits(row['raw15'][10:],5)==saved.raw_bits(expected_tail,5) and
             ''.join(saved.raw_bits(row['raw15'],15))==row['raw15_bits'],'same actual committed raw15 byte identity')
        incoming=np.asarray(row['raw15'][-10:],dtype='<f4');mean=np.asarray(t['action_mean'],np.float32);std=np.asarray(t['action_std'],np.float32)
        tokens=((incoming-mean)/std).reshape(2,10)
        need(incoming.tobytes().hex()==row['causal_raw10_bits'] and same(incoming.tolist(),row['causal_raw10']) and
             tokens.tobytes()==array(row['normalized_action_history'],(2,10),f32=True).astype('<f4').tobytes() and
             hashlib.sha256(tokens.tobytes()).hexdigest()==row['normalized_float32_sha256'], 'both actual history tokens regenerated once')
        need(tokens.tobytes()==array(child['action_history'],(2,10),f32=True).astype('<f4').tobytes(),
             'model replay and actual encoding have identical two incoming action tokens')
        entry=ph['original_receipts'][i];physical=ph['original_rows'][i];actual=None;native=None
        need(row['physical_receipt_entry_sha256']==digest(entry),'whole original physical receipt entry digest')
        need(same(row['physical_counts'],None if physical is None else physical['counts']) and
             row['physical_status']==(None if physical is None else physical['status']), 'native full resource vector/status identity')
        endpoint=None if physical is None else physical['endpoint15']
        if row['source'] is None:
            need(endpoint is None and row['status']=='missing_physical_cp15' and row['latent'] is taskout is None and
                 row['missing_reason'] is not None and row['physical_response'] is None,'early stop/missing null, never substitute terminal')
        else:
            encoded+=1
            need(endpoint is not None and row['status']=='encoded' and row['missing_reason'] is None and
                 all(same(row['source'][k],endpoint[k]) for k in endpoint) and
                 row['source']['path']==str(Path(entry['output'])/endpoint['png']) and
                 row['selection_digest']==physical['job']['selection_digest'] and
                 saved.raw_bits(row['raw15'],15)==saved.raw_bits(physical['job']['raw15'],15), 'actual PNG/state/full snapshot/raw row join')
            response=next(x for x in physical['responses'] if x['step']==15)
            need(same(row['physical_response'],response) and same(endpoint['state'],response['state']), 'same actual native cp15 response')
            native=native_math(endpoint['state'],prior['native_cp10']['goal7'])
            actual=load_latent(Path(request['rgb_report']['path']).parent,row['latent'],c,seen)
            derived=deepcopy(t);newctx=deepcopy(ctx);newctx['latent_history']=ctx['latent_history'][1:]+[actual]
            newctx['action_history']=row['normalized_action_history'];identity=taskout['identity']
            need(identity==dict(namespace=RGB,index=i,root_slot=root,role=role,probe=arm,history_absolute_steps=[15,20,25],
                source=row['source'],cp10_identity=row['cp10_identity'],raw10_sha256=hashlib.sha256(incoming.tobytes()).hexdigest(),
                encoded_latent=row['latent'],cp10_task_sha256=digest(t)), 'entire actual cp15 task identity')
            derived['root_context']=newctx
            derived['binding'].update(context_sha256=digest(newctx),task_id=digest([RGB,identity]),
                task_sha256=digest([t,newctx,identity]),task_contract_sha256=digest(identity))
            need(taskout['index']==i and same(taskout['task'],derived) and taskout['task_sha256']==digest(derived),
                 'actual full three-frame context and unchanged goal/Q/model/normalizer')
        metrics=latent_math(pred,actual,goal)
        original_key='whole' if arm=='U' else 'nominal'
        exact=float(np.mean((array(t['projection'],(192,192))@(array(pred,(192,))-array(goal,(192,))))**2))
        need(exact==plan['Q'][original_key]==prior['Q_original_numpy'][original_key] and
             metrics['predicted_Q_fsum']==prior['Q_fsum'][original_key],'original saved NumPy and independent fsum Q')
        oldbranch=ph['arithmetic']['branches'][i]
        need(oldbranch['index']==i and oldbranch['endpoint']==e and oldbranch['root_slot']==root and
             oldbranch['role']==role and oldbranch['probe']==arm and
             (oldbranch['metrics'] is None)==(native is None), 'fixed scalar summary ten branches and missing')
        if native is not None:
            need(all(native[k]==oldbranch['metrics'][k] for k in AXES) and
                 oldbranch['original_native_success'] is row['physical_response']['native_success'] and
                 oldbranch['reported_D_original_reward']==-row['physical_response']['reward'], 'independent complete native scalar recomputation')
        branches.append(dict(index=i,endpoint=e,root_slot=root,role=role,probe=arm,
            checkpoint=15,absolute_step=25,parent_context_sha256=digest(ctx),task_sha256=digest(t),
            predicted_context=child,goal192=goal,predicted_latent192=pred,actual_latent192=actual,
            latent=metrics,original_predicted_Q_numpy=exact,fsum_minus_original=metrics['predicted_Q_fsum']-exact,
            native_metrics=native,native_success=oldbranch['original_native_success'],
            D_original_reward=oldbranch['D_original_reward'],eligible_same_time_endpoint=oldbranch['eligible_same_time_endpoint'],
            physical_status=row['physical_status'],missing_reason=row['missing_reason'],actual_task=taskout,
            source=row['source'],terminal=ph['mapped_rows'][i]))
    need(s['encoded']==encoded and s['missing']==10-encoded and same(r['counts'],s['counts']), 'all actual encodings and missing census')
    expected={k:dict(attempted=encoded if k in ('encoder','projector') else 0,completed=encoded if k in ('encoder','projector') else 0)
              for k in ('encoder','projector','action_encoder','predict_API','predictor','pred_proj')}
    need(same(s['counts'],expected) and same(p['expected_calls'],{k:v['completed'] for k,v in expected.items()}),'full six-axis neural accounting, zero new predict')
    pairs=[]
    for e,(root,role) in enumerate(ORDER):
        u,z=branches[2*e:2*e+2]
        need(u['parent_context_sha256']==z['parent_context_sha256'] and u['task_sha256']==z['task_sha256'], 'same endpoint/context pair only')
        model=difference(z['latent']['predicted_Q_fsum'],u['latent']['predicted_Q_fsum'])
        real=difference(z['latent']['actual_Q_fsum'],u['latent']['actual_Q_fsum'])
        nat=difference(z['D_original_reward'],u['D_original_reward'])
        axes={k:difference(None if not z['eligible_same_time_endpoint'] else z['native_metrics'][k],
                           None if not u['eligible_same_time_endpoint'] else u['native_metrics'][k]) for k in AXES}
        inherited=ph['arithmetic']['U_Z2_pairs'][e]
        need(nat['value']==inherited['G'] and all(axes[k]['value']==inherited['error_improvements'][k]['A_minus_B'] for k in AXES),
             'all fixed prior native pair margins, no favorable-axis selection')
        pairs.append(dict(index=e,root_slot=root,role=role,U_job=2*e,Z2_job=2*e+1,same_parent_context=True,
            model_gain=model,real_latent_gain=real,native_reward_distance_gain=nat,native_axis_gains=axes,
            real_minus_model_gain=None if real['value'] is None else real['value']-model['value'],
            signs=dict(model_vs_real=alignment(model,real),real_vs_native=alignment(real,nat),
                model_vs_native=alignment(model,nat),real_vs_native_axes={k:alignment(real,v) for k,v in axes.items()}),
            native_success=dict(U=u['native_success'],Z2=z['native_success'],relation=inherited['success_relation'])))
    result=dict(schema=NAME+'.summary',status='fixture_only_NOT_evidence' if fixture else 'saved_three_layer_NOT_admitted',
        request=request,sources=sources,input_sha256=seen,fixture_only=fixture,branches=branches,pairs=pairs,
        fixed_denominators=dict(branches=10,same_context_pairs=5,C_logical_slots=4,C_control_pairs=8,C_no_witness=3),
        original_C_control_pairs=ph['arithmetic']['C_control_pairs'],original_C_dual_control_conditions=ph['arithmetic']['C_dual_control_conditions'],
        cross_context_model_comparisons=[],same_time_actual_step=25,known_roots=2,fresh_roots=0,
        inherited_stageB_resources=b['resource_receipt'],inherited_RGB_counts=s['counts'],
        inherited_RGB_IO_bytes=s['actual_module_IO_bytes'],inherited_native_counts=ph['native_resource_vectors'],
        upstream_fullstrict_reexecuted=False,new_neural_calls=0,new_AD_calls=0,new_physics_calls=0,FLOPs=None,**FLAGS)
    need(closure()==sources and all(sha(f)==h for f,h in seen.items()),'source/data unchanged through saved build')
    return result


def verify(path,expected,*,fixture=False):
    value=saved.load(dict(path=path,sha256=expected),{})
    need(value['fixture_only'] is fixture and same(value,build(value['request'],fixture=fixture)) and sha(path)==expected,'whole saved summary rebuild')
    return dict(schema=NAME+'.verify',summary=path,summary_sha256=expected,fixture_only=fixture,**FLAGS)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);sub=q.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('build')
    for k in PINS:
        p.add_argument('--'+k.replace('_','-'),required=True);p.add_argument('--'+k.replace('_','-')+'-sha',required=True)
    p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--summary',required=True);p.add_argument('--summary-sha',required=True)
    a=q.parse_args(argv)
    if a.cmd=='build':v=saved.write_new(a.output,build({k:dict(path=getattr(a,k),sha256=getattr(a,k+'_sha')) for k in PINS}))
    else:v=verify(a.summary,a.summary_sha)
    print(saved.encoded(v).decode())


if __name__=='__main__':main()
