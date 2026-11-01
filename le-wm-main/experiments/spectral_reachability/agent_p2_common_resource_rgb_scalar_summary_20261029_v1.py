"""Independent saved-only eighteen-slot three-arm common-resource scalar arithmetic. No producer imports."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import zlib
import numpy as np

NAME='agent_p2_common_resource_rgb_scalar_summary_20261029_v1'
RGB='agent_p2_common_resource_rgb_20261029_v1'
PHYSICAL='agent_p2_common_resource_physical_20261029_v1'
MODEL='agent_p2_common_resource_grasp_cuda_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
PLAN_ORDER=((0,'joint'),(0,'composition'),(0,'grasp'),(1,'grasp'),(1,'composition'),(1,'joint'))
ORDER=tuple((root,arm,step) for root,arm in PLAN_ORDER for step in (5,10,15))
DIRECTIONS=(('composition','joint'),('grasp','joint'),('grasp','composition'))
AXES=('joint_xy_error','angle_native_term','angle_wrapped_error','velocity_error','native7_distance','native7_reward')
META=128*1024**2
ARTIFACT_CAP=64*1024**2
ARRAY_CAP=2*1024**2
ARGS=('prepared','prepare_sha','report','report_sha','strict','strict_sha')


def require(ok,message):
    if not ok:raise ValueError(message)


def raw(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def bhash(b):return hashlib.sha256(b).hexdigest()


def sha(path):
    p=Path(path);require(p.is_file() and not p.is_symlink(),'regular non-symlink bound file')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def absolute(value):
    require(type(value) is str and Path(value).is_absolute() and str(Path(value).resolve())==value,'canonical absolute path')
    return Path(value)


def read(path):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=META,'bounded regular JSON')
    body=p.read_bytes();require(len(body)<=META,'bounded actual JSON bytes')
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON field');out[k]=v
        return out
    def invalid(_):raise ValueError('nonfinite JSON literal')
    return json.loads(body,object_pairs_hook=pairs,parse_constant=invalid)


def same(a,b,message):require(raw(a)==raw(b),message)


def f64(value,shape):
    a=np.asarray(value,np.float64);require(a.shape==shape and np.isfinite(a).all(),'finite exact numeric shape')
    return a


def native32(value,shape):
    a=f64(value,shape);b=a.astype(np.float32)
    require(np.isfinite(b).all() and np.array_equal(a,b.astype(np.float64)),'literal float32 values, no lossy conversion')
    return b


def mse(a,b):
    a=f64(a,(192,));b=f64(b,(192,));return float(np.mean((a-b)**2,dtype=np.float64))


def decomposition(predicted,actual,goal):
    p,a,g=(f64(x,(192,)) for x in (predicted,actual,goal))
    e=p-a;d=a-g
    model=mse(p,g);real=mse(a,g);error=mse(p,a)
    cross=float(2*np.mean(d*e,dtype=np.float64));gap=model-real;residual=gap-(cross+error)
    # Forward-rounding allowance for this algebra identity ONLY. Never a gate or data-fit threshold.
    eps=np.finfo(np.float64).eps;ops=16*192+64;gamma=ops*eps/(1-ops*eps)
    scale=float(np.mean((np.abs(p)+np.abs(g))**2+(np.abs(a)+np.abs(g))**2+(np.abs(p)+np.abs(a))**2))
    bound=float(gamma*scale)
    require(abs(residual)<=bound,'float64 quadratic identity exceeds derived arithmetic bound')
    return dict(model_Q=model,real_Q=real,prediction_error_MSE=error,model_minus_real_Q=gap,
        actual_goal_error_cross_prediction_error=cross,prediction_error_quadratic=error,
        identity='model_Q-real_Q = 2 mean((actual-goal)*(pred-actual)) + mean((pred-actual)^2)',
        identity_rounding_residual=residual,identity_rounding_bound=bound,
        dimension=192,normalization='mean, not sum; sum = 192 * mean',scientific_tolerance_changed=False)


def native_scalars(goal,state):
    g,s=(f64(x,(7,)) for x in (goal,state));d=s-g;sq=d*d
    xy=float(np.linalg.norm(d[:4]));distance=float(np.linalg.norm(d));angle=float(abs(d[4]))
    original=float(min(angle,2*np.pi-angle));wrapped=float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    success=bool(xy<20 and original<np.pi/9);wrapped_success=bool(xy<20 and wrapped<np.pi/9)
    native=dict(native_success=success,native_distance=distance,native_angle_term=original,
        negative_native_angle=bool(original<0),wrapped_allxy_success=wrapped_success,
        wrapped_object_angle_error=wrapped,object_position_error=float(np.linalg.norm(d[2:4])),
        object_pose=s[2:5].tolist(),object_success=None)
    original_metrics=dict(native=native,native_success=success,joint_xy_error=xy,angle_native_term=original,
        angle_wrapped_error=wrapped,velocity_error=float(np.linalg.norm(d[5:])),native7_distance=distance,native7_reward=-distance)
    axes=dict(delta_native7=d.tolist(),squared_coordinate_contributions=sq.tolist(),
        agent_xy_squared=float(np.sum(sq[:2])),object_xy_squared=float(np.sum(sq[2:4])),
        raw_angle_squared=float(sq[4]),velocity_squared=float(np.sum(sq[5:])),
        agent_xy_error=float(np.linalg.norm(d[:2])),object_xy_error=float(np.linalg.norm(d[2:4])),
        joint_xy_error=xy,raw_angle_delta=float(d[4]),raw_angle_absolute=angle,
        native_angle_term=original,wrapped_angle_error=wrapped,velocity_error=float(np.linalg.norm(d[5:])),
        pose5_distance=float(np.linalg.norm(d[:5])),native7_distance=distance,native7_reward=-distance,
        native7_squared=float(np.sum(sq)),distance_square_rounding_residual=distance*distance-float(np.sum(sq)),
        original_success=success,wrapped_success_diagnostic=wrapped_success,
        negative_native_angle=bool(original<0),object_only_success=None,
        interpretation='native7 includes velocity; wrapped angle diagnostic never replaces raw/native distance or success')
    return original_metrics,axes


def flat(directory,name):
    require(type(name) is str and Path(name).name==name and name not in ('','.','..'),'flat artifact name')
    p=Path(directory)/name;require(not p.is_symlink(),'no symlink artifact');return p


def load_npy(directory,receipt,artifacts):
    require(set(receipt)=={'file','sha256','array_sha256','shape','dtype'},'original NPY receipt fields')
    path=flat(directory,receipt['file']);require(path.stat().st_size<=ARRAY_CAP,'bounded compressed NPY')
    body=path.read_bytes()
    require(bhash(body)==receipt['sha256']==artifacts[receipt['file']],'NPY actual compressed SHA and report binding')
    d=zlib.decompressobj(31);payload=d.decompress(body,ARRAY_CAP+1)
    require(len(payload)<=ARRAY_CAP and d.eof and not d.unused_data and not d.unconsumed_tail,'single complete bounded gzip')
    stream=io.BytesIO(payload);a=np.load(stream,allow_pickle=False)
    require(stream.tell()==len(payload) and a.dtype==np.dtype('float32') and str(a.dtype)==receipt['dtype']
        and list(a.shape)==receipt['shape'] and np.isfinite(a).all()
        and bhash(np.ascontiguousarray(a).tobytes())==receipt['array_sha256'],'NPY shape/dtype/data SHA, no trailing payload')
    return a


def bind_sources(arguments):
    require(set(arguments)==set(ARGS),'explicit prepare/report/strict paths and actual SHAs')
    bindings={}
    def bind(path,expected):
        p=absolute(str(path))
        require(type(expected) is str and len(expected)==64 and expected!='0'*64
            and all(x in '0123456789abcdef' for x in expected),'actual lowercase SHA256')
        require(str(p) not in bindings or bindings[str(p)]==expected,'nonconflicting provenance binding')
        require(sha(p)==expected,'bound source SHA mismatch: '+str(p));bindings[str(p)]=expected;return p
    for a,b in (('prepared','prepare_sha'),('report','report_sha'),('strict','strict_sha')):bind(arguments[a],arguments[b])
    p,r,s=(read(arguments[k]) for k in ('prepared','report','strict'))
    require(p['schema']==RGB+'.prepare' and r['schema']==RGB+'.report' and s['schema']==RGB+'.consume'
        and r['status']=='common_resource_RGB_encoded_NOT_science'
        and s['status']=='all18_common_resource_RGB_slots_verified_NOT_science','actual own complete RGB authority')
    require(r['prepared']==arguments['prepared'] and r['prepare_sha256']==arguments['prepare_sha']
        and s['report_sha256']==arguments['report_sha'],'explicit prepare/report/strict join')
    same(p['contract'],r['contract'],'same fixed contract')
    require(r['errors']==r['blocked']==[] and r['source_verified'] is r['model_unchanged'] is True
        and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'clean frozen model')
    for obj in (r,s):
        require(all(obj[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','physical_release')),'no scientific admission')
    for path,h in p['bindings'].items():bind(path,h)
    for spec in p['rows']:
        if spec['source'] is not None:bind(spec['source']['path'],spec['source']['file_sha256'])
    directory=absolute(arguments['report']).parent;files=list(directory.iterdir())
    require({f.name for f in files}==set(r['artifact_sha256'])|{'report.json'}
        and sum(f.stat().st_size for f in files)<=ARTIFACT_CAP,'complete bounded artifact census')
    for name,h in r['artifact_sha256'].items():bind(flat(directory,name),h)
    same(read(directory/'binding.json'),dict(prepared=arguments['prepared'],prepare_sha256=arguments['prepare_sha']),'execution binding')
    same(read(directory/'initial_roster.json'),p['rows'],'whole original eighteen-slot roster')
    require(sha(directory/'native_seal.json')==r['native_seal_sha256'],'native seal SHA')
    same(read(directory/'native_seal.json'),{k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native seal')
    a=p['arguments'];pr=read(bind(Path(a['physical_output'])/'report.json',a['physical_report_sha']))
    ps=read(bind(a['physical_receipt'],a['physical_receipt_sha']));b=pr['execution_binding']
    pp=read(bind(b['prepared'],b['prepare_sha256']))
    require(pp['schema']==PHYSICAL+'.prepare' and pr['schema']==PHYSICAL+'.report' and ps['schema']==PHYSICAL+'.consume'
        and pr['status']==ps['status']=='common_resource6_native_recorded_not_admitted'
        and ps['physical_report_sha256']==a['physical_report_sha']==s['physical_report_sha256']==p['physical_report_sha256']
        and a['physical_receipt_sha']==s['physical_receipt_sha256']==p['physical_receipt_sha256'],'actual own physical authority')
    same(pr['rows'],ps['rows'],'whole physical rows')
    for path,h in pp['bindings'].items():bind(path,h)
    a=pp['arguments'];mr=read(bind(Path(a['planner_output'])/'report.json',a['planner_report_sha']))
    ms=read(bind(a['planner_receipt'],a['planner_receipt_sha']))
    mp=read(bind(mr['prepared'],mr['prepare_sha256']))
    require(mp['schema']==MODEL+'.prepare' and mr['schema']==MODEL+'.report' and mr['status']=='records_sealed'
        and ms['schema']==MODEL+'.consume' and ms['status']=='all6_common_cap_records_verified_not_admitted'
        and ms['report_sha256']==a['planner_report_sha'],'actual common-resource six-plan runtime, not legacy root files')
    for k,want in (('runtime_report_sha256',a['planner_report_sha']),('runtime_receipt_sha256',a['planner_receipt_sha']),
                   ('runtime_prepare_sha256',mr['prepare_sha256'])):
        require(p[k]==s[k]==want,'RGB/runtime authority '+k)
    same(mp['slots'],pp['model_slots'],'whole original model slots')
    same(mp['contract'],pp['planner_authority']['runtime_contract'],'runtime contract')
    same(ps['plans'],pp['plans'],'six physical plans including null exports')
    same(ms['plans'],[x['planner_plan'] if x else None for x in pp['plans']],'six exact runtime plans')
    require(mp['expected_model_hash']==pp['expected_model_hash']==p['expected_model_hash'],'same original model identity')
    return p,r,s,dict(physical_prepare=pp,physical_strict=ps,model_prepare=mp,physical_output=p['arguments']['physical_output']),bindings


def eighteen_rows(p,r,s,source,load):
    require(len(p['rows'])==len(r['rows'])==len(s['rows'])==s['fixed_denominator']==18
        and s['plan_denominator']==6 and s['group_denominator']==6 and s['pair_denominator']==18,'fixed18 slots/18pairs/6plans')
    pp,ps,mp=(source[k] for k in ('physical_prepare','physical_strict','model_prepare'))
    require(len(pp['plans'])==len(ps['rows'])==len(pp['model_witnesses'])==6 and len(mp['slots'])==2,'six plans/two known roots')
    output=[]
    for i,(root,arm,step) in enumerate(ORDER):
        spec,row,strict=p['rows'][i],r['rows'][i],s['rows'][i];pi=i//3;j=step//5-1
        for v in (spec,row,strict):
            require(all(type(v[k]) is int for k in ('index','root_slot','checkpoint','plan_index'))
                and (v['index'],v['root_slot'],v['arm'],v['checkpoint'],v['plan_index'])==(i,root,arm,step,pi)
                and v['condition']=='off','exact counterbalanced root/arm/time order')
        same({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')},
            {k:v for k,v in spec.items() if k not in ('status','latent')},'immutable whole RGB request')
        plan,actual,slot=pp['plans'][pi],ps['rows'][pi],mp['slots'][root]
        same(actual['plan'],plan,'whole physical plan')
        require((actual['index'],actual['root_slot'],actual['condition'],actual['arm'])==(pi,root,'off',arm),'physical row identity')
        same(spec['task_id'],slot['task_id'],'original task id');same(spec['task_sha256'],slot['task_sha256'],'task bytes')
        same(spec['goal192'],slot['packet']['goal'],'original goal192');same(spec['root_context'],slot['packet']['root_context'],'original full3 root')
        require(spec['model_hash']==mp['expected_model_hash'],'model identity')
        require(np.asarray(slot['q']).shape==(192,192) and np.array_equal(slot['q'],np.eye(192)),'identity192 Q, no fitted metric')
        # Physical Q schema deliberately uses spaced JSON without LF; runtime uses compact+LF.
        require(spec['Q_sha256']==bhash(json.dumps(slot['q'],sort_keys=True,allow_nan=False).encode()),'physical-schema Q digest codec')
        same(spec['native_goal'],pp['source_roots'][root]['goal'],'native goal including velocity')
        require(spec['actual_returned']==actual['own_returned'] and spec['physical_status']==actual['status'],'actual stop')
        for k in ('task_id','task_sha256','selection_digest','model_selection_digest','source','native_metrics','response',
                  'causal_raw10_bits','physical_status','actual_returned','missing_reason','goal192','Q_sha256','model_hash','root_context'):
            same(spec[k],strict[k],'strict row source binding: '+k)
        model=None
        if plan is None:
            require(pp['model_witnesses'][pi] is None and spec['missing_reason']=='null_runtime_export','null export remains null')
            require(all(spec[k] is None for k in ('selection_digest','model_selection_digest','full15','raw_float32_bits',
                'input_context','output_context','predicted192','model_Q','planned_raw10','causal_raw10','causal_raw10_bits')),'no invented runtime witness')
        else:
            chosen=plan['planner_plan'];witness=pp['model_witnesses'][pi]
            same(chosen['contexts'],witness['contexts'],'original whole runtime export contexts')
            require(len(chosen['contexts'])==3,'all three model boundaries')
            for k,v in (('root_slot',root),('arm',arm),('task_id',slot['task_id']),('task_sha256',slot['task_sha256']),
                        ('model_hash',mp['expected_model_hash']),('root_context',slot['packet']['root_context']),
                        ('goal',slot['packet']['goal']),('warm_raw10',slot['warm_raw10'])):same(chosen[k],v,'runtime source '+k)
            same(spec['selection_digest'],plan['selection_digest'],'physical action witness')
            same(spec['model_selection_digest'],chosen['selection_digest'],'runtime action witness')
            same(spec['full15'],chosen['commands'],'runtime commands');same(spec['full15'],plan['commands'],'physical commands')
            commands=native32(spec['full15'],(15,2))
            require(commands.tobytes().hex()==spec['raw_float32_bits']==chosen['raw_float32_bits'],'all literal raw15 signed bits')
            same(spec['input_context'],chosen['root_context'] if j==0 else chosen['contexts'][j-1],'same incoming full3')
            same(spec['output_context'],chosen['contexts'][j],'same outgoing full3')
            same(spec['predicted192'],chosen['contexts'][j]['latent_history'][-1],'same-time model endpoint')
            history=np.asarray(slot['warm_raw10']+spec['full15'][:step],np.float32)[-10:]
            require(history.shape==(10,2),'chronological raw history shape')
            same(spec['planned_raw10'],history.tolist(),'same planned history including missing future endpoint')
            model=mse(native32(spec['predicted192'],(192,)),native32(spec['goal192'],(192,)))
        require(model==spec['model_Q']==strict['model_Q'],'independent float64 predicted Q')
        have=plan is not None and step<=actual['own_returned']
        require((spec['source'] is not None) is have,'same-time presence, never terminal substitution')
        base=dict(index=i,plan_index=pi,root_slot=root,arm=arm,checkpoint=step,task_id=spec['task_id'],model_Q=model,
            real_Q=None,missing=not have,missing_reason=spec['missing_reason'],latent=None,native_axes=None,native_metrics=None,decomposition=None)
        if not have:
            same(row,spec,'immutable missing row')
            require(strict['latent_missing'] is True and all(strict[k] is None for k in ('real_Q','model_minus_real_Q','prediction_vs_encoded_MSE'))
                and spec['response'] is spec['native_metrics'] is spec['causal_raw10'] is spec['causal_raw10_bits'] is None,'null not zero')
            require(spec['missing_reason']==('null_runtime_export' if plan is None else 'native_stop_before_checkpoint'),'exact missing reason')
            output.append(base);continue
        require(history.tobytes().hex()==spec['causal_raw10_bits'],'causal raw10 signed bits')
        same(spec['causal_raw10'],history.tolist(),'causal raw10 values')
        responses=[v for v in actual['responses'] if v['step']==step];cps=[v for v in actual['checkpoints'] if v['step']==step]
        require(len(responses)==len(cps)==1 and step not in actual['missing_checkpoints'],'unique same-time physical response/checkpoint')
        same(spec['response'],responses[0],'complete native response')
        same({k:spec['source'][k] for k in cps[0]},cps[0],'original PNG/state/snapshot checkpoint')
        same(spec['source']['state'],spec['response']['state'],'native/RGB same state')
        require(spec['source']['path']==str(Path(source['physical_output'])/spec['source']['png']),'original native PNG path')
        require(row['status']=='encoded' and strict['latent_missing'] is False and spec['missing_reason'] is None,'available encoded slot')
        z=load(row['latent']);require(z.shape==(192,) and z.dtype==np.float32,'native actual192 NPY')
        proj=row['modules'][-1];require(proj['resource']=='projector' and proj['completed'] is True,'complete original projector')
        projected=load(proj['output']);require(projected.shape==(1,192) and projected.dtype==np.float32
            and projected[0].tobytes()==z.tobytes(),'real latent exactly equals saved projector output')
        dec=decomposition(spec['predicted192'],z,spec['goal192'])
        require(dec['real_Q']==strict['real_Q'] and dec['model_minus_real_Q']==strict['model_minus_real_Q']
            and dec['prediction_error_MSE']==strict['prediction_vs_encoded_MSE'],'independent all float64 latent scalars')
        native,axes=native_scalars(spec['native_goal'],spec['response']['state'])
        same(native,spec['native_metrics'],'independent full native wrapper')
        same(native['native'],spec['response']['native_metrics'],'original native scalar formula')
        require(spec['response']['reward']==native['native7_reward']
            and spec['response']['native_success'] is spec['response']['terminated'] is native['native_success']
            and type(spec['response']['truncated']) is bool,'original native reward/success/termination')
        base.update(real_Q=dec['real_Q'],latent=deepcopy(row['latent']),native_axes=axes,native_metrics=native,decomposition=dec);output.append(base)
    missing=[x['index'] for x in output if x['missing']];present=18-len(missing)
    same(s['missing_slots'],missing,'all missing incidences')
    expected={k:dict(attempted=present if k in ('encoder','projector') else 0,completed=present if k in ('encoder','projector') else 0)
        for k in ('encoder','projector','action_encoder','predict_API','predictor','pred_proj')}
    same(r['counts'],expected,'actual resources');same(s['counts'],expected,'strict resources')
    return output


def sign(value):
    return 'missing' if value is None else 'negative' if value<0 else 'positive' if value>0 else 'zero'


def relation(a,b):
    if a is None or b is None:return 'missing'
    if a==0 and b==0:return 'both_tie'
    if a==0:return 'left_tie'
    if b==0:return 'right_tie'
    return 'strict_same' if sign(a)==sign(b) else 'strict_reversal'


def paired(rows,strict):
    require(len(rows)==len(strict['pairs'])==18 and len(strict['groups'])==6,'all18 pair and six triple records')
    pairs=[];groups=[]
    for root in (0,1):
        for step in (5,10,15):
            sides={x['arm']:x for x in rows if x['root_slot']==root and x['checkpoint']==step}
            require(len(sides)==3 and set(sides)=={'joint','composition','grasp'},'unique three-arm same-time group')
            task=sides['joint']['task_id'];require(all(x['task_id']==task for x in sides.values()),'same paired task')
            group=dict(root_slot=root,checkpoint=step,task_id=task,indices={a:x['index'] for a,x in sides.items()},
                model_Q={a:x['model_Q'] for a,x in sides.items()},real_Q={a:x['real_Q'] for a,x in sides.items()},
                native_metrics={a:x['native_metrics'] for a,x in sides.items()},missing={a:x['missing'] for a,x in sides.items()})
            for k,v in group.items():same(strict['groups'][len(groups)][k],v,'independent triple '+k)
            groups.append(group)
            for left,right in DIRECTIONS:
                a,b=sides[left],sides[right];old=strict['pairs'][len(pairs)]
                item=dict(root_slot=root,checkpoint=step,task_id=task,direction=left+' - '+right,
                    left_arm=left,right_arm=right,left_index=a['index'],right_index=b['index'])
                for k,v in item.items():same(old[k],v,'fixed pair binding '+k)
                def delta(x,y):return None if x is None or y is None else x-y
                values={}
                for key in ('model_Q','real_Q'):
                    x=dict(left=a[key],right=b[key],delta=delta(a[key],b[key]))
                    same(old[key],x,'independent pair '+key);values[key]=dict(x,sign=sign(x['delta']),orientation='lower_is_better')
                physical={}
                for axis in AXES:
                    x=None if a['native_metrics'] is None else a['native_metrics'][axis]
                    y=None if b['native_metrics'] is None else b['native_metrics'][axis]
                    physical[axis]=dict(left=x,right=y,delta=delta(x,y),sign=sign(delta(x,y)),
                        orientation='higher_is_better' if axis=='native7_reward' else 'lower_is_better')
                    same(old['physical_metrics']['delta'][axis],delta(x,y),'independent physical pair '+axis)
                same(old['physical_metrics']['left'],a['native_metrics'],'left physical wrapper')
                same(old['physical_metrics']['right'],b['native_metrics'],'right physical wrapper')
                missing=a['missing'] or b['missing'];require(old['real_pair_missing'] is missing,'paired missing stays missing')
                identity=None
                if not missing:
                    da,db=a['decomposition'],b['decomposition']
                    cross=da['actual_goal_error_cross_prediction_error']-db['actual_goal_error_cross_prediction_error']
                    quadratic=da['prediction_error_quadratic']-db['prediction_error_quadratic']
                    gap=values['model_Q']['delta']-values['real_Q']['delta'];residual=gap-(cross+quadratic)
                    scale=sum(abs(v[k]) for v in (da,db) for k in
                        ('model_Q','real_Q','actual_goal_error_cross_prediction_error','prediction_error_quadratic'))
                    bound=da['identity_rounding_bound']+db['identity_rounding_bound']+32*np.finfo(np.float64).eps*scale
                    require(abs(residual)<=bound,'paired algebra rounding bound, never sign tolerance')
                    identity=dict(model_margin_minus_real_margin=gap,cross_difference=cross,quadratic_difference=quadratic,
                        identity_rounding_residual=residual,identity_rounding_bound=float(bound),scientific_tolerance_changed=False)
                modeldelta=values['model_Q']['delta'];realdelta=values['real_Q']['delta']
                comparisons={}
                for axis,x in physical.items():
                    # Reward has the opposite preference direction; preserve raw delta/sign and compare using loss=-reward.
                    loss=None if x['delta'] is None else (-x['delta'] if axis=='native7_reward' else x['delta'])
                    comparisons[axis]=dict(physical_loss_delta=loss,physical_loss_sign=sign(loss),
                        model_vs_physical_loss=relation(modeldelta,loss),real_vs_physical_loss=relation(realdelta,loss))
                pairs.append(dict(item,latent=values,physical=physical,real_pair_missing=missing,
                    model_vs_real=relation(modeldelta,realdelta),physical_sign_comparisons=comparisons,
                    scalar_error_identity=identity,
                    native_success=dict(left=None if a['native_axes'] is None else a['native_axes']['original_success'],
                        right=None if b['native_axes'] is None else b['native_axes']['original_success']),physical_benefit_claim=False))
    require(strict['real_pairs_available']==sum(not x['real_pair_missing'] for x in pairs),'all pair availability incidences')
    return groups,pairs


def analyze(**arguments):
    p,r,s,source,bindings=bind_sources(arguments);directory=Path(arguments['report']).parent
    rows=eighteen_rows(p,r,s,source,lambda rec:load_npy(directory,rec,r['artifact_sha256']))
    groups,pairs=paired(rows,s)
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bindings[str(f)]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'unchanged complete source/NPY/authority closure')
    labels=('missing','both_tie','left_tie','right_tie','strict_same','strict_reversal')
    return dict(schema=NAME+'.summary',arguments=arguments,rows=rows,groups=groups,pairs=pairs,bindings=bindings,
        fixed_denominator=18,plan_denominator=6,group_denominator=6,pair_denominator=18,known_roots=2,fresh_roots=0,
        missing_slots=[x['index'] for x in rows if x['missing']],real_pairs_available=sum(not x['real_pair_missing'] for x in pairs),
        model_real_sign_census={k:sum(x['model_vs_real']==k for x in pairs) for k in labels},
        independent_float64_arithmetic=True,producer_math_imported=False,new_neural=0,new_physics=0,
        outcome_selection=False,training=False,Qfit=False,P1_admission=False,P2_admission=False,P2_mechanism=False,
        official_reproduction=False,equal_FLOPs=False,physical_benefit_claim=False,
        interpretation='sealed-action post-hoc diagnostic, not preregistered method or admission; signs use exact float64 zero, no tolerance')


def write(path,value):
    p=absolute(str(path));body=raw(value)
    require(p.name.startswith(NAME+'_') and not p.exists() and not p.is_symlink() and len(body)<=16*1024**2,'new bounded own summary')
    with p.open('xb') as f:require(f.write(body)==len(body),'full summary write');f.flush();os.fsync(f.fileno())
    return dict(path=str(p),sha256=sha(p),bytes=len(body))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for k in (*ARGS,'output','summary','summary_sha'):ap.add_argument('--'+k.replace('_','-'))
    a=ap.parse_args()
    if a.command=='analyze':result=write(a.output,analyze(**{k:getattr(a,k) for k in ARGS}))
    else:
        require(sha(a.summary)==a.summary_sha,'explicit saved summary SHA');saved=read(a.summary)
        same(saved,analyze(**saved['arguments']),'whole independent scalar summary reconstruction')
        result=dict(verified=True,sha256=a.summary_sha,fixed_denominator=18,pair_denominator=18)
    print(raw(result).decode(),end='')


if __name__=='__main__':main()


