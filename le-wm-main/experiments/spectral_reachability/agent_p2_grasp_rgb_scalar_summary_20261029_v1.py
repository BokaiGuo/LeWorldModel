"""Independent saved-only six-slot GRASP scalar arithmetic. No producer imports."""
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

NAME='agent_p2_grasp_rgb_scalar_summary_20261029_v1'
RGB='agent_p2_history_grasp_rgb_20261029_v1'
PHYSICAL='agent_p2_history_grasp_physical_20261029_v2'
MODEL='agent_p2_history_joint30_grasp_cuda_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
ORDER=tuple((root,step) for root in (0,1) for step in (5,10,15))
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
        p=absolute(str(path));require(type(expected) is str and len(expected)==64 and expected!='0'*64
            and all(x in '0123456789abcdef' for x in expected),'actual lowercase SHA256')
        require(str(p) not in bindings or bindings[str(p)]==expected,'nonconflicting provenance binding')
        require(sha(p)==expected,'bound source SHA mismatch: '+str(p));bindings[str(p)]=expected;return p
    for pathkey,hashkey in (('prepared','prepare_sha'),('report','report_sha'),('strict','strict_sha')):
        bind(arguments[pathkey],arguments[hashkey])
    p,r,s=(read(arguments[k]) for k in ('prepared','report','strict'))
    require(p['schema']==RGB+'.prepare' and r['schema']==RGB+'.report' and s['schema']==RGB+'.consume'
        and r['status']=='grasp_RGB_encoded_NOT_science' and s['status']=='all6_grasp_RGB_slots_verified_NOT_science',
        'actual own RGB complete report/strict, not an old or partial schema')
    require(r['prepared']==arguments['prepared'] and r['prepare_sha256']==arguments['prepare_sha']
        and s['report_sha256']==arguments['report_sha'],'explicit report/prepare/strict join')
    same(p['contract'],r['contract'],'same fixed RGB contract')
    require(r['errors']==r['blocked']==[] and r['source_verified'] is r['model_unchanged'] is True
        and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'],'frozen actual model/clean encoding')
    for obj in (r,s):
        require(all(obj[k] is False for k in ('P1_admission','P2_admission','P2_mechanism','physical_release')),'no scientific admission')
    for path,h in p['bindings'].items():bind(path,h)
    for spec in p['rows']:
        if spec['source'] is not None:bind(spec['source']['path'],spec['source']['file_sha256'])
    directory=absolute(arguments['report']).parent
    files=list(directory.iterdir())
    require({f.name for f in files}==set(r['artifact_sha256'])|{'report.json'}
        and sum(f.stat().st_size for f in files)<=ARTIFACT_CAP,'complete bounded actual RGB artifact census')
    for name,h in r['artifact_sha256'].items():bind(flat(directory,name),h)
    same(read(directory/'binding.json'),dict(prepared=arguments['prepared'],prepare_sha256=arguments['prepare_sha']),'actual RGB execution binding')
    same(read(directory/'initial_roster.json'),p['rows'],'fixed initial six-slot roster')
    require(sha(directory/'native_seal.json')==r['native_seal_sha256'],'native seal SHA')
    same(read(directory/'native_seal.json'),{k:r[k] for k in ('rows','modules','guards','blocked','errors')},'complete native seal')
    a=p['arguments'];physical_report=bind(Path(a['physical_output'])/'report.json',a['physical_report_sha'])
    physical_strict=read(bind(a['physical_receipt'],a['physical_receipt_sha']));pr=read(physical_report);b=pr['execution_binding']
    pp=read(bind(b['prepared'],b['prepare_sha256']))
    require(pp['schema']==PHYSICAL+'.prepare' and physical_strict['schema']==PHYSICAL+'.consume'
        and physical_strict['physical_report_sha256']==a['physical_report_sha']==s['physical_report_sha256']==p['physical_report_sha256']
        and a['physical_receipt_sha']==s['physical_receipt_sha256']==p['physical_receipt_sha256'],'actual own physical authority')
    same(pr['rows'],physical_strict['rows'],'whole physical report/strict rows')
    mrpath=Path(pp['arguments']['planner_output'])/'report.json'
    mr=read(bind(mrpath,pp['arguments']['planner_report_sha']))
    require(mr['schema']==MODEL+'.report' and mr['status']=='normalized_model_variant_complete_not_admitted','original complete planner')
    mp=read(bind(mr['prepared'],mr['prepare_sha256']));results=[]
    require(len(mr['rows'])==2,'two original root results')
    for i,row in enumerate(mr['rows']):
        name=f'root{i}.json';require(row['root_slot']==i and row['result_file']==name,'original fixed model root files')
        results.append(read(bind(mrpath.parent/name,mr['artifact_sha256'][name])))
    return p,r,s,dict(physical_prepare=pp,physical_strict=physical_strict,model_prepare=mp,root_results=results,
        physical_output=a['physical_output']),bindings


def six_rows(p,r,s,source,load):
    require(len(p['rows'])==len(r['rows'])==len(s['rows'])==6
        and s['fixed_denominator']==s['model_real_comparison_denominator']==6 and s['plan_denominator']==2,
        'all six slots, no success/time selection')
    require(len(source['physical_prepare']['plans'])==len(source['physical_strict']['rows'])==len(source['root_results'])==2,'both original roots')
    output=[];present=0
    for i,(root,step) in enumerate(ORDER):
        spec,row,strict=p['rows'][i],r['rows'][i],s['rows'][i]
        for v in (spec,row,strict):
            require(type(v['index']) is int and type(v['root_slot']) is int and type(v['checkpoint']) is int
                and (v['index'],v['root_slot'],v['checkpoint'])==(i,root,step) and v['condition']=='off'
                and v['plan_index']==root,'fixed exact root/time/order')
        same({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')},
            {k:v for k,v in spec.items() if k not in ('status','latent')},'unchanged encoded task/goal/prediction/raw/source')
        plan=source['physical_prepare']['plans'][root];actual=source['physical_strict']['rows'][root]
        slot=source['model_prepare']['slots'][root];result=source['root_results'][root];j=step//5-1
        require(actual['index']==actual['root_slot']==root and actual['condition']=='off'
            and result['root_slot']==root and len(result['whole'])==3,'source row/root result identity')
        same(actual['plan'],plan,'whole original physical execution plan')
        require(result['task_id']==plan['task_id']==slot['task_id'] and spec['arm']==strict['arm']==plan['arm'],
            'same original root result task/arm')
        require(plan['planner_plan']['model_sha256']==source['model_prepare']['expected_model_hash'],
            'original source model tensor identity')
        require(np.asarray(slot['model_input']['q']).shape==(192,192)
            and np.array_equal(slot['model_input']['q'],np.eye(192)),'original identity192 metric, not fitted Q')
        same(spec['task_id'],plan['task_id'],'same original task')
        same(spec['selection_digest'],plan['selection_digest'],'same literal action witness')
        same(spec['model_selection_digest'],plan['planner_plan']['selection_digest'],'same model selection witness')
        same(spec['task_sha256'],slot['task_sha256'],'same original task bytes')
        same(spec['goal192'],slot['model_input']['goal'],'original goal, no new encoder')
        same(spec['native_goal'],plan['native_goal'],'same original native goal including velocity')
        same(spec['full15'],result['commands'],'same fixed model/native commands')
        same(spec['full15'],plan['commands'],'same physical raw15 witness')
        same(spec['output_context'],result['whole'][j],'prediction from original full whole edge')
        commands=native32(spec['full15'],(15,2))
        require(commands.tobytes().hex()==spec['raw_float32_bits'],'exact literal raw15 bits')
        history=np.asarray(slot['warm_raw10']+spec['full15'][:step],np.float32)[-10:]
        require(history.shape==(10,2) and history.tobytes().hex()==spec['causal_raw10_bits'],'exact same-time incoming raw10')
        same(spec['predicted192'],result['whole'][j]['latent_history'][-1],'original predicted endpoint')
        pred=native32(spec['predicted192'],(192,));goal=native32(spec['goal192'],(192,))
        model=mse(pred,goal);require(model==spec['model_Q']==strict['model_Q'],'independent model-goal MSE')
        require(spec['actual_returned']==actual['own_returned'] and spec['physical_status']==actual['status'],'actual stop/time identity')
        keyfields=('plan_index','task_id','task_sha256','selection_digest','model_selection_digest','source','native_metrics','response',
                   'causal_raw10_bits','physical_status','actual_returned')
        for k in keyfields:same(spec[k],strict[k],'strict scalar row binding: '+k)
        have=step<=actual['own_returned'];require((spec['source'] is not None) is have,'same-step presence, no endpoint substitution')
        if not have:
            same(row,spec,'missing report row remains immutable null')
            require(strict['latent_missing'] is True and all(strict[k] is None for k in ('real_Q','model_minus_real_Q','prediction_vs_encoded_MSE'))
                and spec['response'] is spec['native_metrics'] is None,'missing is not a zero response')
            output.append(dict(index=i,root_slot=root,checkpoint=step,task_id=spec['task_id'],model_Q=model,
                missing=True,latent=None,native_axes=None,decomposition=None));continue
        present+=1
        responses=[v for v in actual['responses'] if v['step']==step];cps=[v for v in actual['checkpoints'] if v['step']==step]
        require(len(responses)==len(cps)==1 and step not in actual['missing_checkpoints'],'unique actual same-time native response/checkpoint')
        same(spec['response'],responses[0],'actual full native response')
        same({k:spec['source'][k] for k in cps[0]},cps[0],'actual PNG/state/snapshot checkpoint')
        same(spec['source']['state'],spec['response']['state'],'same state for native/RGB')
        require(spec['source']['path']==str(Path(source['physical_output'])/spec['source']['png']),
            'actual original physical PNG path, not an alternate file')
        require(row['status']=='encoded' and strict['latent_missing'] is False,'actual available encoded slot')
        z=load(row['latent']);require(z.shape==(192,) and z.dtype==np.float32,'real native192 NPY')
        proj=row['modules'][-1];require(proj['resource']=='projector' and proj['completed'] is True,'real complete projector output')
        projected=load(proj['output']);require(projected.shape==(1,192) and projected.dtype==np.float32
            and projected[0].tobytes()==z.tobytes(),'real192 NPY exactly equals saved projected output')
        dec=decomposition(pred,z,goal)
        require(dec['real_Q']==strict['real_Q'] and dec['model_minus_real_Q']==strict['model_minus_real_Q']
            and dec['prediction_error_MSE']==strict['prediction_vs_encoded_MSE'],'independent actual/goal/pred float64 scalar checks')
        native,axes=native_scalars(plan['native_goal'],responses[0]['state'])
        same(native,spec['native_metrics'],'independent complete native scalar formula')
        same(native['native'],responses[0]['native_metrics'],'independent native original success/angle/object fields')
        require(responses[0]['reward']==native['native7_reward'] and responses[0]['native_success'] is responses[0]['terminated'] is native['native_success']
            and type(responses[0]['truncated']) is bool,'original full7 reward/native termination')
        output.append(dict(index=i,root_slot=root,checkpoint=step,task_id=spec['task_id'],model_Q=model,
            missing=False,latent=deepcopy(row['latent']),native_axes=axes,decomposition=dec))
    require(s['missing_slots']==[v['index'] for v in output if v['missing']] and s['real_comparisons_available']==present,'all six strict missing incidences')
    expected={k:dict(attempted=present if k in ('encoder','projector') else 0,completed=present if k in ('encoder','projector') else 0)
        for k in ('encoder','projector','action_encoder','predict_API','predictor','pred_proj')}
    same(r['counts'],expected,'actual six-slot resource counts');same(s['counts'],expected,'strict counts unchanged')
    return output


def analyze(**arguments):
    p,r,s,source,bindings=bind_sources(arguments)
    directory=Path(arguments['report']).parent
    rows=six_rows(p,r,s,source,lambda rec:load_npy(directory,rec,r['artifact_sha256']))
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f)]=sha(f)
    for path,h in bindings.items():require(sha(path)==h,'unchanged full source/NPY/authority closure')
    return dict(schema=NAME+'.summary',arguments=arguments,rows=rows,bindings=bindings,
        fixed_denominator=6,known_roots=2,fresh_roots=0,missing_slots=[v['index'] for v in rows if v['missing']],
        independent_float64_arithmetic=True,producer_math_imported=False,new_neural=0,new_physics=0,
        outcome_selection=False,training=False,Qfit=False,P1_admission=False,P2_admission=False,
        official_reproduction=False,equal_FLOPs=False,physical_benefit_claim=False)


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
        same(saved,analyze(**saved['arguments']),'whole scalar summary independent reconstruction')
        result=dict(verified=True,sha256=a.summary_sha,fixed_denominator=6)
    print(raw(result).decode(),end='')


if __name__=='__main__':main()
