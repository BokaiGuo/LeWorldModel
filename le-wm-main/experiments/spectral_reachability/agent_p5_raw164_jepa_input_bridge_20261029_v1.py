"""Saved-only fixed164 LeWM-native -> JEPA input adapter; no implicit neural work.

prepare/verify are executable. load_host_cuda/host_inputs/labels_after_seal are
explicit integration APIs, NOT a claim of a completed CUDA experiment runner.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import agent_raw_raster_factor_20261028_v1 as factor
from experiments.spectral_reachability import agent_observed_raw_coordinate_physical_20261028_v1 as physical

NAME='agent_p5_raw164_jepa_input_bridge_20261029_v1'
ROOT=factor.ROOT
FLOOR=200*1024**2
META_CAP=32*1024**2
LOADER=ROOT/'experiments/spectral_reachability/p5_jepa_native_action_response_20261027_v8.py'
LOADER_SHA='fc915ddee2e53ed58366994684fe2190da5eaaeb8323ad608ffe3292534c936a'
PARENT=ROOT/'results/p5_jepa_native_action_response_20261027_v8/manifest_frozen.json'
PARENT_SHA='0588f825c29b884350e97ed0dc47d8872bb539d0ca7a7faf045190eee5f3340f'
STATS=ROOT/'external/jepa-wms/app/plan_common/datasets/pusht_dset.py'
STATS_SHA='30070c5a9f4770b84e3d6b6e5323df01d9c3405042cb988af07fcdcf43b6bf92'
require,sha=factor.require,factor.sha


def raw(value):return factor.raw(value)+b'\n'
def digest(value):return hashlib.sha256(factor.raw(value)).hexdigest()


def pairs(values):
    out={}
    for key,value in values:
        require(key not in out,'duplicate JSON key');out[key]=value
    return out


def read(path,expected=None,cap=META_CAP):
    path=Path(path).absolute()
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)) and path.stat().st_size<=cap,'bounded nonsymlink input')
    body=path.read_bytes();h=hashlib.sha256(body).hexdigest()
    require(expected is None or h==expected,'exact external SHA')
    value=json.loads(body,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    # Also rejects overflowing float literals such as 1e999.
    factor.raw(value)
    require(sha(path)==h,'input stable');return value


def write_new(path,value):
    path=Path(path).absolute();body=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and
        not any(p.is_symlink() for p in (path,*path.parents)) and len(body)<=META_CAP and
        shutil.disk_usage(path.parent).free>=FLOOR+len(body),'exclusive own metadata/cap/floor')
    with path.open('xb') as f:
        require(f.write(body)==len(body),'full write');f.flush();os.fsync(f.fileno())
    physical.paired.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path),status=value['status'])


def stats_from_source(path):
    require(sha(path)==STATS_SHA,'original JEPA stats source')
    wanted={'ACTION_MEAN','ACTION_STD','PROPRIO_MEAN','PROPRIO_STD'};out={}
    for node in ast.parse(Path(path).read_text()).body:
        if not isinstance(node,ast.Assign) or len(node.targets)!=1 or not isinstance(node.targets[0],ast.Name):continue
        key=node.targets[0].id
        if key not in wanted:continue
        call=node.value
        require(isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and
            call.func.value.id=='torch' and call.func.attr=='tensor' and len(call.args)==1 and not call.keywords,'literal original tensor stats only')
        out[key]=ast.literal_eval(call.args[0])
    require(set(out)==wanted,'all4 unchanged source statistics');return out


def array(value,shape):
    a=np.asarray(value,dtype=np.float64)
    require(a.shape==shape and np.isfinite(a).all(),'finite exact numeric shape');return a


def proprio(frame):
    state=array(frame['state'],(7,));snap=frame['snapshot']
    require(np.array_equal(state,array(snap['state'],(7,))) and
        state.tobytes().hex()==snap['state_float64_bits'],'saved native7 bit identity')
    agent=snap['bodies']['agent']
    actual=np.concatenate([array(agent['position'],(2,)),array(agent['velocity'],(2,))])
    require(actual.tobytes()==state[[0,1,5,6]].tobytes(),'actual saved pre-action agent position+velocity, no estimation')
    return actual


def normalized(prop,commands,stats):
    p=array(prop,(3,4)).astype(np.float32);u=array(commands,(15,2))
    # Original v11 dataset projection, not u/100; retain the float32 round-trip.
    dataset=(100.*u).astype(np.float32)/np.float32(100.)
    am=array(stats['ACTION_MEAN'],(2,)).astype(np.float32);sd=array(stats['ACTION_STD'],(2,)).astype(np.float32)
    pm=array(stats['PROPRIO_MEAN'],(4,)).astype(np.float32);ps=array(stats['PROPRIO_STD'],(4,)).astype(np.float32)
    require(np.all(sd>0) and np.all(ps>0),'original positive std')
    return (p-pm)/ps,((dataset-am)/sd).reshape(3,10)


def image_record(frame,directory):
    path=directory/frame['png']
    require(Path(frame['png']).name==frame['png'] and sha(path)==frame['file_sha256'],'flat source PNG exact bytes')
    return dict(path=str(path),file_sha256=frame['file_sha256'],pixel_sha256=frame['pixel_sha256'],
        phase=frame['phase'],step=frame['step'],overlay_index=frame['overlay_index'],condition=frame['condition'])


def request(row,plan,directory,stats):
    i=row['index'];condition=plan['condition'];overlay=1 if condition=='on' else 0
    require(plan['index']==i and row['plan']==plan and row['condition']==condition and row['root_slot']==plan['root_slot'] and
        row['selection_digest']==plan['selection_digest'] and row['status']=='horizon_completed' and row['own_returned']==10 and
        row['warmup_returned']==10 and row['prefix51']['complete'] is True,'entire fixed branch full chronology/authority')
    frames=row['trajectory']
    def at(phase,step):
        found=[f for f in frames if (f['phase'],f['step'],f['overlay_index'])==(phase,step,overlay)]
        require(len(found)==1,'unique actual fixed-condition frame required, no fill')
        require(found[0]['condition']==condition,'condition consistent');return found[0]
    selected=[at('warmup',5),at('plan',0),at('plan',5)]
    # These are pre-action packet frames at global5,10,15, never successor20.
    props=np.stack([proprio(f) for f in selected])
    warm=np.asarray(physical.moving.warmup_commands(),dtype=np.float32)
    own=array(plan['commands'],(10,2)).astype(np.float32)
    commands=np.concatenate([warm[5:],own]).astype(np.float64)
    np_,packets=normalized(props,commands,stats)
    return dict(index=i,root_slot=plan['root_slot'],condition=condition,selection_digest=plan['selection_digest'],
        task_id=plan['task_id'],context_global_frames=[5,10,15],successor_global_frame=20,
        images=[image_record(f,directory) for f in selected],raw_proprio=props.tolist(),
        normalized_proprio=np_.tolist(),commands=commands.tolist(),packets=packets.tolist(),
        label=image_record(at('plan',10),directory))


def protocol():
    return dict(schema=NAME+'.protocol',requests=164,pairs=factor.schedule(),conditions=['off','on'],known_roots=2,fresh=False,
        context_global_frames=[5,10,15],successor_global_frame=20,packet_raw_steps=5,
        observation='original224 RGB plus saved pre-action agent[x,y,vx,vy]; privileged vs LeWM RGB-only',
        action='original float32 raw; original JEPA pixel32=(100*u64).astype(float32)/float32(100), fixed stats',
        labels='only accessed through labels_after_seal after all164 full3 outputs durable',
        entry='load_host_cuda calls original pinned v8.load_cuda; host_inputs yields no labels',
        target='own factual successor only; no goal selection/Qfit',P5_admission=False,P1_admission=False,
        matched_control_benefit=False,cross_architecture_MSE_magnitude_comparable=False,
        neural_executed=False,physics_executed=False,training=False)


def build():
    fixed={k:factor.read(ROOT/'results'/p,h) for k,(p,h) in factor.PINS.items()}
    p,r,c,a=(fixed[k] for k in ('physical_prepare','physical_report','physical_receipt','observability'))
    factor.validate_rows(p,r,c,a);physical.validate_execution(p)
    bindings=deepcopy(c['bindings'])
    physical.mixed.merge(bindings,p['bindings'])
    for name,h in factor.PINS.values():bindings[str(ROOT/'results'/name)]=h
    directory=(ROOT/'results'/factor.PINS['physical_report'][0]).parent
    bindings[str(directory/'journal.jsonl')]=r['journal_sha256']
    parent=read(PARENT,PARENT_SHA)
    require(sha(LOADER)==LOADER_SHA and sha(STATS)==STATS_SHA,'fixed original loader/stats')
    physical.mixed.merge(bindings,parent['seals'])
    stats=stats_from_source(STATS)
    requests=[request(row,plan,directory,stats) for row,plan in zip(r['rows'],p['plans'])]
    require(len(requests)==164 and [x['index'] for x in requests]==list(range(164)) and
        [(x['root_slot'],x['condition']) for x in requests]==[factor.ORDER[i//41] for i in range(164)],'all164 fixed order, no subset')
    for file in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md'),
        Path(factor.__file__),Path(physical.__file__),LOADER,PARENT,STATS):bindings[str(file)]=sha(file)
    for file,h in bindings.items():require(sha(file)==h,'complete saved source/artifact closure unchanged')
    return dict(schema=NAME+'.prepare',status='saved_inputs_prepared_not_cuda_executed',protocol=protocol(),
        bindings=bindings,requests=requests,stats=stats,parent_manifest_sha256=PARENT_SHA,loader_source_sha256=LOADER_SHA)


def verify(path,expected):
    q=read(path,expected);require(q==build(),'exact rebuilt full prepared closure');return q


def pixels(rec):
    from PIL import Image
    require(sha(rec['path'])==rec['file_sha256'],'PNG bytes before decode')
    with Image.open(rec['path']) as image:
        require(image.mode=='RGB' and image.size==(224,224),'original RGB224, no resize/conversion');value=np.asarray(image).copy()
    require(value.dtype==np.uint8 and hashlib.sha256(value.tobytes()).hexdigest()==rec['pixel_sha256'],'source pixel SHA')
    return value


def host_inputs(q):
    """Non-neural, no target image decode; yields complete factual requests."""
    require(q['protocol']==protocol() and len(q['requests'])==164,'full fixed prepared protocol')
    for i,req in enumerate(q['requests']):
        require(req['index']==i,'fixed all164 order')
        prop,acts=normalized(req['raw_proprio'],req['commands'],q['stats'])
        require(prop.tolist()==req['normalized_proprio'] and acts.tolist()==req['packets'],'unaltered normalizer output')
        yield dict(index=i,condition=req['condition'],root_slot=req['root_slot'],
            rgb=np.stack([pixels(x) for x in req['images']]),normalized_proprio=prop,packets=acts)


def load_host_cuda(prepared,expected,authorization):
    """MAIN-only real loader entry, not a full execution/score/store engine."""
    require(authorization=='MAIN_P5_RAW164_JEPA_HOSTCUDA_ONLY' and not os.environ.get('PYTHONPATH'),'explicit MAIN host authority/no overlay')
    q=verify(prepared,expected)  # all closure checks BEFORE torch import
    import torch
    require(torch.cuda.is_available(),'HOST CUDA required; no CPU neural fallback')
    from experiments.spectral_reachability import p5_jepa_native_action_response_20261027_v8 as loader
    from torch.nn.attention import sdpa_kernel,SDPBackend
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=read(PARENT,PARENT_SHA)
    with sdpa_kernel(SDPBackend.MATH):models,strict,sources,before=loader.load_cuda(ROOT,parent)
    return dict(prepared=q,models=models,strict=strict,sources=sources,model_before=before,
        attention_context_factory=lambda:sdpa_kernel(SDPBackend.MATH))


def labels_after_seal(q,seal_path,seal_sha):
    """All predicted full3 arrays must exist before ANY successor PNG is decoded.

    This is a bridge seal checker, not a replacement for runtime module ledger,
    source/model immutability or a full execution consumer.
    """
    seal=read(seal_path,seal_sha)
    require(seal['schema']==NAME+'.prediction_seal' and seal['prepared_semantic_sha256']==digest(q) and
        len(seal['predictions'])==164,'whole164 prediction seal before labels')
    for i,rec in enumerate(seal['predictions']):
        require(rec['index']==i and Path(rec['path']).is_absolute() and sha(rec['path'])==rec['sha256'] and
            Path(rec['path']).stat().st_size<=2*1024**2,'actual ordered bounded full3 prediction artifact')
        value=np.load(rec['path'],allow_pickle=False)
        require(value.shape==(1,3,256,384) and value.dtype==np.float32 and np.isfinite(value).all(),'all original full3 finite predictions durable')
    return [dict(index=i,rgb=pixels(req['label'])) for i,req in enumerate(q['requests'])]


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',required=True)
    p=sub.add_parser('verify');p.add_argument('--prepared',required=True);p.add_argument('--prepare-sha',required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=write_new(args.output,build())
    else:
        q=verify(args.prepared,args.prepare_sha);result=dict(status='saved_input_closure_verified',requests=len(q['requests']),pairs=80,neural_calls=0,physics_calls=0)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
