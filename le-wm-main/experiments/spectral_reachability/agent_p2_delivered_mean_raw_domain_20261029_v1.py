"""Saved-only P2 engineering inverse/domain/bit checks; NEVER raw release."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from experiments.spectral_reachability import agent_p2_target_controller_cuda_20261029_v1 as controller

TAG='agent_p2_delivered_mean_raw_domain_20261029_v1'
ROOT=Path(__file__).resolve().parents[2]
need=controller.need
sha=controller.sha
read=controller.read
encode=controller.encode
CAP=16*1024**2
PINS={
 'prepare':('results/agent_p2_target_controller_cuda_20261029_v1_prepare_MAIN01.json','1a021116784dfaa5b4ce976f694fbfde377e6b282ce3553cb57d49a5e033c0aa'),
 'report':('results/agent_p2_target_controller_cuda_20261029_v1_cuda_MAIN01/report.json','b1358eb3cec42cdddb30b0d138c2bc2e68773f195aa985977bf9c090527382cb'),
 'receipt':('results/agent_p2_target_controller_cuda_20261029_v1_consume_MAIN01.json','34e0491ca05a1b381db3541997c6215c44f9927761d6123a559e604160f0be4a'),
 'physical_prepare':('results/agent_p2_three_program_physical_20261029_v1_prepare_MAIN01.json','a4fb18b5aedd53e1820717f827d4703774b8c41dd08b0fd3974cca58593ade93'),
}
SOURCE_PINS={
 'experiments/run_pusht_counterfactual.py':'0a4eb4592951139e596b35becede17d175df7b48a2fde16679ab0aac7c4dce34',
 'experiments/spectral_reachability/pusht_d2p_packet_contract.py':'eda631348e929b45591807dac0adcbb72d10d84b7d48a7a558a492eab9c04aef',
 'experiments/spectral_reachability/pusht_full_packet_compiler_20261027.py':'f359e7b21d1970f95f065dbee206d2dc9359c2eaff7ddbcd7611e07686c6cf2a',
 'experiments/spectral_reachability/preflight_pusht_d2p_p0.py':'745a13d47934f1862a2b43a62b20bc07fc7308075b7b6fea26c3236d9ffa4c98',
 'experiments/spectral_reachability/p3_goal_patch_consistent_compiler_20261027_v2.py':'1dfa678a2a89856d557d50fe79a962235efc25d2e36b6114bf98e0a98d133f61',
 'experiments/spectral_reachability/agent_p2_three_program_physical_20261029_v1.py':'b0e7ae15d4810963a64916b1da16e34ae1017b189156a93b635d73520e2666e5',
 '.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py':'c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2',
}


def bits(a):return np.ascontiguousarray(a,dtype=np.float32).view(np.uint32)
def same(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def bithex(u):return f'{int(u):08x}'
def value(x):return float(x) if np.isfinite(x) else None


def ordered(u):
    """Finite IEEE monotone rank; both zeros share rank, signs remain separately."""
    u=int(u);return 0x80000000-(u&0x7fffffff) if u&0x80000000 else 0x80000000+u


def literal_stats(text):
    found={}
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'):
            need(isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='np.asarray' and
                 any(k.arg=='dtype' and ast.unparse(k.value)=='np.float32' for k in n.value.keywords),'original literal float32 stats')
            found[n.targets[0].id]=np.asarray(ast.literal_eval(n.value.args[0]),np.float32)
    need(set(found)=={'PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'},'both original stats')
    mean,std=found['PUBLIC_ACTION_MEAN'],found['PUBLIC_ACTION_STD']
    need(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'finite XY stats')
    return mean,std


def native_contract(text):
    tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PushT')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    assignments={ast.unparse(n.targets[0]):n.value for n in ast.walk(init) if isinstance(n,ast.Assign) and len(n.targets)==1}
    call=assignments['self.action_space']
    need(isinstance(call,ast.Call) and ast.unparse(call.func)=='spaces.Box','original native Box')
    kw={k.arg:k.value for k in call.keywords}
    need(set(kw)=={'low','high','shape','dtype'} and ast.literal_eval(kw['low'])==-1. and
        ast.literal_eval(kw['high'])==1. and ast.literal_eval(kw['shape'])==(2,) and ast.unparse(kw['dtype'])=='np.float32','native float32[-1,1]2')
    defaults={arg.arg:v for arg,v in zip(init.args.args[-len(init.args.defaults):],init.args.defaults,strict=True)}
    need(ast.literal_eval(defaults['relative']) is True and ast.literal_eval(assignments['self.action_scale'])==100 and
         ast.unparse(assignments['self.relative'])=='relative','native relative/action_scale source')
    return dict(low=[-1.,-1.],high=[1.,1.],shape=[2],dtype='<f4',relative=True,action_scale=100,
                source_only_not_env_instantiated=True,extra_div100=False)


def forward(raw,mean,std):
    """Pure NumPy mirror of the bound packet equation, separate float32 ops."""
    centered=np.subtract(raw,mean,dtype=np.float32)
    restored=np.divide(centered,std,dtype=np.float32)
    return centered,restored


def check_mean(mean_actions,mean,std):
    need(type(mean_actions) is np.ndarray and mean_actions.dtype==np.float32 and mean_actions.shape==(1,3,10)
         and np.isfinite(mean_actions).all(),'actual finite float32 mean1x3x10')
    need(mean.dtype==std.dtype==np.float32 and mean.shape==std.shape==(2,) and
         np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(),'literal f32 XY normalizer')
    normalized=mean_actions.reshape(15,2).copy()
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        product=np.multiply(normalized,std,dtype=np.float32)
        raw=np.add(product,mean,dtype=np.float32)
        centered,restored=forward(raw,mean,std)
    nb,rb,tb=bits(normalized),bits(raw),bits(restored)
    coords=[];outside=[];different=[];zeros=[];zero_occurrences=[]
    for step in range(15):
        for axis in range(2):
            x,y,z=normalized[step,axis],raw[step,axis],restored[step,axis]
            exact=bool(nb[step,axis]==tb[step,axis]);finite=bool(np.isfinite(y))
            in_domain=finite and bool(-1.<=y<=1.)
            zero_flip=bool(x==0 and z==0 and np.signbit(x)!=np.signbit(z))
            rec=dict(flat_index=2*step+axis,raw_step=step+1,packet=step//5,within_packet=step%5,axis=axis,
                normalized=value(x),normalized_bits=bithex(nb[step,axis]),raw=value(y),raw_bits=bithex(rb[step,axis]),
                roundtrip=value(z),roundtrip_bits=bithex(tb[step,axis]),raw_finite=finite,in_native_domain=in_domain,
                bitexact=exact,ulp_distance=abs(ordered(nb[step,axis])-ordered(tb[step,axis])) if np.isfinite(z) else None,
                signed_zero_flip=zero_flip,zero_signs={k:('negative' if np.signbit(v[step,axis]) else 'positive')
                    if v[step,axis]==0 else None for k,v in dict(normalized=normalized,product=product,raw=raw,centered=centered,roundtrip=restored).items()})
            coords.append(rec)
            if not in_domain:outside.append(rec)
            if not exact:different.append(rec)
            if zero_flip:zeros.append(rec)
            if any(v is not None for v in rec['zero_signs'].values()):zero_occurrences.append(rec)
    blocked=bool(outside);needs=bool(different)
    return dict(status='blocked_native_domain' if blocked else 'needs_paid_actual_token_replay' if needs else 'bitexact_in_domain_checks_only_not_released',
        normalized_shape=[1,3,10],raw_shape=[15,2],packet_order='H0xy0..4,H1xy0..4,H2xy0..4',
        mean=mean.tolist(),std=std.tolist(),arithmetic='multiply_f32;add_f32;subtract_f32;divide_f32; no FMA/search/retry',
        array_bits={k:np.ascontiguousarray(v).tobytes().hex() for k,v in dict(normalized=normalized,product=product,raw=raw,centered=centered,roundtrip=restored).items()},
        raw_candidate=[[value(x) for x in r] for r in raw],coordinates=coords,out_of_domain=outside,
        roundtrip_differences=different,signed_zero_differences=zeros,signed_zero_occurrences=zero_occurrences,
        ULP_definition='finite monotone IEEE rank distance; +/-0 same rank0 distance but distinct bits/sign inventory',
        domain_ok=not blocked,roundtrip_bitexact=not needs,
        needs_paid_actual_token_replay=needs,blocked_native_domain=blocked,
        original_prediction_cost_reuse_authorized=False,CUDA_token_identity_certified=False,raw_deployment_authorized=False)


def authorities():
    return {k:dict(path=str(ROOT/p),sha256=h) for k,(p,h) in PINS.items()}


def source_authority(cm,physical,sources):
    for rel,h in SOURCE_PINS.items():
        p=str(ROOT/rel);need(sha(p)==h,'frozen normalizer/Jacobian/native source');sources[p]=h
        need(cm['sources'].get(p)==h or physical['bindings'].get(p)==h,'original controller/Jacobian or physical source authority')
    env=str(ROOT/'.venv/lib/python3.12/site-packages/stable_worldmodel/envs/pusht/env.py')
    need(physical['bindings'][env]==sources[env],'original physical native binding')
    mean,std=literal_stats((ROOT/'experiments/run_pusht_counterfactual.py').read_text())
    contract=native_contract(Path(env).read_text())
    return mean,std,dict(scope='MAIN approved P2 known-root engineering convention, NOT Hi original scaler',
        mean=mean.tolist(),std=std.tolist(),mean_bits=mean.tobytes().hex(),std_bits=std.tobytes().hex(),
        source_sha256=SOURCE_PINS['experiments/run_pusht_counterfactual.py'],native=contract,
        original_Hi_scaler_certified=False,training_stats_provenance_verified=False,
        Jacobian_mapping='pusht_full_packet_compiler_20261027.make_transition -> normalize_live_packet(raw,mean,std)',
        forward_equation='(raw.reshape(...,10)-tile(mean,5))/tile(std,5); no extra /100',
        inverse_equation='float32(float32(normalized*std)+mean)',
        CPU_mirror_only=True,CUDA_roundtrip_unchecked=True)


def build():
    refs=authorities();docs={k:read(v['path'],v['sha256']) for k,v in refs.items()}
    cm,r,c,physical=(docs[k] for k in ('prepare','report','receipt','physical_prepare'))
    need(r['manifest_sha256']==refs['prepare']['sha256'] and Path(r['manifest']).resolve()==Path(refs['prepare']['path']) and
        c['manifest_sha256']==refs['prepare']['sha256'] and c['report_sha256']==refs['report']['sha256'],'complete controller authority triple')
    need(r['completed'] is True and r['errors']==[] and c['denominator']==dict(total=6,returned=4,missing=2) and
        c['status']=='all4_target_controller_saved_verified_2_arbitrary_missing' and c['predict_API']==48 and c['predictor_samples']==216,'complete actual four controller results')
    need(r['protocol']==cm['protocol']==controller.protocol() and r['model_before']==r['model_after']==cm['expected_state'],'fixed original model/spec')
    sources={v['path']:v['sha256'] for v in refs.values()}
    helpers=controller.closure()
    need(all(cm['sources'].get(p)==h for p,h in helpers.items()),'exact already executed helpers')
    sources.update(helpers)
    for folder,name in (('experiments/spectral_reachability',TAG+'.py'),('tests','test_'+TAG+'.py'),('innerdocs',TAG+'.md')):
        p=ROOT/folder/name;sources[str(p)]=sha(p)
    mean,std,normalizer=source_authority(cm,physical,sources)
    pa=cm['delivery_artifact']['authorities']['program_receipt'];program=read(pa['path'],pa['sha256']);sources[pa['path']]=pa['sha256']
    tasks=[]
    for i in (1,3):
        p=program['plans'][i];root=i//2;source=cm['delivery_artifact']['rows'][3*root]
        need(p['normalization']==dict(mean=mean.tolist(),std=std.tolist()) and p['task_id']==source['task_id'] and
            p['task_sha256']==source['task_sha256'] and p['blind_sha256']==source['blind_sha256'],'original task normalizer/source identity')
        old=physical['plans'][i]
        need(old['planner_plan']==p,'original physical mapping binds entire same planner plan')
        need(sha(p['task_file'])==p['task_sha256'],'original frozen task bytes, no decode/substitution');sources[p['task_file']]=p['task_sha256']
        warm=np.asarray(p['root_warm_raw10'],np.float32);need(warm.shape==(10,2) and np.isfinite(warm).all() and (np.abs(warm)<=1).all(),'original raw warm10')
        _,history=forward(warm,mean,std)
        need(same(history.reshape(1,2,10),np.asarray(source['controller_payload']['a'],np.float32)) and
            same(history,np.asarray(p['root_context']['action_history'],np.float32).reshape(10,2)),'exact original task between-action normalization')
        tasks.append(dict(root=root,task_file=p['task_file'],task_sha256=p['task_sha256'],task_id=p['task_id'],
            normalization=p['normalization'],warm_history_roundtrip_source_SHA=hashlib.sha256(warm.tobytes()).hexdigest()))
    rows=[];folder=Path(refs['report']['path']).parent
    need(len(r['rows'])==6 and [x['slot'] for x in c['local_checks']]==list(controller.ACTIVE),'complete original slot/check denominator')
    for i,row in enumerate(r['rows']):
        need((row['slot'],row['root'],row['arm'])==(i,i//3,controller.ARMS[i%3]),'fixed slot order')
        if i not in controller.ACTIVE:
            need(row['status']=='missing' and row['debits']==dict(known=True,predict_API=0,predictor_samples=0),'arbitrary still missing')
            rows.append(dict(slot=i,root=row['root'],arm=row['arm'],status='missing',checks=None));continue
        need(row['status']=='returned','all four selected means returned')
        cc=next(x for x in c['local_checks'] if x['slot']==i)
        need(cc==dict(slot=i,root=row['root'],arm=row['arm'],local_check=row['local_check']),'entire imported local check')
        td=row['trace'];need(td==r['files'][td['file']],'actual trace inventory')
        cr=read(folder/td['file'],td['file_sha256']);sources[str(folder/td['file'])]=td['file_sha256']
        need(cr['completed'] is True and cr['errors']==[] and cr['spec']==cm['protocol']['spec'] and
            cr['backend']=='host_cuda_installed_CEM' and cr['host_tensor_seam_checked'] is True,'actual completed core trace')
        def get(key):
            d=row['arrays'][key];need({k:d[k] for k in ('file','file_bytes','file_sha256','root')}==r['files'][d['file']],'array exact inventory')
            a=controller.load_array(folder,d);sources[str(folder/d['file'])]=d['file_sha256'];return a
        key=cr['selected']['mean'];actual=get(key)
        need(same(actual,get(cr['native_return']['actions'])) and same(actual,get(cr['native_return']['mean'])),'actual returned mean, not elite/reselection')
        checks=check_mean(actual,mean,std)
        rows.append(dict(slot=i,root=row['root'],arm=row['arm'],status=checks['status'],checks=checks,
            mean_source=dict(trace=td,key=key,array=row['arrays'][key],normalized_value_sha256=hashlib.sha256(actual.tobytes()).hexdigest())))
    need(all(sha(p)==h for p,h in sources.items()),'all metadata/source/selected arrays unchanged')
    return dict(schema=TAG+'.checks',authorities=refs,imported_complete_receipt=c,sources=sources,normalizer=normalizer,tasks=tasks,rows=rows,
        denominator=dict(total=6,checked=4,missing=2),blocked_slot_ids=[x['slot'] for x in rows if x['checks'] and x['checks']['blocked_native_domain']],
        needs_paid_actual_token_replay_slots=[x['slot'] for x in rows if x['checks'] and x['checks']['needs_paid_actual_token_replay']],
        raw_deployment_authorized=False,original_prediction_cost_reuse_authorized=False,scientific_admission=False,
        original_Hi_scaler_certified=False,prior_physical_time=None,time_matched=False,
        actual_new_calls=dict(NN=0,physics=0,full_strict=0),evidence_scope='fixed complete receipts imported; selected saved means and NumPy f32 arithmetic only')


def output_path(p):
    p=Path(p);need(p.parent.resolve()==ROOT/'results' and p.name.startswith(TAG+'_') and not p.is_symlink(),'exclusive own results child');return p


def prepare(output):
    v=build();b=encode(v);p=output_path(output)
    need(len(b)<=CAP and shutil.disk_usage(p.parent).free>=200*1024**2+len(b),'cap16MiB/floor200MiB')
    controller.tracker.durable(p,b);return dict(path=str(p.resolve()),sha256=sha(p))


def verify(path,h):
    need(read(path,h)==build(),'entire inverse/domain report rebuilt, no tolerance')
    return dict(verified=True,raw_deployment_authorized=False,full_strict_rebuilt=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('authorities','prepare','verify'))
    p.add_argument('--output',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--manifest-sha256');a=p.parse_args(argv)
    out=authorities() if a.command=='authorities' else prepare(a.output) if a.command=='prepare' else verify(a.manifest,a.manifest_sha256)
    print(json.dumps(out,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
