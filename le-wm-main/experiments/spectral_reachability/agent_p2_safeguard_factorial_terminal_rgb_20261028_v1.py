"""Zero-anchor eight real terminal PNGs -> frozen native192. MAIN CUDA only."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
from experiments.spectral_reachability import agent_p2_safeguard_factorial_physical_20261028_v1 as producer
from experiments.spectral_reachability import agent_p2_midfeedback_terminal_rgb_20261028_v1 as core

NAME='agent_p2_safeguard_factorial_terminal_rgb_20261028_v1'
AUTH='MAIN_ZERO8_TERMINAL_RGB_HOST_CUDA_ONLY'
ROOT=producer.ROOT
bridge,physical=core.bridge,core.physical
require,sha,raw=core.require,core.sha,core.raw
PRODUCER_SHA='b296c95ad8624c9e3a1a7ca2cd5c165759ad41068292f94f785a56f5d9da865a'
CORE_SHA='e7bcd2d9a4f0920eb4c715ddd0d095b22d4d6ba3b2b1bd6c49599836b6ab68ac'
REPORT_SHA='999f8278dbda931a2bed1ba1aa0ec202d47ec8444ad090b58f28ce88c74f4a2f'
RECEIPT_SHA='9c125a32268899ece92144d95382529aa2a2f406de36cfd88c05828d4e5ac00d'
PHYSICAL_PREPARE_SHA='7ebd434aaff49d66d65181186569f9511827c263c003d1856c573f99adafae70'


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def contract():
    c=core.contract()
    c.update(scope='zero-anchor unguarded fixed8 terminal only; no selection or dedup',
        authority=producer.NAME+'.consume',physical_report_sha256=REPORT_SHA,physical_receipt_sha256=RECEIPT_SHA,
        mapping='actual factorial32 sealed cell new_suffix_Q/context/witness, never parent graph',
        suffix_replay='previous actual runtime forward authority; no independent suffix prediction in this stage',
        full_upstream_rebuilt_here=False,physical_strict_reconstructed_here=True)
    return c


def fixed_rows(strict,output):
    require(strict['schema']==producer.NAME+'.consume' and strict['status']=='engineering_verified_science_blocked'
        and strict['physical_report_sha256']==REPORT_SHA and strict['runtime_strict_receipt_sha256']==producer.STRICT_SHA
        and strict['factorial_prepare_sha256']==producer.runtime.FACTORIAL_SHA and strict['fixed_denominator']==8
        and strict['known_roots']==2 and strict['P1_admission'] is strict['P2_admission'] is strict['physical_release'] is strict['fresh'] is False,
        'actual zero8 native strict authority, no old-schema projection')
    require(len(strict['plans'])==len(strict['rows'])==8,'all8 native plans and returned rows')
    rows=[]
    for i,(plan,actual) in enumerate(zip(strict['plans'],strict['rows'],strict=True)):
        key=(*physical.saved.producer.ORDER[i//2],physical.ARMS[i%2])
        require(plan['schema']==producer.NAME+'.zero_unguarded_physical_plan'
            and (plan['root_slot'],plan['condition'],plan['arm'])==key
            and (actual['index'],actual['root_slot'],actual['condition'],actual['arm'])==(i,*key)
            and actual['selection_digest']==plan['selection_digest'] and actual['compiler_selection_digest']==plan['compiler_selection_digest']
            and plan['factorial_cell_index']==4*i and plan['factorial_strict_receipt_sha256']==producer.STRICT_SHA,
            'original zero unguarded cell/root/condition/arm/native action identity')
        require('predicted_Q' not in plan and 'witness' not in plan
            and plan['selection_digest']==physical.saved.digest({k:v for k,v in plan.items() if k!='selection_digest'})
            and physical.bits(plan['commands'])==plan['raw_float32_bits'],'own new plan seal/rawfloat32, parent fields not new claims')
        w=plan['new_suffix_witness'];context=plan['new_suffix_context'];producer.runtime.F.core.context(context)
        require(w['schema']==producer.NAME+'.actual_mid_suffix_witness'
            and w['scope']=='actual-mid suffix5 Q only; NOT root-to-terminal imagined prediction'
            and w['raw5']==plan['commands'][5:] and w['output_context']==context and w['goal']==plan['goal']
            and w['Q']=='unchanged identity192' and w['independent_suffix_replay_performed'] is False
            and w['selector_cell_digest']==plan['compiler_selection_digest'] and w['runtime_strict_receipt_sha256']==producer.STRICT_SHA,
            'real factorial suffix endpoint witness, not old compiler prediction')
        goal=bridge.latent(plan['goal']);pred=bridge.latent(context['latent_history'][-1])
        require(bridge.mse(pred,goal)==plan['new_suffix_Q'],'unchanged original identity192 Q')
        frames=actual['rgb_encoding_inputs'];steps=[x['step'] for x in frames]
        require(len(set(steps))==len(steps) and sorted(steps)==[s for s in (0,5,10) if s not in actual['missing_checkpoints']],
            'explicit checkpoint denominator; no duplicate or invented frame')
        cp=[f for f in frames if f['step']==10];source=None
        if cp:
            source=deepcopy(cp[0])
            require(actual['own_returned']==10 and actual['full_horizon'] is True and Path(source['png']).name==source['png'],
                'true terminal10, not early terminal padding')
            require(source['condition']==plan['condition'] and source['compiler_selection_digest']==plan['compiler_selection_digest']
                and source['task_id']==plan['task_id'] and source['original_task_file']==plan['original_task_file']
                and source['original_task_sha256']==plan['original_task_sha256'] and source['original_blind_path']==plan['blind_path']
                and source['original_blind_sha256']==plan['blind_sha256'] and source['encoder_identity']==plan['encoder_identity']
                and source['encoding_performed'] is False and source['model_bound_by_prepare'] is True
                and physical.bits(source['raw_action_history10'])==source['raw_action_history_bits']==plan['raw_float32_bits'],
                'same condition task/model full10 causal raw history and original PNG')
            source['path']=str(Path(output).resolve()/source['png'])
        rows.append(dict(index=i,root_slot=key[0],condition=key[1],arm=key[2],task_id=plan['task_id'],
            selection_digest=plan['selection_digest'],compiler_selection_digest=plan['compiler_selection_digest'],
            parent_selection_digest=plan['parent_selection_digest'],factorial_cell_index=4*i,
            full10=deepcopy(plan['commands']),raw_float32_bits=plan['raw_float32_bits'],goal192=goal.tolist(),predicted192=pred.tolist(),
            new_suffix_Q=plan['new_suffix_Q'],new_suffix_context=deepcopy(context),new_suffix_witness=deepcopy(w),
            parent_predicted_Q=plan['parent_predicted_Q'],physical_row=deepcopy(actual),source=source,
            status='not_started' if source is not None else 'missing_physical_checkpoint10',latent=None))
    return rows


def build(physical_output,physical_report_sha,physical_receipt,physical_receipt_sha):
    require(physical_report_sha==REPORT_SHA and physical_receipt_sha==RECEIPT_SHA,'only actual pinned zero8 authorities')
    require(sha(producer.__file__)==PRODUCER_SHA and sha(core.__file__)==CORE_SHA,'frozen native adapter and policy-free tensor core')
    output=Path(physical_output).resolve();receipt=Path(physical_receipt).resolve()
    require(sha(receipt)==RECEIPT_SHA,'real actual zero8 strict receipt SHA')
    strict=producer.consume(output,REPORT_SHA)
    require(raw(strict)==raw(physical.read(receipt)),'entire actual zero8 physical strict reconstruction')
    report=physical.read(output/'report.json');b=report['execution_binding']
    require(b['prepare_sha256']==PHYSICAL_PREPARE_SHA and sha(b['prepared'])==PHYSICAL_PREPARE_SHA,'actual MAIN02 physical prepare, not failed MAIN01')
    pp=physical.read(b['prepared']);rows=fixed_rows(strict,output)
    runtime_output=Path(pp['arguments']['runtime_output']);runtime_report=physical.read(runtime_output/'report.json')
    require(sha(runtime_output/'report.json')==pp['runtime_report_sha256'],'actual sealed factorial runtime report')
    rp=physical.read(runtime_report['prepared']);require(sha(runtime_report['prepared'])==runtime_report['prepare_sha256'],'runtime model/normalizer preparation')
    bindings=dict(pp['bindings']);physical.merge(bindings,rp['bindings'])
    physical.merge(bindings,{str(receipt):RECEIPT_SHA,str(output/'report.json'):REPORT_SHA,b['prepared']:PHYSICAL_PREPARE_SHA,
        str(runtime_output/'report.json'):pp['runtime_report_sha256'],runtime_report['prepared']:runtime_report['prepare_sha256']})
    for f in output.iterdir():physical.merge(bindings,{str(f):sha(f)})
    for row in rows:
        if row['source'] is not None:bridge.load_pixels(row['source'])
    physical.merge(bindings,{str(f):sha(f) for f in bridge.source_files()+producer.own_files()+own_files()+[Path(core.__file__).resolve()]})
    physical.merge(bindings,bridge.rgb.MODELS);bridge.verify_bindings(bindings)
    n=sum(r['source'] is not None for r in rows)
    return dict(schema=NAME+'.prepare',arguments=dict(physical_output=str(output),physical_report_sha=REPORT_SHA,
        physical_receipt=str(receipt),physical_receipt_sha=RECEIPT_SHA),contract=contract(),rows=rows,bindings=bindings,
        shapes=bridge.rgb.config_shapes(bridge.rgb.read(bridge.rgb.POLICY/'config.json')),
        expected_calls={k:n if k in ('encoder','projector') else 0 for k in bridge.KINDS},policy=rp['policy'],
        expected_model_hash=rp['expected_model_hash'],actual_new_physical_counts=strict['counts'],
        inherited_factorial_model_compute=pp['inherited_model_compute'],guarded_reuse=pp['guarded_reuse'],
        full_upstream_rebuilt_here=False,physical_strict_reconstructed_here=True)


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(payload)<=bridge.META_CAP
        and shutil.disk_usage(path.parent).free>=bridge.FLOOR+len(payload),'exclusive bounded own metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    bridge.rgb.syncdir(path.parent);return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


def verify(path,expected):
    require(sha(path)==expected,'own actual prepare SHA');p=physical.read(path)
    require(raw(p)==raw(build(**p['arguments'])),'whole zero8 terminal preparation reconstruction');return p


class Store(core.Store):
    def __init__(self,output,budget):
        require(Path(output).name.startswith(NAME+'_'),'own zero8 terminal output only')
        bridge.rgb.Store.__init__(self,output,budget);self.index=0


process=core.process


def run(prepared,expected,output,authorization):
    require(authorization==AUTH,'MAIN explicit CUDA before I/O')
    budget=bridge.Budget();p=verify(prepared,expected);budget.check();store=Store(output,budget)
    rows=deepcopy(p['rows']);errors=[];trace=guard=None;model=None;before=after=None;source_ok=False;flags=None
    store.put('binding.json',dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected));store.put('initial_roster.json',rows)
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        flags=torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        bridge.native.frozen(model);before=bridge.native.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model')
        trace=bridge.Trace(model,torch,store,store.check,p['expected_calls'])
        guard=bridge.native.Ledger(2,lambda e:store.event(dict(kind='guard_event',event=e)),
            lambda:torch.cuda.synchronize(device),lambda:not torch.is_grad_enabled())
        transform=p0.image_transform()
        def encode(pixels,row,scope,ordinal):
            guard.image=ordinal
            with physical.saved.overlay.traced_scope(trace,scope,row):
                x=p0._image_tensor(pixels,transform=transform,device=device)
                row['preprocessed']=trace.save(x,f'r{row["index"]}_preprocessed')
                bridge.same(x.detach().cpu().numpy(),bridge.pixels.preprocess_pixels(pixels),'exact original preprocessing')
                out=model.encode({'pixels':x})['emb'];require(tuple(out.shape)==(1,1,192),'native192 only')
                return out[0,0].detach().cpu().numpy().copy()
        with trace,bridge.native.guard(model,torch,guard,errors),torch.inference_mode(),\
            torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
            process(p,rows,store,encode,store.check)
    except BaseException as e:errors.append(physical.kernel.error('encode',e))
    finally:
        if flags is not None:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=flags
        try:
            if model is not None:bridge.native.frozen(model);after=bridge.native.tensor_hash(model)
            bridge.verify_bindings(p['bindings']);require(sha(prepared)==expected,'prepare unchanged');budget.check();source_ok=True
        except BaseException as e:errors.append(physical.kernel.error('final_closure_model_limits',e))
        entries=[] if trace is None else trace.entries;guards=[] if guard is None else guard.entries;blocked=[] if guard is None else guard.blocked
        complete=(not errors and not blocked and source_ok and before==after==p['expected_model_hash'] and
            bridge.counts(entries)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()})
        seal=dict(rows=rows,modules=entries,guards=guards,blocked=blocked,errors=errors)
        h=store.put('native_seal.json',seal,final=True)
        report=dict(schema=NAME+'.report',status='zero8_terminal_encoded_science_blocked' if complete else 'partial_or_failed',
            prepared=str(Path(prepared).resolve()),prepare_sha256=expected,contract=contract(),**seal,
            counts=bridge.counts(entries),model_hash_before=before,model_hash_after=after,model_unchanged=before is not None and before==after,
            source_verified=source_ok,native_seal_sha256=h,artifact_sha256=deepcopy(store.hashes),event_count=store.index,
            elapsed_seconds=time.monotonic()-budget.started,P1_admission=False,P2_admission=False,physical_release=False)
        store.put('report.json',report,final=True)
    require(complete,'partial retained/no retries '+str(errors))
    return dict(report=str(store.path/'report.json'),report_sha256=sha(store.path/'report.json'))


def analyze(spec,row,load):
    z=load(row['latent']) if row['latent'] is not None else None
    actual=bridge.mse(z,spec['goal192']) if z is not None else None;pred=spec['new_suffix_Q']
    return dict(index=spec['index'],root_slot=spec['root_slot'],condition=spec['condition'],arm=spec['arm'],
        selection_digest=spec['selection_digest'],compiler_selection_digest=spec['compiler_selection_digest'],
        factorial_cell_index=spec['factorial_cell_index'],raw_float32_bits=spec['raw_float32_bits'],
        predicted_Q=pred,actual_Q=actual,Q_error=pred-actual if actual is not None else None,
        endpoint_MSE=bridge.mse(z,spec['predicted192']) if z is not None else None,latent_missing=z is None,
        native_metrics=spec['physical_row']['final_metrics'],physical_row=spec['physical_row'],
        goal192=spec['goal192'],actual192=z.tolist() if z is not None else None,physical_benefit_claim=False)


def validate(p,r,events,load):
    require(p['schema']==NAME+'.prepare' and r['schema']==NAME+'.report' and r['contract']==p['contract']==contract()
        and len(p['rows'])==len(r['rows'])==8 and r['errors']==r['blocked']==[],'entire honest zero8 schema/contract/denominator')
    modules=[];guards=[];want=[];ordinal=0
    for spec,row in zip(p['rows'],r['rows'],strict=True):
        require({k:v for k,v in row.items() if k not in ('status','latent','pixels','preprocessed','modules')}==
            {k:v for k,v in spec.items() if k not in ('status','latent')},'whole original zero8 plan/witness/goal/frame identity')
        if spec['source'] is None:require(row==spec,'missing not replaced');continue
        require(row['status']=='encoded','available encoded once');pix=load(row['pixels'])
        bridge.same(pix,bridge.load_pixels(spec['source']),'actual zero8 terminal PNG pixels')
        pre=load(row['preprocessed']);bridge.same(pre,bridge.pixels.preprocess_pixels(pix),'exact preprocessed input')
        z=load(row['latent']);bridge.latent(z);scope=dict(row=row['index'],step=10,phase='encode')
        mods=row['modules'];require(len(mods)==2 and [m['resource'] for m in mods]==['encoder','projector']
            and all(m['scope']==scope and m['completed'] is True and len(m['inputs'])==1 for m in mods),'full scoped encoder/projector')
        en,proj=mods
        physical.saved.overlay.visual_bridge(pre,load(en['inputs'][0]),load(en['output']),load(proj['inputs'][0]),load(proj['output']),z,[1,257,192])
        want.append(dict(kind='image_attempt',scope=scope,guard_image=ordinal))
        for mod in mods:
            g=dict(index=len(guards),image=ordinal,resource=mod['resource'],completed=True);guards.append(g)
            attempt={k:mod[k] for k in ('index','scope','resource')}|dict(completed=False)
            want += [dict(kind='module_attempt',entry=attempt),dict(kind='guard_event',event=dict(kind='module_attempt',entry=dict(g,completed=False))),
                dict(kind='guard_event',event=dict(kind='module_return',entry=g)),dict(kind='module_return',entry=mod)]
        modules.extend(mods);want.append(dict(kind='image_return',scope=scope,row=row));ordinal+=1
    require(modules==r['modules'] and [m['index'] for m in modules]==list(range(len(modules))) and r['guards']==guards
        and events==want and r['counts']==bridge.counts(modules)=={k:dict(attempted=v,completed=v) for k,v in p['expected_calls'].items()},
        'complete event/native/guard chronology/counts, unknowns never filtered')
    return [analyze(spec,row,load) for spec,row in zip(p['rows'],r['rows'],strict=True)]


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual zero8 encoder report SHA')
    r=physical.read(output/'report.json');p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='zero8_terminal_encoded_science_blocked'
        and r['model_unchanged'] is r['source_verified'] is True and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash']
        and r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False and 0<=r['elapsed_seconds']<=600,'whole frozen CUDA only')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'} and
        sum(f.stat().st_size for f in output.iterdir())<=bridge.CAP,'whole output census/cap')
    def blob(n):return bridge.rgb.flat(output,n).read_bytes()
    for n,h in r['artifact_sha256'].items():require(hashlib.sha256(blob(n)).hexdigest()==h,'every actual artifact SHA')
    require(physical.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']) and
        physical.read(output/'initial_roster.json')==p['rows'],'original zero8 execution binding/full roster')
    require(sha(output/'native_seal.json')==r['native_seal_sha256'] and physical.read(output/'native_seal.json')==
        {k:r[k] for k in ('rows','modules','guards','blocked','errors')},'native seal before analysis')
    rows=validate(p,r,[physical.read(output/f'event_{i:04}.json') for i in range(r['event_count'])],lambda e:bridge.rgb.load_array(e,blob))
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'final report unchanged')
    return dict(schema=NAME+'.consume',status='zero8_terminal_RGB_verified_NOT_science',rows=rows,report_sha256=expected,
        physical_report_sha256=REPORT_SHA,physical_receipt_sha256=RECEIPT_SHA,counts=r['counts'],fixed_denominator=8,
        known_roots=2,fresh_roots=0,total_FLOPs=None,actual_new_physical_counts=p['actual_new_physical_counts'],
        inherited_factorial_model_compute=p['inherited_factorial_model_compute'],guarded_reuse=p['guarded_reuse'],
        consume_new_neural_calls=0,consume_new_physics=0,P1_admission=False,P2_admission=False,physical_release=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','physical-output','physical-report-sha','physical-receipt','physical-receipt-sha','report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-host-cuda',action='store_true');a=q.parse_args()
    if a.command=='prepare':v=write(a.output,build(a.physical_output,a.physical_report_sha,a.physical_receipt,a.physical_receipt_sha))
    elif a.command=='verify':p=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',expected_calls=p['expected_calls'])
    elif a.command=='run':
        require(a.authorize_host_cuda,'MAIN explicit CUDA flag');v=run(a.prepared,a.prepare_sha,a.output,AUTH)
    else:
        require(a.receipt,'own saved receipt required');v=write(a.receipt,consume(a.output,a.report_sha))
    print(raw(v).decode(),end='',flush=True)


if __name__=='__main__':main()
