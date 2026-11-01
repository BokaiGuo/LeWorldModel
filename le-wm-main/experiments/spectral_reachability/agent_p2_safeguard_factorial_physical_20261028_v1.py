"""Fixed eight zero-anchor unguarded full10 interventions. MAIN physics only.

Pinned accepted receipts are reused, not re-audited or re-executed upstream.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_p2_safeguard_factorial_cuda_20261028_v1 as runtime

NAME='agent_p2_safeguard_factorial_physical_20261028_v1'
ROOT=runtime.ROOT
v2=runtime.source.v2
physical,kernel,bridge=runtime.physical,runtime.kernel,runtime.bridge
require,sha,raw=runtime.require,runtime.sha,runtime.raw
AUTH='MAIN_ZERO_FACTORIAL_FULL10_PHYSICS_ONLY'
STRICT_SHA='8a59a189feaedf3b215b7aa9245c3b027228e2a51cfef32a47df286c664e5d89'
RUNTIME_SHA='d5fe1002c0437df7c4fb94fac509428be54cca9d63a61823f223476112748e4c'
V2_SHA='e459cdc5de507353873346d6f86bc223afed291982904c4c4c6ce9bdfb7eaeab'
CAP,FINAL,FLOOR,WALL=64*1024**2,2*1024**2,200*1024**2,900


def own_files():return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def contract():
    return dict(plans=8,executed_policy='zero_anchor_unguarded_only',guarded_policy='exact-action reference only',
        warmup=10,prefix=5,suffix=5,horizon=10,known_roots=2,caps=kernel.contract()['caps'],
        same_mid='native RGB bytes/pixels/state/snapshot before suffix entry',
        no_goal_or_state_edit=True,new_neural=0,new_AD=0,new_physics_plans=8,
        wall_seconds=WALL,output_cap_bytes=CAP,final_reserve_bytes=FINAL,disk_floor_bytes=FLOOR,
        total_FLOPs=None,equal_FLOPs=False,full_upstream_rebuilt_here=False,
        authority='pinned MAIN factorial prepare and accepted actual runtime strict receipt; complete file hashes rechecked',
        native_contract_is_compatibility_ABI_only=True,
        P1_admission=False,P2_admission=False,fresh=False,physical_release=False)


def read_bound(path,expected,bindings):
    path=Path(path).resolve();require(sha(path)==expected,'bound actual saved authority')
    physical.merge(bindings,{str(path):expected});return physical.read(path)


def zero_plans(cells,parents,branches,runtime_report_sha):
    require(len(cells)==32 and len(parents)==len(branches)==8,'all32 sealed cells and original8 parents')
    plans=[];reuse=[]
    for i,(parent,b) in enumerate(zip(parents,branches,strict=True)):
        c,g=cells[4*i:4*i+2]
        for cell,guard in ((c,False),(g,True)):
            require(cell['index']==4*i+int(guard) and cell['branch']==i and cell['anchor']=='zero' and cell['guard'] is guard
                and cell['selection_digest']==runtime.F.core.digest({k:v for k,v in cell.items() if k!='selection_digest'}),
                'original fixed zero-cell seal/order')
            require(all(cell[k]==parent[k] for k in ('root_slot','condition','arm','task_id'))
                and cell['input_context_sha256']==runtime.F.core.digest(b['observed_context'])
                and cell['goal_sha256']==runtime.F.core.digest(b['goal'])
                and cell['prefix_raw_bits']==v2.suffix_bits(b['prefix_commands'])==v2.suffix_bits(parent['commands'][:5])
                and physical.bits(cell['full10'])==cell['full10_raw_bits']==cell['prefix_raw_bits']+cell['selection']['raw_bits'],
                'same root/condition/task/context/goal/prefix and literal selected full10')
            require(v2.suffix_bits(cell['full10'][5:])==cell['selection']['raw_bits']
                and cell['selection']['model_Q']==runtime.source.cost(b,cell['new_suffix_context']),
                'selected suffix prediction and original Q')
            runtime.F.planner.advance(b,b['observed_context'],cell['full10'][5:],cell['new_suffix_context'])
        p=deepcopy(parent)
        p.update(schema=NAME+'.zero_unguarded_physical_plan',parent_schema=parent.get('schema'),
            parent_selection_digest=parent['selection_digest'],parent_compiler_selection_digest=parent['compiler_selection_digest'],
            parent_witness=p.pop('witness'),parent_predicted_Q=p.pop('predicted_Q'),
            parent_planner_report_sha256=parent['planner_report_sha256'],planner_report_sha256=runtime_report_sha,commands=deepcopy(c['full10']),
            raw_float32_bits=c['full10_raw_bits'],raw_sha256=hashlib.sha256(bytes.fromhex(c['full10_raw_bits'])).hexdigest(),
            compiler_selection_digest=c['selection_digest'],factorial_cell_index=c['index'],factorial_strict_receipt_sha256=STRICT_SHA,
            factorial_prepare_sha256=runtime.FACTORIAL_SHA,
            new_suffix_Q=c['selection']['model_Q'],new_suffix_context=deepcopy(c['new_suffix_context']),
            new_suffix_witness=dict(schema=NAME+'.actual_mid_suffix_witness',scope=c['scope'],
                input_context=deepcopy(b['observed_context']),raw5=deepcopy(c['full10'][5:]),goal=deepcopy(b['goal']),
                output_context=deepcopy(c['new_suffix_context']),Q='unchanged identity192',
                selector_cell_digest=c['selection_digest'],runtime_strict_receipt_sha256=STRICT_SHA,
                independent_suffix_replay_performed=False))
        p['selection_digest']=physical.saved.digest({k:v for k,v in p.items() if k!='selection_digest'});plans.append(p)
        if g['full10_raw_bits']==c['full10_raw_bits']:
            reuse.append(dict(branch=i,guarded_cell_index=g['index'],guarded_cell_digest=g['selection_digest'],
                full10_raw_bits=g['full10_raw_bits'],source='new_zero_unguarded_branch',source_branch=i,source_plan_digest=p['selection_digest']))
        else:
            require(g['selection']['selected_index']==0 and g['full10_raw_bits']==physical.bits(parent['commands']),
                'different guarded action must be exact original incumbent, not another outcome')
            reuse.append(dict(branch=i,guarded_cell_index=g['index'],guarded_cell_digest=g['selection_digest'],
                full10_raw_bits=g['full10_raw_bits'],source='original_incumbent_control',source_branch=i,
                source_plan_digest=parent['selection_digest'],strict_receipt_sha256=runtime.source.PHYSICAL[1]))
    return plans,reuse


def build(runtime_output,runtime_receipt):
    require(sha(runtime.__file__)==RUNTIME_SHA and sha(v2.__file__)==V2_SHA,'frozen runtime/gate sources')
    output=Path(runtime_output).resolve();bindings={}
    accepted=read_bound(runtime_receipt,STRICT_SHA,bindings)
    require(accepted['schema']==runtime.NAME+'.consume' and accepted['status']=='all32_engineering_verified_science_blocked'
        and accepted['factorial_prepare_sha256']==runtime.FACTORIAL_SHA and accepted['fixed_cell_denominator']==accepted['fixed_request_denominator']==32
        and len(accepted['cells'])==32 and accepted['P1_admission'] is accepted['P2_admission'] is accepted['physical_release'] is False,
        'accepted MAIN actual strict32 nonadmission authority')
    r=read_bound(output/'report.json',accepted['report_sha256'],bindings)
    rp=read_bound(r['prepared'],accepted['prepare_sha256'],bindings)
    require(r['prepare_sha256']==accepted['prepare_sha256'] and r['schema']==runtime.NAME+'.report'
        and r['status']=='all32_forward_and_cells_sealed_science_blocked' and r['errors']==[] and len(r['rows'])==32
        and r['model_before']==r['model_after']==rp['expected_model_hash'],'same completed actual frozen-model run')
    require({f.name for f in output.iterdir()}==set(r['artifact_sha256'])|{'report.json'},'runtime full census')
    for n,h in r['artifact_sha256'].items():read_path=bridge.rgb.flat(output,n);physical.merge(bindings,{str(read_path):h})
    require(r['selection']['file']=='selections.json' and sha(output/'selections.json')==r['selection']['sha256']
        and physical.read(output/'selections.json')==accepted['cells'],'actual sealed cells equal accepted receipt')
    require(rp['arguments']['factorial_sha']==runtime.FACTORIAL_SHA and rp['contract']==runtime.contract()
        and rp['full_upstream_rebuilt_here'] is False,'actual runtime prep authority')
    fp=read_bound(rp['arguments']['factorial_prepared'],runtime.FACTORIAL_SHA,bindings)
    material=runtime.material(fp)
    require(material['requests']==rp['requests'] and material['selector_rows']==rp['selector_rows'],'literal prepared ABI reconstructed')
    physical.merge(bindings,rp['bindings']);physical.merge(bindings,fp['bindings'])
    cr=physical.read(Path(fp['arguments']['incumbent_output'])/'report.json')
    require(sha(Path(fp['arguments']['incumbent_output'])/'report.json')==runtime.source.REPORTS[1],'original v2 compiler report')
    old=read_bound(cr['prepared'],cr['prepare_sha256'],bindings)
    pp=read_bound(old['physical_prepare'],old['physical_prepare_sha256'],bindings)
    require(old['plans']==pp['plans'] and len(old['midpoints'])==8,'original physical parents and encoded mids')
    plans,reuse=zero_plans(accepted['cells'],old['plans'],old['branches'],accepted['report_sha256'])
    controls=physical.read(fp['arguments']['incumbent_physical_receipt'])
    require(sha(fp['arguments']['incumbent_physical_receipt'])==runtime.source.PHYSICAL[1],'accepted original incumbent native authority')
    for i,row in enumerate(controls['rows']):
        require(row['index']==i and row['control']['selection_digest']==old['plans'][i]['selection_digest'],'original control exact plan reference')
    data=kernel.source_material();physical.merge(bindings,data['bindings']);physical.merge(bindings,pp['bindings'])
    physical.merge(bindings,{str(f):sha(f) for f in own_files()});bridge.verify_bindings(bindings)
    pp=deepcopy(pp);pp.update(plans=plans,bindings=bindings,planner_report_sha256=accepted['report_sha256'],
        strict_receipt=str(Path(runtime_receipt).resolve()),strict_receipt_sha256=STRICT_SHA,
        planner_output=str(output),adapter_schema=NAME,upstream_authority_reuse_not_reaudit=True)
    # The unchanged native runner requires its original schema/contract ABI.
    # Its legacy compiler-count labels are NOT new compute claims.
    physical.validate_plans(plans,data)
    return dict(schema=NAME+'.prepare',arguments=dict(runtime_output=str(output),runtime_receipt=str(Path(runtime_receipt).resolve())),
        contract=contract(),native_prepare=pp,midpoints=deepcopy(old['midpoints']),guarded_reuse=reuse,
        runtime_strict_receipt_sha256=STRICT_SHA,runtime_report_sha256=accepted['report_sha256'],factorial_prepare_sha256=runtime.FACTORIAL_SHA,
        bindings=bindings,full_upstream_rebuilt_here=False,planned_native_caps=kernel.contract()['caps'],
        inherited_model_compute=deepcopy(accepted['actual_new_modules']),inherited_model_compute_scope='factorial32 forward paid previously',
        total_FLOPs=None,P1_admission=False,P2_admission=False,physical_release=False),data


def verify(prepared,expected):
    require(sha(prepared)==expected,'actual own prepare SHA');p=physical.read(prepared);now,data=build(**p['arguments'])
    require(raw(p)==raw(now),'whole saved identity/plan/gate/closure reconstruction');return p,data


def write(path,value):
    path=Path(path).absolute();payload=raw(value)
    require(path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink() and len(payload)<=8*1024**2
        and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'exclusive bounded own metadata')
    with path.open('xb') as f:
        require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    kernel.paired.fsync_dir(path.parent)
    return dict(path=str(path),sha256=sha(path),status=value.get('status','prepared'))


class Store(v2.ReplayGateStore):
    """Original exact-mid emit gate; only prefix/caps/final reserve are adapted."""
    def __init__(self,path,midpoints,started):
        require(Path(path).name.startswith(NAME+'_'),'own fixed physical output prefix')
        kernel.Store.__init__(self,path);self.started=started;self.midpoints=midpoints;self.matched=set()
    def check(self,size=4096,final=False):
        require(final or time.monotonic()-self.started<WALL,'whole-stage900s, not native-call preemption')
        used=sum(f.stat().st_size for f in self.path.iterdir())
        require(used+size<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.path).free>=FLOOR+size,'64MiB /2MiB final /200MiB floor')
    def emit(self,kind,**fields):
        if kind!='run_finished':return super().emit(kind,**fields)
        e=dict(index=self.index,previous_sha256=self.previous,kind=kind,**deepcopy(fields));e['event_sha256']=kernel.source.digest(e)
        self.write('journal.jsonl',kernel.raw(e),append=True,final=True);self.previous=e['event_sha256'];self.index+=1


def execution_binding(prepared,expected,p):
    return dict(prepared=str(Path(prepared).resolve()),prepare_sha256=expected,planner_report_sha256=p['runtime_report_sha256'],
        strict_receipt_sha256=STRICT_SHA,factorial_prepare_sha256=runtime.FACTORIAL_SHA,adapter_schema=NAME,
        full_upstream_rebuilt_here=False,upstream_authority='accepted MAIN runtime strict receipt, not original compiler receipt')


def run(prepared,expected,output,factory,authorization):
    require(authorization==AUTH,'MAIN explicit native physics authorization before I/O/factory')
    started=time.monotonic();p,data=verify(prepared,expected)
    store=Store(output,p['midpoints'],started);store.check()
    return physical.execute(p['native_prepare'],data,store,factory,authorization=physical.AUTH,
        binding=execution_binding(prepared,expected,p),started=started)


def consume(output,expected):
    output=Path(output).resolve();require(sha(output/'report.json')==expected,'external actual physical report SHA')
    r=physical.read(output/'report.json');b=r['execution_binding'];p,data=verify(b['prepared'],b['prepare_sha256'])
    require(b==execution_binding(b['prepared'],b['prepare_sha256'],p) and sha(output/'execution_binding.json')==r['execution_binding_sha256']
        and physical.read(output/'execution_binding.json')==b,'actual native adapter execution binding')
    require(sha(output/'journal.jsonl')==r['journal_sha256'],'actual complete native journal SHA')
    events=physical.journal(output/'journal.jsonl',r['journal_chain_tail'])
    def image(f):
        from PIL import Image
        path=bridge.rgb.flat(output,f['png']);require(sha(path)==f['file_sha256'],'actual native PNG bytes')
        with Image.open(path) as im:return np.asarray(im).copy()
    strict=physical.assemble(r,events,p['native_prepare'],data,image)
    require(len(strict['rows'])==8,'retain complete8 native denominator')
    require({f.name for f in output.iterdir()}==set(strict['PNG_files'])|{'report.json','journal.jsonl','execution_binding.json'}
        and sum(f.stat().st_size for f in output.iterdir())<=CAP,'complete native output census/cap')
    for i,mid in enumerate(p['midpoints']):
        own=[e for e in events if e.get('branch')==i]
        next_steps=[e for e in own if e['kind']=='operation_attempted' and e['resource']=='public_step' and e.get('phase')=='plan' and e['step']>5]
        if next_steps:
            frames=[e for e in own if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan' and e['receipt']['step']==5
                and e['receipt']['condition']==mid['condition'] and e['receipt']['overlay_index']==(1 if mid['condition']=='on' else 0)]
            require(len(frames)==1 and all(frames[0]['receipt'][k]==mid[k] for k in ('file_sha256','pixel_sha256','state','snapshot'))
                and all(events.index(frames[0])<events.index(e) for e in next_steps),'exact encoded mid before any suffix native entry')
    bridge.verify_bindings(p['bindings']);require(sha(output/'report.json')==expected,'physical report unchanged after consume')
    return dict(schema=NAME+'.consume',status=strict['status'],physical_report_sha256=expected,
        plans=deepcopy(p['native_prepare']['plans']),rows=strict['rows'],counts=strict['counts'],fixed_denominator=8,known_roots=2,
        guarded_reuse=p['guarded_reuse'],runtime_strict_receipt_sha256=STRICT_SHA,factorial_prepare_sha256=runtime.FACTORIAL_SHA,
        full_upstream_rebuilt_here=False,native_contract_is_compatibility_ABI_only=True,
        inherited_model_compute=p['inherited_model_compute'],new_neural_calls=0,total_FLOPs=None,
        P1_admission=False,P2_admission=False,physical_release=False,fresh=False)


def main():
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('runtime-output','runtime-receipt','prepared','prepare-sha','output','physical-report-sha','receipt'):q.add_argument('--'+k)
    q.add_argument('--authorize-physics',action='store_true');a=q.parse_args()
    if a.command=='prepare':p,_=build(a.runtime_output,a.runtime_receipt);v=write(a.output,p)
    elif a.command=='verify':p,_=verify(a.prepared,a.prepare_sha);v=dict(status='verified_no_execution',plans=len(p['native_prepare']['plans']))
    elif a.command=='consume':
        require(a.receipt,'own durable strict receipt');v=write(a.receipt,consume(a.output,a.physical_report_sha))
    else:
        require(a.authorize_physics,'MAIN native physics flag')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False,render_action=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,AUTH)
        v=dict(status=r['status'],report=str(Path(a.output).resolve()/'report.json'),report_sha256=sha(Path(a.output)/'report.json'))
    print(raw(v).decode(),end='',flush=True)
    return 1 if v.get('status') in ('partial_or_failed','partial_failclosed') else 0


if __name__=='__main__':raise SystemExit(main())
