"""Four actual feedback terminal10 RGB encodes; frozen old prefix NPY reused.

MAIN-only CUDA, saved-only prepare. New own compiler and physical schemas;
no physics/rollout/training/Q fit, retry, root selection or prefix re-encoding.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import moving_history_observed_suffix_cuda_20261027_v1 as compiler
from experiments.spectral_reachability import moving_history_observed_suffix_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_constructive_moving_history_rgb_cuda_20261027_v1 as parent

NAME='moving_history_observed_suffix_rgb_cuda_20261027_v1'
ROOT=physical.ROOT
rgb,enc,moving=parent.rgb,parent.enc,parent.moving
require,sha,read=rgb.require,rgb.sha,rgb.read
CAP,FLOOR,FINAL=8*1024**2,200*1024**2,256*1024
PHYSICAL_SHA='07215161b9ea620efe83ff7ca7ae2838d2f27b40ca16f5cc3aa969626813b2ce'
COMPILER_SHA='3e72cef9375c5fa5b010593e2285e8eeb8d6b51b71310b70663b7f7406a7eb56'
PARENT_SHA='ed9b80bebb212434d60ddd55d3a3cba01e5cc55306170a0ea6f9a167bb156469'
REPORT=ROOT/'results/moving_history_observed_suffix_physical_20261027_v1_main01/report.json'
REPORT_SHA='3376eda7d0a88b31e8fb56a5f3b2cc19ef4854ef5b12f35ae196da0744c71f0e'
PHYSICAL_PREPARE_SHA='5bcd47b14a02a07162640349b26d315e11b7ee7b016628556eef42162f760972'
COMPILER_REPORT=ROOT/'results/moving_history_observed_suffix_cuda_20261027_v1_cuda_main01/report.json'
COMPILER_REPORT_SHA='79f07b28722991ce40a01cdfa2c40fa0e51f149803db00e1617d1ccecb0966fc'
ARMS=list(parent.ARMS)


def assemble(report,events,prepared,branches,selected,results,tasks,prefixes):
    """Pure new-schema terminal join. Physical state is never a selection input."""
    census=[(s,a) for s in (0,1) for a in ARMS]
    require(report['schema']==physical.NAME+'.report' and report['status']=='moving_history_observed_suffix_physical_recorded' and
        report['errors']==[] and report['P1_admission'] is False and report['P4_admission'] is False and
        report['original_physical_report_sha256']==parent.PHYSICAL_REPORT_SHA,'actual NEW complete feedback physical report')
    require(prepared['schema']==physical.NAME+'.prepare' and prepared['contract']==report['contract']==physical.contract(prepared['contract']['wall_seconds']) and
        prepared['sourcepack_sha256']==report['sourcepack_sha256']==parent.SOURCEPACK_SHA and
        prepared['original_physical_report_sha256']==parent.PHYSICAL_REPORT_SHA,'same feedback physical contract/source')
    require(events[0]['kind']=='run_started' and events[0]['prepare']==prepared and events[-1]['kind']=='run_finished' and
        events[-1]['status']==report['status'] and events[-1]['counts']==report['counts'] and events[-1]['errors']==[] and
        events[-1]['event_sha256']==report['journal_chain_tail'],'original NEW prepare and terminal journal')
    require([(r['root_slot'],r['arm']) for r in report['rows']]==[(r['root_slot'],r['arm']) for r in prepared['plans']]==census and
        set(branches)==set(selected)==set(results)==set(prefixes)==set(census) and len(tasks)==2,'fixed four branches, no missing/duplicate/replacement')
    counts={k:dict(attempted=0,completed=0) for k in rgb.counts_for(4)};stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            k=e['resource'];f='attempted' if e['kind']=='operation_attempted' else 'completed';counts[k][f]+=1
            if f=='attempted':stack.append(k)
            else:require(stack and stack.pop()==k,'balanced feedback physical nested operations')
    require(not stack and counts==report['counts']==rgb.counts_for(4),'actual full80 public/816physics/88render census')
    starts=[e for e in events if e['kind']=='branch_started'];closes=[e for e in events if e['kind']=='branch_closed']
    require(len(starts)==len(closes)==4,'all four durable branch receipts')
    images=[]
    for row,plan in zip(report['rows'],prepared['plans'],strict=True):
        identity=physical.key(row);require(identity==physical.key(plan),'root/arm/selection digest join')
        slot,arm,_=identity;k=(slot,arm);b=branches[k];chosen=selected[k];result=results[k];prefix=prefixes[k];t=tasks[slot]['gn_task']
        require(type(chosen['source_slot']) is int and chosen['source_slot']==slot and chosen['arm']==arm and
            chosen['schema']==compiler.NAME+'.model_plan' and chosen['whole_run_report_sha256']==COMPILER_REPORT_SHA and
            chosen['commands']==plan['commands'] and chosen['branch_id']==b['branch_id'] and chosen['parent_task_id']==tasks[slot]['task_id'] and
            chosen['parent_task_id']==b['parent_task_id'] and chosen['observed_context_sha256']==compiler.core.digest(b['observed_context']) and
            chosen['blind_sha256']==b['blind_sha256'] and chosen['planner_selection_sha256']==b['planner_selection_sha256'] and
            chosen['prefix_commands']==b['prefix_commands'] and chosen['suffix_commands']==result['new_suffix_commands'],'own NEW consumed commands/task/observed branch')
        compiler.check_result(b,result)
        body={f:v for f,v in chosen.items() if f not in ('selection_sha256','whole_run_report_sha256')}
        require(compiler.core.digest(body)==chosen['selection_sha256'] and result['plan']==dict(body,selection_sha256=chosen['selection_sha256']), 'NEW model-plan self seal')
        physical.base.commands(plan['commands'])
        require(physical.source.digest({f:v for f,v in plan.items() if f!='selection_digest'})==plan['selection_digest'],'physical default-spaces seal')
        require(row['status'] in ('horizon_completed','native_success') and row['first_error'] is None and row['secondary_errors']==[] and
            row['initial_exact'] is True and row['prefix_exact'] is True and row['prefix_goal_exact'] is True and row['full_horizon'] is True and
            row['warmup_returned']==row['own_attempted']==row['own_returned']==10 and row['PNG']==22 and row['truncated'] is False,'full10 only; missing terminal fails closed')
        start=[e for e in starts if physical.key(e)==identity];close=[e for e in closes if physical.key(e['receipt'])==identity]
        require(len(start)==len(close)==1 and start[0]['plan']==plan and close[0]['receipt']==row and
            start[0]['original_selection_digest']==row['original_selection_digest']==b['physical_selection_digest'],'new/old branch selection provenance')
        branch=start[0]['branch'];scoped=[e for e in events if e.get('branch')==branch]
        require(all(physical.key(e)==identity for e in scoped),'same branch scope throughout')
        obs=[e for e in scoped if e['kind']=='fresh_observation'];guards=[e for e in scoped if e['kind']=='guard_receipt']
        order=[(phase,n) for phase in ('warmup','plan') for n in range(11)]
        require([(e['receipt']['phase'],e['receipt']['step']) for e in obs]==[(e['phase'],e['step']) for e in guards]==order and
            all(e['accepted'] is True and e['render_preserved_dynamics'] is True for e in guards) and
            all(o['index']<g['index'] and o['receipt']['state']==o['receipt']['snapshot']['state']==o['receipt']['observation_state'] for o,g in zip(obs,guards,strict=True)), 'all22 durable observations/guards')
        trans=[e for e in scoped if e['kind']=='returned_transition' and e['phase']=='plan']
        require([e['step'] for e in trans]==list(range(1,11)) and [e['command'] for e in trans]==plan['commands'] and
            all(not e['terminated'] and not e['truncated'] for e in trans[:9]),'all ten own commands and no early terminal')
        checks=[e for e in scoped if e['kind']=='paired_prefix_exact']
        require([(e['phase'],e['step']) for e in checks]==[('warmup',n) for n in range(11)]+[('plan',n) for n in range(6)] and
            all(e['accepted'] is True for e in checks),'full moving warmup and prefix0..5 replays exact')
        for e,o in zip(checks,obs[:17],strict=True):
            require(e['new_png']==o['receipt']['png'] and e['original_file_sha256']==o['receipt']['file_sha256'] and
                e['original_pixel_sha256']==o['receipt']['pixel_sha256'],'paired prefix file/pixel receipts')
        admitted=[e for e in scoped if e['kind']=='suffix_admitted_after_exact_prefix']
        step6=[e for e in scoped if e['kind']=='operation_attempted' and e['resource']=='public_step' and e.get('phase')=='plan' and e.get('step')==6]
        require(len(admitted)==len(step6)==1 and admitted[0]['index']<step6[0]['index'] and
            admitted[0]['prefix_snapshot']==obs[16]['receipt']['snapshot'] and admitted[0]['goal']==row['goal'],'prefix exact BEFORE suffix dispatch')
        checkpoints=[rgb.old.checkpoint(row,n) for n in (0,5,10)]
        require(checkpoints==[obs[11+n]['receipt'] for n in (0,5,10)] and checkpoints[0]['state']==row['root_state'] and
            checkpoints[-1]['state']==row['final_state'] and all(e['state']==obs[11+n]['receipt']['state'] for n,e in enumerate(trans,1)), 'root/prefix/terminal state/receipt join')
        require(prefix['checkpoint']['phase']=='plan' and prefix['checkpoint']['step']==5 and
            all(prefix['checkpoint'][f]==checkpoints[1][f] for f in ('file_sha256','pixel_sha256','state','snapshot','observation_state')) and
            prefix['entry']==b['prefix_npy'] and prefix['latent']==b['observed_context']['latent_history'][-1] and
            prefix['planner_selection_sha256']==b['planner_selection_sha256'],'original sealed prefix PNG/NPY reused, never encoded again')
        require(b['goal']==t['goal'] and np.array_equal(b['q'],t['projection']) and np.array_equal(t['projection'],np.eye(192)),'unchanged original native goal mean-Q')
        goal=[e for e in scoped if e['kind']=='goal_attached']
        require(len(goal)==1 and goal[0]['goal']==row['goal'] and goal[0]['snapshot']==checkpoints[0]['snapshot'] and
            goal[0]['goal_rgb_sha256']==t['goal_source_sha256'] and admitted[0]['goal_rgb_sha256']==t['goal_source_sha256'] and
            all(e['returned_goal_state']==row['goal'] and e['returned_goal_rgb_sha256']==t['goal_source_sha256'] for e in trans),'same original task goal throughout')
        images.append(dict(index=len(images),branch=branch,root_slot=slot,source_slot=slot,arm=arm,role='terminal10',
            task_id=tasks[slot]['task_id'],task_export=b['task_export'],task_sha256=b['task_sha256'],blind_sha256=b['blind_sha256'],
            checkpoint=checkpoints[-1],selection_digest=row['selection_digest'],compiler_branch_id=b['branch_id'],
            compiler_selection_sha256=chosen['selection_sha256'],predicted=result['evaluations'][1]['context']['latent_history'][-1],
            root=t['root_context']['latent_history'][-1],goal=t['goal'],Q='identity192',reused_prefix=deepcopy(prefix)))
    require(len(images)==4,'four real terminal10 frames, no prefix encode or padding')
    return images


def build():
    bindings={}
    def bind(path,h):
        path=str(Path(path).resolve());require(sha(path)==h and (path not in bindings or bindings[path]==h),'frozen input/source seal: '+path)
        bindings[path]=h;return Path(path)
    for path,h in ((compiler.__file__,COMPILER_SHA),(physical.__file__,PHYSICAL_SHA),(parent.__file__,PARENT_SHA),
                   (moving.__file__,parent.MOVING_SHA),(rgb.__file__,parent.RGB_SHA),(rgb.old.__file__,rgb.OLD_SHA)):
        bind(path,h)
    cr=bind(COMPILER_REPORT,COMPILER_REPORT_SHA);r=read(cr,COMPILER_REPORT_SHA)
    consumed=compiler.consume(cr.parent,COMPILER_REPORT_SHA)
    require(consumed['schema']==compiler.NAME+'.consume' and consumed['status']=='observed_suffix_commands_verified_not_admitted' and
        consumed['errors']==[] and consumed['new_physics']==0 and consumed['P1_admission'] is False and consumed['P4_admission'] is False,'NEW compiler own consume authority')
    cp=compiler.verify(r['prepared'],r['prepare_sha256']);bind(r['prepared'],r['prepare_sha256'])
    for f,h in cp['bindings'].items():bind(f,h)
    for name,h in r['artifact_sha256'].items():bind(rgb.old.local(cr.parent,name),h)
    keys=lambda xs:{(x['source_slot'],x['arm']):x for x in xs}
    branches=keys(cp['branches']);selected=keys(consumed['plans']);results=keys(r['rows'])
    require(len(cp['branches'])==len(consumed['plans'])==len(r['rows'])==len(branches)==len(selected)==len(results)==4,'complete NEW compiler denominator')
    oldpath=bind(compiler.RGB_REPORT,compiler.RGB_REPORT_SHA);old=read(oldpath,compiler.RGB_REPORT_SHA)
    require(old['schema']==parent.NAME+'.report' and old['status']=='moving_selected_rgb_encoded_not_admitted' and
        old['errors']==[] and old['model_unchanged'] is True,'original RGB source, never fake new report')
    for name,h in old['artifact_sha256'].items():bind(rgb.old.local(oldpath.parent,name),h)
    prefixes={}
    for k,b in branches.items():
        matches=[x for x in old['rows'] if (x['source_slot'],x['arm'])==k and x['role']=='prefix5']
        require(len(matches)==1,'exact original fixed prefix RGB row')
        row=matches[0];entry=b['prefix_npy'];p=bind(entry['path'],entry['sha256']);z,_=compiler.load_npy(p,entry)
        require(row['status']=='encoded' and all(row['latent'][f]==entry[f] for f in ('file','sha256','dtype','shape')) and
            p==rgb.old.local(oldpath.parent,row['latent']['file']).resolve(),'original prefix native NPY receipt')
        prefixes[k]=dict(entry=deepcopy(entry),latent=z.astype(float).tolist(),checkpoint=deepcopy(row['provenance']['checkpoint']),
            planner_selection_sha256=row['provenance']['planner_selection_sha256'],original_rgb_report_sha256=compiler.RGB_REPORT_SHA)
    tasks=[]
    for f,h in zip(parent.EXPORTS,parent.EXPORT_SHAS,strict=True):bind(f,h);tasks.append(moving.load_task(f,h))
    ph=bind(REPORT,REPORT_SHA);report=read(ph,REPORT_SHA)
    bp=ph.parent/'execution_binding.json';bind(bp,sha(bp));binding=read(bp)
    require(binding['prepare_sha256']==PHYSICAL_PREPARE_SHA,'actual feedback physical prepare SHA')
    pp=bind(binding['prepared'],PHYSICAL_PREPARE_SHA);prep=read(pp,PHYSICAL_PREPARE_SHA)
    mp=bind(prep['manifest'],prep['manifest_sha256']);manifest=read(mp,prep['manifest_sha256'])
    require(manifest['schema']==physical.NAME+'.selection' and manifest['plans']==prep['plans'] and manifest['arms']==ARMS and
        manifest['original_physical_report_sha256']==parent.PHYSICAL_REPORT_SHA and manifest['sourcepack_sha256']==prep['sourcepack_sha256'], 'NEW physical manifest/prepare/source binding')
    a=manifest['compiler_authority'];bind(a['source'],a['source_sha256']);bind(a['report'],a['report_sha256']);bind(a['consume'],a['consume_sha256'])
    require(Path(a['source']).resolve()==Path(compiler.__file__).resolve() and a['source_sha256']==COMPILER_SHA and
        Path(a['report']).resolve()==cr and a['report_sha256']==COMPILER_REPORT_SHA and
        read(a['consume'],a['consume_sha256'])==consumed,'physical authority is actual NEW own consume, not old metadata')
    for p in prep['plans']:
        require(Path(p['compiler_report']).resolve()==Path(a['consume']).resolve() and p['compiler_report_SHA']==a['consume_sha256'] and
            p['compiler_schema']==compiler.NAME+'.consume' and p['compiler_status']==consumed['status'],'literal NEW consume mapping')
        for field,want in (('commands_pointer',p['commands']),('source_slot_pointer',p['root_slot']),('arm_pointer',p['arm']),
                           ('blind_sha_pointer',selected[p['root_slot'],p['arm']]['blind_sha256'])):
            require(physical.base.pointer(consumed,p[field])==want,'explicit NEW compiler pointer')
    for f,h in prep['bindings'].items():bind(f,h)  # bytes only, no privileged source parsing
    jp=bind(ph.parent/'journal.jsonl',report['journal_sha256']);events=physical.base.journal(jp,report['journal_chain_tail'])
    images=assemble(report,events,prep,branches,selected,results,tasks,prefixes)
    for item in images:
        c=item['checkpoint'];f=bind(rgb.old.local(ph.parent,c['png']),c['file_sha256']);rgb.old.png(f,c['file_sha256'],c['pixel_sha256']);item['png_path']=str(f)
    for f,h in enc.MODELS.items():bind(f,h)
    for f in moving.source_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]:bind(f,sha(f))
    for f,h in list(bindings.items()):bind(f,h)
    return dict(schema=NAME+'.prepare',images=images,bindings=bindings,policy=str(enc.POLICY),expected_model_hash=enc.MODEL_TENSOR_SHA,
        physical_report_sha256=REPORT_SHA,physical_prepare_sha256=PHYSICAL_PREPARE_SHA,compiler_report_sha256=COMPILER_REPORT_SHA,
        original_rgb_report_sha256=compiler.RGB_REPORT_SHA,contract=dict(roots=2,arms=ARMS,branches=4,images=4,encoder=4,projector=4,
        new_prefix_encodes=0,predict=0,action_encoder=0,AD=0,physics=0,training=False,Q='original_identity_mean192_no_fit',no_dedup=True,
        P1_admission=False,P4_admission=False,selection=False,prediction='actual NEW observed-context new_suffix seal; no rollout',
        inherited_costs='old RGB8/compiler/physical paid separately; no refund or full-compute equality',
        numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),output_cap_bytes=CAP,disk_floor_bytes=FLOOR))


def prepare(output):
    p=build();path=Path(output).resolve();value=rgb.canonical(p)
    require(not path.exists() and len(value)<256*1024 and shutil.disk_usage(path.parent).free>=FLOOR+len(value),'NEW small prepare +200MiB')
    with path.open('xb') as f:require(f.write(value)==len(value),'complete prepare');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),sha256=sha(path),bytes=len(value))


def verify(path,expected):
    p=read(path,expected);require(p==build(),'complete NEW feedback RGB reconstruction');return p


class Store(rgb.old.Store):
    """Lexical 8MiB contract; no modification of frozen generic module globals."""
    def __init__(self,output):
        self.output=Path(output).resolve()
        require(not self.output.exists() and shutil.disk_usage(self.output.parent).free>=FLOOR+CAP,'NEW8MiB +200MiB reserve')
        self.output.mkdir()
    def write(self,name,value,append=False,final=False):
        path=rgb.old.local(self.output,name);require(not path.is_symlink(),'no symlink output')
        used=sum(f.stat().st_size for f in self.output.iterdir() if f.is_file())
        require(used+len(value)<=CAP-(0 if final else FINAL) and shutil.disk_usage(self.output).free>=FLOOR+len(value),'whole8MiB cap/floor incl partial')
        with path.open('ab' if append else 'xb') as f:require(f.write(value)==len(value),'complete artifact write');f.flush();os.fsync(f.fileno())
        fd=os.open(self.output,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)


def compare(item,z):
    d=parent.compare(item,z);prefix=rgb.old.vector(item['reused_prefix']['latent'],192);real=z.astype(float);goal=rgb.old.vector(item['goal'],192)
    d.update(prefix_Q=float(np.mean((prefix-goal)**2)),actual_vs_prefix_mse=float(np.mean((real-prefix)**2)))
    d['actual_Q_change_from_prefix']=d['actual_Q']-d['prefix_Q']
    return d


def encode_images(p,callback,ledger,store,rows):
    for item,row in zip(p['images'],rows,strict=True):
        ledger.image=item['index'];row['status']='attempted';store.event(dict(kind='image_attempt',provenance=item))
        a=rgb.old.png(item['png_path'],item['checkpoint']['file_sha256'],item['checkpoint']['pixel_sha256'])
        z=callback(a.copy());enc.native(z)
        require([(e['resource'],e['completed']) for e in ledger.entries if e['image']==item['index']]==
            [('encoder',True),('projector',True)],'exact terminal native module completions')
        row.update(status='encoded',latent=store.latent(item['index'],z),metrics=compare(item,z))
        store.event(dict(kind='image_saved',receipt=deepcopy(row)));ledger.image=None

def run(prepared,expected,output):
    # Own verifier, not rgb.verify() / generic physical.verify(). Encoding,
    # PNG decode, ledger, NPY writes and metrics use unchanged generic core.
    p=verify(prepared,expected);store=Store(output);errors=[];model=None
    before=after=host=None;source_ok=False;ledger=enc.Ledger(1,store.event)
    rows=[dict(**{k:i[k] for k in ('index','branch','root_slot','source_slot','arm','role','task_id','task_export','task_sha256','blind_sha256')},
        status='not_started',provenance=i) for i in p['images']]
    complete=False
    try:
        store.put('binding.json',dict(prepared=str(Path(prepared).absolute()),prepare_sha256=expected))
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        from torch.nn.attention import SDPBackend,sdpa_kernel
        from experiments.checkpoint_compat import load_compatible_pretrained
        from experiments.gpu_preflight import require_cuda_allocation
        from experiments.spectral_reachability import preflight_pusht_d2p_p0 as p0
        host=require_cuda_allocation('cuda:0');device=torch.device('cuda:0')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=load_compatible_pretrained(p['policy'],device=device).eval().requires_grad_(False)
        enc.frozen(model);before=enc.tensor_hash(model);require(before==p['expected_model_hash'],'original frozen model tensor SHA')
        transform=p0.image_transform();ledger.sync=lambda:torch.cuda.synchronize(device);ledger.no_grad=lambda:not torch.is_grad_enabled()
        with sdpa_kernel(SDPBackend.MATH),enc.guard(model,torch,ledger,errors):
            def callback(a):
                with torch.inference_mode():
                    x=p0._image_tensor(a,transform=transform,device=device)
                    require(x.device.type=='cuda' and tuple(x.shape)==(1,1,3,224,224),'native CUDA RGB batch1')
                    z=model.encode({'pixels':x})['emb']
                    require(z.device.type=='cuda' and z.dtype==torch.float32 and tuple(z.shape)==(1,1,192),'native projected CLS192')
                    return z[0,0].detach().cpu().numpy().copy()
            encode_images(p,callback,ledger,store,rows)
    except BaseException as exc:
        errors.insert(0,enc.error('execution',exc))
        try:store.event(dict(kind='execution_error',error=errors[0]))
        except BaseException as e:errors.append(enc.error('error_journal',e))
    finally:
        ledger.image=None
        for row in rows:
            if row['status']=='attempted':row['status']='failed_partial'
        if model is not None:
            try:enc.frozen(model);after=enc.tensor_hash(model)
            except BaseException as e:errors.append(enc.error('final_model',e))
        try:verify(prepared,expected);source_ok=True
        except BaseException as e:errors.append(enc.error('final_source',e))
        complete=not errors and not ledger.blocked and source_ok and before is not None and before==after and all(r['status']=='encoded' for r in rows) and ledger.counts()=={k:dict(attempted=4,completed=4) for k in ('encoder','projector')}
        hashes={}
        try:artifacts=list(store.output.iterdir())
        except BaseException as e:errors.append(enc.error('artifact_listing',e));complete=False;artifacts=[]
        for f in artifacts:
            try:
                if f.is_file():hashes[f.name]=sha(f)
            except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='observed_suffix_terminal_rgb_encoded_not_admitted' if complete else 'partial_or_failed',
            prepare_sha256=expected,physical_report_sha256=REPORT_SHA,compiler_report_sha256=COMPILER_REPORT_SHA,
            physical_prepare_sha256=PHYSICAL_PREPARE_SHA,original_rgb_report_sha256=compiler.RGB_REPORT_SHA,
            rows=rows,resources=ledger.counts(),errors=errors,first_error=errors[0] if errors else None,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,host=host,artifact_sha256=hashes,
            contract=p['contract'],new_predict=0,new_AD=0,new_physics=0,training=False,P1_admission=False,P4_admission=False)
        try:store.put('report.json',report,final=True)
        except BaseException as e:errors.append(enc.error('final_report',e));complete=False;print(rgb.canonical(dict(errors=errors)).decode(),flush=True)
    require(complete,'partial output retained; no retry: '+str(errors));return report



def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify','run'))
    for k in ('prepared','prepare-sha','output'):p.add_argument('--'+k)
    a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='verify':r=dict(status='verified_no_execution',images=len(verify(a.prepared,a.prepare_sha)['images']))
    else:r=run(a.prepared,a.prepare_sha,a.output)
    print(rgb.canonical(r).decode(),end='')


if __name__=='__main__':main()
