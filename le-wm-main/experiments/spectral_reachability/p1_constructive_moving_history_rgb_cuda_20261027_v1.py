"""Moving-history GN/D2P selected physical RGB evaluation; MAIN-only CUDA.

Own saved schema join and moving load_task. No zero-history loaders, physics,
new predictions, outcome selection, training or CPU neural fallback.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import p1_constructive_moving_history_physical_20261027_v1 as physical
from experiments.spectral_reachability import p1_constructive_moving_history_encode_cuda_20261027_v1 as moving
from experiments.spectral_reachability import moving_history_planners_cuda_20261027_v1 as planner
from experiments.spectral_reachability import p1_constructive_selected_rgb_cuda_20261027_v2 as rgb

NAME='p1_constructive_moving_history_rgb_cuda_20261027_v1'
ROOT=physical.ROOT
PHYSICAL_SHA='44d90599eab881c773d5efa09177dfe6b395fa3f4eef41070340681fea5ec96e'
MOVING_SHA='246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63'
PLANNER_SHA='f8f1d5507e2474fbbbfa56cae98f7bfa99f5929829946a2d7760c2c30ccc03c1'
RGB_SHA='e1448739bbae827a2e3b1f62bf50f0f27aa06d47aef60163483fa289e6eb432e'
SOURCEPACK_SHA='c2d8d573f94884ffc081163f25f2bfa269f98ffe8155887704109219910ffec6'
PHYSICAL_REPORT=ROOT/'results/p1_constructive_moving_history_physical_20261027_v1_main01/report.json'
PHYSICAL_REPORT_SHA='a5322dc4f483e82b5ed9d7b3996cf8a63845261d65774f90a4ae341773016aaf'
PHYSICAL_PREPARE_SHA='226749e2bf8acdcd7a4a4a08be7336445daad0b6ad12185b24cbc0fb4560b1ff'
PLANNER_REPORT=ROOT/'results/moving_history_planners_cuda_20261027_v1_cuda_main01/report.json'
PLANNER_REPORT_SHA='cd8432c06288577d25d9d9b92c7eb3673de592499ee64cae93238156f3f19aa0'
EXPORTS=[str(ROOT/f'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/task_{s:02d}.json.gz') for s in (0,1)]
EXPORT_SHAS=['a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b',
             '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74']
ARMS=['line_search_GN','full10_conditional_D2P']
enc=rgb.enc
require,sha,read=rgb.require,rgb.sha,rgb.read


def key(row):
    require(type(row['root_slot']) is int and row['root_slot'] in (0,1),'literal physical root_slot')
    return row['root_slot'],row['arm'],row['selection_digest']


def assemble(report,events,prepared,tasks,selected):
    """Own moving schema join. No file I/O; state is provenance, never selection."""
    expected=[(s,a) for s in (0,1) for a in ARMS]
    require(report['schema']==physical.NAME+'.report' and report['status']=='moving_history_physical_recorded' and
        report['errors']==[] and report['P1_admission'] is False,'complete moving physical record')
    c=prepared['contract']
    require(prepared['schema']==physical.NAME+'.prepare' and c==physical.contract(c['wall_seconds']) and
        c==report['contract'] and prepared['sourcepack_sha256']==report['sourcepack_sha256'],'same moving physical contract')
    require(events and events[0]['kind']=='run_started' and events[0]['prepare']==prepared and
        events[-1]['kind']=='run_finished' and events[-1]['status']==report['status'] and
        events[-1]['counts']==report['counts'] and events[-1]['errors']==[] and
        events[-1]['event_sha256']==report['journal_chain_tail'],'original prepare and terminal journal')
    require([(r['root_slot'],r['arm']) for r in report['rows']]==expected and
        [(r['root_slot'],r['arm']) for r in prepared['plans']]==expected and len(tasks)==2,'fixed four root-major rows')
    counts={k:dict(attempted=0,completed=0) for k in rgb.counts_for(4)};stack=[]
    for e in events:
        if e['kind'] in ('operation_attempted','operation_completed'):
            k=e['resource'];field='attempted' if e['kind']=='operation_attempted' else 'completed';counts[k][field]+=1
            if field=='attempted':stack.append(k)
            else:require(stack and stack.pop()==k,'balanced actual physical operation nesting')
    require(not stack and counts==report['counts']==rgb.counts_for(4),'full80 public/816 physics/88 render census')
    starts=[e for e in events if e['kind']=='branch_started'];closed=[e for e in events if e['kind']=='branch_closed']
    require(len(starts)==len(closed)==4 and len({key(e) for e in starts})==4,'all four durable branches')
    images=[]
    for row,plan in zip(report['rows'],prepared['plans'],strict=True):
        identity=key(row);require(identity==key(plan),'root/arm/selection_digest join')
        slot,arm,_=identity;task=tasks[slot];t=task['interface']['gn_task']
        chosen=selected[(slot,arm)]
        require(chosen['source_slot']==slot and chosen['arm']==arm and chosen['task_id']==task['interface']['task_id'] and
            chosen['blind_sha256']==task['blind_sha256'] and chosen['task_sha256']==task['task_sha256'] and
            chosen['task_export']==task['task_export'] and chosen['commands']==plan['commands'],'own moving task/blind/raw commands')
        physical.commands(plan['commands'])
        require(physical.source.digest({k:v for k,v in plan.items() if k!='selection_digest'})==plan['selection_digest'],'physical default-spaces row seal')
        predictions=rgb.old.selected_task(chosen,task['interface'])  # array math only; never its old loader
        require(row['status'] in ('horizon_completed','native_success') and row['first_error'] is None and
            row['secondary_errors']==[] and row['initial_exact'] is True and row['full_horizon'] is True and
            row['own_attempted']==row['own_returned']==row['warmup_returned']==10 and row['PNG']==22 and
            row['truncated'] is False,'full10 branches only; never substitute missing checkpoint')
        start=[e for e in starts if key(e)==identity];close=[e for e in closed if key(e['receipt'])==identity]
        require(len(start)==len(close)==1 and start[0]['plan']==plan and close[0]['receipt']==row,'durable exact selected branch')
        branch=start[0]['branch'];scoped=[e for e in events if e.get('branch')==branch]
        require(all(key(e)==identity for e in scoped),'branch scope identity throughout journal')
        warm=np.asarray(start[0]['raw_warmup'],np.float32)
        require(np.array_equal(warm,moving.fixed_raw()) and start[0]['raw_float32_bits']==warm.tobytes().hex(),'actual nonzero warmup declaration')
        warm_returns=[e for e in scoped if e['kind']=='warmup_return']
        require([e['step'] for e in warm_returns]==list(range(1,11)) and
            [e['raw_float32_bits'] for e in warm_returns]==[a.tobytes().hex() for a in warm] and
            [e['raw_action'] for e in warm_returns]==warm.astype(float).tolist(),'all actual moving raw-history returns')
        transitions=[e for e in scoped if e['kind']=='returned_transition' and e['phase']=='plan']
        require([e['step'] for e in transitions]==list(range(1,11)) and
            [e['command'] for e in transitions]==plan['commands'] and
            all(not e['truncated'] and not e['terminated'] for e in transitions[:-1]),'own10 literal commands and chronology')
        observed=[e for e in scoped if e['kind']=='fresh_observation'];guards=[e for e in scoped if e['kind']=='guard_receipt']
        census=[(phase,n) for phase in ('warmup','plan') for n in range(11)]
        require([(e['receipt']['phase'],e['receipt']['step']) for e in observed]==census and
            [(e['phase'],e['step']) for e in guards]==census and
            all(e['accepted'] is True and e['render_preserved_dynamics'] is True for e in guards),'all22 fresh frames/guards, no repair')
        for o,g in zip(observed,guards,strict=True):
            require(o['index']<g['index'] and o['receipt']['state']==o['receipt']['observation_state']==o['receipt']['snapshot']['state'],'persisted fresh observation precedes guard')
        tc=task['task_contract']
        require([observed[n]['receipt']['pixel_sha256'] for n in (0,5,10)]==tc['pixel_sha256'][:3],'physical moving frames match actual task history')
        checkpoints=[rgb.old.checkpoint(row,n) for n in (0,5,10)]
        root=checkpoints[0]
        require(root['state']==row['root_state'] and checkpoints[-1]['state']==row['final_state'] and
            root['pixel_sha256']==tc['pixel_sha256'][2] and root['snapshot']==observed[10]['receipt']['snapshot'],'exact post-bind root and terminal identity')
        require(prepared['source_slots'][slot]==dict(root_slot=slot,root_sha256=physical.source.root_hash(
            np.asarray(row['root_state']),np.asarray(row['goal']))),'moving root plus goal identity')
        attached=[e for e in scoped if e['kind']=='goal_attached']
        require(len(attached)==1 and attached[0]['goal']==row['goal'] and attached[0]['snapshot']==root['snapshot'] and
            attached[0]['goal_rgb_sha256']==t['goal_source_sha256']==tc['pixel_sha256'][3] and
            all(e['returned_goal_state']==row['goal'] and e['returned_goal_rgb_sha256']==t['goal_source_sha256'] for e in transitions),'same goal throughout own commands')
        for j,step in enumerate((5,10)):
            checkpoint=checkpoints[j+1]
            require(observed[11+step]['receipt']==checkpoint and transitions[step-1]['state']==checkpoint['state'],'checkpoint tied to returned state and durable frame')
            images.append(dict(index=len(images),branch=branch,root_slot=slot,source_slot=slot,arm=arm,
                task_id=task['interface']['task_id'],task_export=task['task_export'],task_sha256=task['task_sha256'],
                blind_sha256=task['blind_sha256'],role='prefix5' if step==5 else 'terminal10',checkpoint=checkpoint,
                root=t['root_context']['latent_history'][-1],goal=t['goal'],Q='identity192',predicted=predictions[j],
                selection_digest=row['selection_digest'],planner_selection_sha256=chosen['selection_sha256']))
    require(len(images)==8,'eight real selected images; no padding')
    return images


def handoff_plan(handoff,plan,selected,planner_sha):
    require(plan['commands_pointer'].endswith('/commands'),'explicit consume-plan commands leaf')
    parent=plan['commands_pointer'].rsplit('/',1)[0]
    authority=physical.pointer(handoff,parent)
    if 'prediction' not in authority:
        require(handoff['projection']['omitted_plan_fields']==['prediction'] and
            handoff['projection']['source_report_sha256']==planner_sha,'explicit prediction-only handoff projection')
        wanted={k:v for k,v in selected.items() if k!='prediction'}
    else:wanted=selected
    require(authority==wanted and selected['whole_run_report_sha256']==planner_sha,
        'handoff exact own consumed moving plan projection; full prediction from planner seal')


def build(physical_report=PHYSICAL_REPORT,physical_sha=PHYSICAL_REPORT_SHA,
          physical_prepare_sha=PHYSICAL_PREPARE_SHA,planner_report=PLANNER_REPORT,planner_sha=PLANNER_REPORT_SHA):
    arguments=dict(physical_report=str(Path(physical_report).resolve()),physical_sha=physical_sha,
        physical_prepare_sha=physical_prepare_sha,planner_report=str(Path(planner_report).resolve()),planner_sha=planner_sha)
    for h in (physical_sha,physical_prepare_sha,planner_sha):moving.core.hash_value(h)
    require((physical_sha,physical_prepare_sha,planner_sha)==(PHYSICAL_REPORT_SHA,PHYSICAL_PREPARE_SHA,PLANNER_REPORT_SHA),
        'fixed MAIN actual physical/prepare/planner run')
    pins={physical.__file__:PHYSICAL_SHA,moving.__file__:MOVING_SHA,planner.__file__:PLANNER_SHA,
        rgb.__file__:RGB_SHA,rgb.old.__file__:rgb.OLD_SHA}
    bindings={}
    def bind(path,h):
        path=str(Path(path).resolve());require(sha(path)==h,'external saved/source seal: '+path)
        require(path not in bindings or bindings[path]==h,'non-conflicting closure seal');bindings[path]=h;return Path(path)
    for path,h in pins.items():bind(path,h)
    pr=bind(arguments['planner_report'],planner_sha)
    require(pr.name=='report.json','explicit own planner report.json')
    consumed=planner.consume(pr.parent,planner_sha)  # saved only, own moving loader; no model/physics
    require(consumed['schema']==planner.NAME+'.consume' and consumed['status']=='moving_commands_verified_not_admitted' and
        consumed['new_physics']==0 and consumed['P1_admission'] is False,'own moving model command authority')
    selected={(p['source_slot'],p['arm']):p for p in consumed['plans']}
    require(len(consumed['plans'])==len(selected)==4 and set(selected)=={(s,a) for s in (0,1) for a in ARMS},'all four moving model plans')
    model_report=read(pr,planner_sha);pp=planner.verify(model_report['prepared'],model_report['prepare_sha256'])
    bind(model_report['prepared'],model_report['prepare_sha256'])
    for path,h in pp['bindings_sha256'].items():bind(path,h)
    for name,h in model_report['artifact_sha256'].items():bind(rgb.old.local(pr.parent,name),h)
    tasks=[]
    for slot,(path,h) in enumerate(zip(EXPORTS,EXPORT_SHAS,strict=True)):
        bind(path,h);interface=moving.load_task(path,h);export=read(path,h)
        require(type(export['source_slot']) is int and export['source_slot']==slot and
            export['blind_sha256']==export['blind_source_sha256'] and np.array_equal(interface['q'],np.eye(192)),'moving load_task own identity Q')
        tasks.append(dict(source_slot=slot,task_export=path,task_sha256=h,blind_sha256=export['blind_sha256'],
            task_contract=export['task_contract'],interface=interface))
    ph=bind(arguments['physical_report'],physical_sha);report=read(ph,physical_sha)
    bp=ph.parent/'execution_binding.json';bind(bp,sha(bp));binding=read(bp)
    require(binding['prepare_sha256']==physical_prepare_sha,'external physical prepare SHA')
    ppath=bind(binding['prepared'],physical_prepare_sha);prep=read(ppath,physical_prepare_sha)
    mp=bind(prep['manifest'],prep['manifest_sha256']);manifest=read(mp,prep['manifest_sha256'])
    require(manifest['schema']==physical.NAME+'.selection' and manifest['arms']==ARMS and
        manifest['plans']==prep['plans'] and manifest['roots']==prep['source_slots'] and
        manifest['sourcepack_sha256']==prep['sourcepack_sha256']==SOURCEPACK_SHA,'moving manifest/prepare/source identity')
    refs=[dict(root_sha256=s['root_sha256'],blind_sha256=t['blind_sha256']) for s,t in zip(prep['source_slots'],tasks,strict=True)]
    plans,compiler_bindings=physical.selection(manifest,dict(references=refs))  # saved pointers only
    require(plans==prep['plans'],'own selected handoff mappings')
    for path,h in compiler_bindings.items():bind(path,h)
    for p in plans:
        handoff=read(p['compiler_report'],p['compiler_report_SHA'])
        handoff_plan(handoff,p,selected[(p['root_slot'],p['arm'])],planner_sha)
    # Evaluator/source bindings are hashed, NEVER parsed for witness/state selection.
    for path,h in prep['bindings'].items():bind(path,h)
    jp=bind(ph.parent/'journal.jsonl',report['journal_sha256'])
    events=physical.journal(jp,report['journal_chain_tail'])
    images=assemble(report,events,prep,tasks,selected)
    for item in images:
        c=item['checkpoint'];f=bind(rgb.old.local(ph.parent,c['png']),c['file_sha256'])
        rgb.old.png(f,c['file_sha256'],c['pixel_sha256']);item['png_path']=str(f)
    for path,h in enc.MODELS.items():bind(path,h)
    files=moving.source_files()+[Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    for f in files:bind(f,sha(f))
    for path,h in list(bindings.items()):bind(path,h)
    return dict(schema=NAME+'.prepare',arguments=arguments,images=images,bindings=bindings,policy=str(enc.POLICY),
        expected_model_hash=enc.MODEL_TENSOR_SHA,contract=dict(roots=2,arms=ARMS,branches=4,images=8,encoder=8,projector=8,
        predict=0,action_encoder=0,AD=0,physics=0,training=False,Q='identity_mean192_no_fit',no_dedup=True,
        prediction='same-command sealed model prefix/terminal; no new rollout',selection=False,P1_admission=False,P2_admission=False,
        inherited_costs='model planning/original encoding/physical separately paid, not refunded',
        numerics=dict(SDP='MATH',matmul_TF32=False,cudnn_TF32=False),output_cap_bytes=rgb.CAP,disk_floor_bytes=rgb.FLOOR))


def prepare(output,**arguments):
    p=build(**arguments);path=Path(output).resolve();value=rgb.canonical(p)
    require(not path.exists() and len(value)<256*1024 and shutil.disk_usage(path.parent).free>=rgb.FLOOR+len(value),'exclusive small prepare +200MiB')
    with path.open('xb') as f:require(f.write(value)==len(value),'complete prepare write');f.flush();os.fsync(f.fileno())
    return dict(prepared=str(path),sha256=sha(path),bytes=len(value))


def verify(path,expected):
    p=read(path,expected);require(p==build(**p['arguments']),'complete moving RGB reconstruction');return p


def compare(item,z):
    d=rgb.compare(item,z);predicted=rgb.old.vector(item['predicted'],192);goal=rgb.old.vector(item['goal'],192)
    d.update(predicted_Q=float(np.mean((predicted-goal)**2)),predicted_vs_real_mse=float(np.mean((predicted-z.astype(float))**2)),
        prediction_status='sealed_same_command_model_not_recomputed')
    d['actual_minus_predicted_Q']=d['actual_Q']-d['predicted_Q']
    d['Q_improvement_sign_reversed']=bool((d['actual_Q']-d['root_Q'])*(d['predicted_Q']-d['root_Q'])<0)
    return d


def encode_images(p,callback,ledger,store,rows):
    for item,row in zip(p['images'],rows,strict=True):
        ledger.image=item['index'];row['status']='attempted';store.event(dict(kind='image_attempt',provenance=item))
        a=rgb.old.png(item['png_path'],item['checkpoint']['file_sha256'],item['checkpoint']['pixel_sha256'])
        z=callback(a.copy());enc.native(z)
        require([(e['resource'],e['completed']) for e in ledger.entries if e['image']==item['index']]==
            [('encoder',True),('projector',True)],'actual encoder/projector completions per image')
        row.update(status='encoded',latent=store.latent(item['index'],z),metrics=compare(item,z))
        store.event(dict(kind='image_saved',receipt=deepcopy(row)));ledger.image=None

def run(prepared,expected,output):
    # Own verifier, not rgb.verify() / generic physical.verify(). Encoding,
    # PNG decode, ledger, NPY writes and metrics use unchanged generic core.
    p=verify(prepared,expected);store=rgb.Store(output);errors=[];model=None
    before=after=host=None;source_ok=False;ledger=enc.Ledger(2,store.event)
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
        complete=not errors and not ledger.blocked and source_ok and before is not None and before==after and all(r['status']=='encoded' for r in rows) and ledger.counts()=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')}
        hashes={}
        try:artifacts=list(store.output.iterdir())
        except BaseException as e:errors.append(enc.error('artifact_listing',e));complete=False;artifacts=[]
        for f in artifacts:
            try:
                if f.is_file():hashes[f.name]=sha(f)
            except BaseException as e:errors.append(enc.error('artifact_hash',e));complete=False
        report=dict(schema=NAME+'.report',status='moving_selected_rgb_encoded_not_admitted' if complete else 'partial_or_failed',
            prepare_sha256=expected,inputs=p['arguments'],
            rows=rows,resources=ledger.counts(),errors=errors,first_error=errors[0] if errors else None,
            source_verified=source_ok,model_hash_before=before,model_hash_after=after,
            model_unchanged=before==after if before is not None else None,host=host,artifact_sha256=hashes,
            contract=p['contract'],new_predict=0,new_AD=0,new_physics=0,training=False,P1_admission=False,P2_admission=False)
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
