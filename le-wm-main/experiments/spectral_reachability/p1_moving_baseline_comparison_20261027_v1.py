"""Saved moving2 model comparison assembler; no model/physics execution.

Strict external-SHA manifest, own-consume inputs, missing/partial kept distinct.
Physical/RGB schemas not frozen here: v1 NEVER interprets those metrics.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np

NAME='p1_moving_baseline_comparison_20261027_v1'
ROOT=Path(__file__).resolve().parents[2]
PREFIX='experiments.spectral_reachability.'
FLOOR,CAP=200*1024**2,1024**2
ENCODER=ROOT/'results/p1_constructive_moving_history_encode_cuda_20261027_v1_cuda_main01/report.json'
ENCODER_SHA='e7fb554327b3bb5a96190f6291f8450b2b0723cd981817715cfb7e50aa266bf6'
LOADER='p1_constructive_moving_history_encode_cuda_20261027_v1'
LOADER_SHA='246b9194fd31c874d7f0da67c4dec17365f93294e6ed460fe46e2509673e1b63'
TASK_SHAS=('a42bfbb1ea692782355e2572a5df2cd048b72cf3e6c41fee40eb12503674959b',
            '101add21a7a40591167a6ab8e8b22c3c62cb1fd700c741fd36cd2073e2d5df74')
TASK_IDS=('bf89d0d3c5738763c40a2d6481c6782df218de398d48bd7df33543d3444d21fd',
          '6643e3cbd4322303fdd8eb158532d75e1c0f2ce720d13a8258d12a9527801b12')
ARMS=('line_search_GN','full10_conditional_D2P','CEM','kernel_faithful_mppi_nominal',
      'moving_gradient_open_loop','fixed_H2_kinodynamic_beam16x4')
PROVIDERS={
 'moving':dict(module='moving_history_planners_cuda_20261027_v1',sha='f8f1d5507e2474fbbbfa56cae98f7bfa99f5929829946a2d7760c2c30ccc03c1',
     arms=ARMS[:2],consume_rows='plans',status='moving_history_plans_sealed_not_admitted'),
 'sampling':dict(module='p1_moving_sampling_baselines_cuda_20261027_v1',sha='f30a1c5ac5ef618cfe3add37326aae29513278fe7c4319b5a743d9f6422300e6',
     arms=ARMS[2:4],consume_rows='rows',status='moving_sampling_plans_sealed_not_admitted'),
 'gradient':dict(module='p1_moving_gradient_baseline_cuda_20261027_v1',sha='3224d6a094e74b3845c2c695c6c85fe29c6522cd4a88c91c57f2cc03d127a70e',
     arms=ARMS[4:5],consume_rows='rows',status='moving_gradient_model_plans_sealed_not_admitted'),
 'kino':dict(module='p1_moving_kinodynamic_baseline_cuda_20261027_v1',sha='66c090cb98ad8a954bae5c94e39a8cbb707b5a2e8dc82027e63fa00f2583ae6d',
     arms=ARMS[5:],consume_rows='plans',status='fixed_H2_beam_plans_sealed_not_admitted')}


def require(ok,message):
    if not ok:raise ValueError(message)


def fields(value,keys,label):require(isinstance(value,dict) and set(value)==set(keys.split()),label+' fields')


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def hash_value(value):require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'literal SHA256')


def sha(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'regular nonsymlink input')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path,cap=8*1024**2):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'bounded plain JSON input')
    def pairs(items):
        out={}
        for k,v in items:require(k not in out,'duplicate JSON key');out[k]=v
        return out
    result=json.loads(path.read_bytes(),object_pairs_hook=pairs);canonical(result);return result


def reference(ref):
    fields(ref,'path sha256','external reference');hash_value(ref['sha256'])
    p=Path(ref['path']);require(p.is_absolute() and sha(p)==ref['sha256'],'absolute external input SHA');return p


def load_ref(ref):
    p=reference(ref);v=read(p);require(sha(p)==ref['sha256'],'input TOCTOU');return v


def missing(reason,status='missing'):
    require(status in ('missing','partial','unmeasured'),'explicit nonactual status')
    return dict(status=status,value=None,reason=reason)


def actual(value,basis):
    canonical(value);return dict(status='actual',value=deepcopy(value),basis=basis)


def number(value):return type(value) in (int,float) and math.isfinite(value)


def counter(value):
    fields(value,'attempted completed','resource counter')
    require(all(type(x) is int and x>=0 for x in value.values()) and value['completed']<=value['attempted'],'literal nonnegative attempt/return counts')
    return deepcopy(value)


def count_cell(resources,key,basis):
    return actual(counter(resources[key]),basis) if key in resources else missing('not recorded by this source','unmeasured')


def validate_tasks(tasks):
    require(len(tasks)==2,'fixed two moving tasks')
    for s,task in enumerate(tasks):
        require(type(task['source_slot']) is int and task['source_slot']==s and task['task_id']==TASK_IDS[s] and task['task_sha256']==TASK_SHAS[s], 'not old zero/replacement/root order')
        require(task['with_target'] is False,'target-free original observations only')
        z=np.asarray(task['root_context']['latent_history']);g=np.asarray(task['goal'])
        require(z.shape==(3,192) and g.shape==(192,) and np.isfinite(z).all() and np.isfinite(g).all(),'native192 history/goal')
        raw=np.asarray(task['raw_history'],np.float32);want=np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)
        require(np.array_equal(raw,want),'exact moving warmup raw history')
        c=task['root_context'];norm=((want-np.asarray([-.0087,.0068],np.float32))/np.asarray([.2019,.2002],np.float32)).reshape(2,10)
        require(c['layout']=='states_with_between_state_actions_v1' and c['context_length']==3 and c['latent_dim']==192 and c['action_dim']==10 and
            np.array_equal(c['action_history'],norm),'actual normalized NONZERO between-state packets')
        for k in ('blind_sha256','task_contract_sha256'):hash_value(task[k])


def task_inputs():
    p=ROOT/'experiments/spectral_reachability'/(LOADER+'.py');require(sha(p)==LOADER_SHA,'frozen public nonzero loader')
    loader=importlib.import_module(PREFIX+LOADER);tasks=[]
    for s,h in enumerate(TASK_SHAS):
        path=ENCODER.parent/f'task_{s:02d}.json.gz';i=loader.load_task(path,h);e=loader.read(path);t=i['gn_task']
        tasks.append(dict(source_slot=s,task_id=i['task_id'],task_export=str(path),task_sha256=h,blind_sha256=e['blind_sha256'],
            task_contract_sha256=t['task_contract_sha256'],with_target=e['with_target'],root_context=t['root_context'],
            goal=t['goal'],raw_history=t['root_raw_action_history']))
    validate_tasks(tasks);return tasks


def downstream_check(downstream):
    fields(downstream,'physical rgb','downstream')
    for v in downstream.values():
        fields(v,'state reason','pending downstream')
        require(v['state'] in ('missing','schema_unfrozen') and type(v['reason']) is str and 0<len(v['reason'])<=500,'v1 does not guess downstream schema')


def manifest_check(m):
    fields(m,'schema purpose encoder sources downstream','manifest')
    require(m['schema']==NAME+'.manifest' and m['purpose']=='fixed_known_moving2_model_comparison','own purpose/schema')
    require(m['encoder']==dict(path=str(ENCODER),sha256=ENCODER_SHA),'fixed original shared encoder report')
    require(isinstance(m['sources'],list) and len(m['sources'])<=6,'bounded explicit source list')
    covered=set()
    for s in m['sources']:
        require(isinstance(s,dict) and s.get('provider') in PROVIDERS and s.get('state') in ('complete','partial','missing'),'known provider and source state')
        state=s['state'];want='provider state arms reason' if state=='missing' else 'provider state arms report consume' if state=='complete' else 'provider state arms report reason'
        fields(s,want,'source');spec=PROVIDERS[s['provider']]
        require(isinstance(s['arms'],list) and len(s['arms'])>0 and s['arms']==[a for a in spec['arms'] if a in s['arms']],'fixed provider arm order/no duplicates')
        for a in s['arms']:require(a not in covered,'duplicate algorithm across source runs');covered.add(a)
        if state!='complete':require(type(s['reason']) is str and 0<len(s['reason'])<=500,'explicit bounded missing/partial reason')
    downstream_check(m['downstream'])


def load_sources(m):
    bundles=[]
    for source in m['sources']:
        b=dict(entry=deepcopy(source))
        if source['state']!='missing':
            spec=PROVIDERS[source['provider']];report=load_ref(source['report'])
            require(report['schema']==spec['module']+'.report','correct provider report schema')
            b['report']=report
            if source['state']=='complete':
                consumer=load_ref(source['consume']);module_path=ROOT/'experiments/spectral_reachability'/(spec['module']+'.py')
                require(sha(module_path)==spec['sha'],'frozen own consumer source')
                module=importlib.import_module(PREFIX+spec['module'])
                replay=module.consume(Path(source['report']['path']).parent,source['report']['sha256'])
                require(consumer==replay,'saved ownconsume must equal its saved-data re-execution')
                b['consume']=consumer
        bundles.append(b)
    return bundles


def complete_header(report,spec):
    require(report['schema']==spec['module']+'.report' and report['status']==spec['status'] and report['errors']==[] and
        report['commands_released'] is True and report['source_verified'] is True and report['model_unchanged'] is True and
        report['model_hash_before']==report['model_hash_after'],'whole actual model source')
    require(report['contract'].get('with_target',False) is False,'no target-on contract')


def resources_for(provider,arm,row,contract):
    raw=row.get('resources',{});api=row.get('AD_API_resources',{})
    for value in raw.values():counter(value)
    for value in api.values():counter(value)
    out={key:count_cell(raw,key,'actual ownconsume-verified module/operation receipts') for key in
        ('predict','action_encoder','backward','jacobian','svd','pred_projector')}
    # Predictor count includes all AD forwards; batch1 is frozen by every
    # provider, not inferred from a walltime or guessed FLOPs conversion.
    out['forward_samples']=deepcopy(out['predict'])
    out['grad_API']=count_cell(api,'grad','actual autograd API receipts; NOT additive to backward/VJP')
    out['jacobian_API']=count_cell(api,'jacobian','actual functional.jacobian API; distinct from GN manual Jacobian operation')
    if provider in ('sampling','kino'):
        for k in ('backward','jacobian','svd','grad_API','jacobian_API'):
            if out[k]['value'] is None:out[k]=actual(dict(attempted=0,completed=0),'frozen no-AD/no-math algorithm contract; not a separate runtime hook')
    if provider=='gradient':
        for k in ('jacobian','svd'):
            require(contract[k]==0,'explicit gradient zero math contract');out[k]=actual(dict(attempted=0,completed=0),'explicit ownconsume-verified zero contract')
        out['jacobian_API']=actual(dict(attempted=0,completed=0),'gradient gate forbids Jacobian API')
    ad=contract.get('ADforward_in_predict_per_root',{}).get(arm) if provider=='moving' else contract.get('gradient_forward_predict_per_root') if provider=='gradient' else 0
    vjp=contract.get('mathematical_vjp_per_root',{}).get(arm) if provider=='moving' else contract.get('scalar_reverse_sweeps_per_root') if provider=='gradient' else 0
    out['AD_forward_in_predict']=actual(ad,'ownconsume-verified schedule, ALREADY INCLUDED in predict') if ad is not None else missing('AD forwards not separately recorded','unmeasured')
    out['mathematical_reverse_directions']=actual(vjp,'declared ownconsume-verified mathematical output directions; not extra API calls') if vjp is not None else missing('reverse directions unknown','unmeasured')
    out['raw_resources']=deepcopy(raw);out['raw_AD_API_resources']=deepcopy(api)
    out['raw_predict_API_resources']=deepcopy(row.get('predict_API_resources',{}))
    out['additive_warning']='grad API/backward/VJP/Jacobian may describe the SAME derivative work; never sum; ADforward already in predict'
    return out


def row_blank(task,arm,downstream):
    return dict(source_slot=task['source_slot'],task_id=task['task_id'],task_sha256=task['task_sha256'],blind_sha256=task['blind_sha256'],
        arm=arm,model_status='missing',model_Q=missing('no complete ownconsume provided'),model_root_Q=actual(float(np.mean((np.asarray(task['root_context']['latent_history'][-1])-np.asarray(task['goal']))**2)),'original saved task root/goal arithmetic'),
        commands_sha256=missing('no released commands'),resources=missing('no actual receipt'),latency_seconds=missing('per-root/per-arm walltime not metered','unmeasured'),
        native_success=missing('physical '+downstream['physical']['state']+': '+downstream['physical']['reason']),
        realQ_root=missing('RGB '+downstream['rgb']['state']+': '+downstream['rgb']['reason']),
        realQ_prefix5=missing('RGB schema not connected'),realQ_terminal10=missing('RGB schema not connected'),
        contract=None,source_id=None,source_row_status=None)


def validate_plan(task,arm,plan,provider):
    require(type(plan['source_slot']) is int and plan['source_slot']==task['source_slot'] and plan['task_id']==task['task_id'] and
        plan['task_sha256']==task['task_sha256'] and plan['blind_sha256']==task['blind_sha256'] and plan['task_export']==task['task_export'] and plan['arm']==arm,'exact approved moving task/arm join')
    if provider!='kino':
        require(plan['with_target'] is False and plan['selection_uses_physics'] is False and plan['physical_execution'] is False,'blind model-only selected plan')
        require(plan['task_contract_sha256']==task['task_contract_sha256'],'same nonzero task contract')
    raw=np.asarray(plan['commands']);require(raw.shape==(10,2) and raw.dtype.kind in 'fi' and np.isfinite(raw).all() and np.abs(raw).max()<=1,'literal complete raw10, no padding')
    if provider=='kino':
        cost=plan['terminal_cost'];require(len(plan['path'])==2,'fixed H2 kino path')
    else:
        require(plan['prediction']['layout']==task['root_context']['layout'],'native same-layout prediction')
        last=np.asarray(plan['prediction']['latent_history'][-1]);require(last.shape==(192,) and np.isfinite(last).all(),'native terminal192')
        cost=float(np.mean((last-np.asarray(task['goal']))**2))
        require(cost==plan['model_terminal_cost'],'recomputed selected model Q')
    require(number(cost) and cost>=0,'nonnegative finite model MSE')
    return float(cost),digest(plan['commands'])


def aggregate_counts(rows,key):
    values=[]
    for row in rows:
        cell=row['resources']
        if cell['value'] is None:return missing('not all two roots have complete verified resource receipts', 'partial' if any(r['model_status']=='partial' for r in rows) else 'missing')
        v=cell['value'][key]['value']
        if v is None:return missing('resource not measured for all roots','unmeasured')
        values.append(counter(v))
    return actual({k:sum(v[k] for v in values) for k in ('attempted','completed')},'sum over fixed two roots; not cross-resource compute equivalence')


def assemble(tasks,encoder,bundles,downstream):
    """Pure RAM assembler. IO boundary verifies external SHAs/ownconsumers."""
    validate_tasks(tasks);downstream_check(downstream)
    require(encoder['schema']==LOADER+'.report' and encoder['status']=='moving_blind_tasks_encoded_and_sealed' and encoder['errors']==[] and
        encoder['with_target'] is False and encoder['model_unchanged'] is True,'actual target-free shared encoder source')
    enc={r:counter(encoder['resources'][r]) for r in ('encoder','projector')}
    require(enc=={r:dict(attempted=8,completed=8) for r in ('encoder','projector')},'shared inherited actual8+8')
    require([(x['source_slot'],x['task_id'],x['task_file_sha256']) for x in encoder['rows']]==[(s,TASK_IDS[s],TASK_SHAS[s]) for s in (0,1)],'shared encoding task census')
    rows={(s,a):row_blank(tasks[s],a,downstream) for a in ARMS for s in (0,1)};runs=[];covered=set()
    for b in bundles:
        e=b['entry'];provider=e['provider'];spec=PROVIDERS[provider];state=e['state'];arms=e['arms'];run_id=digest(e)
        require(all(a not in covered for a in arms),'no algorithm overwritten by another source');covered.update(arms)
        run=dict(source_id=run_id,provider=provider,arms=arms,state=state,provenance=deepcopy(e),latency_seconds=missing('whole run latency not recorded','unmeasured'),contract=None)
        if state=='missing':
            for a in arms:
                for s in (0,1):rows[s,a]['model_Q']=missing(e['reason']);rows[s,a]['source_id']=run_id
            runs.append(run);continue
        r=b['report'];require(r['schema']==spec['module']+'.report','source report schema')
        require(provider=='kino' or r['contract'].get('with_target') is False,'no target-on observations/partial contracts')
        run['contract']=deepcopy(r['contract']);limit=r.get('limits')
        if isinstance(limit,dict) and 'elapsed_seconds' in limit:
            elapsed=limit['elapsed_seconds']
            if elapsed is not None:
                require(number(elapsed) and elapsed>=0,'finite actual run latency')
                run['latency_seconds']=actual(elapsed,'report.limits.elapsed_seconds WHOLE RUN guard scope, includes all its roots/arms; not per arm/isolated planner GPU time')
        census=[(s,a) for s in (0,1) for a in arms]
        rr={(x['source_slot'],x['arm']):x for x in r['rows']}
        require(len(rr)==len(r['rows']) and set(rr)==set(census),'source all fixed two roots and declared arm denominators')
        if state=='complete':
            complete_header(r,spec);c=b['consume'];require(c['schema']==spec['module']+'.consume' and c['P1_admission'] is False,'ownconsume schema/admission')
            pp={(x['source_slot'],x['arm']):x for x in c[spec['consume_rows']]}
            require(len(pp)==len(c[spec['consume_rows']]) and set(pp)==set(census),'complete ownconsume denominator')
        else:require(r['status']!=spec['status'] or r.get('errors') or r.get('commands_released') is not True,'do not label a complete released run partial')
        for s,a in census:
            row=rows[s,a];raw=rr[s,a]
            require(type(raw['source_slot']) is int and raw['task_id']==tasks[s]['task_id'],'source fixed task id, no old zero mixing')
            row.update(source_id=run_id,source_row_status=raw['status'],contract=deepcopy(r['contract']),model_status='actual' if state=='complete' else 'partial')
            if state=='complete':
                cost,commands_sha=validate_plan(tasks[s],a,pp[s,a],provider)
                row['model_Q']=actual(cost,'ownconsume-released native identity192 H2 prediction; NOT native success or realQ')
                row['commands_sha256']=actual(commands_sha,'canonical literal raw10x2 command digest')
                row['resources']=actual(resources_for(provider,a,raw,r['contract']),'actual counters + explicitly labeled frozen zero/schedule contracts')
            else:
                row['model_Q']=missing('partial run: '+e['reason'],'partial')
                partial=raw.get('resources',{})
                for v in partial.values():counter(v)
                row['resources']=dict(status='partial',value=None,reason='unreleased partial counters are not a completed budget',reported_counters=deepcopy(partial),reported_AD_API_resources=deepcopy(raw.get('AD_API_resources',{})))
        runs.append(run)
    ordered=[rows[s,a] for a in ARMS for s in (0,1)];summary=[]
    for a in ARMS:
        group=[rows[s,a] for s in (0,1)];complete=all(x['model_status']=='actual' for x in group)
        summary.append(dict(arm=a,expected_roots=2,actual_model_roots=sum(x['model_status']=='actual' for x in group),
            partial_model_roots=sum(x['model_status']=='partial' for x in group),
            mean_model_Q=actual(sum(x['model_Q']['value'] for x in group)/2,'mean of BOTH fixed roots') if complete else missing('fixed two-root mean unavailable; never average only available roots'),
            native_successes=missing('physical schema not connected; modelQ cannot substitute'),native_denominator=2,actual_physical_roots=0,
            mean_realQ_terminal10=missing('RGB schema not connected'),
            resources={k:aggregate_counts(group,k) for k in ('predict','action_encoder','grad_API','backward','jacobian','jacobian_API','svd','pred_projector')},
            standalone_inherited_encoding=actual(enc,'8+8 per standalone TWO-ROOT algorithm invocation; NOT sum into shared comparison ledger')))
    return dict(schema=NAME+'.comparison',status='model_layer_complete_physical_rgb_missing' if all(x['model_status']=='actual' for x in ordered) else 'partial_model_inventory_physical_rgb_missing',
        rows=ordered,summary=summary,runs=runs,downstream=deepcopy(downstream),
        inherited_encoding_shared_once=actual(enc,'single original encoder report SHA '+ENCODER_SHA+'; do not multiply by6 algorithms'),
        denominator=dict(algorithms=6,roots_per_algorithm=2,expected_model_rows=12,actual_model_rows=sum(x['model_status']=='actual' for x in ordered),expected_physical_rows=12,actual_physical_rows=0),
        P1_admission=False,P2_admission=False,independent_confirmation=False,equal_total_compute=False,equal_search_domain=False,official_reproduction=False,
        warnings=['known moving2, not independent confirmation','forward samples are NOT total compute; AD already in forward, reverse counters not additive',
            'standalone inherited8/8 each is a counterfactual accounting view; shared ledger pays it ONCE',
            'run latency is shared by all rows of that source_id; never divide or duplicate to invent per-arm latency',
            'physical and RGB schema unconnected: missing native success/realQ is NOT zero/failure'])


def md_cell(cell):
    if cell['value'] is None:return cell['status']
    v=cell['value']
    if isinstance(v,dict) and set(v)=={'attempted','completed'}:return f"{v['completed']}/{v['attempted']}"
    return format(v,'.6g') if type(v) is float else str(v)


def markdown(result):
    lines=['# Moving 固定两根基线结果（模型层）','',
        '已知两根；P1/P2未通过。native成功和realQ未接入，missing不代表0。以下模型Q不是物理效果。','',
        '| 算法 | 模型完成/2 | 两根平均modelQ | forward样本 完成/尝试 | grad API | Jacobian操作 | SVD | native成功/2 | actualQ终点 |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for s in result['summary']:
        r=s['resources'];lines.append('| '+' | '.join([s['arm'],str(s['actual_model_roots'])+'/2',md_cell(s['mean_model_Q']),
            md_cell(r['predict']),md_cell(r['grad_API']),md_cell(r['jacobian']),md_cell(r['svd']),md_cell(s['native_successes']),md_cell(s['mean_realQ_terminal10'])])+' |')
    lines+=['','原编码共享仅一次8 encoder +8 projector；每算法standalone视图各列8/8，不累加到共享账。',
        'forward已含AD forward；grad/backward/Jacobian/VJP不能机械相加，也不能凭forward数宣称等总算力。','',
        '| 运行provider | 状态 | 整次运行秒数（不可均摊） |','|---|---|---:|']
    for run in result['runs']:lines.append(f"| {run['provider']} | {run['state']} | {md_cell(run['latency_seconds'])} |")
    lines+=['','每根明细、数学VJP、backward、完整信息合同、来源SHA和缺失原因见同目录 comparison.json。','']
    return '\n'.join(lines)


def template():
    return dict(schema=NAME+'.manifest',purpose='fixed_known_moving2_model_comparison',encoder=dict(path=str(ENCODER),sha256=ENCODER_SHA),
        sources=[dict(provider=p,state='missing',arms=list(s['arms']),reason='MAIN填入actual report及ownconsume外部SHA后改complete') for p,s in PROVIDERS.items()],
        downstream={k:dict(state='schema_unfrozen',reason='下游接口尚未绑定，v1不猜字段') for k in ('physical','rgb')})


def run(manifest,manifest_sha,output):
    hash_value(manifest_sha);ref=dict(path=str(Path(manifest).absolute()),sha256=manifest_sha);m=load_ref(ref);manifest_check(m)
    tasks=task_inputs();encoder=load_ref(m['encoder']);bundles=load_sources(m)
    result=assemble(tasks,encoder,bundles,m['downstream'])
    bindings={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'tests'/('test_'+NAME+'.py')):sha(ROOT/'tests'/('test_'+NAME+'.py')),
        str(ROOT/'innerdocs'/(NAME+'.md')):sha(ROOT/'innerdocs'/(NAME+'.md')),
        str(ROOT/'experiments/spectral_reachability'/(LOADER+'.py')):LOADER_SHA}
    bindings.update({t['task_export']:t['task_sha256'] for t in tasks})
    for b in bundles:
        if b['entry']['state']=='complete':
            s=PROVIDERS[b['entry']['provider']];bindings[str(ROOT/'experiments/spectral_reachability'/(s['module']+'.py'))]=s['sha']
    result.update(manifest=ref,implementation_bindings_sha256=bindings)
    data=canonical(result);text=markdown(result).encode();require(len(data)+len(text)<=CAP,'<=1MiB output cap')
    # Check all explicit sources again BEFORE opening a NEW output directory.
    reference(ref);reference(m['encoder'])
    for b in bundles:
        for k in ('report','consume'):
            if k in b['entry']:reference(b['entry'][k])
    for p,h in bindings.items():require(sha(p)==h,'implementation final TOCTOU')
    path=Path(output).absolute();require(not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+len(data)+len(text),'NEW output +200MiB reserve')
    path.mkdir()
    for name,raw in (('comparison.json',data),('comparison.md',text)):
        require(shutil.disk_usage(path).free>=FLOOR+len(raw),'per-write reserve')
        with (path/name).open('xb') as f:require(f.write(raw)==len(raw),'short write');f.flush();os.fsync(f.fileno())
    return dict(schema=NAME+'.output',status=result['status'],files={str(path/n):sha(path/n) for n in ('comparison.json','comparison.md')},P1_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('template','assemble'))
    for k in ('manifest','manifest-sha','output'):parser.add_argument('--'+k)
    a=parser.parse_args();result=template() if a.command=='template' else run(a.manifest,a.manifest_sha,a.output)
    print(canonical(result).decode(),end='')


if __name__=='__main__':main()
