"""Reader MD/CSV from externally sealed moving comparison v3; saved JSON ONLY.

Six fixed baselines, two calibration roots each. No averages, ranking, neural
execution, ownconsumer replay or inferred equal-compute/official status.
"""
from __future__ import annotations
import argparse
import csv
import io
import os
from pathlib import Path
import shutil
from experiments.spectral_reachability import p1_moving_baseline_comparison_20261027_v3 as comparison

base=comparison.base
NAME='p1_moving_baseline_reader_table_20261027_v1'
ROOT=base.ROOT
CAP,FLOOR=256*1024,200*1024**2
COMPARISON_SHA='506d4abc6e962e2374ccbc813c0d8b953275cecba9a5df741c5dc71c1a3b3477'
require,sha,canonical=base.require,base.sha,base.canonical
ARMS=comparison.BASELINES
LABELS=dict(zip(ARMS,('梯度开环（本地）','CEM（本地）','nominal MPPI（本地非官方）',
    '固定H2离散beam（本地）','GRASP bounded扩展（非官方等价）','无组合谱bank映射（本地）')))
COUNTERS=('predict','predict_API','action_encoder','backward','grad_API','jacobian','jacobian_API','svd','bounded_solve','pred_projector')
SCALARS=('AD_forward_in_predict','mathematical_reverse_directions')
METRICS=('model_Q','model_root_Q','realQ_root','realQ_prefix5','realQ_terminal10','native_success','full_horizon','physical_returned_steps')


def cell(cell,kind='number'):
    require(isinstance(cell,dict) and cell.get('status') in ('actual','missing','partial','unmeasured') and 'value' in cell,'explicit typed evidence cell')
    if cell['status']!='actual':
        require(cell['value'] is None and type(cell.get('reason')) is str,'unknown MUST be null with reason');return None
    v=cell['value']
    if kind=='boolean':require(type(v) is bool,'native/horizon actual boolean, never Q threshold')
    elif kind=='counter':base.counter(v)
    elif kind=='text':require(type(v) is str and bool(v),'actual nonempty text')
    else:require(base.number(v) and v>=0,'actual finite nonnegative numeric metric')
    return v


def resource(row,key):
    enclosing=row['resources'];require(isinstance(enclosing,dict),'model resources envelope')
    if enclosing.get('value') is None:
        cell(enclosing);return base.missing('unreleased model resource ledger',enclosing['status'])
    require(enclosing.get('status')=='actual' and isinstance(enclosing['value'],dict),'actual resource map')
    if key=='predict_API':
        raw=enclosing['value'].get('raw_predict_API_resources',{}).get('predict')
        return base.actual(base.counter(raw),'actual predict API; overlaps module predict, never additive') if raw is not None else base.missing('predict API not separately metered','unmeasured')
    return enclosing['value'].get(key,base.missing('not separately metered','unmeasured'))


def declared_sources(m):return m['existing']['sources']+m['additional_sources']


def downstream(m):
    out=[(list(comparison.prior.COHORT),m['existing']['physical'],m['existing']['rgb'])]
    out.extend((e['arms'],e['physical'],e['rgb']) for e in m['existing']['additional_cohorts'])
    if m['existing']['legacy']['state']=='complete':
        e=m['existing']['legacy'];out.append((list(comparison.prior.LEGACY),dict(state='complete',**e['physical']),e['rgb']))
    out.append((list(comparison.EXTRA_ARMS),m['new_downstream']['physical'],m['new_downstream']['rgb']))
    return out


def validate(m,r,manifest_ref):
    """Validate table input contract only, NOT independent artifact/model audit."""
    comparison.manifest_check(m)
    for ref in [manifest_ref]+[e[k] for e in declared_sources(m) for k in ('report','consume') if k in e]+[e['report'] for _,p,z in downstream(m) for e in (p,z) if e['state']!='missing']:
        base.fields(ref,'path sha256','literal saved reference');base.hash_value(ref['sha256'])
        require(type(ref['path']) is str and Path(ref['path']).is_absolute(),'absolute saved source path, no dynamic lookup')
    require(r['schema']==comparison.NAME+'.comparison' and r['manifest']==manifest_ref,'comparison binds EXACT supplied v3 manifest path/SHA')
    require(r['bindings_sha256'].get(str(Path(comparison.__file__).resolve()))==COMPARISON_SHA,'comparison output binds frozen v3 implementation')
    require(r['status'] in ('comparison_recorded_not_admitted','model_complete_downstream_partial_not_admitted','partial_comparison_not_admitted'),'own v3 result status')
    require(all(r[k] is False for k in ('P1_admission','P2_admission','independent_confirmation','official_reproduction','equal_total_compute','equal_search_domain')),'calibration/nonofficial/nonmatched boundaries')
    require(r['cohort']==m['existing']['cohort']+list(comparison.EXTRA_ARMS) and r['fixed_baselines']==list(ARMS),'predeclared algorithm cohort, no outcome subset')
    expected=[(s,a) for a in r['cohort'] for s in (0,1)]
    require([(x['source_slot'],x['arm']) for x in r['rows']]==expected and all(type(x['source_slot']) is int for x in r['rows']),'all per-root rows fixed order, no root filtering')
    source_by_arm={a:e for e in declared_sources(m) for a in e['arms']}
    require(set(source_by_arm)==set(r['cohort']),'each declared arm has explicit complete/partial/missing authority')
    runmap={x['source_id']:x for x in r['runs']}
    require(len(runmap)==len(r['runs'])==len(declared_sources(m)),'unique model run ledgers')
    for e in declared_sources(m):
        run=runmap[base.digest(e)]
        require(run['provenance']==e and run['state']==e['state'] and run['arms']==e['arms'] and run['provider']==e['provider'],'exact model provenance, no duplicate run costs')
        cell(run['latency_seconds'])
    phys={x['source_id']:x for x in r['physical_runs']};require(len(phys)==len(r['physical_runs']),'unique physical run ledgers')
    rgb_seen=set();allowed_ph=set()
    for arms,p,z in downstream(m):
        for rows,entry,which in ((r['physical_runs'],p,'physical'),(r['RGB_runs'],z,'rgb')):
            match=[x for x in rows if x['arms']==arms]
            if entry['state']=='complete':require(len(match)==1 and match[0]['provenance']==entry,'same '+which+' cohort authority, never reuse old arm report')
            else:require(not match,'unreleased downstream MUST NOT have actual run ledger')
        if p['state']=='complete':
            key=base.digest(p);allowed_ph.add(key);run=phys[key];n=2*len(arms)
            require(run['actual_branches']==run['expected_branches']==n and run['full_horizons']==n and
                type(run['native_successes']) is int and 0<=run['native_successes']<=n,'fixed cohort physical denominator')
            for v in run['counts'].values():base.counter(v)
            cell(run['latency_seconds'])
        if z['state']=='complete':
            rgb_seen.add(base.digest(z));run=next(x for x in r['RGB_runs'] if x['arms']==arms);n=4*len(arms)
            require(run['image_denominator']==run['actual_images']==n and run['resources']=={k:dict(attempted=n,completed=n) for k in ('encoder','projector')},'complete RGB endpoint denominator')
            require(run['new_predict']==run['new_AD']==run['new_physics']==0,'encoding-only actual RGB')
    require(set(phys)==allowed_ph and len(r['RGB_runs'])==len(rgb_seen),'no extra/overlapping downstream invocation')
    down_by_arm={a:(p,z) for arms,p,z in downstream(m) for a in arms}
    for row in r['rows']:
        s,a=row['source_slot'],row['arm'];e=source_by_arm[a];p,z=down_by_arm[a]
        require(row['task_id']==base.TASK_IDS[s] and row['task_sha256']==base.TASK_SHAS[s] and row['source_id']==base.digest(e),'same original calibration task/source')
        base.hash_value(row['blind_sha256'])
        require(row['model_status']==dict(complete='actual',partial='partial',missing='missing')[e['state']],'model completion agrees with manifest')
        require(row['physical_status']==('actual' if p['state']=='complete' else p['state']) and row['rgb_status']==('actual' if z['state']=='complete' else z['state']),'downstream completion agrees with explicit manifest')
        for k in METRICS:
            value=cell(row.get(k,base.missing('not recorded')), 'boolean' if k in ('native_success','full_horizon') else 'number')
            if k=='model_Q':require((value is not None)==(row['model_status']=='actual'),'missing model cannot have actual Q')
            if k in ('native_success','full_horizon','physical_returned_steps'):require((value is not None)==(row['physical_status']=='actual'),'unknown physical is NOT actual0/false')
            if k in ('realQ_prefix5','realQ_terminal10'):require((value is not None)==(row['rgb_status']=='actual'),'unknown RGB is NOT Q0')
        if row['physical_status']=='actual':require(row['physical_source_id']==base.digest(p),'per-root physical cohort reference')
        for k in COUNTERS:cell(resource(row,k),'counter')
        for k in SCALARS:cell(resource(row,k))
    for run in phys.values():
        group=[x for x in r['rows'] if x.get('physical_source_id')==run['source_id']]
        require(len(group)==run['actual_branches'] and sum(x['native_success']['value'] for x in group)==run['native_successes'],'native boolean census, not model/actual Q')
    n=len(expected);d=r['denominator']
    require(d==dict(algorithms=len(r['cohort']),roots_per_algorithm=2,expected_model_rows=n,
        actual_model_rows=sum(x['model_status']=='actual' for x in r['rows']),expected_physical_rows=n,
        actual_physical_rows=sum(x['physical_status']=='actual' for x in r['rows']),expected_RGB_images=2*n,
        actual_RGB_images=2*sum(x['rgb_status']=='actual' for x in r['rows'])),'unshrunk all-root denominators')
    require(r['model_layer_complete']==all(x['model_status']=='actual' for x in r['rows']),'model census')
    enc=r['inherited_encoding_shared_once']
    require(enc['status']=='actual' and enc['value']=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')},'shared inherited8/8 ONCE')
    return r


def shown(value):
    if value is None:return 'NA'
    if type(value) is bool:return 'true' if value else 'false'
    if type(value) is float:return repr(value)
    return str(value)


def flat_rows(r,manifest_ref,comparison_ref):
    out=[]
    for a in ARMS:
        for row in (x for x in r['rows'] if x['arm']==a):
            d=dict(algorithm=a,label=LABELS[a],calibration_slot=row['source_slot'],task_id=row['task_id'],task_sha256=row['task_sha256'],
                scope='known_two_root_calibration_only',implementation='local_not_official_reproduction',equal_total_compute='false',equal_search_domain='false',
                model_status=row['model_status'],physical_status=row['physical_status'],RGB_status=row['rgb_status'],model_cohort_id=row['source_id'],
                physical_cohort_id=row.get('physical_source_id','NA'),manifest_sha256=manifest_ref['sha256'],comparison_sha256=comparison_ref['sha256'])
            for k in METRICS:
                c=row.get(k,base.missing('not recorded'));d[k]=shown(cell(c,'boolean' if k in ('native_success','full_horizon') else 'number'))
                d[k+'_status']=c['status']
            for k in COUNTERS:
                c=resource(row,k);v=cell(c,'counter')
                for phase in ('attempted','completed'):d[k+'_'+phase]=shown(None if v is None else v[phase])
                d[k+'_status']=c['status']
            for k in SCALARS:
                c=resource(row,k);d[k]=shown(cell(c));d[k+'_status']=c['status']
            out.append(d)
    require(len(out)==12,'exact six baselines x both roots');return out


def cohort_rows(r,encoder_ref):
    """Run-level cost ledger; NEVER divide costs among arms/roots."""
    out=[]
    inherited=dict(provenance=dict(report=encoder_ref),arms=list(ARMS),resources=r['inherited_encoding_shared_once']['value'],
        latency_seconds=base.missing('historical encoding latency not metered','unmeasured'),state='complete')
    for kind,runs in [('model',r['runs']),('physical',r['physical_runs']),('RGB',r['RGB_runs']),('inherited_encoding_shared_once',[inherited])]:
        for run in runs:
            ref=run['provenance'];resources=run.get('counts',run.get('resources',{}))
            latency=run['latency_seconds'];cell(latency)
            d=dict(kind=kind,cohort_id=run.get('source_id',base.digest(ref)),algorithms=';'.join(run['arms']),
                state=run.get('state','complete'),scope='whole_invocation_not_per_root_or_arm',latency_seconds=shown(latency['value']),latency_status=latency['status'],
                report_path=ref.get('report',{}).get('path','NA'),report_sha256=ref.get('report',{}).get('sha256','NA'))
            keys=('environment_construct','environment_close','reset','set_state','goal_bind','goal_render','public_step','physics_step','explicit_render','encoder','projector')
            for k in keys:
                v=resources.get(k)
                if v is not None:base.counter(v)
                for phase in ('attempted','completed'):d[k+'_'+phase]=shown(None if v is None else v[phase])
            out.append(d)
    return out


def csv_bytes(rows):
    require(bool(rows),'nonempty CSV table');s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=list(rows[0]),lineterminator='\n')
    w.writeheader();w.writerows(rows);return s.getvalue().encode('utf-8')


def markdown(rows,cohorts,manifest_ref,comparison_ref,r):
    lines=['# 固定两根 calibration：六基线逐根结果','',
        '仅同一已知两根的校准记录，不是独立确认、总体统计、SOTA排名或P1/P2通过。所有算法为本地实现；GRASP/nominal MPPI不是官方等价复现。搜索域与预算不同，不等总算力。',
        '', '| 本地算法 | 根slot | model H2 Q | actual Q root / 5 / 10 | native成功 | 完整10步 | 物理cohort |',
        '|---|---:|---:|---|---|---|---|']
    for x in rows:
        q=' / '.join(x[k] for k in ('realQ_root','realQ_prefix5','realQ_terminal10'))
        lines.append('| '+' | '.join((x['label'],str(x['calibration_slot']),x['model_Q'],q,x['native_success'],x['full_horizon'],x['physical_cohort_id'][:12]))+' |')
    lines+=['','Q为原native192 identity mean squared error；model Q来自模型终态，actual Q来自真实观测编码，二者不能互代。native成功来自实际物理receipt，与Q改善或完整10步不同。NA是缺失/partial/未meter，不是0。',
        '', '| 本地算法 | 根slot | predict完成/尝试 | AE完成/尝试 | ADforward子集 | scalar backward完成/尝试 | J完成/尝试 | reverse计数 | SVD完成/尝试 | solve完成/尝试 |',
        '|---|---:|---|---|---:|---|---|---|---|---|']
    def count(x,k):return x[k+'_completed']+'/'+x[k+'_attempted'] if x[k+'_completed']!='NA' else 'NA'
    for x in rows:
        lines.append('| '+' | '.join((x['label'],str(x['calibration_slot']),count(x,'predict'),count(x,'action_encoder'),x['AD_forward_in_predict'],
            count(x,'backward'),count(x,'jacobian'),x['mathematical_reverse_directions'],count(x,'svd'),count(x,'bounded_solve')))+' |')
    lines+=['','ADforward已包含于predict；backward/grad API/VJP/Jacobian可能是同一导数工作的不同记法，不相加。GRASP的reverse为scalar loss sweeps，谱基线为输出VJP方向，非同等工作；未单独meter的API不能猜0。完整API计数及status见CSV。',
        '', '| 物理cohort（只计一次） | 算法 | public完成/尝试 | 内部physics完成/尝试 | render完成/尝试 | 整次秒数 |',
        '|---|---|---|---|---|---|']
    for x in cohorts:
        if x['kind']=='physical':lines.append('| '+' | '.join((x['cohort_id'][:12],x['algorithms'],count(x,'public_step'),count(x,'physics_step'),count(x,'explicit_render'),x['latency_seconds']))+' |')
    lines+=['','物理数值是整个cohort调用账，逐根只关联ID，不均摊、不重复相加。模型整次latency与RGB调用另列reader_cohorts.csv，未测为NA。',
        '共享原encoder8/projector8只记一次；各算法standalone继承8/8不是可相加的共享成本。新端点评价编码在各RGB cohort另计。',
        'kino全库80 planning/根不是自适应预算优势；离散constant packet域不同于D2P逐时刻logitball。SRA joint谱bank后packet球映射改变支持域；非精确双约束ridge。',
        '',f"输入manifest：`{manifest_ref['path']}`；SHA `{manifest_ref['sha256']}`。",f"输入comparison：`{comparison_ref['path']}`；SHA `{comparison_ref['sha256']}`。",
        f"原comparison全cohort行数 {len(r['rows'])}；本表固定六基线12行。若额外纳入GN/D2P，只在输入原cohort保留，未按结果筛选。",
        '本工具仅格式/接线合同检查，不重跑ownconsume或原始artifact审计；不是独立科学准入证明。','']
    return '\n'.join(lines).encode('utf-8')


def render(m,r,manifest_ref,comparison_ref):
    base.hash_value(comparison_ref['sha256']);require(Path(comparison_ref['path']).is_absolute(),'absolute comparison reference')
    validate(m,r,manifest_ref);rows=flat_rows(r,manifest_ref,comparison_ref);cohorts=cohort_rows(r,m['existing']['encoder'])
    outputs=dict(reader_rows_csv=csv_bytes(rows),reader_cohorts_csv=csv_bytes(cohorts),reader_table_md=markdown(rows,cohorts,manifest_ref,comparison_ref,r))
    require(sum(map(len,outputs.values()))<=CAP,'<=256KiB all reader outputs');return outputs


def run(manifest,manifest_sha,comparison_path,comparison_sha,output):
    require(sha(comparison.__file__)==COMPARISON_SHA,'frozen v3 source identity')
    mr=dict(path=str(Path(manifest).absolute()),sha256=manifest_sha);cr=dict(path=str(Path(comparison_path).absolute()),sha256=comparison_sha)
    m=base.load_ref(mr);r=base.load_ref(cr);outputs=render(m,r,mr,cr)
    base.reference(mr);base.reference(cr)
    path=Path(output).absolute();size=sum(map(len,outputs.values()))
    require(not path.exists() and shutil.disk_usage(path.parent).free>=FLOOR+size,'NEW output directory +200MiB floor')
    path.mkdir();files={}
    for key,data in outputs.items():
        name={'reader_rows_csv':'reader_rows.csv','reader_cohorts_csv':'reader_cohorts.csv','reader_table_md':'reader_table.md'}[key]
        require(shutil.disk_usage(path).free>=FLOOR+len(data),'per-write reserve')
        with (path/name).open('xb') as f:require(f.write(data)==len(data),'short reader output');f.flush();os.fsync(f.fileno())
        files[str(path/name)]=sha(path/name)
    return dict(schema=NAME+'.output',status='reader_tables_written_calibration_only',files=files,bytes=size,rows=12,P1_admission=False,equal_total_compute=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('manifest','manifest-sha','comparison','comparison-sha','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(canonical(run(a.manifest,a.manifest_sha,a.comparison,a.comparison_sha,a.output)).decode(),end='')


if __name__=='__main__':main()
