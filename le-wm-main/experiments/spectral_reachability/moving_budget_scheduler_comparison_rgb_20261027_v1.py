"""Saved-only scheduler comparison + own RGB receipt; no neural/physics/selection.

Preserve the original comparison verbatim. Add eight strictly paired actual
prefix/terminal RGB descriptions, never repair an unknown prediction.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import numpy as np
from experiments.spectral_reachability import moving_budget_scheduler_comparison_20261027_v1 as comparison
from experiments.spectral_reachability import moving_budget_scheduler_rgb_cuda_20261027_v1 as rgb

NAME='moving_budget_scheduler_comparison_rgb_20261027_v1'
ROOT=comparison.ROOT
require,sha,canonical=comparison.require,comparison.sha,comparison.canonical
CAP,FLOOR=1024**2,200*1024**2
COMPARISON_SHA='1eb2c51d6d4459b4aa6723fafed40ca8034729aa4f27d4359c28964125840b64'
PINS={str(ROOT/'experiments/spectral_reachability/moving_budget_scheduler_comparison_20261027_v1.py'):'977cb1e991ecf95752ea9c7d568b53f076c116ad9639be6faec657338989a55b',
    str(ROOT/'tests/test_moving_budget_scheduler_comparison_20261027_v1.py'):'25abcf8d00847452859c7ba92319480dba21207686ec1a211691f4fef52c7c41',
    str(ROOT/'innerdocs/moving_budget_scheduler_comparison_20261027_v1.md'):'c3b913991f12ca0ba50a7f0899122a859b0024335ff59f2eaafdaf22a22abda8',
    str(ROOT/'experiments/spectral_reachability/moving_budget_scheduler_rgb_cuda_20261027_v1.py'):'7279022b1990f9a629d785412ae8428b1a1719fd9b814af89b50fd77677c944a',
    str(ROOT/'tests/test_moving_budget_scheduler_rgb_cuda_20261027_v1.py'):'c3d79ea1a799d04249d3eb0269b96998ce7d08232db328bed877c409d3d19f20',
    str(ROOT/'innerdocs/moving_budget_scheduler_rgb_cuda_20261027_v1.md'):'b3a9126f2431c492f240d29f1574afea7d22beca88c91801e18ffc9e32fe5b48'}


def key(row,slot='root_slot'):
    s=row[slot];require(type(s) is int and s in (0,1) and row['arm'] in rgb.ARMS,'literal original root and fixed strategy')
    return s,row['arm']


def assemble(base,report,receipt):
    """Pure RAM join after both frozen providers have verified their closures."""
    require(base['schema']==comparison.NAME+'.report' and base['status']=='saved_comparison_assembled_not_admitted' and
        base['errors']==[] and base['fixed_branch_denominator']==8 and all(base[k] is False for k in comparison.FLAGS),
        'original comparison scope/admission')
    require(report['schema']==rgb.NAME+'.report' and report['status']=='moving_budget_scheduler_rgb_encoded_not_admitted' and
        report['errors']==[] and report['contract']==rgb.contract() and report['fixed_branch_denominator']==8 and report['fixed_image_denominator']==16 and
        all(report[k] is False for k in ('P1_admission','P2_admission','P3_admission','training')) and
        receipt['schema']==rgb.NAME+'.consume' and receipt['status']=='scheduler_rgb_verified_not_admitted' and receipt['errors']==[] and
        receipt['fixed_branch_denominator']==8 and receipt['fixed_image_denominator']==16 and
        receipt['prediction_status']=='unknown_not_recomputed' and all(receipt[k] is False for k in ('P1_admission','P2_admission','P3_admission')),
        'complete own RGB receipt with original scope')
    bi=base['inputs'];ri=report['inputs']
    require(bi['physical_report_sha256']==ri['physical_sha'] and bi['physical_prepare_sha256']==ri['physical_prepare_sha'] and
        bi['physical_source_sha256']==ri['physical_source_sha'] and
        str(Path(bi['physical_output'])/'report.json')==ri['physical_report'],'same actual physical report/prepare/source/path')
    expected={(s,a) for s in (0,1) for a in rgb.ARMS}
    originals={}
    for row in base['rows']:
        k=key(row,'source_slot');require(k not in originals,'unique original comparison cell');originals[k]=row
        require(row['full_horizon'] is True and row['native_steps']==10 and row['truncated'] is False,'full same H2, not native partial endpoint')
    require(set(originals)==expected and len(base['rows'])==8,'all eight comparison branches')
    images={};own={}
    for collection,index in ((report['rows'],images),(receipt['rows'],own)):
        for row in collection:
            k=(*key(row),row['role']);require(row['role'] in ('prefix5','terminal10') and k not in index,'unique image role/identity');index[k]=row
        require(len(collection)==16 and set(index)=={(*k,role) for k in expected for role in ('prefix5','terminal10')},'fixed sixteen image denominator')
    rows=[];root_latents={}
    for s,a in [(s,a) for s in (0,1) for a in rgb.ARMS]:
        b=originals[s,a];paired=[]
        for role,step in (('prefix5',5),('terminal10',10)):
            row=images[s,a,role];o=own[s,a,role];p=row['provenance']
            require(row['status']=='encoded' and key(p)==(s,a) and p['role']==role and
                all(row[k]==p[k] for k in ('root_slot','arm','role','selection_digest') if k in row) and
                all(p[k]==b[k] for k in ('root_sha256','task_id','blind_sha256')) and
                p['selection_digest']==b['physical_selection_digest'] and p['compiler_selection_sha256']==b['model_selection_sha256'] and
                p['compiler_authority']==rgb.physical.AUTHORITY and p['compiler_report_sha256']==bi['scheduler_report_sha256'] and
                p['Q']=='identity192' and p['predicted'] is None,'stable original task/root/blind/model/physical selection join')
            require(all(o[k]==p[k] for k in ('root_slot','arm','role','selection_digest','task_id','root_sha256','compiler_report_sha256')) and
                all(o[k]==row[k] for k in ('metrics','rawpixel','latent')),'actual RGB ownconsume receipt joins raw report row')
            cp=[c for c in b['checkpoint_references'] if c['step']==step and c['phase']=='plan']
            require(len(cp)==1 and cp[0]==p['checkpoint'],'actual selected prefix/terminal state/pixel/PNG provenance')
            require(rgb.physical.source.root_hash(np.asarray(b['raw_initial_state']),np.asarray(b['raw_goal']))==b['root_sha256'],
                'comparison raw root/goal stable identity')
            root=rgb.rgb.old.vector(p['root'],192);goal=rgb.rgb.old.vector(p['goal'],192)
            latent_identity=(root.tolist(),goal.tolist(),p['task_export'],p['task_sha256'],p['task_contract_sha256'])
            require(s not in root_latents or root_latents[s]==latent_identity,'same original native192 root/goal/task/Q for all strategies')
            root_latents[s]=latent_identity;m=row['metrics']
            require(m['root_Q']==float(np.mean((root-goal)**2)) and
                all(m[k] is None for k in ('predicted_Q','predicted_vs_real_mse','Q_improvement_sign_reversed')) and
                m['prediction_status']=='unknown_not_recomputed' and np.isfinite(m['actual_Q']) and m['actual_Q']>=0 and
                m['actual_Q_change']==m['actual_Q']-m['root_Q'],'same identity mean192 Q, prediction remains null')
            paired.append(deepcopy(o))
        prefix,terminal=paired
        require(prefix['metrics']['rawpixel_vs_prefix5_mse']==0 and prefix['metrics']['actual_Q_change_from_prefix5']==0 and
            terminal['metrics']['actual_Q_change_from_prefix5']==terminal['metrics']['actual_Q']-prefix['metrics']['actual_Q'],
            'own prefix to terminal Q/pixel convention')
        rows.append(dict(source_slot=s,arm=a,root_sha256=b['root_sha256'],task_id=b['task_id'],blind_sha256=b['blind_sha256'],
            physical_selection_digest=b['physical_selection_digest'],comparison_row=deepcopy(b),prefix5=prefix,terminal10=terminal,
            original_planner_terminal_Q=b['model_Q'],predicted_native192=None,predicted_prefix_Q=None,
            actual_terminal_minus_original_model_Q=terminal['metrics']['actual_Q']-b['model_Q']))
    differences={}
    for reference in (rgb.ARMS[0],rgb.ARMS[2],rgb.ARMS[3]):
        cells=[]
        for s in (0,1):
            d=next(r for r in rows if key(r,'source_slot')==(s,rgb.ARMS[1]));r=next(r for r in rows if key(r,'source_slot')==(s,reference))
            model=d['original_planner_terminal_Q']-r['original_planner_terminal_Q']
            actual=d['terminal10']['metrics']['actual_Q']-r['terminal10']['metrics']['actual_Q']
            cells.append(dict(source_slot=s,root_sha256=d['root_sha256'],original_model_Q_D_minus_reference=model,
                actual_terminal_Q_D_minus_reference=actual,
                actual_prefix_Q_D_minus_reference=d['prefix5']['metrics']['actual_Q']-r['prefix5']['metrics']['actual_Q'],
                terminal_model_actual_order_reversed=None if model==0 or actual==0 else bool(model*actual<0)))
        differences[reference]=dict(per_root=cells,mean_actual_terminal_Q_D_minus_reference=float(np.mean([c['actual_terminal_Q_D_minus_reference'] for c in cells])))
    return dict(schema=NAME+'.report',status='saved_comparison_RGB_joined_not_admitted',errors=[],
        original_comparison=deepcopy(base),rows=rows,D_minus_reference=differences,fixed_branch_denominator=8,fixed_image_denominator=16,
        resources=dict(new_neural_calls=0,new_physics_calls=0,inherited_RGB=deepcopy(report['resources']),
            inherited_comparison=deepcopy(base['resources']),warning='inherited costs are not added twice or equal total compute'),
        interpretation='Two reused moving roots only; model Q, actual RGB Q, native reward and object pose remain distinct. No CI/generalization/causal/P3 claim.',
        **comparison.FLAGS)


def build(comparison_report,comparison_sha,rgb_output,rgb_report_sha):
    require(comparison_sha==COMPARISON_SHA,'only MAIN-approved original comparison report')
    for h in (comparison_sha,rgb_report_sha):comparison.core.hash_value(h)
    bindings={}
    def bind(path,h):
        path=Path(path).absolute();require(not path.is_symlink() and sha(path)==h,'saved external/closure seal '+str(path))
        require(str(path) not in bindings or bindings[str(path)]==h,'nonconflicting closure');bindings[str(path)]=h
    for f,h in PINS.items():bind(f,h)
    path=Path(comparison_report).resolve();bind(path,comparison_sha);base=comparison.read(path,CAP)
    i=base['inputs']
    rebuilt=comparison.inputs(scheduler_output=i['scheduler_output'],scheduler_report_sha=i['scheduler_report_sha256'],
        physical_output=i['physical_output'],physical_report_sha=i['physical_report_sha256'],physical_source_sha=i['physical_source_sha256'])
    require(canonical(rebuilt)==canonical(base),'original saved comparison fully reproduced, not trusted self-assertions')
    output=Path(rgb_output).absolute();bind(output/'report.json',rgb_report_sha)
    receipt=rgb.consume(output,rgb_report_sha)  # own saved-only provider, never run/encode
    report=rgb.read(output/'report.json',rgb_report_sha)
    require(receipt['report_sha256']==rgb_report_sha,'actual own RGB report SHA')
    bind(report['prepared'],report['prepare_sha256']);p=rgb.read(report['prepared'],report['prepare_sha256'])
    result=assemble(base,report,receipt)
    for mapping in (base['inputs']['bindings_sha256'],p['bindings']):
        for f,h in mapping.items():bind(f,h)
    for name,h in report['artifact_sha256'].items():bind(rgb.rgb.old.local(output,name),h)
    for f in (Path(__file__),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bind(f,sha(f))
    for f,h in bindings.items():require(sha(f)==h,'final readback seal')
    result['inputs']=dict(comparison_report=str(path),comparison_sha256=comparison_sha,rgb_output=str(output),
        rgb_report_sha256=rgb_report_sha,rgb_prepare_sha256=report['prepare_sha256'],bindings_sha256=bindings)
    return result


def write_new(output,result):
    path=Path(output).absolute();payload=canonical(result)
    require(not path.exists() and len(payload)<=CAP and shutil.disk_usage(path.parent).free>=FLOOR+CAP,'NEW <=1MiB saved report +200MiB floor')
    path.mkdir()
    with (path/'report.json').open('xb') as f:require(f.write(payload)==len(payload),'short report');f.flush();os.fsync(f.fileno())
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(output=str(path),report_sha256=sha(path/'report.json'),bytes=len(payload),status=result['status'],**comparison.FLAGS)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('comparison-report','comparison-sha','rgb-output','rgb-report-sha','output'):p.add_argument('--'+k,required=True)
    a=vars(p.parse_args());output=a.pop('output');print(canonical(write_new(output,build(**a))).decode(),end='')


if __name__=='__main__':main()
