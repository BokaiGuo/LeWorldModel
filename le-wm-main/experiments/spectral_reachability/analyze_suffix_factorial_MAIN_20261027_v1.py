"""Fixed all24 real suffix endpoints; no learned metric or outcome selection."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import capability_suffix_factorial_physical_20261027_v1 as physical
from experiments.spectral_reachability import audit_capability_native_goal_MAIN_20261027_v1 as native

require,sha=physical.require,physical.sha
AXES=('allxy_l2','object_xy_l2','agent_xy_l2','native_angle_term','wrapped_angle','velocity_l2','state7_distance')


def analyze_rows(rows,original):
    require(len(rows)==24 and len(original)==8,'fixed all24/eight denominator')
    result=[]
    for i,row in enumerate(rows):
        base=original[i//3]
        require(row['index']==i and row['variant']==physical.VARIANTS[i%3] and
                all(row[k]==base[k] for k in ('root_slot','condition','arm','selection_digest','task_id')),'exact fixed identity')
        require(row['goal']==base['goal'],'same task goal')
        prefix=[x for x in base['checkpoints'] if x['step']==5]
        require(len(prefix)==1,'original prefix checkpoint')
        item={k:deepcopy(row[k]) for k in ('index','variant','root_slot','condition','arm','selection_digest','task_id','status')}
        item.update(complete=row['status']=='suffix_completed',final=None,baseline=native.metrics(base['goal'],base['final_state']),
                    prefix=native.metrics(base['goal'],prefix[0]['state']),minus_baseline=None,minus_prefix=None)
        # Engineering failures remain null; never silently impute old endpoints.
        if row['status'] in ('suffix_completed','native_stopped'):
            require(row['start_verified'] is True and row['start_state']==prefix[0]['state'] and
                    len(row['responses'])>0 and row['final_state']==row['responses'][-1]['state'],'actual observed suffix endpoint')
            final=native.metrics(row['goal'],row['final_state'])
            require(final['native_success'] is row['native_success'] and final['native_success'] is row['responses'][-1]['terminated'],'exact installed native success')
            item.update(final=final,minus_baseline={k:final[k]-item['baseline'][k] for k in AXES},
                        minus_prefix={k:final[k]-item['prefix'][k] for k in AXES})
        result.append(item)
    comparisons=[]
    for anchor in range(8):
        a,b,c=result[anchor*3:anchor*3+3]
        for left,right in ((a,b),(b,c)):
            valid=left['final'] is not None and right['final'] is not None
            comparisons.append(dict(anchor=anchor,root_slot=left['root_slot'],condition=left['condition'],arm=left['arm'],
                left=left['variant'],right=right['variant'],available=valid,
                right_minus_left={k:right['final'][k]-left['final'][k] for k in AXES} if valid else None))
    summaries={}
    for variant in physical.VARIANTS:
        rr=[r for r in result if r['variant']==variant];valid=[r for r in rr if r['final'] is not None]
        summaries[variant]=dict(denominator=8,observed=len(valid),complete=sum(r['complete'] for r in rr),
            unknown=8-len(valid),native_success=sum(r['final']['native_success'] for r in valid),
            wrapped_success=sum(r['final']['wrapped_success'] for r in valid),
            mean_minus_baseline={k:float(np.mean([r['minus_baseline'][k] for r in valid])) for k in AXES} if valid else None)
    return dict(rows=result,paired_comparisons=comparisons,summaries=summaries)


def run(directory,expected,output):
    print('All24 fixed endpoints, native task axes and original suffix reference; no best-condition selection.',flush=True)
    accepted=physical.consume(directory,expected)
    d=Path(directory).resolve();report=physical.read(d/'report.json',physical.CAP)
    p,_=physical.verify(report['execution_binding']['prepared'],report['execution_binding']['prepare_sha256'])
    original_dir=Path(p['arguments']['physical_output']);op=original_dir/'report.json'
    require(sha(op)==p['physical_report_sha256'],'original8 report SHA')
    original=physical.read(op,physical.CAP)['rows']
    result=analyze_rows(accepted['rows'],original)
    bindings=deepcopy(accepted['bindings'])
    for f in (Path(__file__),Path(native.__file__)):
        bindings[str(f.resolve())]=sha(f)
    result.update(schema='analyze_suffix_factorial_MAIN_20261027_v1.report',
        status='all24_native_axes_recorded',physical_report_sha256=expected,input_bindings=bindings,
        fixed_branches=24,known_roots=2,fresh_roots=0,new_neural_calls=0,new_physics_calls=0,P1_admission=False,
        interpretation='Exploratory fixed known-root ablation; original suffix reference differs in compiler history costs.',
        limitations=['No independent confirmation; 24 branches are not 24 independent roots.',
                     'Native raw angle formula retained including negative terms; wrapped metrics are diagnostics.',
                     'No latent predicted Q is substituted for native task success.',
                     'No best variant selected from outcomes; all variants and failed rows retained.'])
    for f,h in bindings.items():require(sha(f)==h,'final input/source seal')
    data=json.dumps(result,sort_keys=True,allow_nan=False).encode()
    with Path(output).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(status=result['status'],sha256=sha(output),summaries=result['summaries']),sort_keys=True))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('input','report-sha','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run(a.input,a.report_sha,a.output)
