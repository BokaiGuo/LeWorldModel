"""Saved-only fixed8 CEM64 versus spectral24 teacher_full physical analysis.

Calls both complete physical.consume validators; never prepares or runs a model/env.
MAIN supplies actual terminal report SHAs and authorizes writing one NEW JSON file.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np

NAME = 'agent_CEM8_physical_analysis_20261028_v1'
CEM = 'agent_moving_CEM8_physical_20261028_v1'
SPECTRAL = 'capability_suffix_factorial_physical_20261027_v1'
SOURCE_SHA = {
    CEM: 'd9c72806141f1291aa1e7d088c0419035b34b9b529a452af2ae1ea0921b33fc2',
    SPECTRAL: '15125f4d86e5a6b3344f750205101a0e8a6f4a8dfba82ce46ce3d925f4ea8bd6',
}
ARMS = ('line_search_GN', 'full10_conditional_D2P')
VARIANTS = ('open_full', 'teacher_full', 'teacher_small')
ORDER = [(s,c,a) for s in (0,1) for c in ('off','on') for a in ARMS]
OUTPUT_CAP, FLOOR, PREPARE_CAP = 32*1024**2, 200*1024**2, 16*1024**2
CP_FIELDS = ('step','file_sha256','pixel_sha256','state','snapshot')
SCALARS = ('native7_distance','native7_reward','joint_xy_error','agent_xy_error',
           'object_xy_error','velocity_error','angle_wrapped_error','joint_xy_wrapped_angle_error')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def actual_hash(value):
    require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None and
            value != '0'*64, 'actual nonzero lowercase SHA256 required')
    return value


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False)+'\n').encode()


def sha(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'regular nonsymlink input')
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_bound(path, expected, cap):
    actual_hash(expected); p = Path(path)
    require(p.is_absolute() and p.stat().st_size <= cap, 'absolute bounded saved metadata')
    require(sha(p) == expected, 'saved metadata SHA')
    body = p.read_bytes()
    require(len(body) <= cap and hashlib.sha256(body).hexdigest() == expected, 'bounded read SHA')
    def pairs(items):
        out = {}
        for key,value in items:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    value = json.loads(body, object_pairs_hook=pairs,
                       parse_constant=lambda x:require(False,'nonfinite JSON'))
    require(isinstance(value,dict), 'JSON object')
    return value


def vector(value, size=7):
    require(isinstance(value,list) and len(value) == size and
            all(type(x) in (int,float) for x in value), 'literal finite native/raw vector')
    result = np.asarray(value,np.float64)
    require(np.isfinite(result).all(), 'finite vector')
    return result


def raw_commands(value, size):
    require(isinstance(value,list) and len(value) == size, 'literal raw command count')
    array = np.asarray([vector(v,2) for v in value],np.float64)
    require(np.isfinite(array.astype(np.float32)).all() and
            np.array_equal(array,array.astype(np.float32).astype(np.float64)), 'literal float32 raw, no scale/tanh/recast')
    return np.ascontiguousarray(array,np.float32).tobytes()


def metrics(goal, state):
    """Independent native implementation, INCLUDING the native multi-turn quirk."""
    s,g = vector(state),vector(goal); d = s-g
    absolute_angle = float(abs(d[4]))
    native_angle = min(absolute_angle,2*np.pi-absolute_angle)
    wrapped = float(abs(np.arctan2(np.sin(d[4]),np.cos(d[4]))))
    xy = float(np.linalg.norm(d[:4])); distance = float(np.linalg.norm(d))
    native = dict(native_success=bool(xy < 20 and native_angle < np.pi/9),
                  native_distance=distance,native_angle_term=float(native_angle),
                  negative_native_angle=bool(native_angle < 0),
                  wrapped_allxy_success=bool(xy < 20 and wrapped < np.pi/9),
                  wrapped_object_angle_error=wrapped,object_position_error=float(np.linalg.norm(d[2:4])),
                  object_pose=s[2:5].tolist(),object_success=None)
    return dict(native=native,native7_distance=distance,native7_reward=-distance,
                joint_xy_error=xy,agent_xy_error=float(np.linalg.norm(d[:2])),
                object_xy_error=float(np.linalg.norm(d[2:4])),velocity_error=float(np.linalg.norm(d[5:7])),
                angle_wrapped_error=wrapped,joint_xy_wrapped_angle_error=float(np.linalg.norm(np.r_[d[:4],wrapped])))


def same_checkpoint(left, right):
    require(all(left[k] == right[k] for k in CP_FIELDS), 'exact common checkpoint state/body/file/pixel, not filename')


def row_metrics(row, anchor):
    responses = row['responses']; cps = row['checkpoints']
    require(isinstance(responses,list) and len(responses) <= 5 and
            [x['step'] for x in responses] == list(range(1,len(responses)+1)), 'contiguous suffix responses')
    require(all(type(x['step']) is int for x in responses+cps), 'literal step ordinal')
    require(row['status'] in ('suffix_completed','native_stopped','engineering_failed','not_started'), 'terminal row only')
    require(len({cp['step'] for cp in cps}) == len(cps) and all(cp['step'] in (0,5,10) for cp in cps), 'unique actual checkpoints')
    cpmap = {cp['step']:cp for cp in cps}
    if 5 in cpmap:
        same_checkpoint(cpmap[5],anchor['checkpoint'])
    if row['start_state'] is not None:
        require(row['start_state'] == anchor['checkpoint']['state'], 'exact source prefix start state')
    if row.get('root_state') is not None and row.get('goal') is not None:
        body = np.r_[vector(row['root_state']),vector(row['goal'])].astype(np.float64).tobytes()
        require(hashlib.sha256(body).hexdigest() == row['root_sha256'], 'root native7/goal SHA')
    recomputed = []
    for response in responses:
        require(row['start_verified'] is True, 'verified original51 prefix required')
        calc = metrics(row['goal'],response['state'])
        require(response['native_metrics'] == calc['native'] and response['reward'] == calc['native7_reward'] and
                response['native_success'] is calc['native']['native_success'] and
                response['terminated'] is response['native_success'] and type(response['truncated']) is bool,
                'independent native7/reward/success/termination recomputation')
        delta = vector(response['state'])-vector(row['start_state'])
        require(np.array_equal(delta,vector(response['delta_native7'])) and
                response['delta_angle_wrapped'] == float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))), 'exact response delta')
        recomputed.append(dict(step=response['step'],state=deepcopy(response['state']),metrics=calc,
                               terminated=response['terminated'],truncated=response['truncated']))
    require(all(not r['terminated'] and not r['truncated'] for r in responses[:-1]), 'no transitions after native stop')
    if responses:
        last = responses[-1]
        require(row['final_state'] == last['state'] and row['final_metrics'] == last['native_metrics'] and
                all(row[k] is last[k] for k in ('native_success','terminated','truncated')), 'actual returned endpoint linkage')
    else:
        require(row['final_state'] is row['final_metrics'] is row['native_success'] is None, 'unknown endpoint not zero')
    if 10 in cpmap:
        require(len(responses) == 5 and cpmap[10]['state'] == responses[-1]['state'], 'H10 from actual fifth suffix return only')
    successful_execution = row['status'] in ('suffix_completed','native_stopped')
    if successful_execution:
        require(responses and row['first_error'] is None and row['secondary_errors'] == [] and
                row['suffix_returned'] == len(responses) and row['full_suffix'] is (len(responses) == 5) and
                list(cpmap) == [0,5]+([10] if len(responses) == 5 else []) and
                row['missing_checkpoints'] == ([] if len(responses) == 5 else [10]) and
                ((row['status'] == 'native_stopped') == (responses[-1]['terminated'] or responses[-1]['truncated'])),
                'real native endpoint, no padding/full-horizon substitution')
    checkpoints = [dict(step=cp['step'],metrics=metrics(row['goal'],cp['state'])) for cp in cps]
    observed = recomputed[-1]['metrics'] if recomputed else None
    return dict(status=row['status'],first_error=deepcopy(row['first_error']),secondary_errors=deepcopy(row['secondary_errors']),
                returned_suffix_steps=len(responses),missing_checkpoints=deepcopy(row['missing_checkpoints']),
                responses=recomputed,checkpoints=checkpoints,observed_endpoint=observed,
                endpoint=observed if successful_execution else None,
                horizon10=observed if successful_execution and 10 in cpmap else None)


def check_receipt(receipt, prepared, name, denominator):
    require(receipt['schema'] == name+'.consume' and receipt['fixed_denominator'] == denominator and
            type(receipt['fixed_denominator']) is int and len(receipt['rows']) == denominator and
            prepared['schema'] == name+'.prepare', 'full real physical consume ABI/denominator')
    require(receipt['status'] in ('partial_or_failed', 'CEM8_suffix_physical_recorded_not_admitted' if name == CEM else
                                 'factorial24_suffix_physical_recorded_not_admitted'), 'terminal consume status')
    require(all(receipt[k] is False for k in ('P1_admission','scientific_admission','independent_confirmation')) and
            receipt['new_neural_calls'] == 0, 'no admission upgrade')
    actual_hash(receipt['physical_report_sha256'])
    for key in ('compiler_report_sha256','consumer_report_sha256'):
        require(receipt[key] == prepared[key], 'consume/prepare source compiler and acceptance')
        actual_hash(receipt[key])
    require(len(prepared['plans']) == len(prepared['anchors']) == 8 and
            len(prepared['suffix_plans']) == denominator, 'complete prepared ancestry')


def identify(row, prepared, i, ordinal, cem, variant=None):
    original = prepared['plans'][i]; plan = row['selected_suffix']['plan']; key = ORDER[i]
    require(type(row['index']) is type(row['anchor_index']) is type(row['root_slot']) is int and
            row['index'] == ordinal and row['anchor_index'] == i and
            (row['root_slot'],row['condition'],row['prefix_arm'] if cem else row['arm']) == key and
            row['selected_suffix'] == prepared['suffix_plans'][ordinal], 'exact row key/ordinal/selected raw source')
    require((original['root_slot'],original['condition'],original['arm']) == key and
            all(row[k] == original[k] for k in ('task_id','selection_digest')), 'original task/selection join')
    require(type(plan['anchor']) is int and plan['anchor'] == i and type(plan['root_slot']) is int and
            (plan['root_slot'],plan['condition'],plan['prefix_arm'] if cem else plan['arm']) == key and
            plan['task_id'] == original['task_id'], 'selected plan anchor/key/task')
    prefix = plan['prefix_commands'] if cem else plan['prefix']
    suffix = plan['suffix_commands'] if cem else plan['suffix']
    require(raw_commands(prefix,5) == raw_commands(original['commands'][:5],5) and
            raw_commands(plan['commands'],10) == raw_commands(prefix+suffix,10), 'literal original raw prefix and own full command bytes')
    if cem:
        require(row['arm'] == plan['arm'] == 'teacher_full_CEM64' and type(plan['source_slot']) is int and
                plan['source_slot'] == key[0] and plan['blind_sha256'] == original['blind_sha256'] and
                plan['original_selection_digest'] == original['selection_digest'] and
                plan['whole_run_report_sha256'] == prepared['compiler_report_sha256'], 'own CEM64 source/context/selection identity')
    else:
        require(row['variant'] == plan['variant'] == variant and type(plan['index']) is int and plan['index'] == ordinal and
                plan['selection_digest'] == original['selection_digest'] and
                all(plan['original_plan'][k] == value for k,value in original.items()),
                'spectral24 original source/raw/variant ordinal, never observed4')


def summarize(values):
    n = len(values); available = [v for v in values if v is not None]; complete = len(available) == n
    return dict(denominator=n,available=len(available),unknown=n-len(available),
                mean={k:float(np.mean([v[k] for v in available])) for k in SCALARS} if complete else None,
                native_success_count=sum(v['native']['native_success'] for v in available) if complete else None,
                native_success_rate=float(np.mean([v['native']['native_success'] for v in available])) if complete else None,
                wrapped_joint_success_count=sum(v['native']['wrapped_allxy_success'] for v in available) if complete else None)


def group_summary(rows):
    result = {method:{endpoint:summarize([r[method][endpoint] for r in rows]) for endpoint in ('endpoint','horizon10')}
              for method in ('cem','spectral_teacher_full')}
    differences = {}
    for endpoint in ('endpoint','horizon10'):
        pairs = [(r['cem'][endpoint],r['spectral_teacher_full'][endpoint]) for r in rows]
        available = [(c,s) for c,s in pairs if c is not None and s is not None]
        differences[endpoint] = dict(denominator=len(rows),available=len(available),
            mean_CEM_minus_spectral={k:float(np.mean([c[k]-s[k] for c,s in available])) for k in SCALARS}
            if len(available) == len(rows) else None)
    result['paired_difference'] = differences
    return result


def assemble(cem, spectral, cem_prepare, spectral_prepare):
    """Pure RAM seam. Production must enter through analyze(), never bypass consume."""
    check_receipt(cem,cem_prepare,CEM,8); check_receipt(spectral,spectral_prepare,SPECTRAL,24)
    require(cem_prepare['plans'] == spectral_prepare['plans'] and cem_prepare['anchors'] == spectral_prepare['anchors'] and
            cem_prepare['physical_report_sha256'] == spectral_prepare['physical_report_sha256'], 'exact common original raw/source/planner/51-observation ancestry')
    rows = []
    for i,key in enumerate(ORDER):
        anchor = cem_prepare['anchors'][i]
        require(type(anchor['index']) is int and anchor['index'] == i and len(anchor['trace']) == 51, 'exact51 source anchor ordinal')
        c = cem['rows'][i]; identify(c,cem_prepare,i,i,True)
        for j,variant in enumerate(VARIANTS):
            identify(spectral['rows'][3*i+j],spectral_prepare,i,3*i+j,False,variant)
        s = spectral['rows'][3*i+1]
        unknown = []
        for field in ('root_state','root_sha256','goal','goal_pixel_sha256','goal_pose'):
            cv,sv = c.get(field),s.get(field)
            if cv is None or sv is None:
                unknown.append(field)
            else:
                require(cv == sv, 'exact paired physical root/goal/condition identity: '+field)
        for step in (0,5):
            cc = next((cp for cp in c['checkpoints'] if cp['step'] == step),None)
            sc = next((cp for cp in s['checkpoints'] if cp['step'] == step),None)
            if cc is None or sc is None:
                unknown.append('checkpoint'+str(step))
            else:
                same_checkpoint(cc,sc)
        cm,sm = row_metrics(c,anchor),row_metrics(s,anchor)
        # Incomplete identity cannot be silently scored as a paired task result.
        if unknown:
            cm['endpoint']=cm['horizon10']=sm['endpoint']=sm['horizon10']=None
        rows.append(dict(index=i,root_slot=key[0],condition=key[1],prefix_arm=key[2],spectral_ordinal=3*i+1,
                         task_id=c['task_id'],selection_digest=c['selection_digest'],original_plan=deepcopy(cem_prepare['plans'][i]),
                         anchor_sha256=hashlib.sha256(raw(anchor)).hexdigest(),identity_unknown=unknown,
                         cem=cm,spectral_teacher_full=sm))
    return dict(schema=NAME+'.analysis',status='saved_fixed8_analysis_not_admitted',fixed_denominator=8,
                known_roots=2,fresh_roots=0,new_independent_roots=0,new_independent_samples=0,
                P1_admission=False,scientific_admission=False,independent_confirmation=False,
                equal_total_compute=False,equal_FLOPs=False,freshness=False,new_neural_calls=0,new_physics_calls=0,
                budget_comparison=dict(equal=False,scope='saved actual per-producer resource receipts; no nested summation',
                    warning='CEM64 and spectral candidate budgets differ; paired task outcomes do not establish equal-compute superiority'),
                rows=rows,consumes=dict(cem8=deepcopy(cem),spectral24=deepcopy(spectral)),
                summaries=dict(all=group_summary(rows),
                    by_prefix_arm={a:group_summary([r for r in rows if r['prefix_arm']==a]) for a in ARMS},
                    by_condition={c:group_summary([r for r in rows if r['condition']==c]) for c in ('off','on')},
                    by_root_slot={str(s):group_summary([r for r in rows if r['root_slot']==s]) for s in (0,1)},
                    by_condition_prefix={c+'|'+a:group_summary([r for r in rows if r['condition']==c and r['prefix_arm']==a])
                                         for c in ('off','on') for a in ARMS}))


def merge(bindings, additions):
    for path,expected in additions.items():
        actual_hash(expected)
        require(Path(path).is_absolute() and (path not in bindings or bindings[path] == expected), 'nonconflicting absolute source/input seals')
        bindings[path] = expected


def analyze(cem_output, cem_report_sha, spectral_output, spectral_report_sha):
    """Complete saved consumers, independently pinned before lazy import. No execution API."""
    for h in (cem_report_sha,spectral_report_sha): actual_hash(h)
    bindings = {}; consumed = []; prepared = []; reports = []; directories = []
    merge(bindings,{str(Path(__file__).resolve()):sha(Path(__file__).resolve())})
    # CEM imports the spectral helper transitively: pin BOTH before any import.
    for name,expected in SOURCE_SHA.items():
        require(sha(Path(__file__).resolve().with_name(name+'.py')) == expected, 'frozen real physical consumer source')
    for name,output,expected,cap in ((CEM,cem_output,cem_report_sha,64*1024**2),
                                   (SPECTRAL,spectral_output,spectral_report_sha,192*1024**2)):
        directory = Path(output)
        require(directory.is_absolute() and directory.is_dir() and not directory.is_symlink(), 'absolute saved physical directory')
        source = Path(__file__).resolve().with_name(name+'.py')
        require(sha(source) == SOURCE_SHA[name], 'frozen real physical consumer source')
        rp = directory/'report.json'; report = read_bound(rp,expected,cap)
        require(report['schema'] == name+'.report' and report['status'] in
                ('partial_or_failed','CEM8_suffix_physical_recorded_not_admitted' if name == CEM else
                 'factorial24_suffix_physical_recorded_not_admitted'), 'terminal own physical report only')
        b = report['execution_binding']; p = read_bound(b['prepared'],b['prepare_sha256'],PREPARE_CAP)
        merge(bindings,p['bindings'])
        merge(bindings,{str(source):SOURCE_SHA[name],str(rp):expected,b['prepared']:b['prepare_sha256']})
        # The real complete consume verifies journal, PNG/pixels, exact51, literal raw,
        # full source/task closure and native responses. No fake receipt CLI exists.
        module = importlib.import_module('experiments.spectral_reachability.'+name)
        require(Path(module.__file__).resolve() == source, 'actual pinned module path')
        receipt = module.consume(str(directory),expected)
        require(receipt['rows'] == report['rows'] and receipt['physical_report_sha256'] == expected,
                'complete actual consumer result, not prepare/test/acceptance surrogate')
        if 'bindings' in receipt: merge(bindings,receipt['bindings'])
        for f in directory.iterdir():
            merge(bindings,{str(f):sha(f)})
        consumed.append(receipt); prepared.append(p); reports.append(report); directories.append(str(directory))
    result = assemble(*consumed,*prepared)
    # Original report errors/resources retained, not flattened into an invented budget.
    result['physical_report_diagnostics'] = {name:{k:deepcopy(v) for k,v in r.items() if k != 'rows'}
                                             for name,r in zip(('cem8','spectral24'),reports,strict=True)}
    result['input_directories'] = directories
    result['input_sha256'] = bindings
    for f,h in bindings.items(): require(sha(f) == h, 'final saved input/source drift')
    require(len(raw(result)) <= OUTPUT_CAP, 'bounded analysis output')
    return result


def free_bytes(path):
    return shutil.disk_usage(path).free


def write_new(path, result):
    p = Path(path); body = raw(result)
    require(len(body) <= OUTPUT_CAP, 'output byte cap')
    require(p.parent.is_dir() and not p.is_symlink(), 'existing output parent, no symlink')
    require(free_bytes(p.parent) >= FLOOR+len(body), '200MiB free floor')
    for directory in result.get('input_directories',[]):
        require(not p.resolve().is_relative_to(Path(directory).resolve()), 'no writes into producer output')
    require(str(p.resolve()) not in result.get('input_sha256',{}), 'no input overwrite')
    for f,h in result.get('input_sha256',{}).items():
        actual_hash(h)
        require(sha(f) == h, 'final input drift before exclusive write')
    with p.open('xb') as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(p.parent,os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return hashlib.sha256(body).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('cem-output','cem-report-sha','spectral-output','spectral-report-sha','output'):
        parser.add_argument('--'+flag,required=True)
    args = parser.parse_args(argv)
    # Refuse reentry before the potentially expensive read-only saved consumes.
    require(not Path(args.output).exists() and not Path(args.output).is_symlink(), 'NEW output required')
    result = analyze(args.cem_output,args.cem_report_sha,args.spectral_output,args.spectral_report_sha)
    print(json.dumps(dict(output=str(Path(args.output).absolute()),sha256=write_new(args.output,result),
                          status=result['status'],P1_admission=False),sort_keys=True))


if __name__ == '__main__':
    main()
