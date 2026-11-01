"""Dynamic, read-only join of candidate cp10 rows to physical strict receipts."""
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p3_cp10_saved_source_closure_20261029_v1 as common
from experiments.spectral_reachability import agent_p3_future_cp10_receipt_join_20261029_v1 as reader
from experiments.spectral_reachability import agent_p3_same_time_cp10_rgb_bridge_20261029_v1 as original

NAME='agent_p3_dynamic_cp10_physical_inputs_20261029_v1'
ORDER=((0,'C'),(0,'Z'),(0,'P'),(1,'Z'),(1,'P'))
OLD_ROOTS={'a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7',
           'df038a122a660da57675fa281ec7dd358c9f9f67cb387aca9595f31711ca60ad'}
SCHEMAS={'C':('agent_p3_secant_commit_physical_20261029_v1','secant_commit5_native_recorded_NOT_admitted'),
         'ZP':('agent_p3_secant_controls_first5_physical_20261029_v1','all4_ZP_first5_native_recorded_NOT_admitted')}
need=common.need
raw=common.stage.raw


def physical(refs,kind):
    need(set(refs)=={'prepare','report','completion','receipt','source'},'physical reference vector')
    p,r,c,s=(reader.read(refs[k]) for k in ('prepare','report','completion','receipt'))
    common.data(refs['source']['path'],refs['source']['sha256'])
    name,status=SCHEMAS[kind]
    need([v['schema'] for v in (p,r,c,s)]==[name+'.prepare',name+'.report',name+'.completion',name+'.consume'], 'exact known physical ABI')
    need(r['status']==s['status']==status and c['status']=='published_within_cap', 'complete physical')
    need(s['physical_report_sha256']==c['report_sha256']==refs['report']['sha256'] and
         s['completion_sha256']==refs['completion']['sha256'] and s['completion']==c and
         r['execution_binding']['prepared']==c['prepared']==refs['prepare']['path'] and
         r['execution_binding']['prepare_sha256']==c['prepare_sha256']==refs['prepare']['sha256'], 'physical four-way join')
    need(r['rows']==s['rows'] and len(r['rows'])==len(p['plans'])==4 and s['fixed_denominator']==4 and
         r['errors']==s['errors']==[] and r['counts']==s['counts'] and r['inputs_final_verified'] is True, 'physical full rows and accounting')
    need(p['planner_authority']==r['execution_binding']['planner_authority']==s['planner_authority'], 'planner authority identity')
    need(all(all(v[k] is False for k in original.FLAGS) for v in (r,c,s)), 'all physical scientific flags false')
    need(0<=r['wall_seconds']<=c['inclusive_wall_seconds']<=original.controls.WALL, 'physical completion wall')
    for k in ('prepare','report','completion','source'):
        need(s['bindings'][refs[k]['path']]==refs[k]['sha256'], 'physical receipt input binding')
    need(all(row['index']==i and row['plan']==p['plans'][i] for i,row in enumerate(r['rows'])), 'all physical plans')
    return p,s


def assemble(specs, groups, source_class):
    """Input rows are untrusted: reconstruct identity from actual inherited physical rows."""
    need(source_class in ('historical','future_candidate','fixture'), 'explicit source class')
    need(set(groups)=={'C','ZP'} and len(specs)==5, 'fixed source denominator')
    loaded={k:physical(groups[k],k) for k in ('C','ZP')}
    roots={}; models=set()
    for i,(spec,(root,role)) in enumerate(zip(specs,ORDER,strict=True)):
        group='C' if role=='C' else 'ZP'; slot=0 if role=='C' else i-1
        prep,receipt=loaded[group];plan=prep['plans'][slot];actual=receipt['rows'][slot];anchor=prep['prefix_reference'][slot]
        need(spec['index']==i and spec['root_slot']==root and spec['role']==role and spec['source_slot']==slot and spec['source_kind']==group, 'slot identity')
        need(spec['source_report']==groups[group]['report'] and spec['source_receipt']==groups[group]['receipt'], 'exact physical provenance')
        need(actual['status']=='horizon_completed' and actual['root_slot']==root and actual['arm']==('H1' if role=='C' else role)
             and actual['task_id']==plan['task_id']==spec['base_task_id'] and actual['selection_digest']==plan['selection_digest']==spec['selection_digest']
             and actual['initial_exact'] is actual['prefix_reference_exact'] is True and actual['own_returned']==10
             and actual['prefix51']['complete'] is True and len(actual['prefix51']['frames'])==51 and actual['PNG']==66 and actual['missing_checkpoints']==[], 'native history and guard')
        task=common.stage.core.validate_task(plan['model_task']);need(raw(task)==raw(spec['cp5_task']),'exact cp5 task')
        need(task['binding']['model_sha256']==original.bridge.rgb.MODEL_TENSOR_SHA and np.array_equal(task['projection'],np.eye(192)), 'original model and projection')
        root_hash=task['binding']['root_sha256']; models.add(task['binding']['model_sha256'])
        if root in roots: need(roots[root]==root_hash,'same root binding')
        else: roots[root]=root_hash
        if source_class=='future_candidate':need(root_hash not in OLD_ROOTS,'historical root cannot be relabeled future')
        same=loaded['ZP'][0]['plans'][0 if root==0 else 2]
        need(all(raw(plan[k])==raw(same[k]) for k in ('model_task','base_task','provenance','prefix_commands','native_goal','native_goal_pose','native_root_sha256','native_goal_pixel_sha256')), 'same-root exact context/goal')
        commands=np.asarray(plan['commands'],dtype='<f4')
        need(commands.shape==(10,2) and commands.tobytes().hex()==spec['causal_raw10_bits'] and
             commands.tobytes()==np.asarray(spec['causal_raw10'],dtype='<f4').tobytes() and
             commands[:5].tobytes()==np.asarray(anchor['prefix_raw5'],dtype='<f4').tobytes() and
             original.controls.bytehash(commands)==plan['raw_float32_sha256'], 'literal raw10/prefix')
        tokens=original.controls.bridge.typed.tokens(commands,np.asarray(task['action_mean'],np.float32),np.asarray(task['action_std'],np.float32)).reshape(2,10)
        need(tokens.tobytes()==np.asarray(spec['normalized_action_history'],dtype='<f4').tobytes() and
             original.controls.bytehash(tokens)==spec['normalized_float32_sha256'], 'original normalization once')
        need([x['absolute_step'] for x in anchor['history_frames']]==[5,10,15] and raw(anchor['actual_context'])==raw(task['root_context']), 'anchor context')
        frames=[f for f in actual['trajectory'] if f['phase']=='plan' and f['overlay_index']==0 and f['step'] in (0,5,10)]
        hist=spec['history_sources'];need(len(hist)==3 and [x['absolute_step'] for x in hist]==[10,15,20] and [x['step'] for x in frames]==[0,5,10], 'three frame times')
        for f,h in zip(frames,hist,strict=True):
            need(f['condition']=='off' and all(raw(f[k])==raw(h[k]) for k in ('file_sha256','pixel_sha256','state','snapshot')), 'frame identity')
            common.image(h)
        for h,a in zip(hist[:2],anchor['history_frames'][1:],strict=True):
            need(all(raw(h[k])==raw(a[k]) for k in ('file_sha256','pixel_sha256','state','snapshot')), 'real prior history')
        need(hist[-1]['png']==spec['source']['path'] and all(raw(hist[-1][k])==raw(spec['source'][k]) for k in ('file_sha256','pixel_sha256','state','snapshot')), 'endpoint image')
        checkpoints=[x for x in actual['checkpoints'] if x['step']==10]
        need(len(checkpoints)==1 and checkpoints[0]=={k:frames[-1][k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}, 'checkpoint identity')
        need(actual['responses'][-1]['step']==10 and raw(actual['responses'][-1]['state'])==raw(frames[-1]['state']), 'terminal native response')
    need(len(roots)==2 and len(set(roots.values()))==2 and len(models)==1, 'two distinct roots / same model')
    return dict(schema=NAME+'.validated',source_class=source_class,slots=5,physical_fullstrict_inherited_not_rebuilt=True,
        source_authenticity_independently_verified=False,fresh_roots_certified=False,execution_authorized=False,scientific_admission=False)
