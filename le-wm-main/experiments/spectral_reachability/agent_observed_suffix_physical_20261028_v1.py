"""Four fresh native raw10 runs for actual-cp5 observed suffix reoptimization.

Own schema/authority; original native episode and hooks only, no model loading.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import shutil
import time
import numpy as np
from experiments.spectral_reachability import agent_nonlinear_suffix_physical_20261028_v1 as base
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v2 as compiler

NAME='agent_observed_suffix_physical_20261028_v1'
AUTH='MAIN_OBSERVED_SUFFIX4_RAW10_PHYSICS_ONLY'
RECORDED='observed_suffix4_physical_recorded_not_admitted'
mixed=base.mixed
ROOT,ORDER=base.ROOT,base.ORDER
require,sha,raw,read=base.require,base.sha,base.raw,base.read
source,moving,paired=base.source,base.moving,base.paired
# Own execution plans/proofs retain the physical adapter's canonical encoding.
# Compiler-owned hashes MUST use its distinct compact canonical encoding.
digest=mixed.core.digest
producer_digest=compiler.digest
CAP,FLOOR,FINAL,WALL=base.CAP,base.FLOOR,base.FINAL,base.WALL
META_CAP=128*1024**2
ARGS={'compile_output','compile_report_sha','compile_receipt','compile_receipt_sha','compile_source_sha'}


def contract():
    c=deepcopy(base.contract())
    c.update(schema=NAME+'.contract',arm='actual_cp5_observed_suffix_reoptimized_raw10',producer=compiler.NAME,
        producer_denominator=4,raw_authority='full real observed v2 consume plans; fixed original prefix5 plus new selected suffix5',
        objective='original full task/goal192/identity meanQ preserved; new observed suffix scores separate',
        producer_compute='actual72forward/32scalar reverse plus complete inherited midpoint/physical/model resources',
        prefix_reference='all51 warmup/plan<=5 triplets and cp5 raw/state/snapshot/PNG match original physical; no inheritance',
        prepare_cap_bytes=META_CAP,producer_metadata_cap_bytes=META_CAP)
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


class Store(base.Store):
    def __init__(self,path):
        self.path=Path(mixed.absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink()
            and self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=FLOOR+CAP,'NEW own native output32MiB plus200MiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent);self.previous=None;self.index=0


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and p['source_report_sha256']==mixed.SOURCE_SHA
        and len(p['plans'])==len(p['prefix_reference'])==4 and len(p['source_roots'])==len(p['source_slots'])==len(p['warm_traces'])==2,
        'own observed four plans/two original native roots')
    a=p['planner_authority'];require(set(p['arguments'])==ARGS,'explicit actual compile inputs')
    require(a['producer']==compiler.NAME and a['whole_strict_reconstructed'] is a['persisted_receipt_equal'] is True
        and a['complete_four'] is True and a['selection_uses_physics'] is a['P1_admission'] is False
        and a['plans_sha256']==digest(p['plans']) and a['actual_new_compute']=={k:compiler.F.caps(72,32) for k in ('attempted','completed')},'complete actual observed compiler authority')
    for k,v in p['arguments'].items():
        if k.endswith('_sha'):mixed.actual_hash(v)
        else:mixed.absolute(v)
        require(a[k]==v,'exact external actual compile arguments')
    for i,(plan,ref) in enumerate(zip(p['plans'],p['prefix_reference'],strict=True)):
        require(type(plan['index']) is type(plan['root_slot']) is int and plan['index']==i
            and (plan['root_slot'],plan['condition'])==ORDER[i] and plan['source_slot']==plan['root_slot']
            and plan['schema']==NAME+'.execution_plan' and plan['arm']==contract()['arm'] and plan['raw_steps']==10
            and plan['selection_uses_physics'] is plan['P1_admission'] is False,'own fixed ordered raw10 plans, no filtering')
        require(plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
            and plan['planner_report_sha256']==a['compile_report_sha'] and plan['planner_receipt_sha256']==a['compile_receipt_sha'], 'own immutable compiler-derived plan')
        task=compiler.core.validate_task(plan['model_task']);commands=mixed.raw10(plan['commands'])
        require(plan['model_task_binding']==task['binding'] and plan['task_id']==task['binding']['task_id']
            and np.array_equal(task['projection'],np.eye(192))
            and task['action_mean']==compiler.F.source.kernel.ACTION_MEAN.tolist()
            and task['action_std']==compiler.F.source.kernel.ACTION_STD.tolist(), 'unchanged full original task/goal/Q/normalizer')
        require(hashlib.sha256(commands.tobytes()).hexdigest()==plan['raw_float32_sha256']
            and compiler.F.bits(commands[:5])==plan['prefix_bits']==ref['raw_prefix_bits']
            and plan['prefix_reference_sha256']==digest(ref),'new full raw10 and original prefix bytes')
        native=p['source_roots'][plan['root_slot']]
        require(all(plan[k]==native[v] for k,v in (('native_goal','goal'),('native_goal_pose','goal_pose'),('native_root_sha256','root_sha256')))
            and plan['native_goal_pixel_sha256']==native['goal_pixels'][plan['condition']], 'original physical root/goal/renderer condition')
        require(ref['index']==i and ref['complete'] is True and len(ref['frames'])==51 and ref['checkpoint']['step']==5
            and ref['checkpoint']['state']==ref['checkpoint']['snapshot']['state'],'complete authenticated original prefix/cp5')


def assemble(arguments,c,r,u,bp,br,data):
    """Pure real-schema join after actual compiler.consume and native authority checks."""
    require(c['schema']==compiler.NAME+'.prepare' and c['contract']==compiler.contract() and c['production_ready'] is True
        and r['schema']==compiler.NAME+'.report' and r['status']=='observed_suffix4_recompiled_not_admitted'
        and r['errors']==[] and r['contract']==c['contract'] and r['fixed_denominator']==r['known_roots']*2==4,'complete actual observed compile, not old all168 schema')
    require(u['schema']==compiler.NAME+'.consume' and u['status']=='observed_suffix4_verified_science_blocked' and u['complete'] is True
        and u['report_sha256']==arguments['compile_report_sha'] and u['fixed_denominator']==4 and u['known_roots']==2
        and raw(u['rows'])==raw(r['rows']) and len(u['plans'])==len(c['rows'])==len(r['rows'])==4,'complete actual four observed plans')
    require(all(obj[k] is False for obj in (r,u) for k in ('P1_admission','scientific_admission','freshness','physical_release'))
        and r['source_verified'] is r['model_unchanged'] is True and r['model_before']==r['model_after']
        and r['model_before']['original_state_dict']==c['expected_model_hash']
        and u['actual_new_compute']=={k:compiler.F.caps(72,32) for k in ('attempted','completed')},'actual scalar compute/frozen model, no release')
    base.validate_execution(bp)
    require(br['schema']==base.NAME+'.report' and br['status']==base.RECORDED and br['errors']==[] and len(br['rows'])==4
        and c['original_physical_authority']==bp['planner_authority']==u['original_physical_authority'],'exact original native prefix authority')
    plans=[];refs=[]
    for i,(spec,row,chosen,old,actual) in enumerate(zip(c['rows'],r['rows'],u['plans'],bp['plans'],br['rows'],strict=True)):
        key=ORDER[i];task=compiler.core.validate_task(spec['request']['task']);binding=task['binding'];inc=spec['request']['incumbent']
        require(type(spec['index']) is type(row['index']) is type(chosen['slot']) is int and spec['index']==row['index']==chosen['slot']==i
            and (spec['root_slot'],spec['condition'])==(chosen['root_slot'],chosen['condition'])==key
            and spec['status']=='ready' and row['status']=='sealed' and raw(row['result']['plan'])==raw(chosen),'all4 observed original ordered slot joins')
        require(chosen['schema']==compiler.NAME+'.plan' and chosen['selection_digest']==producer_digest({k:v for k,v in chosen.items() if k!='selection_digest'})
            and chosen['model_task_binding']==binding==old['model_task_binding'] and chosen['task_id']==binding['task_id']
            and chosen['goal_sha256']==producer_digest(task['goal']) and chosen['observed_context_sha256']==producer_digest(spec['request']['observed_context'])
            and chosen['parent_plan_sha256']==spec['parent_plan_sha256']
            and chosen['parent_selection_digest']==spec['parent_selection_digest']==old['producer_selection_digest']
            and chosen['midpoint_row_binding_sha256']==spec['midpoint_row_binding_sha256'], 'exact full task/observed context/parent model witness')
        require(all(chosen[k] is False for k in ('P1_admission','physical_release','freshness')),'model-only plan flags')
        a=mixed.raw10(chosen['commands']);oa=mixed.raw10(old['commands'])
        require(compiler.F.bits(a)==chosen['raw_float32_bits'] and compiler.F.bits(a[:5])==chosen['prefix_bits']==inc['prefix_bits']==compiler.F.bits(inc['prefix'])
            and a[:5].tobytes()==oa[:5].tobytes() and inc['parent_prefix_index']==old['selected_index']
            and inc['parent_iteration']==old['iterate_index'],'original prefix5 unmodified, only new suffix5')
        require(type(chosen['winner_iteration']) is int and 0<=chosen['winner_iteration']<=8
            and type(chosen['new_suffix_Q']) in (int,float) and np.isfinite(chosen['new_suffix_Q']) and chosen['new_suffix_Q']>=0,
            'finite original-Q observed suffix iterate; no outcome-based threshold')
        native=bp['source_roots'][key[0]]
        require(actual['index']==i and actual['plan']==old and actual['status']=='horizon_completed' and actual['initial_exact'] is True
            and actual['root_state']==native['root_state'] and actual['root_sha256']==native['root_sha256']
            and actual['goal']==native['goal'] and actual['goal_pose']==native['goal_pose']
            and actual['goal_pixel_sha256']==native['goal_pixels'][key[1]],'complete original physical native identity, never copied as new outcome')
        cp=[x for x in actual['checkpoints'] if x['step']==5]
        require(len(cp)==1 and actual['prefix51']['complete'] is True and len(actual['prefix51']['frames'])==51,'genuine original full prefix51 and cp5 required')
        ref=dict(index=i,complete=True,raw_prefix_bits=compiler.F.bits(oa[:5]),frames=deepcopy(actual['prefix51']['frames']),checkpoint=deepcopy(cp[0]))
        refs.append(ref)
        plan=dict(schema=NAME+'.execution_plan',index=i,root_slot=key[0],source_slot=key[0],condition=key[1],arm=contract()['arm'],
            commands=deepcopy(chosen['commands']),raw_float32_sha256=hashlib.sha256(a.tobytes()).hexdigest(),raw_steps=10,prefix_bits=chosen['prefix_bits'],
            task_id=binding['task_id'],model_task=deepcopy(task),model_task_binding=deepcopy(binding),
            source_report_sha256=mixed.SOURCE_SHA,planner_report_sha256=arguments['compile_report_sha'],planner_receipt_sha256=arguments['compile_receipt_sha'],
            selected_index=old['selected_index'],iterate_index=chosen['winner_iteration'],new_suffix_Q=chosen['new_suffix_Q'],
            observed_incumbent_Q=chosen['observed_incumbent_Q'],original_model_terminal_Q=old['nonlinear_terminal_Q'],
            compiler_plan=deepcopy(chosen),prefix_reference_sha256=digest(ref),native_goal=deepcopy(native['goal']),
            native_goal_pose=deepcopy(native['goal_pose']),native_root_sha256=native['root_sha256'],native_goal_pixel_sha256=native['goal_pixels'][key[1]],
            selection_uses_physics=False,P1_admission=False)
        plan['selection_digest']=digest(plan);plans.append(plan)
    authority=dict(producer=compiler.NAME,**deepcopy(arguments),whole_strict_reconstructed=True,persisted_receipt_equal=True,complete_four=True,
        actual_new_compute=deepcopy(u['actual_new_compute']),inherited_resources=deepcopy(c['inherited_resources']),
        selection_uses_physics=False,P1_admission=False,plans_sha256=digest(plans))
    p=dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),plans=plans,prefix_reference=refs,
        planner_authority=authority,bindings={},source_report_sha256=mixed.SOURCE_SHA,
        source_slots=deepcopy(bp['source_slots']),warm_traces=deepcopy(bp['warm_traces']),source_roots=deepcopy(bp['source_roots']))
    require(p['source_slots']==[x['slot'] for x in data['roots']] and p['source_roots']==mixed.allmode.root_metadata(data)
        and p['warm_traces']==[x['warm_trace'] for x in data['roots']],'complete original source identity/warmup')
    validate_execution(p);return p


def bound(c,path,expected):
    path=mixed.absolute(str(path));require(sha(path)==expected==c['bindings'].get(path),'exact inherited actual authority SHA: '+path)
    return read(path,META_CAP)


def reference(c):
    """Follow real midpoint -> terminal -> native schema, never guess old output."""
    a=c['arguments'];mr=bound(c,Path(a['midpoint_output'])/'report.json',a['midpoint_report_sha'])
    mp=bound(c,mr['prepared'],mr['prepare_sha256']);ta=mp['arguments']
    tr=bound(c,Path(ta['terminal_output'])/'report.json',ta['terminal_report_sha'])
    tp=bound(c,tr['prepared'],tr['prepare_sha256']);pa=tp['arguments']
    br=bound(c,Path(pa['physical_output'])/'report.json',pa['physical_report_sha'])
    bp=bound(c,pa['physical_prepared'],pa['physical_prepare_sha'])
    receipt=bound(c,pa['physical_receipt'],pa['physical_receipt_sha'])
    # compiler.consume has already performed full midpoint/terminal/native consumes.
    # The exact persisted native receipt and bound report are joined, not reissued.
    require(receipt['schema']==base.NAME+'.consume' and receipt['status']==base.RECORDED and receipt['errors']==[]
        and receipt['physical_report_sha256']==pa['physical_report_sha'] and raw(receipt['rows'])==raw(br['rows'])
        and br['execution_binding']['prepared']==pa['physical_prepared'] and br['execution_binding']['prepare_sha256']==pa['physical_prepare_sha']
        and receipt['planner_authority']==bp['planner_authority']==c['original_physical_authority'], 'actual complete original native strict/report/source plan')
    return bp,br


def material(arguments):
    require(type(arguments) is dict and set(arguments)==ARGS,'explicit external actual compile report/receipt/source SHA')
    for k,v in arguments.items():
        if k.endswith('_sha'):mixed.actual_hash(v)
        else:mixed.absolute(v)
    output=Path(arguments['compile_output']);receipt=Path(arguments['compile_receipt']);code=Path(compiler.__file__).resolve()
    external={str(code):arguments['compile_source_sha'],str(output/'report.json'):arguments['compile_report_sha'],str(receipt):arguments['compile_receipt_sha']}
    for f,h in external.items():require(sha(f)==h,'external actual observed compiler SHA before consume')
    u=compiler.consume(output,arguments['compile_report_sha'])
    require(raw(u)==raw(read(receipt,META_CAP)),'entire persisted real observed compile receipt equal')
    r=read(output/'report.json',META_CAP);c=read(r['prepared'],META_CAP)
    require(sha(r['prepared'])==r['prepare_sha256'] and c['bindings'][str(code)]==arguments['compile_source_sha'],'actual compile preparation/source identity')
    bp,br=reference(c);data=mixed.source_material();p=assemble(arguments,c,r,u,bp,br,data)
    bindings=deepcopy(c['bindings']);mixed.merge(bindings,data['bindings']);mixed.merge(bindings,external)
    mixed.merge(bindings,{r['prepared']:r['prepare_sha256'],**{str(output/n):h for n,h in r['artifact_sha256'].items()}})
    for f in [*own_files(),Path(base.__file__).resolve(),Path(mixed.__file__).resolve(),Path(mixed.allmode.__file__).resolve()]:mixed.merge(bindings,{str(f):sha(f)})
    path,h=mixed.allmode.constructor_contract();mixed.merge(bindings,{path:h});p['bindings']=bindings
    for f,h in bindings.items():require(sha(f)==h,'complete original/task/model/source closure')
    require(len(raw(p))<=META_CAP,'128MiB metadata cap without truncating source bindings');return p,data


def write_new(path,value,cap=META_CAP):
    path=Path(mixed.absolute(str(path)));payload=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists() and not path.is_symlink()
        and len(payload)<=cap and shutil.disk_usage(path.parent).free>=FLOOR+len(payload),'NEW bounded own metadata')
    with path.open('xb') as f:require(f.write(payload)==len(payload),'full metadata');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(path.parent);return dict(output=str(path),sha256=sha(path),bytes=len(payload))


def prepare(output,arguments):
    require(not Path(output).exists(),'exclusive prepare before strict rebuild');p,_=material(arguments);return write_new(output,p)


def verify(prepared,expected):
    mixed.actual_hash(expected);mixed.absolute(str(prepared));require(sha(prepared)==expected,'actual own native prepare SHA')
    p=read(prepared,META_CAP);now,data=material(p['arguments'])
    require(raw(p)==raw(now) and sha(prepared)==expected,'whole original source/compile/prefix preparation reconstruction');return p,data


def prefix_match(row,ref):
    """Actual own render IDs differ, all physical/RGB bytes and history must not."""
    frames=row['prefix51']['frames'];n=len(frames)
    require(n<=51 and row['prefix51']['complete'] is (n==51),'all original prefix frames, no padding')
    for actual,expected in zip(frames,ref['frames'][:n],strict=True):
        for key in ('pre','post'):require(raw(actual[key])==raw(expected[key]),'prefix native snapshot/renderer/goal identity')
        fields=('phase','step','condition','overlay_index','file_sha256','pixel_sha256','state','snapshot')
        require(raw({k:actual['receipt'][k] for k in fields})==raw({k:expected['receipt'][k] for k in fields}),'prefix raw state/snapshot/PNG exact')
    cp=[f for f in row['checkpoints'] if f['step']==5]
    if cp:
        fields=('step','file_sha256','pixel_sha256','state','snapshot')
        require(len(cp)==1 and all(raw(cp[0][k])==raw(ref['checkpoint'][k]) for k in fields),'cp5 exact original physical state/snapshot/PNG')
    if row['own_returned']>=5:require(len(cp)==1 and n==51,'full actual cp5 and51 prefix frames')
    return n==51 and len(cp)==1


class Meter(base.Meter):
    """Local causal guard before step6 intent; original accounting unchanged."""
    def guard_context(self,row,ref):self.row,self.ref=row,ref;self.guarded=False
    def call(self,kind,fn,*,cleanup=False,**fields):
        if kind=='public_step' and fields.get('phase')=='plan' and fields.get('step')==6:
            require(not self.guarded and self.row['index']==self.where['branch'],'single own cp5 guard context')
            proof=dict(prefix_reference_sha256=digest(self.ref),prefix51_sha256=digest(self.row['prefix51']),
                checkpoint_sha256=digest([c for c in self.row['checkpoints'] if c['step']==5]))
            try:
                ops=[o for o in self.receipts if o['resource']=='public_step' and o['branch']==self.row['index'] and o['phase']=='plan']
                require([o['step'] for o in ops]==[1,2,3,4,5] and all(o['status']=='returned' for o in ops),'exact actual prefix5 returned before suffix')
                require(prefix_match(self.row,self.ref),'authenticated full51/cp5 before any suffix dispatch')
            except BaseException as exc:
                self.store.emit('cp5_guard',**self.where,**proof,accepted=False,error=mixed.base.error('cp5_guard',exc))
                raise
            self.store.emit('cp5_guard',**self.where,**proof,accepted=True,error=None)
            self.guarded=True
        return super().call(kind,fn,cleanup=cleanup,**fields)


def audit_guard(r,events,p):
    for i,(row,ref) in enumerate(zip(r['rows'],p['prefix_reference'],strict=True)):
        scoped=[e for e in events if e.get('branch')==i];guards=[e for e in scoped if e['kind']=='cp5_guard']
        suffix=[e for e in events if e['kind']=='operation_dispatch_intent' and e['receipt']['branch']==i
            and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan' and e['receipt'].get('step',0)>=6]
        if not guards:
            require(not suffix,'no suffix intent without causal cp5 guard');continue
        require(len(guards)==1,'single causal cp5 guard per branch');g=guards[0]
        require(set(g)=={'kind','index','previous_sha256','event_sha256','branch','root_slot','condition','arm','selection_digest',
            'prefix_reference_sha256','prefix51_sha256','checkpoint_sha256','accepted','error'} and type(g['accepted']) is bool,'exact cp5 guard event fields')
        require(g['prefix_reference_sha256']==digest(ref) and g['prefix51_sha256']==digest(row['prefix51'])
            and g['checkpoint_sha256']==digest([c for c in row['checkpoints'] if c['step']==5]),'actual causal cp5 proof buffers')
        frames=[e for e in scoped if e['kind']=='fresh_observation' and (e['receipt']['phase']=='warmup' or e['receipt']['step']<=5)]
        prefix_ops=[e for e in events if e['kind']=='operation_returned' and e['receipt']['branch']==i
            and e['receipt']['resource']=='public_step' and e['receipt'].get('phase')=='plan' and e['receipt'].get('step',0)<=5]
        require(len(frames)==51 and len(prefix_ops)==5 and max(e['index'] for e in frames+prefix_ops)<g['index'],'all native prefix frames/returns before guard')
        if g['accepted']:
            require(g['error'] is None and prefix_match(row,ref) and all(g['index']<e['index'] for e in suffix),'successful guard before every suffix intent')
            if suffix:require(suffix[0]['receipt']['step']==6,'suffix starts exactly at6')
        else:require(not suffix and g['error'] is not None and row['status']=='engineering_failed','failed cp5 guard forbids suffix dispatch')


def audit_all_labels(r,events,p):
    """Every returned native7 label uses its authenticated actual phase goal."""
    for i,(row,plan,ref) in enumerate(zip(r['rows'],p['plans'],p['prefix_reference'],strict=True)):
        scoped=[e for e in events if e.get('branch')==i]
        placements=[e for e in scoped if e['kind']=='placement_recipe']
        attached=[e for e in scoped if e['kind']=='goal_attached']
        warm_goal=ref['frames'][0]['pre']['goal_state']
        if placements:
            require(len(placements)==1 and placements[0]['native_goal']==warm_goal
                and placements[0]['slot']==p['source_slots'][plan['root_slot']], 'canonical original seed placement goal and root slot')
        if row['initial_exact']:
            require(len(attached)==1 and attached[0]['goal']==plan['native_goal']
                and attached[0]['goal_pose']==plan['native_goal_pose']
                and attached[0]['goal_rgb_sha256']==plan['native_goal_pixel_sha256'],'canonical final task bind, not arbitrary event goal')
        else:require(not attached,'no goal binding claimed without root identity')
        for e in scoped:
            if e['kind']=='fresh_observation' and e['receipt']['phase']=='plan':
                require(len(attached)==1 and attached[0]['index']<e['index'],'actual final binding before plan0 RGB')
            if e['kind']!='returned_transition':continue
            phase=e['phase'];require(phase in ('warmup','plan') and len(placements)==1,'every return has actual placement/phase')
            require(placements[0]['index']<e['index'],'placement before native return')
            if phase=='warmup':
                require(not attached or e['index']<attached[0]['index'],'warmup before task goal binding')
                goal=warm_goal
            else:
                require(len(attached)==1 and attached[0]['index']<e['index'],'task goal attached before every prefix/suffix return')
                goal=plan['native_goal']
            require(raw(e['goal_state'])==raw(goal) and raw(e['state'])==raw(e['snapshot']['state']),'full native goal/state identity')
            calc=mixed.metrics(goal,e['state'])
            require(type(e['reward']) is float and e['reward']==calc['native7_reward']
                and type(e['terminated']) is bool and e['terminated'] is calc['native_success']
                and type(e['truncated']) is bool,'all warmup/plan native7 reward + jointXY/angle success labels')

def rows_for(p):
    return [dict(index=i, root_slot=plan['root_slot'], condition=plan['condition'], arm=plan['arm'],
        task_id=plan['task_id'], selection_digest=plan['selection_digest'], plan=deepcopy(plan),
        status='not_started', first_error=None, secondary_errors=[], warmup_attempted=0, warmup_returned=0,
        own_attempted=0, own_returned=0, PNG=0, initial_exact=False, trajectory=[], checkpoints=[],
        missing_checkpoints=[0, 5, 10], last_observation=None, responses=[], final_state=None,
        final_metrics=None, native_success=None, terminated=None, truncated=None,
        prefix51=dict(complete=False, frames=[]), source_unpaired_on_steps=[], prefix_reference_exact=False,
        evidence_origin='new_native_execution_known_root') for i, plan in enumerate(p['plans'])]


def execute(p, data, store, factory, *, authorization, prepared, prepare_sha256, started=None):
    require(authorization == AUTH, 'MAIN explicit physical authority before callbacks')
    mixed.actual_hash(prepare_sha256); validate_execution(p)
    require(p['source_slots'] == [r['slot'] for r in data['roots']] and
            p['source_roots'] == mixed.allmode.root_metadata(data) and
            p['warm_traces'] == [r['warm_trace'] for r in data['roots']], 'actual source identity/warmup history')
    meter = Meter(store, started); rows = rows_for(p); errors = []; seen = []
    binding = dict(prepared=mixed.absolute(str(prepared)), prepare_sha256=prepare_sha256,
        source_report_sha256=mixed.SOURCE_SHA, planner_authority=deepcopy(p['planner_authority']))
    def fail(stage, exc, row=None):
        item = mixed.base.error(stage, exc); errors.append(item)
        if row is not None:
            if row['first_error'] is None: row['first_error'] = deepcopy(item)
            else: row['secondary_errors'].append(deepcopy(item))
            row['status'] = 'engineering_failed'
    try:
        store.put('execution_binding.json', binding)
        store.emit('run_started', prepare_sha256=prepare_sha256, contract=contract())
        for row, plan in zip(rows, p['plans'], strict=True):
            meter.where = dict(branch=row['index'], **{k: row[k] for k in
                ('root_slot', 'condition', 'arm', 'selection_digest')})
            meter.guard_context(row,p['prefix_reference'][row['index']])
            env = None; row['status'] = 'running'
            try:
                env = meter.call('constructor', factory)
                require(all(env is not prior and env.unwrapped is not prior.unwrapped for prior in seen),
                        'fresh native env per arm'); seen.append(env)
                mixed.validate_env(env.unwrapped)
                require(env.unwrapped.render_action is False and env.unwrapped.render_size == 224 and
                        env.unwrapped.render_mode == 'rgb_array', 'original renderer/raw control semantics')
                with mixed.native_hooks(env, meter, row['secondary_errors']):
                    mixed.episode(env, data['roots'][plan['root_slot']], data['source_plan'], plan, row, meter, store)
                require(not row['secondary_errors'], 'native hooks restored')
            except mixed.allmode.NativeStop as exc:
                row.update(status='native_stopped', stop_reason=str(exc))
            except BaseException as exc:
                fail('branch', exc, row)
            finally:
                try: row['prefix_reference_exact'] = prefix_match(row, p['prefix_reference'][row['index']])
                except BaseException as exc: fail('prefix_reference', exc, row)
                if env is not None:
                    try: meter.call('close', env.close, cleanup=True)
                    except BaseException as exc: fail('close', exc, row)
                if row['secondary_errors'] and row['first_error'] is None:
                    fail('cleanup', ValueError('native cleanup secondary error'), row)
                for phase, label in (('warmup', 'warmup'), ('plan', 'own')):
                    ops = [op for op in meter.receipts if op['resource'] == 'public_step' and
                           op.get('branch') == row['index'] and op['phase'] == phase]
                    row[label + '_attempted'] = len(ops)
                    row[label + '_returned'] = sum(op['status'] == 'returned' for op in ops)
            store.emit('branch_closed', receipt=row)
    except BaseException as exc:
        fail('execution', exc)
    source_ok = False
    try:
        for file, expected in p['bindings'].items(): require(sha(file) == expected, 'final full input/source closure')
        source_ok = True
    except BaseException as exc:
        fail('final_inputs', exc)
    errors.extend(meter.secondary)
    if time.monotonic() - meter.started >= WALL: fail('final_wall', ValueError('900s wall'))
    good = not errors and source_ok and all(r['status'] in ('horizon_completed', 'native_stopped') for r in rows)
    status = RECORDED if good else 'partial_or_failed'
    try: store.emit('run_finished', status=status, counts=meter.counts, errors=errors)
    except BaseException as exc: fail('final_journal', exc); status = 'partial_or_failed'
    report = dict(schema=NAME + '.report', status=status, contract=contract(), execution_binding=binding,
        rows=rows, errors=errors, counts=deepcopy(meter.counts), actual_operation_receipts=deepcopy(meter.receipts),
        fixed_denominator=4, known_roots=2, fresh_roots=0, inherited=0, inputs_final_verified=source_ok,
        new_neural_calls=0, new_independent_samples=0, P1_admission=False, P3_admission=False,
        scientific_admission=False, freshness=False, equal_total_compute=False, equal_FLOPs=False,
        training=False, independent_confirmation=False,
        total_internal_physics_scope='metered_native_calls' if meter.counts['constructor']['attempted'] ==
            meter.counts['constructor']['completed'] else 'unknown_constructor_work',
        journal_sha256=None, execution_binding_sha256=None, journal_chain_tail=store.previous,
        wall_seconds=time.monotonic() - meter.started)
    for name, key in (('journal.jsonl', 'journal_sha256'), ('execution_binding.json', 'execution_binding_sha256')):
        try: report[key] = sha(store.path / name)
        except BaseException as exc: fail('final_' + key, exc); report['status'] = 'partial_or_failed'
    try: store.put('report.json', report, final=True)
    except BaseException as exc: fail('final_report', exc); report['status'] = 'partial_or_failed'
    return report


def validate_saved(r, events, p):
    """Own-schema native trace validator, never a projected mixedmode report."""
    validate_execution(p)
    require(r['schema'] == NAME + '.report' and r['contract'] == contract() and
            r['status'] in (RECORDED, 'partial_or_failed') and r['fixed_denominator'] == len(r['rows']) == 4 and
            r['known_roots'] == 2 and r['fresh_roots'] == r['inherited'] == 0 and
            r['execution_binding']['planner_authority'] == p['planner_authority'] and
            r['execution_binding']['source_report_sha256'] == mixed.SOURCE_SHA, 'own full4 observed suffix physical report authority')
    require(all(r[k] is False for k in ('P1_admission', 'P3_admission', 'scientific_admission', 'freshness',
            'equal_total_compute', 'equal_FLOPs', 'training', 'independent_confirmation')) and
            r['new_neural_calls'] == r['new_independent_samples'] == 0, 'known-root engineering only')
    require(events and events[0]['kind'] == 'run_started' and events[0]['contract'] == contract() and
            events[0]['prepare_sha256'] == r['execution_binding']['prepare_sha256'] and
            events[-1]['kind'] == 'run_finished' and all(events[-1][k] == r[k] for k in ('status', 'counts', 'errors')),
            'actual terminal bookends')
    allowed = {'run_started', 'run_finished', 'operation_dispatch_intent', 'operation_returned', 'operation_failed',
        'placement_recipe', 'goal_attached', 'invalid_observation', 'fresh_observation', 'returned_transition',
        'own_response', 'branch_closed', 'cp5_guard'}
    tail = None; intents = {}; terminals = {}; stack = []; closed = 0
    for i, event in enumerate(events):
        require(event['kind'] in allowed and type(event['index']) is int and event['index'] == i and
                event['previous_sha256'] == tail and event['event_sha256'] ==
                source.digest({k: v for k, v in event.items() if k != 'event_sha256'}), 'known physical event/hash chain')
        tail = event['event_sha256']
        if event['kind'] == 'branch_closed':
            require(closed < 4 and event['receipt'] == r['rows'][closed], 'ordered exact full branch closure')
            closed += 1
        if 'branch' in event:
            require(type(event['branch']) is int and event['branch'] == closed and closed < 4 and
                    all(event[k] == r['rows'][closed][k] for k in ('root_slot', 'condition', 'arm', 'selection_digest')),
                    'active exact branch identity')
        if event['kind'] not in ('operation_dispatch_intent', 'operation_returned', 'operation_failed'): continue
        op = event['receipt']; n = op['index']; b = op['branch']
        require(type(n) is type(b) is int and b == closed and 0 <= b < 4 and op['resource'] in contract()['caps'] and
                all(op[k] == r['rows'][b][k] for k in ('root_slot', 'condition', 'arm', 'selection_digest')),
                'native resource exact selected branch')
        if event['kind'] == 'operation_dispatch_intent':
            require(n == len(intents) and op['status'] == 'pending', 'ordered actual operation intent')
            intents[n] = op; stack.append(n)
        else:
            require(stack and stack.pop() == n and n in intents and n not in terminals and
                    op['status'] == ('returned' if event['kind'] == 'operation_returned' else 'failed') and
                    dict({k: v for k, v in op.items() if k != 'error'}, status='pending') == intents[n],
                    'nested native return/failure exact closure')
            terminals[n] = op
    require(not stack and len(intents) == len(terminals) and tail == r['journal_chain_tail'] and
            r['actual_operation_receipts'] == [terminals[i] for i in range(len(terminals))] and
            [e['receipt'] for e in events if e['kind'] == 'branch_closed'] ==
            [row for row in r['rows'] if row['status'] != 'not_started'], 'complete attempted resource/failure denominator')
    counts = {k: dict(attempted=0, completed=0) for k in contract()['caps']}
    for op in r['actual_operation_receipts']:
        counts[op['resource']]['attempted'] += 1
        counts[op['resource']]['completed'] += op['status'] == 'returned'
    require(counts == r['counts'] and all(c['attempted'] <= contract()['caps'][k] for k, c in counts.items()),
            'all-resource independent actual counts/caps')
    for i, row in enumerate(r['rows']):
        plan = p['plans'][i]; task = p['source_roots'][plan['root_slot']]
        require(type(row['index']) is int and row['index'] == i and row['plan'] == plan and
                (row['root_slot'], row['condition']) == ORDER[i] and
                all(row[k] == plan[k] for k in ('arm', 'selection_digest', 'task_id')) and
                row['evidence_origin'] == 'new_native_execution_known_root' and
                row['status'] in ('not_started', 'engineering_failed', 'native_stopped', 'horizon_completed'),
                'all4 new physical rows, no old schema/plan/row inheritance')
        scoped = [e for e in events if e.get('branch') == i]
        ops = [op for op in r['actual_operation_receipts'] if op['branch'] == i]
        public = [op for op in ops if op['resource'] == 'public_step']
        transitions = [e for e in scoped if e['kind'] == 'returned_transition']
        observations = [e for e in scoped if e['kind'] == 'fresh_observation']
        for phase, label, commands in (('warmup', 'warmup', moving.warmup_commands()),
                                       ('plan', 'own', mixed.raw10(plan['commands']))):
            current = [op for op in public if op['phase'] == phase]
            require(len(current) <= 10 and [op['step'] for op in current] == list(range(1, len(current) + 1)) and
                    all(op['raw_bits'] == commands[j].tobytes().hex() for j, op in enumerate(current)) and
                    row[label + '_attempted'] == len(current) and
                    row[label + '_returned'] == sum(op['status'] == 'returned' for op in current),
                    'complete contiguous sealed raw10 authority, no changed actions or padding')
            returned = [t for t in transitions if t['phase'] == phase]
            require([t['step'] for t in returned] == list(range(1, len(returned) + 1)) and
                    len(returned) <= row[label + '_returned'], 'actual returned transition chronology')
            for t in returned:
                require(t['actual_dtype'] == np.dtype(np.float32).str and
                        t['actual_bits'] == t['requested_bits'] == commands[t['step'] - 1].tobytes().hex(),
                        'actual raw float32 return bits')
        require(row['PNG'] == len(observations), 'all actual rendered observations counted')
        bridge = [{k: deepcopy(e[k]) for k in ('receipt', 'pre', 'post')} for e in observations
                  if e['receipt']['phase'] == 'warmup' or e['receipt']['step'] <= 5]
        require(row['prefix51'] == dict(complete=len(bridge) == 51, frames=bridge), 'actual prefix51, no artificial splicing')
        unknown = []; selected = []; wanted_overlay = 1 if row['condition'] == 'on' else 0
        for event in observations:
            f = event['receipt']; phase, step, overlay = f['phase'], f['step'], f['overlay_index']
            require(phase in ('warmup', 'plan') and type(step) is type(overlay) is int and
                    0 <= step <= 10 and overlay in (0, 1, 2) and f['condition'] == ('on' if overlay == 1 else 'off') and
                    event['pre'] == event['post'] and f['snapshot'] == event['post']['snapshot'] and
                    f['state'] == f['snapshot']['state'] and len(f['state']) == 7 and
                    event['pre']['renderer'] == task['renderer'] and event['pre']['goal_pose'] == task['goal_pose'],
                    'native7 observation/renderer/goalpose identity')
            if overlay == wanted_overlay: selected.append(f)
            if phase == 'warmup':
                ref = mixed.allmode.reference_for(p['warm_traces'][plan['root_slot']], step, overlay)
                if ref is None: unknown.append(step)
                else: mixed.allmode.check_warm_frame(event, ref)
            else:
                require(event['pre']['goal_state'] == task['goal'] and
                        event['pre']['goal_rgb_sha256'] == task['goal_pixels'][row['condition']],
                        'same actual task/goal, condition applied to observed history and goalRGB')
                if step:
                    t = next(t for t in transitions if t['phase'] == 'plan' and t['step'] == step)
                    require(f['state'] == t['state'] and f['snapshot'] == t['snapshot'], 'RGB/native dispatch state identity')
        require(row['source_unpaired_on_steps'] == unknown and row['trajectory'] == selected,
                'no invented historical on evidence; actual selected-condition whole trajectory')
        if row['initial_exact']:
            require(all(row[k] == task[k] for k in ('root_state', 'root_sha256', 'goal', 'goal_pose')) and
                    row['goal_pixel_sha256'] == task['goal_pixels'][row['condition']] and
                    source.root_hash(source.vector(row['root_state']), source.vector(row['goal'])) == row['root_sha256'],
                    'exact original native7 root and goal identities')
            goal_events = [e for e in scoped if e['kind'] == 'goal_attached']
            require(len(goal_events) == 1 and all(goal_events[0][k] == row[k] for k in ('goal', 'goal_pose')) and
                    goal_events[0]['goal_rgb_sha256'] == row['goal_pixel_sha256'], 'single original goal bind receipt')
        responses = row['responses']
        require([x['step'] for x in responses] == list(range(1, len(responses) + 1)) and
                len(responses) <= row['own_returned'] and
                [e['response'] for e in scoped if e['kind'] == 'own_response'] == responses, 'all actual own responses')
        for response in responses:
            calc = mixed.metrics(row['goal'], response['state'])
            delta = source.vector(response['state']) - source.vector(row['root_state'])
            require(calc['native'] == response['native_metrics'] and calc['native7_reward'] == response['reward'] and
                    response['native_success'] is response['terminated'] is calc['native_success'] and
                    type(response['truncated']) is bool and np.array_equal(delta, response['delta_native7']) and
                    response['delta_angle_wrapped'] == float(np.arctan2(np.sin(delta[4]), np.cos(delta[4]))),
                    'unchanged native7 reward and jointXY/angle success, wrapped angle diagnostic only')
            t = next(t for t in transitions if t['phase'] == 'plan' and t['step'] == response['step'])
            require(all(t[k] == response[k] for k in ('state', 'reward', 'terminated', 'truncated')) and
                    t['goal_state'] == row['goal'], 'actual dispatch response identity')
        if responses:
            last = responses[-1]
            require(row['final_state'] == last['state'] and row['final_metrics'] == last['native_metrics'] and
                    all(row[k] is last[k] for k in ('native_success', 'terminated', 'truncated')) and
                    all(not x['terminated'] and not x['truncated'] for x in responses[:-1]), 'actual native-stop/final endpoint')
        else:
            require(row['final_state'] is row['final_metrics'] is row['native_success'] is None, 'missing outcomes unknown, not zero')
        cp = [f for f in selected if f['phase'] == 'plan' and f['step'] in (0, 5, 10)]
        require(row['checkpoints'] == [{k: f[k] for k in ('step', 'png', 'file_sha256', 'pixel_sha256', 'state', 'snapshot')}
                for f in cp] and row['missing_checkpoints'] == [b for b in (0, 5, 10) if b not in [f['step'] for f in cp]],
                'checkpoint RGB/native7 identity and explicit missing denominator')
        if row['status'] in ('horizon_completed', 'native_stopped'):
            require(row['first_error'] is None and row['secondary_errors'] == [], 'no engineering error hidden as native stop')
            framekeys = [('warmup', b, k) for b in range(row['warmup_returned'] + 1) for k in (0, 1, 2)]
            if row['initial_exact']: framekeys += [('plan', b, k) for b in range(row['own_returned'] + 1) for k in (0, 1, 2)]
            require([(e['receipt']['phase'], e['receipt']['step'], e['receipt']['overlay_index']) for e in observations] ==
                    framekeys, 'all unpadded real RGB triplets')
            for j in range(0, len(observations), 3):
                a, b, c = observations[j:j + 3]
                require(a['receipt']['file_sha256'] == c['receipt']['file_sha256'] and
                        a['receipt']['pixel_sha256'] == c['receipt']['pixel_sha256'] and
                        a['receipt']['state'] == b['receipt']['state'] == c['receipt']['state'], 'off/on/off unchanged physics')
            if responses:
                require(row['warmup_returned'] == 10 and row['initial_exact'] and row['own_returned'] == len(responses) and
                        ((row['status'] == 'native_stopped') == (last['terminated'] or last['truncated'])), 'native stop never padded')
            if row['status'] == 'horizon_completed': require(row['own_returned'] == 10, 'actual full10, not only first5')
    if r['status'] == RECORDED:
        require(closed == 4 and r['errors'] == [] and r['inputs_final_verified'] is True and
                0 <= r['wall_seconds'] < WALL and all(x['status'] in ('horizon_completed', 'native_stopped') for x in r['rows']),
                'complete engineering four denominator only')
        c = {k: v['completed'] for k, v in counts.items()}
        require(all(v['attempted'] == v['completed'] for v in counts.values()) and
                c['constructor'] == c['close'] == 4 and c['reset'] == 8 and
                c['goal_bind'] == sum(x['initial_exact'] for x in r['rows']) and
                c['set_state'] == 2 * c['reset'] and c['goal_render'] == c['reset'] and
                c['goal_setter'] == c['reset'] + c['goal_bind'] and
                c['physics_step'] == 2 * c['reset'] + 10 * c['public_step'] and
                c['public_step'] == sum(x['warmup_returned'] + x['own_returned'] for x in r['rows']) and
                c['explicit_render'] == sum(x['PNG'] for x in r['rows']), 'independent full native physical census')
    audit_guard(r, events, p)
    audit_all_labels(r, events, p)
    for row, ref in zip(r['rows'], p['prefix_reference'], strict=True):
        require(row['prefix_reference_exact'] is prefix_match(row, ref), 'actual exact prefix-match claim')
    return deepcopy(r['rows'])


def run(prepared, expected, output, factory, *, authorization=None):
    require(authorization == AUTH, 'MAIN authority before input reads/output/env')
    p, data = verify(prepared, expected); store = Store(output)
    return execute(p, data, store, factory, authorization=authorization, prepared=prepared, prepare_sha256=expected)


def consume(output, expected):
    mixed.actual_hash(expected); directory = Path(mixed.absolute(str(output)))
    files = mixed.saved_artifacts(directory)
    require(hashlib.sha256(files['report.json']).hexdigest() == expected, 'actual external observed suffix physical report SHA')
    r = mixed.decode(files['report.json']); binding = r['execution_binding']
    require(hashlib.sha256(files['execution_binding.json']).hexdigest() == r['execution_binding_sha256'] and
            mixed.decode(files['execution_binding.json']) == binding, 'actual execution authority binding')
    p, _ = verify(binding['prepared'], binding['prepare_sha256'])
    journal = files['journal.jsonl']
    require(journal.endswith(b'\n') and hashlib.sha256(journal).hexdigest() == r['journal_sha256'], 'saved physical journal SHA')
    events = [mixed.decode(line) for line in journal.splitlines()]
    rows = validate_saved(r, events, p); names = {'report.json', 'execution_binding.json', 'journal.jsonl'}
    from PIL import Image
    png_count = 0
    for event in events:
        if event['kind'] == 'invalid_observation':
            name = event['file']; require(Path(name).name == name and name not in ('', '.', '..') and
                hashlib.sha256(files[name]).hexdigest() == event['file_sha256'], 'retained invalid observation incident')
            names.add(name)
        if event['kind'] != 'fresh_observation': continue
        f = event['receipt']; name = f['png']
        require(Path(name).name == name and name not in ('', '.', '..') and name not in names and
                len(files[name]) <= 1024**2 and hashlib.sha256(files[name]).hexdigest() == f['file_sha256'],
                'unique own flat PNG bytes/SHA')
        with Image.open(io.BytesIO(files[name])) as image:
            require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (224, 224), 'actual native RGB224')
            rgb = np.asarray(image).copy()
        require(moving.image_sha(rgb) == f['pixel_sha256'], 'saved actual RGB pixels')
        moving.guard(rgb, source.vector(f['state']))
        names.add(name); png_count += 1
    require(set(files) == names, 'no unaccounted output artifacts')
    bindings = deepcopy(p['bindings']); mixed.merge(bindings, {binding['prepared']: binding['prepare_sha256']})
    mixed.merge(bindings, {str(directory / name): hashlib.sha256(body).hexdigest() for name, body in files.items()})
    for file, h in p['bindings'].items(): require(sha(file) == h, 'final original input/source seals')
    require(mixed.saved_artifacts(directory) == files, 'saved artifact bytes unchanged')
    return dict(schema=NAME + '.consume', status=r['status'], rows=rows, counts=r['counts'], errors=r['errors'],
        fixed_denominator=4, known_roots=2, fresh_roots=0, inherited=0, saved_PNG_verified=png_count,
        physical_report_sha256=expected, planner_authority=p['planner_authority'], bindings=bindings,
        metric_breakdown=[dict(root_slot=row['root_slot'], condition=row['condition'],
            metrics=None if row['final_state'] is None else mixed.metrics(row['goal'], row['final_state'])) for row in rows],
        P1_admission=False, P3_admission=False, scientific_admission=False, freshness=False,
        equal_total_compute=False, equal_FLOPs=False, independent_confirmation=False, new_neural_calls=0, new_physics_calls=0)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__);q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','receipt','authorization','compile-output',
              'compile-report-sha','compile-receipt','compile-receipt-sha','compile-source-sha'):q.add_argument('--'+k)
    a=q.parse_args(argv)
    if a.command=='prepare':
        result=prepare(a.output,dict(compile_output=a.compile_output,compile_report_sha=a.compile_report_sha,
            compile_receipt=a.compile_receipt,compile_receipt_sha=a.compile_receipt_sha,compile_source_sha=a.compile_source_sha))
    elif a.command=='verify':
        p,_=verify(a.prepared,a.prepare_sha);result=dict(status='verified_no_execution',fixed_denominator=4,P1_admission=False)
    elif a.command=='consume':
        require(a.receipt,'own durable strict receipt required');result=write_new(a.receipt,consume(a.output,a.report_sha))
    else:
        require(a.authorization==AUTH,'MAIN CLI authority before physics imports')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401; original native registration only
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
    print(source.canonical(result));return 2 if result.get('status')=='partial_or_failed' else 0


if __name__=='__main__':raise SystemExit(main())
