"""Fixed164 ambient-raw probes. MAIN native execution only; no model loading."""
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
from experiments.spectral_reachability import agent_observed_suffix_physical_20261028_v1 as observed
from experiments.spectral_reachability import agent_observed_raw_coordinate_fd_20261028_v1 as producer

NAME='agent_observed_raw_coordinate_physical_20261028_v1'
AUTH='MAIN_OBSERVED_RAW164_PHYSICS_ONLY'
RECORDED='observed_raw164_physical_recorded_not_admitted'
mixed,source,moving,paired=observed.mixed,observed.source,observed.moving,observed.paired
ROOT,ORDER=observed.ROOT,observed.ORDER
require,sha,raw,read=observed.require,observed.sha,observed.raw,observed.read
# Physical schema/proofs: spaced codec. FD producer fields: compact codec.
digest=observed.digest
producer_digest=producer.digest
CAP=FLOOR=1024**3
FINAL=128*1024**2
META_CAP=128*1024**2
WALL=1800
ARGS={'fd_prepared','fd_prepare_sha','fd_source_sha'}
prefix_match=observed.prefix_match
audit_guard=observed.audit_guard
audit_all_labels=observed.audit_all_labels
rows_for=observed.rows_for


def contract():
    c=deepcopy(observed.contract())
    c.update(schema=NAME+'.contract',arm='fixed164_original_suffix_raw_coordinate_probes',producer=producer.NAME,
        fixed_denominator=164,panels=4,jobs_per_panel=41,producer_denominator=164,
        order='panel,baseline,h(.01,.001),coordinate0..9,sign(-1,+1)',
        domain='ambient_raw_box_not_original_logit_L2',deployment_allowed=False,
        raw_authority='literal FD plan_jobs original incumbent iterate0; no winner substitution/no clipping',
        wall_seconds=WALL,output_cap_bytes=CAP,disk_floor_bytes=FLOOR,final_reserve_bytes=FINAL,
        prepare_cap_bytes=META_CAP,producer_metadata_cap_bytes=META_CAP,
        caps=dict(constructor=164,close=164,reset=328,public_step=3280,physics_step=33456,
            set_state=656,goal_setter=492,goal_render=328,goal_bind=164,explicit_render=10824),
        interpretation='164 paired probes of4 known-root conditions,2 known roots, not164 independent samples',
        producer_compute='no FD CUDA receipt needed for physical; complete observed-v2 compile strict rebuilt by FD prepare',
        objective='all original task/Q/goal retained; native state used only for evaluation')
    return c


def own_files():
    return [Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]


def validate_manifest(jobs):
    require(len(jobs)==164,'all164 fixed probes including baseline')
    schedule=[(None,None,0)]+[(c,h,s) for h in producer.STEPS for c in range(10) for s in (-1,1)]
    for i,j in enumerate(jobs):
        slot,local=divmod(i,41);coord,h,sign=schedule[local]
        require(type(j['index']) is type(j['slot']) is type(j['local_index']) is int
            and (j['index'],j['slot'],j['local_index'])==(i,slot,local)
            and (j['root_slot'],j['condition'])==ORDER[slot]
            and (j['coord'],j['h'],j['sign'])==(coord,h,sign),'fixed panel/local/coordinate/h/sign order')
        require(j['job_sha256']==producer_digest({k:v for k,v in j.items() if k!='job_sha256'})
            and j['legal'] is True and j['missing_reason'] is None
            and j['deployment_allowed'] is j['physical_release'] is False,'exact legal producer manifest, never deploy')
        anchor=producer.packet(j['anchor_raw_suffix5'])
        a,delta,reason=(anchor.copy(),0.,None) if coord is None else producer.perturb(j['anchor_raw_suffix5'],coord,h,sign)
        prefix=producer.packet(j['prefix5']);commands=mixed.raw10(j['full10'])
        require(reason is None and producer.F.bits(a)==j['raw_bits']==producer.F.bits(j['raw_suffix5'])
            and producer.F.bits(prefix)==j['prefix_bits'] and commands.tobytes()==np.concatenate((prefix,a)).tobytes()
            and j['actual_delta']==delta,'independent literal float32 perturbation/full10 bit identity')
        base=jobs[slot*41]
        expected_id=f's{slot}_baseline' if coord is None else f's{slot}_h{producer.STEPS.index(h)}_c{coord}_{"minus" if sign<0 else "plus"}'
        require(j['id']==expected_id and j['baseline_reference_id']==base['id']
            and type(j['sign']) is int,'literal producer IDs and same-panel baseline context reference')
        if local==0:
            producer.core.context(j['original_AD_context']);producer.core.context(j['original_iterate0_context'])
        else:
            require(j['original_AD_context'] is j['original_iterate0_context'] is None,
                'only baseline owns terminal contexts; probes reference it, no replacement context')
        for k in ('task','task_id','observed_context','observed_raw10','root_raw10','original_AD_raw_gradient',
                  'provenance','anchor_raw_suffix5','prefix5'):
            require(raw(j[k])==raw(base[k]),'one fixed original task/anchor/context/authority per panel')
        task=producer.core.validate_task(j['task'])
        root=mixed.raw10(j['root_raw10']);incoming=mixed.raw10(j['observed_raw10'])
        require(incoming.tobytes()==np.concatenate((root,prefix))[-10:].tobytes(),
            'observed_raw10 is causal incoming history, never the future full10')
        require(task['binding']['task_id']==j['task_id'] and np.array_equal(task['projection'],np.eye(192))
            and task['action_mean']==producer.F.source.kernel.ACTION_MEAN.tolist()
            and task['action_std']==producer.F.source.kernel.ACTION_STD.tolist(),'original Q/normalizer/full task')


def assemble(arguments,f,bp,br,data):
    """Pure join of actual FD prepare manifest and original native rows; no outcomes select probes."""
    require(f['schema']==producer.NAME+'.prepare' and f['contract']==producer.contract()
        and f['full_upstream_strict_rebuilt_here'] is True and f['production_ready'] is True
        and f['manifest_sha256']==producer_digest(f['jobs']),'whole actual FD ready prepare manifest')
    jobs=f['jobs'];validate_manifest(jobs);observed.base.validate_execution(bp)
    require(br['schema']==observed.base.NAME+'.report' and br['status']==observed.base.RECORDED
        and br['errors']==[] and len(br['rows'])==4,'original fully strict physical4 authority')
    plans=[];refs=[]
    for j in jobs:
        i=j['index'];slot=j['slot'];old=bp['plans'][slot];actual=br['rows'][slot]
        task=deepcopy(j['task']);native=bp['source_roots'][j['root_slot']]
        commands=mixed.raw10(j['full10']);prior=mixed.raw10(old['commands'])
        require(actual['plan']==old and actual['status']=='horizon_completed' and actual['initial_exact'] is True
            and actual['prefix51']['complete'] is True and len(actual['prefix51']['frames'])==51
            and task['binding']==old['model_task_binding']
            and j['provenance']['parent_selection_digest']==old['producer_selection_digest']
            and commands[:5].tobytes()==prior[:5].tobytes()
            and producer.packet(j['anchor_raw_suffix5']).tobytes()==prior[5:].tobytes(),
            'original unchanged prefix and original baseline suffix; observed_raw10 is history, not future commands')
        if j['local_index']==0:
            require(commands.tobytes()==prior.tobytes(),'baseline full10 exactly original physical plan')
        # The entire original task (not just task_id) is authenticated through the FD verifier.
        require(all(task['binding'][k]==old[v] for k,v in (('task_sha256','task_sha256'),
            ('goal_sha256','goal_sha256'),('projection_sha256','q_sha256'),('context_sha256','context_sha256')))
            and all(actual[k]==native[k] for k in
            ('root_state','root_sha256','goal','goal_pose')) and
            actual['goal_pixel_sha256']==native['goal_pixels'][j['condition']],'complete original task/native source identity')
        cps=[cp for cp in actual['checkpoints'] if cp['step']==5]
        require(len(cps)==1,'exact genuine original cp5')
        ref=dict(index=i,complete=True,raw_prefix_bits=producer.F.bits(prior[:5]),
            frames=deepcopy(actual['prefix51']['frames']),checkpoint=deepcopy(cps[0]));refs.append(ref)
        plan=dict(schema=NAME+'.execution_plan',index=i,panel=slot,local_index=j['local_index'],probe_id=j['id'],
            root_slot=j['root_slot'],source_slot=j['root_slot'],condition=j['condition'],arm=contract()['arm'],
            commands=deepcopy(j['full10']),raw_steps=10,raw_float32_sha256=hashlib.sha256(commands.tobytes()).hexdigest(),
            prefix_bits=j['prefix_bits'],prefix_reference_sha256=digest(ref),task_id=j['task_id'],
            model_task=task,model_task_binding=deepcopy(task['binding']),producer_job_sha256=j['job_sha256'],
            native_goal=deepcopy(native['goal']),native_goal_pose=deepcopy(native['goal_pose']),
            native_root_sha256=native['root_sha256'],native_goal_pixel_sha256=native['goal_pixels'][j['condition']],
            selection_uses_physics=False,P1_admission=False,deployment_allowed=False)
        plan['selection_digest']=digest(plan);plans.append(plan)
    require(bp['source_slots']==[x['slot'] for x in data['roots']]
        and bp['source_roots']==mixed.allmode.root_metadata(data)
        and bp['warm_traces']==[x['warm_trace'] for x in data['roots']],'complete source warmup/root identity')
    return dict(schema=NAME+'.prepare',contract=contract(),arguments=deepcopy(arguments),plans=plans,prefix_reference=refs,
        jobs=deepcopy(jobs),planner_authority=dict(producer=producer.NAME,**arguments,manifest_sha256=f['manifest_sha256'],
            full_upstream_strict_rebuilt_here=True,physical_selection=False,P1_admission=False),
        source_report_sha256=mixed.SOURCE_SHA,source_slots=deepcopy(bp['source_slots']),
        source_roots=deepcopy(bp['source_roots']),warm_traces=deepcopy(bp['warm_traces']),bindings={})


def validate_execution(p):
    require(p['schema']==NAME+'.prepare' and p['contract']==contract() and len(p['plans'])==len(p['prefix_reference'])==164
        and p['source_report_sha256']==mixed.SOURCE_SHA and set(p['arguments'])==ARGS
        and len(p['source_slots'])==len(p['source_roots'])==len(p['warm_traces'])==2,'own164 schema/source/authority')
    validate_manifest(p['jobs'])
    require(p['planner_authority']==dict(producer=producer.NAME,**p['arguments'],
        manifest_sha256=producer_digest(p['jobs']),full_upstream_strict_rebuilt_here=True,
        physical_selection=False,P1_admission=False),'full FD authority binding')
    for i,(plan,j,ref) in enumerate(zip(p['plans'],p['jobs'],p['prefix_reference'],strict=True)):
        a=mixed.raw10(plan['commands']);native=p['source_roots'][j['root_slot']]
        require(plan['schema']==NAME+'.execution_plan' and plan['index']==ref['index']==i
            and (plan['panel'],plan['local_index'],plan['probe_id'])==(j['slot'],j['local_index'],j['id'])
            and (plan['root_slot'],plan['condition'])==ORDER[i//41] and plan['source_slot']==plan['root_slot']
            and plan['arm']==contract()['arm'] and plan['raw_steps']==10
            and plan['selection_digest']==digest({k:v for k,v in plan.items() if k!='selection_digest'})
            and plan['producer_job_sha256']==j['job_sha256'] and raw(plan['model_task'])==raw(j['task'])
            and plan['task_id']==j['task_id'] and plan['model_task_binding']==j['task']['binding']
            and producer.F.bits(a)==producer.F.bits(j['full10'])
            and hashlib.sha256(a.tobytes()).hexdigest()==plan['raw_float32_sha256']
            and plan['prefix_bits']==ref['raw_prefix_bits']==producer.F.bits(a[:5])
            and plan['prefix_reference_sha256']==digest(ref),'all immutable own plans/producer compact hashes/raw bytes')
        require(ref['complete'] is True and len(ref['frames'])==51 and ref['checkpoint']['step']==5
            and ref['checkpoint']['state']==ref['checkpoint']['snapshot']['state']
            and all(plan[k]==native[v] for k,v in (('native_goal','goal'),('native_goal_pose','goal_pose'),
                ('native_root_sha256','root_sha256')))
            and plan['native_goal_pixel_sha256']==native['goal_pixels'][j['condition']]
            and all(plan[k] is False for k in ('selection_uses_physics','P1_admission','deployment_allowed')),
            'complete cp5/native task/renderer and no selection/release')


def material(arguments):
    require(type(arguments) is dict and set(arguments)==ARGS,'FD actual prepare/source SHA inputs')
    for k,v in arguments.items():
        if k.endswith('_sha'):mixed.actual_hash(v)
        else:mixed.absolute(v)
    require(sha(producer.__file__)==arguments['fd_source_sha'],'actual FD source seal before full verifier')
    f=producer.verify(arguments['fd_prepared'],arguments['fd_prepare_sha'])
    a=f['authority'];c=read(a['prepared'],META_CAP)
    require(sha(a['prepared'])==a['prepare_sha256'],'actual observed compiler prepare')
    bp,br=observed.reference(c);data=mixed.source_material();p=assemble(arguments,f,bp,br,data)
    bindings=deepcopy(f['bindings'])
    mixed.merge(bindings,{arguments['fd_prepared']:arguments['fd_prepare_sha'],
        str(Path(producer.__file__).resolve()):arguments['fd_source_sha']})
    mixed.merge(bindings,data['bindings'])
    for file in [*own_files(),Path(observed.__file__).resolve(),Path(observed.base.__file__).resolve(),
                 Path(mixed.__file__).resolve(),Path(mixed.allmode.__file__).resolve()]:
        mixed.merge(bindings,{str(file):sha(file)})
    path,h=mixed.allmode.constructor_contract();mixed.merge(bindings,{path:h});p['bindings']=bindings
    for file,h in bindings.items():require(sha(file)==h,'all original source/artifact bytes')
    validate_execution(p);require(len(raw(p))<=META_CAP,'128MiB full prepare, no truncation');return p,data


def write_new(path,value,cap=META_CAP):
    path=Path(mixed.absolute(str(path)));body=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and not path.exists()
        and not path.is_symlink() and len(body)<=cap and shutil.disk_usage(path.parent).free>=FLOOR+len(body),
        'exclusive own bounded metadata')
    with path.open('xb') as f:require(f.write(body)==len(body),'full metadata');f.flush();os.fsync(f.fileno())
    paired.fsync_dir(path.parent);return dict(output=str(path),sha256=sha(path),bytes=len(body))


def prepare(output,arguments):
    require(not Path(output).exists(),'new exclusive prepare');p,_=material(arguments);return write_new(output,p)


def verify(prepared,expected):
    mixed.actual_hash(expected);mixed.absolute(str(prepared));require(sha(prepared)==expected,'own actual prepare SHA')
    p=read(prepared,META_CAP);now,data=material(p['arguments'])
    require(raw(p)==raw(now) and sha(prepared)==expected,'whole source/manifest/native reference rebuild');return p,data


class Store(observed.Store):
    """Incremental actual bytes/SHA; no quadratic journal rehash or directory rescans."""
    def __init__(self,path):
        self.path=Path(mixed.absolute(str(path)))
        require(self.path.name.startswith(NAME+'_') and not self.path.exists() and not self.path.is_symlink()
            and self.path.parent.is_dir() and shutil.disk_usage(self.path.parent).free>=CAP+FLOOR,'new1GiB+1GiB floor')
        self.path.mkdir();paired.fsync_dir(self.path.parent)
        self.previous=None;self.index=0;self.bytes=0;self.hashes={};self.sizes={}
    def check(self,size=4096,final=False):
        require(type(size) is int and size>=0 and self.bytes+size<=CAP-(0 if final else FINAL)
            and shutil.disk_usage(self.path).free>=FLOOR+size,'actual1GiB cap/128MiB final reserve/1GiB floor')
    def write(self,name,value,append=False,final=False):
        require(type(name) is str and Path(name).name==name and name not in ('','.','..')
            and not (self.path/name).is_symlink() and type(value) is bytes,'flat nonsymlink bytes')
        path=self.path/name
        require((append and (not path.exists() or name in self.hashes)) or not path.exists(),'exclusive artifact/owned append')
        require(not path.exists() or path.stat().st_size==self.sizes[name],'no external partial overwrite')
        self.check(len(value),final)
        before=self.sizes.get(name,0)
        try:
            with path.open('ab' if append else 'xb') as f:
                require(f.write(value)==len(value),'short write');f.flush();os.fsync(f.fileno())
        finally:
            after=path.stat().st_size if path.exists() else 0;self.bytes+=after-before;self.sizes[name]=after
        h=self.hashes.setdefault(name,hashlib.sha256());h.update(value)
        paired.fsync_dir(self.path);return h.hexdigest()


class Meter:
    def __init__(self,store,started=None):
        self.store=store;self.started=time.monotonic() if started is None else started
        self.where={};self.receipts=[];self.secondary=[];self.fault=False
        self.counts={k:dict(attempted=0,completed=0) for k in contract()['caps']};self.active={k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        self.guard_call(kind,fn,cleanup=cleanup,**fields)
        require(not cleanup or kind=='close','only nonphysical close may bypass predispatch floor/wall fault')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started<WALL,'wall/fault predispatch');self.store.check()
        count=self.counts[kind];require(count['attempted']<contract()['caps'][kind],'cap before callback: '+kind)
        receipt=dict(index=len(self.receipts),resource=kind,**deepcopy(self.where),**deepcopy(fields),status='pending')
        try:self.store.emit('operation_dispatch_intent',receipt=receipt)
        except BaseException as exc:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(mixed.base.error('cleanup_intent',exc))
        self.receipts.append(receipt);count['attempted']+=1;self.active[kind]+=1
        try:
            value=fn();count['completed']+=1;receipt['status']='returned'
        except BaseException as exc:
            receipt.update(status='failed',error=mixed.base.error(kind,exc))
            try:self.store.emit('operation_failed',receipt=receipt)
            except BaseException as extra:self.fault=True;self.secondary.append(mixed.base.error('failure_journal',extra))
            raise
        finally:self.active[kind]-=1
        try:self.store.emit('operation_returned',receipt=receipt)
        except BaseException as exc:
            self.fault=True
            if not cleanup:raise
            self.secondary.append(mixed.base.error('cleanup_return',exc))
        return value

    def guard_context(self,row,ref):self.row,self.ref=row,ref;self.guarded=False
    def guard_call(self,kind,fn,*,cleanup=False,**fields):
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
    if time.monotonic() - meter.started >= WALL: fail('final_wall', ValueError('1800s wall'))
    good = not errors and source_ok and all(r['status'] in ('horizon_completed', 'native_stopped') for r in rows)
    status = RECORDED if good else 'partial_or_failed'
    try: store.emit('run_finished', status=status, counts=meter.counts, errors=errors)
    except BaseException as exc: fail('final_journal', exc); status = 'partial_or_failed'
    report = dict(schema=NAME + '.report', status=status, contract=contract(), execution_binding=binding,
        rows=rows, errors=errors, counts=deepcopy(meter.counts), actual_operation_receipts=deepcopy(meter.receipts),
        fixed_denominator=164, known_roots=2, fresh_roots=0, inherited=0, inputs_final_verified=source_ok,
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
            r['status'] in (RECORDED, 'partial_or_failed') and r['fixed_denominator'] == len(r['rows']) == 164 and
            r['known_roots'] == 2 and r['fresh_roots'] == r['inherited'] == 0 and
            r['execution_binding']['planner_authority'] == p['planner_authority'] and
            r['execution_binding']['source_report_sha256'] == mixed.SOURCE_SHA, 'own full164 raw-coordinate physical report authority')
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
            require(closed < 164 and event['receipt'] == r['rows'][closed], 'ordered exact full branch closure')
            closed += 1
        if 'branch' in event:
            require(type(event['branch']) is int and event['branch'] == closed and closed < 164 and
                    all(event[k] == r['rows'][closed][k] for k in ('root_slot', 'condition', 'arm', 'selection_digest')),
                    'active exact branch identity')
        if event['kind'] not in ('operation_dispatch_intent', 'operation_returned', 'operation_failed'): continue
        op = event['receipt']; n = op['index']; b = op['branch']
        require(type(n) is type(b) is int and b == closed and 0 <= b < 164 and op['resource'] in contract()['caps'] and
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
                (row['root_slot'], row['condition']) == ORDER[i//41] and
                all(row[k] == plan[k] for k in ('arm', 'selection_digest', 'task_id')) and
                row['evidence_origin'] == 'new_native_execution_known_root' and
                row['status'] in ('not_started', 'engineering_failed', 'native_stopped', 'horizon_completed'),
                'all164 new physical rows, no old schema/plan/row inheritance')
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
        require(closed == 164 and r['errors'] == [] and r['inputs_final_verified'] is True and
                0 <= r['wall_seconds'] < WALL and all(x['status'] in ('horizon_completed', 'native_stopped') for x in r['rows']),
                'complete engineering 164 denominator only')
        c = {k: v['completed'] for k, v in counts.items()}
        require(all(v['attempted'] == v['completed'] for v in counts.values()) and
                c['constructor'] == c['close'] == 164 and c['reset'] == 328 and
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


def run(prepared,expected,output,factory,*,authorization=None):
    require(authorization==AUTH,'MAIN authority before reads/output/env')
    started=time.monotonic();p,data=verify(prepared,expected);store=Store(output)
    return execute(p,data,store,factory,authorization=authorization,prepared=prepared,prepare_sha256=expected,started=started)


def inventory(directory):
    require(directory.is_dir() and not directory.is_symlink(),'real flat own output directory')
    files=list(directory.iterdir())
    require(len(files)<=10824+3 and all(f.is_file() and not f.is_symlink() for f in files)
        and sum(f.stat().st_size for f in files)<=CAP,'full flat1GiB census')
    return {f.name:sha(f) for f in files}


def consume(output,expected):
    mixed.actual_hash(expected);directory=Path(mixed.absolute(str(output)));seals=inventory(directory)
    require(seals.get('report.json')==expected,'external actual164 report SHA')
    r=read(directory/'report.json',META_CAP);binding=r['execution_binding']
    require(seals.get('execution_binding.json')==r['execution_binding_sha256']
        and read(directory/'execution_binding.json',META_CAP)==binding,'full execution binding')
    p,_=verify(binding['prepared'],binding['prepare_sha256'])
    require(seals.get('journal.jsonl')==r['journal_sha256'],'full saved journal hash')
    events=[]
    with (directory/'journal.jsonl').open('rb') as f:
        while True:
            line=f.readline(META_CAP+1)
            if not line:break
            require(len(line)<=META_CAP and line.endswith(b'\n'),'whole bounded journal record')
            events.append(mixed.decode(line))
    rows=validate_saved(r,events,p)
    names={'report.json','execution_binding.json','journal.jsonl'};png_count=0
    from PIL import Image
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file']
            require(Path(name).name==name and name not in names and seals.get(name)==e['file_sha256'],
                'retained unique invalid observation incident');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png']
        require(Path(name).name==name and name not in names and seals.get(name)==f['file_sha256']
            and (directory/name).stat().st_size<=1024**2,'unique flat actual PNG bytes/SHA')
        body=(directory/name).read_bytes()
        require(hashlib.sha256(body).hexdigest()==f['file_sha256'],'PNG unchanged at read')
        with Image.open(io.BytesIO(body)) as image:
            require(image.format=='PNG' and image.mode=='RGB' and image.size==(224,224),'native actual RGB224')
            rgb=np.asarray(image).copy()
        require(moving.image_sha(rgb)==f['pixel_sha256'],'all saved pixel hashes')
        moving.guard(rgb,source.vector(f['state']));names.add(name);png_count+=1
    require(set(seals)==names,'no extra/missing artifacts')
    bindings=deepcopy(p['bindings']);mixed.merge(bindings,{binding['prepared']:binding['prepare_sha256']})
    mixed.merge(bindings,{str(directory/name):h for name,h in seals.items()})
    for file,h in p['bindings'].items():require(sha(file)==h,'final upstream/source bytes intact')
    require(inventory(directory)==seals,'saved complete bytes unchanged after strict consume')
    return dict(schema=NAME+'.consume',status=r['status'],rows=rows,counts=r['counts'],errors=r['errors'],
        physical_report_sha256=expected,fixed_denominator=164,panels=4,jobs_per_panel=41,
        known_roots=2,fresh_roots=0,inherited=0,saved_PNG_verified=png_count,
        planner_authority=p['planner_authority'],bindings=bindings,
        metric_breakdown=[dict(index=row['index'],panel=i//41,local_index=i%41,probe_id=p['jobs'][i]['id'],
            root_slot=row['root_slot'],condition=row['condition'],status=row['status'],
            metrics=None if row['final_state'] is None else mixed.metrics(row['goal'],row['final_state']))
            for i,row in enumerate(rows)],
        P1_admission=False,P3_admission=False,scientific_admission=False,freshness=False,
        deployment_allowed=False,equal_total_compute=False,equal_FLOPs=False,independent_confirmation=False,
        new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    q=argparse.ArgumentParser(description=__doc__)
    q.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','fd-prepared','fd-prepare-sha','fd-source-sha',
              'report-sha','receipt','authorization'):q.add_argument('--'+k)
    a=q.parse_args(argv)
    if a.command=='prepare':
        result=prepare(a.output,dict(fd_prepared=a.fd_prepared,fd_prepare_sha=a.fd_prepare_sha,fd_source_sha=a.fd_source_sha))
    elif a.command=='verify':
        verify(a.prepared,a.prepare_sha);result=dict(status='verified_no_execution',fixed_denominator=164,P1_admission=False)
    elif a.command=='consume':
        require(a.receipt,'durable own receipt required');result=write_new(a.receipt,consume(a.output,a.report_sha))
    else:
        require(a.authorization==AUTH,'MAIN authorization before physics imports')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401; original registration only
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        r=run(a.prepared,a.prepare_sha,a.output,factory,authorization=a.authorization)
        result=dict(status=r['status'],counts=r['counts'],errors=r['errors'],fixed_denominator=164)
    print(source.canonical(result));return 2 if result.get('status')=='partial_or_failed' else 0


if __name__=='__main__':raise SystemExit(main())






