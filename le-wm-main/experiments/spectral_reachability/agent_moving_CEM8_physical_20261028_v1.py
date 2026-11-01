"""MAIN-only eight known-root CEM suffixes after exact original51 prefix replay.

Actual successful producer own-consume and external SHA closure are mandatory.
No CUDA/model runner, saved-poststate restore, old outcome selection or freshness.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.spectral_reachability import capability_suffix_factorial_physical_20261027_v1 as replay
from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as core

NAME = 'agent_moving_CEM8_physical_20261028_v1'
PRODUCER = 'agent_moving_CEM8_20261028_v1'
RELEASE = 'agent_CEM8_release_20261028_v1'
ARM = 'teacher_full_CEM64'
AUTH = 'MAIN_FIXED8_CEM64_EXACT51_SUFFIX_PHYSICS_ONLY'
RECORDED = 'CEM8_suffix_physical_recorded_not_admitted'
base, probe, ROOT = replay.base, replay.probe, replay.ROOT
require, sha, raw, read = replay.require, replay.sha, replay.raw, replay.read
raw5, absolute, merge = replay.raw5, replay.absolute, replay.merge
HELPER_SHA = '15125f4d86e5a6b3344f750205101a0e8a6f4a8dfba82ce46ce3d925f4ea8bd6'
PRODUCER_SHA = '6dcb425c13ea8cade14bf23c97ecfe87b004b7804d155c5ef06282045e99c251'
RELEASE_SHA = '6bde2af9b57cdd037d74f0268f0dca4cdfd76d72db7145a7877ae905b95fad55'
ORIGINAL_SHA = '56962e9370fcf249a2ed88f004a400414b491f503a4c3a9db1bd951cf984a521'
ORDER = [(s, c, a) for s in (0, 1) for c in base.CONDITIONS for a in base.ARMS]
CAP, FLOOR, FINAL, PREPARE_CAP = 64*1024**2, 200*1024**2, 4*1024**2, 16*1024**2
ARGUMENTS = {'compiler_output', 'compiler_report_sha256', 'producer_source_sha256', 'consumer_source_sha256',
             'consumer_report', 'consumer_report_sha256', 'physical_output', 'physical_report_sha256'}


def contract():
    c = deepcopy(replay.contract())
    c.pop('variants'); c.pop('CUDA_v1_accepted')
    c.update(fixed_denominator=8, known_roots=2, independent_roots=0, fresh_roots=0,
             arm=ARM, producer=PRODUCER, consumer=RELEASE+'.verify -> '+PRODUCER+'.consume',
             order=[list(k) for k in ORDER], output_cap_bytes=CAP, final_reserve_bytes=FINAL,
             suffix_source='own strict-consumed CEM literal float32 raw5, same model/dispatch; no tanh/clip/scale',
             interpretation='fixed2 known-root teacher8 suffix solver calibration, not fresh or full CEM-MPC',
             caps=dict(constructor=8, close=8, reset=16, public_step=160, physics_step=1632,
                       set_state=32, goal_setter=24, goal_render=16, goal_bind=8, explicit_render=528))
    return c


def actual_hash(h):
    base.hash_value(h); require(h != '0'*64, 'actual SHA required, no missing/future zero seal')


def own_files():
    return [Path(__file__).resolve(), ROOT/'tests'/('test_'+NAME+'.py'), ROOT/'innerdocs'/(NAME+'.md'),
            Path(core.__file__).resolve(), Path(replay.__file__).resolve(), *replay.own_files()]


def validate_verified(compiler, prepared, consumed, originals, compiler_sha):
    """Pure real producer ABI seam, not a model runner or a substitute strict-consume."""
    actual_hash(compiler_sha)
    require(compiler['schema'] == PRODUCER+'.report' and compiler['status'] == 'teacher8_CEM_commands_sealed_not_admitted' and
            compiler['errors'] == [] and compiler['first_error'] is None and
            compiler['commands_released'] is compiler['model_unchanged'] is compiler['source_verified'] is True and
            compiler['P1_admission'] is compiler['P3_admission'] is False, 'complete actual CEM8 producer, not raw partial release')
    require(prepared['schema'] == PRODUCER+'.prepare' and compiler['contract'] == prepared['contract'] and
            len(compiler['rows']) == len(prepared['branches']) == len(originals) == 8 and
            [(p['root_slot'], p['condition'], p['arm']) for p in originals] == ORDER, 'all8 original known anchors')
    require(consumed['schema'] == PRODUCER+'.consume' and consumed['status'] == 'teacher8_CEM_commands_verified_not_admitted' and
            consumed['errors'] == [] and consumed['P1_admission'] is consumed['P3_admission'] is consumed['equal_total_compute'] is False and
            consumed['new_neural_calls'] == consumed['new_physics_calls'] == 0 and len(consumed['plans']) == 8,
            'successful actual own strict-consume, not producer commands_released alone')
    require(compiler['resources'] == {k:dict(attempted=520, completed=520) for k in ('predict', 'action_encoder', 'pred_proj')},
            'complete actual520 axes, not sum of unlike units')
    selected = []
    for i, (branch, row, plan, original) in enumerate(zip(prepared['branches'], compiler['rows'], consumed['plans'], originals, strict=True)):
        key = ORDER[i]
        require(type(branch['anchor']) is type(plan['anchor']) is type(row['index']) is int and
                branch['anchor'] == plan['anchor'] == row['index'] == i and
                type(plan['root_slot']) is type(plan['source_slot']) is int and
                plan['root_slot'] == plan['source_slot'] == branch['source_slot'] == key[0] and
                (branch['source_slot'], branch['condition'], branch['arm']) == key and
                (plan['root_slot'], plan['condition'], plan['prefix_arm']) == key and plan['arm'] == ARM,
                'literal8 ordinal/root/condition/prefix/CEM identity')
        require(all(branch['original_plan'][k] == original[k] for k in base.PLAN_KEYS) and
                plan['task_id'] == branch['parent_task_id'] == original['task_id'] and
                plan['original_selection_digest'] == branch['planner_selection_sha256'] == original['selection_digest'] and
                plan['blind_sha256'] == branch['blind_sha256'] == original['blind_sha256'] and
                plan['branch_id'] == branch['branch_id'] and plan['teacher_context_sha256'] == core.digest(branch['observed_context']),
                'complete original planner/source/blind/task/teacher context join')
        body = {k:v for k,v in plan.items() if k not in ('selection_sha256', 'whole_run_report_sha256')}
        require(plan['schema'] == PRODUCER+'.model_plan' and plan['selection_sha256'] == core.digest(body) and
                plan['whole_run_report_sha256'] == compiler_sha and
                all(plan[k] is False for k in ('P1_admission', 'P3_admission', 'scientific_admission', 'independent_confirmation', 'equal_total_compute')) and
                plan['raw_authority'] == 'literal_float32_same_model_and_dispatch' and plan['selection'] == 'model_only_stable64_argmin' and
                plan['horizon'] == 2 and plan['reoptimized_horizon'] == 1 and plan['raw_steps'] == 10, 'actual own sealed raw model-only winner')
        prefix, suffix = raw5(plan['prefix_commands']), raw5(plan['suffix_commands'])
        require(prefix.tobytes() == raw5(branch['prefix_commands']).tobytes() == base.previous.commands(original['commands'])[:5].tobytes() and
                plan['commands'] == plan['prefix_commands']+plan['suffix_commands'], 'original prefix5 bytes and literal raw10')
        require(type(plan['selected_index']) is int and 0 <= plan['selected_index'] < 64, 'literal own64 winner index')
        winner = row['evaluations'][plan['selected_index']]
        require(row['status'] == 'suffix_sealed' and row['replay_exact'] is True and len(row['evaluations']) == 65 and
                type(plan['selected_index']) is int and 0 <= plan['selected_index'] < 64 and row['selected_index'] == plan['selected_index'] and
                row['suffix_commands'] == plan['suffix_commands'] and row['model_Q'] == plan['model_Q'] == winner['cost'] and
                np.isfinite(plan['model_Q']) and raw5(winner['commands']).tobytes() == raw5(row['evaluations'][64]['commands']).tobytes() == suffix.tobytes(),
                'own evaluated winner and independent replay literal raw, never rederived from logits')
        require(row['plan'] == {k:v for k,v in plan.items() if k != 'whole_run_report_sha256'}, 'producer row and consumed plan exact')
        item = dict(plan=deepcopy(plan), branch_sha256=core.digest(branch), compiler_report_sha256=compiler_sha)
        item['execution_sha256'] = base.source.digest(item); selected.append(item)
    return selected


def compiler_material(arguments):
    for k in ('compiler_report_sha256', 'producer_source_sha256', 'consumer_source_sha256', 'consumer_report_sha256'):
        actual_hash(arguments[k])
    require(arguments['producer_source_sha256'] == PRODUCER_SHA and arguments['consumer_source_sha256'] == RELEASE_SHA,
            'reviewed frozen CEM8 own consume AND actual full-envelope release wrapper')
    output = Path(absolute(arguments['compiler_output'])); acceptance = Path(absolute(arguments['consumer_report']))
    source_path = ROOT/'experiments/spectral_reachability'/(PRODUCER+'.py'); rp = output/'report.json'
    release_path = source_path.with_name(RELEASE+'.py')
    require(sha(source_path) == PRODUCER_SHA and sha(release_path) == RELEASE_SHA and sha(rp) == arguments['compiler_report_sha256'] and
            sha(acceptance) == arguments['consumer_report_sha256'], 'actual producer/source/strict-consume SHA BEFORE import')
    envelope = read(acceptance, PREPARE_CAP)
    require(acceptance.name == 'receipt.json' and envelope['schema'] == RELEASE+'.receipt' and
            envelope['status'] == 'strict_CEM8_release_persisted_not_admitted' and
            envelope['arguments']['directory'] == str(output) and
            envelope['arguments']['report_sha'] == arguments['compiler_report_sha256'], 'actual release envelope/report invocation')
    release = importlib.import_module('experiments.spectral_reachability.'+RELEASE)
    require(Path(release.__file__).resolve() == release_path and sha(release_path) == RELEASE_SHA, 'fixed reviewed wrapper identity')
    checked = release.verify(str(acceptance.parent), arguments['consumer_report_sha256'])
    require(checked['status'] == 'release_reverified_not_admitted' and checked['plans'] == 8 and
            checked['sha256'] == arguments['consumer_report_sha256'] and
            sha(acceptance) == arguments['consumer_report_sha256'], 'actual envelope/inventory/own-consume reverified, no raw flag bypass')
    saved = envelope['producer_result']
    require(saved['schema'] == PRODUCER+'.consume' and saved['status'] == 'teacher8_CEM_commands_verified_not_admitted', 'no raw/partial/uncompleted acceptance')
    producer = importlib.import_module('experiments.spectral_reachability.'+PRODUCER)
    require(Path(producer.__file__).resolve() == source_path and sha(source_path) == PRODUCER_SHA, 'fixed reviewed producer identity')
    # release.verify has just compared the ENTIRE producer_result against the
    # real saved-only producer.consume and exactly reconstructed the envelope.
    report = read(rp, 32*1024**2); pp = Path(absolute(report['prepared']))
    require(sha(pp) == report['prepare_sha256'], 'actual own producer prepare SHA')
    compiled = read(pp, 2*1024**2); bindings = {}
    merge(bindings, compiled['bindings'])
    require(envelope['input_count'] == len(envelope['input_sha256']), 'complete release source inventory')
    merge(bindings, envelope['input_sha256'])
    merge(bindings, {str(source_path):PRODUCER_SHA, str(rp):arguments['compiler_report_sha256'],
                     str(release_path):RELEASE_SHA, str(acceptance.parent/'request.json'):envelope['request_sha256'],
                     str(acceptance):arguments['consumer_report_sha256'], str(pp):report['prepare_sha256']})
    require(envelope['arguments']['prepared'] == str(pp) and
            envelope['arguments']['prepare_sha'] == report['prepare_sha256'], 'release exact prepare identity')
    for name,h in report['artifact_sha256'].items():
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat actual CEM artifact')
        merge(bindings, {str(output/name):h})
    # Own strict-consume already reconstructs the parent. Explicitly retain its
    # full source closure too, not merely a short list of direct parent references.
    parent_path = Path(absolute(str(producer.FACTORIAL_PREP)))
    require(bindings[str(parent_path)] == compiled['inherited_prepare_sha256'] == sha(parent_path), 'actual inherited v2 prepare closure')
    parent = read(parent_path, PREPARE_CAP); merge(bindings, parent['bindings'])
    parent_report = producer.FACTORIAL_OUTPUT/'report.json'
    require(sha(parent_report) == bindings[str(parent_report)], 'actual inherited v2 report seal')
    pr = read(parent_report, 64*1024**2)
    for name,h in pr['artifact_sha256'].items():
        require(Path(name).name == name and name not in ('', '.', '..'), 'flat inherited artifact')
        merge(bindings, {str(parent_report.parent/name):h})
    return report, compiled, saved, bindings


def material(arguments):
    require(type(arguments) is dict and set(arguments) == ARGUMENTS, 'all external actual authorities required; no future defaults')
    require(sha(replay.__file__) == HELPER_SHA and sha(probe.__file__) == replay.PROBE_SHA and sha(base.__file__) == replay.BASE_SHA,
            'sealed exact51 replay primitives')
    actual_hash(arguments['physical_report_sha256'])
    require(arguments['physical_report_sha256'] == ORIGINAL_SHA, 'original8 prefix evidence only, never new suffix outcome report')
    report, compiled, consumed, bindings = compiler_material(arguments)
    original, data = probe.material(absolute(arguments['physical_output']), arguments['physical_report_sha256'])
    selected = validate_verified(report, compiled, consumed, original['plans'], arguments['compiler_report_sha256'])
    merge(bindings, original['bindings']); native,h = replay.constructor_contract(); merge(bindings, {native:h})
    for path in own_files(): merge(bindings, {str(path.resolve()):sha(path)})
    for path,h in bindings.items(): require(sha(path) == h, 'final full source/task/raw/artifact closure')
    p = dict(schema=NAME+'.prepare', arguments=deepcopy(arguments), contract=contract(),
             plans=deepcopy(original['plans']), anchors=deepcopy(original['anchors']), suffix_plans=selected,
             physical_report_sha256=arguments['physical_report_sha256'], planner_report_sha256=original['planner_report_sha256'],
             compiler_report_sha256=arguments['compiler_report_sha256'], consumer_report_sha256=arguments['consumer_report_sha256'],
             MAIN_verified8=True, bindings=bindings)
    require(len(raw(p)) <= PREPARE_CAP, 'bounded complete physical prepare; no dropping source seals')
    return p, data


def prepare(output, arguments):
    p,_ = material(arguments); value = raw(p); path = Path(output).absolute()
    require(path.parent.is_dir() and not path.exists() and not path.is_symlink() and
            shutil.disk_usage(path.parent).free >= FLOOR+len(value), 'NEW prepare +200MiB floor')
    with path.open('xb') as f:
        require(f.write(value) == len(value), 'complete prepare'); f.flush(); os.fsync(f.fileno())
    base.paired.fsync_dir(path.parent)
    return dict(prepared=str(path), prepare_sha256=sha(path), bytes=len(value))


def verify(path, expected):
    actual_hash(expected); require(sha(path) == expected, 'external own physical prepare SHA')
    p = read(path, PREPARE_CAP); now,data = material(p['arguments'])
    require(p == now and sha(path) == expected, 'full actual strict-consume/original51/selected raw reconstruction')
    return p,data


class Store(base.Store):
    def __init__(self,path):
        self.path = Path(path).absolute()
        require(self.path.parent.is_dir() and not self.path.exists() and not self.path.is_symlink() and
                shutil.disk_usage(self.path.parent).free >= FLOOR+CAP, 'NEW64MiB output +200MiB floor')
        self.path.mkdir(); base.paired.fsync_dir(self.path.parent); self.previous = None; self.index = 0
    def check(self,size=4096,final=False):
        files = list(self.path.iterdir())
        require(all(f.is_file() and not f.is_symlink() for f in files) and
                sum(f.stat().st_size for f in files)+size <= CAP-(0 if final else FINAL) and
                shutil.disk_usage(self.path).free >= FLOOR+size, 'output cap/reserve/floor including partial')


class Meter:
    """Eight-branch caps, real callback entry/return, no refund on failure."""
    def __init__(self,store,started=None):
        self.store = store; self.started = time.monotonic() if started is None else started
        self.where = {}; self.secondary = []; self.fault = False; self.receipts = []
        self.counts = {k:dict(attempted=0,completed=0) for k in contract()['caps']}
        self.active = {k:0 for k in self.counts}
    def call(self,kind,fn,*,cleanup=False,**fields):
        require(not cleanup or kind == 'close', 'cleanup only close')
        if not cleanup:
            require(not self.fault and time.monotonic()-self.started < 900, 'wall/fault predispatch closure'); self.store.check()
        c = self.counts[kind]; require(c['attempted'] < contract()['caps'][kind], 'cap BEFORE dispatch: '+kind)
        r = dict(index=len(self.receipts),resource=kind,**deepcopy(self.where),**deepcopy(fields),status='pending')
        try: self.store.emit('operation_dispatch_intent',receipt=r)
        except BaseException as exc:
            self.fault = True
            if not cleanup: raise
            self.secondary.append(base.error('cleanup_intent',exc))
        self.receipts.append(r); c['attempted'] += 1; self.active[kind] += 1
        try:
            result = fn(); c['completed'] += 1; r['status'] = 'returned'
        except BaseException as exc:
            r.update(status='failed',error=base.error(kind,exc))
            try: self.store.emit('operation_failed',receipt=r)
            except BaseException as secondary:
                self.fault = True; self.secondary.append(base.error('failed_journal',secondary))
            raise
        finally: self.active[kind] -= 1
        try: self.store.emit('operation_returned',receipt=r)
        except BaseException as exc:
            self.fault = True
            if not cleanup: raise
            self.secondary.append(base.error('cleanup_return',exc))
        return result


def validate_execution(p):
    require(p['schema'] == NAME+'.prepare' and p['contract'] == contract() and p['MAIN_verified8'] is True and
            len(p['plans']) == len(p['anchors']) == len(p['suffix_plans']) == 8 and
            [(x['root_slot'],x['condition'],x['arm']) for x in p['plans']] == ORDER, 'own fixed8 verified prepare')
    for i,(selected,original) in enumerate(zip(p['suffix_plans'],p['plans'],strict=True)):
        plan = selected['plan']
        require(type(plan['anchor']) is int and plan['anchor'] == i and plan['arm'] == ARM and
                (plan['root_slot'],plan['condition'],plan['prefix_arm']) == ORDER[i] and
                plan['task_id'] == original['task_id'] and plan['original_selection_digest'] == original['selection_digest'] and
                raw5(plan['prefix_commands']).tobytes() == base.previous.commands(original['commands'])[:5].tobytes() and
                plan['commands'] == plan['prefix_commands']+plan['suffix_commands'], 'exact own CEM/raw/prefix identity')
        raw5(plan['suffix_commands']); actual_hash(selected['branch_sha256'])
        require(selected['compiler_report_sha256'] == plan['whole_run_report_sha256'] == p['compiler_report_sha256'] and
                selected['execution_sha256'] == base.source.digest({k:v for k,v in selected.items() if k != 'execution_sha256'}), 'immutable physical execution seal')


def census(rows,counts):
    c = {k:v['completed'] for k,v in counts.items()}
    require(all(v['attempted'] == v['completed'] for v in counts.values()) and c['constructor'] == c['close'] == 8 and
            c['reset'] == 16 and c['goal_bind'] == 8 and c['set_state'] == 2*c['reset'] and
            c['goal_setter'] == c['reset']+c['goal_bind'] and c['goal_render'] == c['reset'] and
            c['physics_step'] == 2*c['reset']+10*c['public_step'] and
            c['public_step'] == sum(r['warmup_returned']+r['prefix_returned']+r['suffix_returned'] for r in rows) and
            c['explicit_render'] == sum(r['PNG'] for r in rows), 'actual independent resource axes; no padded caps')


def execute(p,data,store,factory,*,authorization,prepared,prepare_sha256,started=None):
    require(authorization == AUTH, 'MAIN explicit physics authorization BEFORE factory'); actual_hash(prepare_sha256); validate_execution(p)
    rows = [dict(index=i,anchor_index=i,root_slot=x['plan']['root_slot'],condition=x['plan']['condition'],arm=ARM,
                 prefix_arm=x['plan']['prefix_arm'],task_id=x['plan']['task_id'],selection_digest=x['plan']['original_selection_digest'],
                 selected_suffix=deepcopy(x),status='not_started',first_error=None,secondary_errors=[],
                 start_verified=False,start_state=None,final_state=None,final_metrics=None,native_success=None,terminated=None,truncated=None,
                 constructor_internal_work='not_attempted',responses=[],trajectory=[],checkpoints=[],missing_checkpoints=[0,5,10],PNG=0,
                 warmup_attempted=0,warmup_returned=0,own_attempted=0,own_returned=0,prefix_attempted=0,prefix_returned=0,suffix_attempted=0,suffix_returned=0)
            for i,x in enumerate(p['suffix_plans'])]
    meter = Meter(store,started); errors = []; seen = []; final_ok = False
    binding = dict(prepared=str(Path(prepared).absolute()),prepare_sha256=prepare_sha256,compiler_report_sha256=p['compiler_report_sha256'],
                   consumer_report_sha256=p['consumer_report_sha256'],physical_report_sha256=p['physical_report_sha256'])
    def fail(stage,exc,row=None):
        e = base.error(stage,exc); errors.append(e)
        if row is not None:
            if row['first_error'] is None: row['first_error'] = deepcopy(e)
            else: row['secondary_errors'].append(deepcopy(e))
            row['status'] = 'engineering_failed'
    try:
        store.put('execution_binding.json',binding); store.emit('run_started',prepare_sha256=prepare_sha256,contract=contract())
        for row in rows:
            i = row['index']; meter.where = dict(branch=i,anchor_index=i,**{k:row[k] for k in ('root_slot','condition','arm','prefix_arm','selection_digest')})
            env = None; holder = []
            def construct():
                row['constructor_internal_work'] = 'unknown'; value = factory(); holder.append(value); return value
            try:
                env = meter.call('constructor',construct)
                require(all(env is not x and env.unwrapped is not x.unwrapped for x in seen), 'fresh env for EACH CEM suffix')
                seen.append(env); base.validate_env(env.unwrapped)
                require(all(getattr(env.unwrapped,k,None) is None for k in ('space','latest_action','render_buffer')) and
                        env.unwrapped.render_action is False and env.unwrapped.render_mode == 'rgb_array', 'pristine native target-free env')
                row.update(status='running',constructor_internal_work='sealed_native_pristine_no_physics')
                with base.native_hooks(env,meter,row['secondary_errors']):
                    # The frozen kernel only reads selected.plan.suffix. This is
                    # an internal literal raw view, NEVER a fabricated producer plan.
                    view = dict(plan=dict(suffix=deepcopy(p['suffix_plans'][i]['plan']['suffix_commands'])))
                    replay.episode(env,data['roots'][row['root_slot']],data['source_plan'],p['plans'][i],p['anchors'][i],view,row,meter,store)
                require(not row['secondary_errors'], 'native hook cleanup failed')
            except BaseException as exc: fail('branch',exc,row)
            finally:
                if env is None and holder: env = holder[0]
                if env is not None:
                    try: meter.call('close',env.close,cleanup=True)
                    except BaseException as exc: fail('close',exc,row)
                steps = [r for r in meter.receipts if r.get('branch') == i and r['resource'] == 'public_step']
                for phase in ('warmup','plan','suffix'):
                    label = 'prefix' if phase == 'plan' else phase
                    row[label+'_attempted'] = sum(r.get('phase') == phase for r in steps)
                    row[label+'_returned'] = sum(r.get('phase') == phase and r['status'] == 'returned' for r in steps)
                row['own_attempted'] = row['prefix_attempted']+row['suffix_attempted']; row['own_returned'] = row['prefix_returned']+row['suffix_returned']
                row['full_suffix'] = row['suffix_returned'] == 5
                if row['secondary_errors'] and row['first_error'] is None: fail('cleanup',ValueError('see secondary errors'),row)
            store.emit('branch_closed',branch=i,receipt=row)
    except BaseException as exc: fail('execution',exc)
    try:
        for path,h in p['bindings'].items(): require(sha(path) == h, 'final input/source seal')
        final_ok = True
    except BaseException as exc: fail('final_inputs',exc)
    errors.extend(meter.secondary)
    if time.monotonic()-meter.started >= 900: fail('wall',ValueError('900s exhausted; cleanup/finalization retained'))
    if not errors:
        try: census(rows,meter.counts)
        except BaseException as exc: fail('census',exc)
    good = not errors and final_ok and all(r['status'] in ('suffix_completed','native_stopped') for r in rows)
    status = RECORDED if good else 'partial_or_failed'
    report = dict(schema=NAME+'.report',status=status,rows=rows,errors=errors,fixed_denominator=8,known_roots=2,fresh_roots=0,
                  execution_binding=binding,contract=contract(),counts=meter.counts,actual_operation_receipts=meter.receipts,inputs_final_verified=final_ok,
                  full_suffixes=sum(r.get('full_suffix',False) for r in rows),early_stops=sum(r['status'] == 'native_stopped' for r in rows),
                  engineering_failures=sum(r['status'] == 'engineering_failed' for r in rows),not_started=sum(r['status'] == 'not_started' for r in rows),
                  new_neural_calls=0,new_independent_samples=0,training=False,P1_admission=False,scientific_admission=False,independent_confirmation=False,
                  equal_total_compute=False,equal_FLOPs=False,wall_seconds=time.monotonic()-meter.started,
                  total_internal_physics_scope='unknown_constructor_work' if any(r['constructor_internal_work'] == 'unknown' for r in rows) else 'hook_observed_native_work')
    try:
        store.emit('run_finished',status=status,counts=meter.counts,errors=errors)
        report.update(journal_sha256=sha(store.path/'journal.jsonl'),journal_chain_tail=store.previous,
                      execution_binding_sha256=sha(store.path/'execution_binding.json'))
    except BaseException as exc: fail('final_journal',exc); report['status'] = 'partial_or_failed'
    try: store.put('report.json',report,final=True)
    except BaseException as exc: fail('final_report',exc); report['status'] = 'partial_or_failed'
    return report


def run(prepared,expected,output,factory,*,authorization):
    require(authorization == AUTH, 'MAIN authorization BEFORE verify/factory')
    started = time.monotonic(); p,data = verify(prepared,expected)
    return execute(p,data,Store(output),factory,authorization=authorization,prepared=prepared,prepare_sha256=expected,started=started)


def validate_saved(report,events,p):
    validate_execution(p)
    require(report['schema'] == NAME+'.report' and report['status'] in (RECORDED,'partial_or_failed') and
            report['fixed_denominator'] == len(report['rows']) == 8 and report['known_roots'] == 2 and report['fresh_roots'] == 0 and
            report['contract'] == contract() and all(report['execution_binding'][k] == p[k] for k in
                ('compiler_report_sha256','consumer_report_sha256','physical_report_sha256')), 'own fixed8 report and actual input identity')
    require(all(report[k] is False for k in ('training','P1_admission','scientific_admission','independent_confirmation','equal_total_compute','equal_FLOPs')) and
            report['new_neural_calls'] == report['new_independent_samples'] == 0, 'no scientific/freshness/compute upgrade')
    require(events[0]['kind'] == 'run_started' and events[0]['prepare_sha256'] == report['execution_binding']['prepare_sha256'] and
            events[0]['contract'] == contract() and events[-1]['kind'] == 'run_finished' and events[-1]['status'] == report['status'] and
            events[-1]['counts'] == report['counts'] and events[-1]['errors'] == report['errors'], 'durable begin/end closure')
    intents,terminals,stack = {},{},[]; tail = None
    for i,e in enumerate(events):
        require(e['index'] == i and e['previous_sha256'] == tail and e['event_sha256'] == base.source.digest({k:v for k,v in e.items() if k != 'event_sha256'}), 'physical hashchain')
        tail = e['event_sha256']
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'): continue
        r = e['receipt']; n = r['index']; require(type(n) is int and r['resource'] in contract()['caps'], 'literal resource/index')
        if e['kind'] == 'operation_dispatch_intent':
            require(n == len(intents) and r['status'] == 'pending', 'ordered durable dispatch intent'); intents[n] = r; stack.append(n)
        else:
            require(n in intents and n not in terminals and stack and stack.pop() == n and
                    r['status'] == ('returned' if e['kind'] == 'operation_returned' else 'failed') and
                    dict({k:v for k,v in r.items() if k != 'error'},status='pending') == intents[n], 'exact nested actual return/failure')
            terminals[n] = r
    require(tail == report['journal_chain_tail'] and not stack and len(intents) == len(terminals) and
            report['actual_operation_receipts'] == [terminals[i] for i in range(len(terminals))], 'all actual operations durably matched')
    counts = {k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for r in report['actual_operation_receipts']:
        counts[r['resource']]['attempted'] += 1; counts[r['resource']]['completed'] += r['status'] == 'returned'
    require(counts == report['counts'] and all(v['attempted'] <= contract()['caps'][k] for k,v in counts.items()), 'actual caps, no refund/fixedcount fill')
    require([e['receipt'] for e in events if e['kind'] == 'branch_closed'] == [r for r in report['rows'] if r['status'] != 'not_started'], 'fixed8 failure denominator')
    for i,row in enumerate(report['rows']):
        selected,original,anchor = p['suffix_plans'][i],p['plans'][i],p['anchors'][i]
        require(type(row['index']) is type(row['anchor_index']) is int and row['index'] == row['anchor_index'] == i and
                row['selected_suffix'] == selected and row['arm'] == ARM and
                (row['root_slot'],row['condition'],row['prefix_arm']) == ORDER[i] and
                row['selection_digest'] == original['selection_digest'] and row['task_id'] == original['task_id'], 'own anchor/task/context/winner identity')
        scoped = [e for e in events if e.get('branch') == i]
        for e in scoped:
            if e['kind'] == 'branch_closed': continue
            require(all(e[k] == row[k] for k in ('root_slot','condition','arm','prefix_arm','selection_digest')), 'branch-scoped observation identity')
        stepops = [r for r in report['actual_operation_receipts'] if r['resource'] == 'public_step' and r.get('branch') == i]
        for phase,commands in (('warmup',base.moving.warmup_commands()),('plan',base.previous.commands(original['commands'])[:5]),('suffix',raw5(selected['plan']['suffix_commands']))):
            ops = [r for r in stepops if r['phase'] == phase]; label = 'prefix' if phase == 'plan' else phase
            require([r['step'] for r in ops] == list(range(1,len(ops)+1)) and len(ops) <= len(commands) and
                    all(r['raw_bits'] == commands[j].tobytes().hex() for j,r in enumerate(ops)) and
                    row[label+'_attempted'] == len(ops) and row[label+'_returned'] == sum(r['status'] == 'returned' for r in ops), 'contiguous literal raw bits/actual attempts/returns')
        observations = [e for e in scoped if e['kind'] == 'fresh_observation']
        require(row['PNG'] == len(observations), 'actual PNG denominator')
        if row['suffix_attempted']:
            verified = [e for e in scoped if e['kind'] == 'suffix_start_verified']
            require(len(verified) == 1 and row['start_verified'] is True and row['start_state'] == anchor['checkpoint']['state'], 'all51 exact start mandatory')
            before = [e for e in observations if e['index'] < verified[0]['index']]
            suffix_intents = [e for e in events if e['kind'] == 'operation_dispatch_intent' and e['receipt'].get('branch') == i and
                              e['receipt']['resource'] == 'public_step' and e['receipt'].get('phase') == 'suffix']
            require(len(before) == 51 and len(suffix_intents) == row['suffix_attempted'] and
                    all(e['index'] > verified[0]['index'] for e in suffix_intents), '51 restores BEFORE suffix dispatch')
            for actual,expected in zip(before,anchor['trace'],strict=True):
                require(actual['pre'] == actual['post'] == expected['pre'] == expected['post'] and
                        all(actual['receipt'][k] == expected['receipt'][k] for k in ('phase','step','condition','overlay_index','file_sha256','pixel_sha256','state','snapshot')), 'all51 exact goal/render/native7/body/RGB')
        responses = row['responses']; require([r['step'] for r in responses] == list(range(1,len(responses)+1)) and len(responses) <= row['suffix_returned'], 'actual response chronology')
        trans = [e for e in scoped if e['kind'] == 'returned_transition' and e['phase'] == 'suffix']
        for response in responses:
            state,start = base.source.vector(response['state']),base.source.vector(row['start_state']); delta = state-start
            require(np.array_equal(delta,response['delta_native7']) and response['delta_angle_wrapped'] == float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))) and
                    response['native_metrics'] == base.previous.metric(row['goal'],state), 'actual7/native labels, not modelQ/old endpoint')
            t = next(e for e in trans if e['step'] == response['step'])
            require(t['state'] == response['state'] and t['reward'] == response['reward'] == -response['native_metrics']['native_distance'] and
                    t['terminated'] == response['terminated'] == response['native_success'] == response['native_metrics']['native_success'] and
                    t['truncated'] == response['truncated'], 'native return/reward/stop join')
        if responses:
            require(all(row[k] == responses[-1][k] for k in ('native_success','terminated','truncated')) and row['final_state'] == responses[-1]['state'] and
                    row['final_metrics'] == responses[-1]['native_metrics'], 'own new endpoint only')
        else: require(row['final_state'] is row['final_metrics'] is row['native_success'] is None, 'unknown endpoint stays null')
        require(row['own_attempted'] == row['prefix_attempted']+row['suffix_attempted'] and row['own_returned'] == row['prefix_returned']+row['suffix_returned'], 'prefix/suffix separate own accounting')
        if row['status'] in ('suffix_completed','native_stopped'):
            require(row['first_error'] is None and not row['secondary_errors'] and row['start_verified'] and row['warmup_returned'] == 10 and row['prefix_returned'] == 5 and
                    len(responses) == row['suffix_returned'] and row['full_suffix'] == (len(responses) == 5) and
                    all(not r['terminated'] and not r['truncated'] for r in responses[:-1]) and
                    ((row['status'] == 'native_stopped') == (responses[-1]['terminated'] or responses[-1]['truncated'])), 'no padding/retry after native stop')
            expected_steps = [('warmup',n) for n in range(11)]+[('plan',n) for n in range(6)]+[('suffix',n) for n in range(1,len(responses)+1)]
            require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in observations] ==
                    [(phase,n,k) for phase,n in expected_steps for k in (0,1,2)], 'complete new off/on/off chronology')
            expected_cp = [0,5]+([10] if len(responses) == 5 else [])
            require([c['step'] for c in row['checkpoints']] == expected_cp and row['missing_checkpoints'] == ([] if len(responses) == 5 else [10]), 'only reached actual checkpoints')
            for cp in row['checkpoints']:
                phase,step = ('suffix',5) if cp['step'] == 10 else ('plan',cp['step'])
                f = next(e['receipt'] for e in observations if e['receipt']['phase'] == phase and e['receipt']['step'] == step and e['receipt']['overlay_index'] == (1 if row['condition'] == 'on' else 0))
                require(all(cp[k] == f[k] for k in ('png','file_sha256','pixel_sha256','state','snapshot')), 'checkpoint from own fresh chosen-condition frame')
    for label,computed in dict(full_suffixes=sum(r.get('full_suffix',False) for r in report['rows']),early_stops=sum(r['status']=='native_stopped' for r in report['rows']),
                               engineering_failures=sum(r['status']=='engineering_failed' for r in report['rows']),not_started=sum(r['status']=='not_started' for r in report['rows'])).items():
        require(report[label] == computed, 'complete8 outcome/failure summary')
    if report['status'] == RECORDED:
        require(report['errors'] == [] and report['inputs_final_verified'] is True and report['wall_seconds'] < 900 and
                all(r['status'] in ('suffix_completed','native_stopped') for r in report['rows']), 'engineering complete, not scientific pass')
        census(report['rows'],counts)
    return deepcopy(report['rows'])


def consume(output,expected):
    actual_hash(expected); directory = Path(absolute(str(output))); files = list(directory.iterdir())
    require(all(f.is_file() and not f.is_symlink() for f in files) and sum(f.stat().st_size for f in files) <= CAP, 'bounded physical output')
    rp = directory/'report.json'; require(sha(rp) == expected, 'external actual physical report SHA')
    report = read(rp,CAP); binding = report['execution_binding']
    require(sha(directory/'execution_binding.json') == report['execution_binding_sha256'] and read(directory/'execution_binding.json') == binding, 'actual execution binding')
    p,_ = verify(binding['prepared'],binding['prepare_sha256']); jp = directory/'journal.jsonl'
    require(sha(jp) == report['journal_sha256'] and jp.stat().st_size <= CAP, 'actual journal SHA/size')
    content = jp.read_bytes(); require(content.endswith(b'\n'), 'complete journal'); events = [json.loads(line) for line in content.splitlines()]
    rows = validate_saved(report,events,p); names = {'report.json','execution_binding.json','journal.jsonl'}
    from PIL import Image
    for e in events:
        if e['kind'] == 'invalid_observation':
            name = e['file']; require(Path(name).name == name and name not in ('','.','..'), 'flat invalid artifact')
            require(sha(directory/name) == e['file_sha256'], 'partial incident seal'); names.add(name)
        if e['kind'] != 'fresh_observation': continue
        r = e['receipt']; name = r['png']; require(Path(name).name == name and name not in ('','.','..'), 'flat PNG')
        path = directory/name; names.add(name); require(sha(path) == r['file_sha256'] and path.stat().st_size <= 1024**2, 'actual fresh PNG SHA')
        with Image.open(io.BytesIO(path.read_bytes())) as image:
            require(image.mode == 'RGB' and image.size == (224,224) and base.moving.image_sha(np.asarray(image)) == r['pixel_sha256'], 'actual RGB224 pixel SHA')
    require({f.name for f in files} == names, 'no unaccounted artifacts/partial writes')
    for path,h in p['bindings'].items(): require(sha(path) == h, 'final full source/raw/task closure')
    require(sha(rp) == expected, 'final report seal')
    return dict(schema=NAME+'.consume',status=report['status'],rows=rows,fixed_denominator=8,known_roots=2,fresh_roots=0,
                counts=report['counts'],physical_report_sha256=expected,compiler_report_sha256=p['compiler_report_sha256'],
                consumer_report_sha256=p['consumer_report_sha256'],P1_admission=False,scientific_admission=False,independent_confirmation=False,
                equal_total_compute=False,equal_FLOPs=False,new_neural_calls=0,new_physics_calls=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('command',choices=('prepare','verify','run','consume'))
    for k in ('output','prepared','prepare-sha','report-sha','compiler-output','compiler-report-sha','producer-source-sha',
              'consumer-source-sha','consumer-report','consumer-report-sha','physical-output','physical-report-sha'):
        parser.add_argument('--'+k)
    parser.add_argument('--authorize-physics',action='store_true'); a = parser.parse_args(argv)
    if a.command == 'prepare':
        args = dict(compiler_output=a.compiler_output,compiler_report_sha256=a.compiler_report_sha,producer_source_sha256=a.producer_source_sha,
                    consumer_source_sha256=a.consumer_source_sha,consumer_report=a.consumer_report,consumer_report_sha256=a.consumer_report_sha,
                    physical_output=a.physical_output,physical_report_sha256=a.physical_report_sha)
        result = prepare(a.output,args)
    elif a.command == 'verify': p,_ = verify(a.prepared,a.prepare_sha); result = dict(status='verified_no_execution',contract=p['contract'])
    elif a.command == 'consume': result = consume(a.output,a.report_sha)
    else:
        require(a.authorize_physics, 'MAIN CLI explicit physics authorization')
        def factory():
            import gymnasium as gym
            import stable_worldmodel  # noqa: F401
            return gym.make('swm/PushT-v1',render_mode='rgb_array',resolution=224,with_target=False)
        result = run(a.prepared,a.prepare_sha,a.output,factory,authorization=AUTH)
    print(base.source.canonical({k:v for k,v in result.items() if k not in ('rows','actual_operation_receipts')}))
    return 2 if result.get('status') == 'partial_or_failed' else 0


if __name__ == '__main__': raise SystemExit(main())
