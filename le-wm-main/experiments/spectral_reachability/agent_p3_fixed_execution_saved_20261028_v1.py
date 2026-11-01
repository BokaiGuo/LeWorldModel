"""Read-only strict consumer of frozen fixed5 physical v2 artifacts.

Only repaired boundary: native observation wraps placement angle modulo 2pi;
the actual snapshot must retain the original unwrapped placement angle.
No prepare/run/environment/model API, schema projection or producer mutation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import io
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_fixed_execution_physical_20261028_v2 as physical

NAME='agent_p3_fixed_execution_saved_20261028_v1'
PHYSICAL=physical.NAME
ROOT=physical.ROOT
FROZEN_SHAS=(
    '5b9532d57564472abf07f305ab62ffeef98ba046b9ac7104dac478c42d33dc92',
    '207f610b30c1581898dc7eb483f324b8be65d573a05c66be1071603f9e746a8c',
    '3e390d1c32ccdaa4c693a1a8ae15225e6d846044d865ffa455f70302449aeff2')
ACTUAL_REPORT=ROOT/'results/agent_p3_fixed_execution_physical_20261028_v2_MAIN01/report.json'
ACTUAL_REPORT_SHA='5fdfc8cfea7f6b4fe78d3e19b59fe946aad73a537850c52a86f2d395055b5e45'
ACTUAL_PREPARE_SHA='51837e4f8a97d6bed08031ceea70b549ea1bf893351c77e6061b698dc5763e85'
require,sha,raw,decode=physical.require,physical.sha,physical.raw,physical.decode
actual_hash,absolute,merge=physical.actual_hash,physical.absolute,physical.merge
source,moving,allmode,fixed=physical.source,physical.moving,physical.allmode,physical.fixed
FLAGS,RECORDED,WALL=physical.FLAGS,physical.RECORDED,physical.WALL
contract,commands,metrics=physical.contract,physical.commands,physical.metrics
validate_execution,physical_receipt,actual_equal=physical.validate_execution,physical.physical_receipt,physical.actual_equal
SOURCE_SHA=physical.SOURCE_SHA
saved_artifacts=physical.saved_artifacts
frozen_verify,frozen_material=physical.verify,physical.material


def own_files():
    return [Path(__file__).resolve(),ROOT/f'tests/{NAME}_test.py',ROOT/f'innerdocs/{NAME}.md']


def pin_frozen():
    paths=[ROOT/f'experiments/spectral_reachability/{PHYSICAL}.py',ROOT/f'tests/{PHYSICAL}_test.py',ROOT/f'innerdocs/{PHYSICAL}.md']
    require(Path(physical.__file__).resolve()==paths[0] and all(sha(p)==h for p,h in zip(paths,FROZEN_SHAS,strict=True)),
        'frozen physical v2 source/tests/doc exact SHA authority')
    return {str(p):h for p,h in zip(paths,FROZEN_SHAS,strict=True)}


def checked_files():
    values=pin_frozen()
    merge(values,{str(p):sha(p) for p in own_files()})
    return values


def material(arguments):
    pin_frozen();result=frozen_material(arguments);pin_frozen()
    return result


def verify(prepared,expected):
    # Frozen verify itself reconstructs frozen material and real producer.consume.
    pin_frozen();result=frozen_verify(prepared,expected);pin_frozen()
    return result


def validate_initial_observation(frame,recipe):
    """Exactly native _get_obs, not modulo-equivalence of physical identities."""
    requested=source.vector(recipe['requested_state'])
    observed=source.vector(frame['state'])
    require(requested.shape==observed.shape==(7,),'seven native placement/observation coordinates')
    expected=requested.copy();expected[4]=requested[4] % (2*np.pi)
    block=frame['snapshot']['bodies']['block']
    angle=np.asarray(block['unwrapped_angle'],dtype='<f8')
    bits=np.frombuffer(bytes.fromhex(block['float64_bits']),dtype='<f8')
    require(angle.shape==() and bits.shape==(9,) and np.isfinite(bits).all() and
        angle.tobytes()==np.asarray(requested[4],dtype='<f8').tobytes()==bits[4:5].tobytes(),
        'original requested unwrapped angle exact numeric AND snapshot float64 bits, never 2pi-equivalence')
    require(moving.exact(observed,expected),'native modulo observation; other six coordinates exact, no tolerance')
    return True


def validate_saved(report,events,p):
    """Own schema only. Rebuild actual dispatches, source bridge and task labels."""
    validate_execution(p)
    require(report['schema']==PHYSICAL+'.report' and report['contract']==contract() and
        report['status'] in (RECORDED,'partial_or_failed') and report['fixed_denominator']==len(report['rows'])==4 and
        report['known_roots']==2 and report['fresh_roots']==0 and
        report['new_neural_calls']==report['new_independent_samples']==0 and all(report[k] is False for k in FLAGS),
        'own full4 schema and closed scientific gates')
    binding=report['execution_binding']
    require(binding['producer_report_sha256']==p['producer_report_sha256'] and binding['source_report_sha256']==SOURCE_SHA,
        'actual producer and original physical source ancestry')
    require(events and events[0]['kind']=='run_started' and events[0]['contract']==contract() and
        events[0]['prepare_sha256']==binding['prepare_sha256'] and events[-1]['kind']=='run_finished' and
        all(events[-1][k]==report[k] for k in ('status','counts','errors')),'durable run boundaries')
    tail=None;intents={};terminals={};stack=[]
    allowed={'run_started','run_finished','operation_dispatch_intent','operation_returned','operation_failed',
        'placement_recipe','returned_transition','fresh_observation','invalid_observation','goal_attached',
        'own_response','task_terminal_at_root','branch_closed'}
    for i,e in enumerate(events):
        require(e['kind'] in allowed and type(e['index']) is int and e['index']==i and e['previous_sha256']==tail and
            e['event_sha256']==source.digest({k:v for k,v in e.items() if k!='event_sha256'}),'journal chain/type')
        tail=e['event_sha256']
        if 'branch' in e:
            b=e['branch'];require(type(b) is int and 0<=b<4 and all(e[k]==report['rows'][b][k]
                for k in ('mode','root_slot','condition','arm','selection_digest')),'event exact branch identity')
        if e['kind'] not in ('operation_dispatch_intent','operation_returned','operation_failed'):continue
        op=e['receipt'];n=op['index'];b=op['branch']
        require(type(n) is int and type(b) is int and 0<=b<4 and op['resource'] in contract()['caps'] and
            all(op[k]==report['rows'][b][k] for k in ('mode','root_slot','condition','arm','selection_digest')),'operation scope')
        if e['kind']=='operation_dispatch_intent':
            require(n==len(intents) and op['status']=='pending','ordered actual intent');intents[n]=op;stack.append(n)
        else:
            require(n in intents and n not in terminals and stack and stack.pop()==n and
                op['status']==('returned' if e['kind']=='operation_returned' else 'failed') and
                dict({k:v for k,v in op.items() if k!='error'},status='pending')==intents[n],'nested actual callback receipt')
            terminals[n]=op
    require(not stack and len(intents)==len(terminals) and tail==report['journal_chain_tail'] and
        report['actual_operation_receipts']==[terminals[i] for i in range(len(terminals))],'complete operation closure')
    counts={k:dict(attempted=0,completed=0) for k in contract()['caps']}
    for op in report['actual_operation_receipts']:
        counts[op['resource']]['attempted']+=1;counts[op['resource']]['completed']+=op['status']=='returned'
    require(counts==report['counts'] and all(type(report['counts'][k][q]) is int for k in counts for q in ('attempted','completed')) and
        all(v['attempted']<=contract()['caps'][k] for k,v in counts.items()),'actual counts and caps')
    require([e['receipt'] for e in events if e['kind']=='branch_closed']==[r for r in report['rows'] if r['status']!='not_started'],
        'fixed4 branch closure including failures')
    for i,row in enumerate(report['rows']):
        plan=p['plans'][i];slot=plan['slot'];task=p['source_roots'][slot];scoped=[e for e in events if e.get('branch')==i]
        require(type(row['index']) is type(row['slot']) is type(row['root_slot']) is int and row['index']==i and
            all(row[k]==plan[k] for k in ('slot','mode','root_slot','condition','arm','task_id','selection_digest','commands','request')) and
            row['status'] in ('five_steps_completed','native_stopped','engineering_failed','not_started'),'all4 exact original requests')
        ops=[op for op in report['actual_operation_receipts'] if op['branch']==i and op['resource']=='public_step']
        trans=[e for e in scoped if e['kind']=='returned_transition']
        resets=[op for op in report['actual_operation_receipts'] if op['branch']==i and op['resource']=='reset']
        recipe=[e for e in scoped if e['kind']=='placement_recipe']
        require(len(resets)<=2 and all(op['role']==('seed','placement')[j] and
            op['seed']==(p['source_slots'][slot]['reset_seed'] if j==0 else 0) for j,op in enumerate(resets)),
            'original reset seed then literal placement reset, never task reset padding')
        require(len(recipe)<=1 and (not recipe or recipe[0]['slot']==p['source_slots'][slot]),'original literal seed/placement recipe')
        for phase,label,actions in (('warmup','warmup',moving.warmup_commands()),('plan','own',commands(plan['commands']))):
            current=[op for op in ops if op['phase']==phase];ts=[t for t in trans if t['phase']==phase]
            require(len(current)<=len(actions) and [op['step'] for op in current]==list(range(1,len(current)+1)) and
                all(op['raw_bits']==actions[j].tobytes().hex() for j,op in enumerate(current)) and
                row[label+'_attempted']==len(current) and row[label+'_returned']==sum(op['status']=='returned' for op in current),
                'exact contiguous actual raw actions, no H3 tail/padding')
            require([t['step'] for t in ts]==list(range(1,len(ts)+1)) and len(ts)<=row[label+'_returned'],'actual return chronology')
            for t in ts:
                require(t['requested_bits']==actions[t['step']-1].tobytes().hex(),'returned request bits')
                if row['status']!='engineering_failed':require(t['actual_bits']==t['requested_bits'] and
                    t['actual_dtype']==np.dtype(np.float32).str,'no raw substitution')
            require(all(not t['terminated'] and not t['truncated'] for t in ts[:-1]) and
                (not ts or not (ts[-1]['terminated'] or ts[-1]['truncated']) or len(current)==ts[-1]['step']),
                'no dispatch after any actual native stop, even before label validation')
        obs=[e for e in scoped if e['kind']=='fresh_observation'];frames=[{k:deepcopy(e[k]) for k in ('receipt','pre','post')} for e in obs]
        require(row['PNG']==len(obs) and row['prefix51']==dict(complete=len(frames)==51,frames=frames),'actual51 no padding')
        unknown=[]
        for e in obs:
            f=e['receipt'];require(f['phase'] in ('warmup','plan') and type(f['step']) is int and
                0<=f['step']<=(10 if f['phase']=='warmup' else 5) and type(f['overlay_index']) is int and
                f['overlay_index'] in (0,1,2) and f['condition']==('on' if f['overlay_index']==1 else 'off') and
                f['png']==f'b{i}_{f["phase"]}_{f["step"]:02}_{f["overlay_index"]}.png','frame horizon and overlays')
            require(e['pre']==e['post'] and f['snapshot']==e['post']['snapshot'] and f['state']==f['snapshot']['state'] and
                e['pre']['renderer']==task['renderer'] and e['pre']['goal_pose']==task['goal_pose'],'original renderer/body invariant')
            if f['step']:
                ts=[t for t in trans if t['phase']==f['phase'] and t['step']==f['step']]
                require(len(ts)==1 and f['state']==ts[0]['state'] and f['snapshot']==ts[0]['snapshot'],
                    'render is actual returned transition, not substituted old frame')
            elif f['phase']=='warmup':
                require(len(recipe)==1,'one actual placement recipe')
                validate_initial_observation(f,recipe[0])
            else:require(f['state']==task['root_state'],'actual requested native root')
            if f['phase']=='warmup':
                require(len(recipe)==1 and e['pre']['goal_state']==recipe[0]['native_goal'],'actual original native warmup goal')
                ref=allmode.reference_for(p['warm_traces'][slot],f['step'],f['overlay_index'])
                if ref is not None:allmode.check_warm_frame(e,ref)
                else:unknown.append(f['step'])
            else:require(e['pre']['goal_state']==task['goal'] and e['pre']['goal_rgb_sha256']==task['goal_pixels']['off'],
                'actual original off goal, not constructed goal_pose')
        require(unknown==row['source_unpaired_on_steps'],'unavailable historical on frames remain unknown')
        for e in obs:
            f=e['receipt']
            if f['overlay_index']==2:
                first=[x['receipt'] for x in obs if x['receipt']['phase']==f['phase'] and x['receipt']['step']==f['step'] and
                    x['receipt']['overlay_index']==0]
                require(len(first)==1 and all(first[0][k]==f[k] for k in ('pixel_sha256','state','snapshot')),'actual off/on/off replay')
        selected=[e['receipt'] for e in obs if e['receipt']['overlay_index']==0 and
            any(x['receipt']['phase']==e['receipt']['phase'] and x['receipt']['step']==e['receipt']['step'] and
                x['receipt']['overlay_index']==2 for x in obs)]
        require(row['trajectory']==selected and row['last_observation']==(obs[-1]['receipt'] if obs else None),
            'actual selected off trajectory and last rendered identity')
        actual=[{k:deepcopy(v) for k,v in e.items() if k not in ('kind','index','previous_sha256','event_sha256')}
            for e in trans if e['phase']=='plan']
        require(actual==row['actual_returns'],'durable actual returns distinct from validated labels')
        responses=row['responses'];require([e['response'] for e in scoped if e['kind']=='own_response']==responses and
            [x['step'] for x in responses]==list(range(1,len(responses)+1)) and len(responses)<=len(actual),'accepted response chronology')
        if row['initial_exact']:
            require(all(row[k]==task[k] for k in ('root_state','root_sha256','goal','goal_pose')) and
                row['goal_pixel_sha256']==task['goal_pixels']['off'] and
                source.root_hash(source.vector(row['root_state']),source.vector(row['goal']))==plan['request']['binding']['root_sha256'],
                'native task root and model request source binding')
            attached=[e for e in scoped if e['kind']=='goal_attached']
            require(len(attached)==1 and attached[0]['goal']==row['goal'] and attached[0]['goal_pose']==row['goal_pose'] and
                attached[0]['goal_rgb_sha256']==row['goal_pixel_sha256'],'actual true goal attach')
            warmroot=[e['receipt'] for e in obs if e['receipt']['phase']=='warmup' and e['receipt']['step']==10]
            require(warmroot and attached[0]['snapshot']==warmroot[-1]['snapshot'],'goal attach preserves complete native body state')
            require(all(e['receipt']['snapshot']==attached[0]['snapshot'] for e in obs if
                e['receipt']['phase']=='plan' and e['receipt']['step']==0),'plan0 is original body after pure goal attach')
        for j,r in enumerate(responses):
            t=actual[j];calc=metrics(row['goal'],r['state']);delta=source.vector(r['state'])-source.vector(row['root_state'])
            require(r['native_metrics']==calc['native'] and r['reward']==calc['native7_reward'] and
                r['native_success'] is r['terminated'] is calc['native_success'] and r['delta_native7']==delta.tolist() and
                r['delta_angle_wrapped']==float(np.arctan2(np.sin(delta[4]),np.cos(delta[4]))) and
                all(r[k]==t[k] for k in ('step','state','reward','terminated','truncated')) and t['goal_state']==row['goal'] and
                t['actual_bits']==t['requested_bits'] and t['actual_dtype']==np.dtype(np.float32).str,'independent native metrics/return/raw')
        require(all(not r['terminated'] and not r['truncated'] for r in responses[:-1]),'no execution after native stop')
        if responses:
            last=responses[-1];require(row['final_state']==last['state'] and row['final_metrics']==last['native_metrics'] and
                all(row[k] is last[k] for k in ('native_success','terminated','truncated')),'last validated endpoint')
        elif row['final_state'] is not None:
            terminal=[e for e in scoped if e['kind']=='task_terminal_at_root'];calc=metrics(row['goal'],row['root_state'])
            require(row['initial_exact'] and len(terminal)==1 and terminal[0]['state']==row['root_state']==row['final_state'] and
                terminal[0]['metrics']==row['final_metrics']==calc['native'] and row['native_success'] is row['terminated'] is True and
                row['truncated'] is False and calc['native_success'] is True,'native terminal at bound root')
        else:require(row['final_metrics'] is row['native_success'] is row['terminated'] is row['truncated'] is None,'unknown never zero')
        cps=[]
        for e in obs:
            f=e['receipt']
            if f['phase']=='plan' and f['step'] in (0,5) and f['overlay_index']==0:
                cps.append({k:deepcopy(f[k]) for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')})
        require(row['checkpoints']==cps and row['missing_checkpoints']==[n for n in (0,5) if n not in [c['step'] for c in cps]],'actual checkpoint identities')
        require(row['physical_receipt']==physical_receipt(row),'actual physical receipt, no invented missing return')
        if row['physical_receipt'] is not None:fixed.audit_physical_receipt(row['request'],row['physical_receipt'])
        if row['status'] in ('five_steps_completed','native_stopped'):
            require(row['first_error'] is None and row['secondary_errors']==[] and len(responses)==row['own_returned'],'clean engineering closure')
            expected=[('warmup',n,k) for n in range(row['warmup_returned']+1) for k in (0,1,2)]
            if row['initial_exact']:expected += [('plan',n,k) for n in range(row['own_returned']+1) for k in (0,1,2)]
            require([(e['receipt']['phase'],e['receipt']['step'],e['receipt']['overlay_index']) for e in obs]==expected,'complete actual frame chronology')
            if row['status']=='five_steps_completed':require(row['initial_exact'] and row['own_returned']==5 and
                not row['terminated'] and not row['truncated'],'five actual returns only')
            elif row['initial_exact']:require(row['terminated'] or row['truncated'],'native stop label')
    require(report['equal_requested_execution_horizon'] is True and
        report['equal_actual_physical_execution_horizon'] is actual_equal(report['rows']) and
        report['all_requested_steps_completed'] is all(r['status']=='five_steps_completed' and r['own_returned']==5 for r in report['rows']),
        'actual length summary distinct from equal requested length')
    scope='metered_native_calls' if counts['constructor']['attempted']==counts['constructor']['completed'] else 'unknown_constructor_work'
    require(report['total_internal_physics_scope']==scope,'unknown constructor internal work retained')
    if report['status']==RECORDED:
        require(not report['errors'] and report['inputs_final_verified'] is True and 0<=report['wall_seconds']<WALL and
            all(r['status'] in ('five_steps_completed','native_stopped') for r in report['rows']),'engineering complete not admission')
        c={k:v['completed'] for k,v in counts.items()}
        require(c['constructor']==c['close']==4 and c['reset']==8 and c['set_state']==2*c['reset'] and
            c['physics_step']==2*c['reset']+10*c['public_step'] and c['goal_render']==c['reset'] and
            c['goal_setter']==c['reset']+c['goal_bind'] and c['goal_bind']==sum(r['initial_exact'] for r in report['rows']) and
            c['public_step']==sum(r['warmup_returned']+r['own_returned'] for r in report['rows']) and
            c['explicit_render']==sum(r['PNG'] for r in report['rows']) and
            all(v['attempted']==v['completed'] for v in counts.values()),'nested native census, not added axes')
    return deepcopy(report['rows'])


def consume(output,expected):
    pin_frozen();actual_hash(expected);directory=Path(absolute(str(output)));files=saved_artifacts(directory)
    require(hashlib.sha256(files['report.json']).hexdigest()==expected,'external actual physical report SHA')
    r=decode(files['report.json']);binding=r['execution_binding']
    require(hashlib.sha256(files['execution_binding.json']).hexdigest()==r['execution_binding_sha256'] and
        decode(files['execution_binding.json'])==binding,'actual external execution binding')
    p,_=verify(binding['prepared'],binding['prepare_sha256']);body=files['journal.jsonl']
    require(body.endswith(b'\n') and hashlib.sha256(body).hexdigest()==r['journal_sha256'],'complete actual journal SHA')
    events=[decode(line) for line in body.splitlines()];rows=validate_saved(r,events,p)
    names={'report.json','journal.jsonl','execution_binding.json'}
    from PIL import Image
    for e in events:
        if e['kind']=='invalid_observation':
            name=e['file'];require(Path(name).name==name and name not in ('','.','..') and
                hashlib.sha256(files[name]).hexdigest()==e['file_sha256'],'sealed invalid incident');names.add(name)
        if e['kind']!='fresh_observation':continue
        f=e['receipt'];name=f['png'];require(Path(name).name==name and name not in ('','.','..') and name not in names and
            len(files[name])<=1024**2 and hashlib.sha256(files[name]).hexdigest()==f['file_sha256'],'actual unique flat PNG SHA');names.add(name)
        with Image.open(io.BytesIO(files[name])) as image:
            require(image.mode=='RGB' and image.size==(224,224) and moving.image_sha(np.asarray(image))==f['pixel_sha256'],'actual RGB224 pixels')
    require(set(files)==names,'no unaccounted artifacts')
    bindings=deepcopy(p['bindings']);merge(bindings,checked_files());merge(bindings,{binding['prepared']:binding['prepare_sha256']})
    merge(bindings,{str(directory/name):hashlib.sha256(value).hexdigest() for name,value in files.items()})
    for path,h in p['bindings'].items():require(sha(path)==h,'final full source/input closure')
    require(saved_artifacts(directory)==files,'final actual artifact drift')
    pin_frozen()
    outcomes=[]
    for row in rows:
        valid=row['initial_exact'] and row['own_returned']==len(row['responses']) and row['physical_receipt'] is not None
        state=row['final_state'] if valid else None
        outcomes.append(dict(slot=row['slot'],mode=row['mode'],returned_steps=row['own_returned'],
            validated_steps=len(row['responses']),metrics=None if state is None else metrics(row['goal'],state),
            contract_audit=None if row['physical_receipt'] is None else fixed.audit_physical_receipt(row['request'],row['physical_receipt'])))
    return dict(schema=NAME+'.consume',physical_report_schema=r['schema'],status=r['status'],rows=rows,outcomes=outcomes,fixed_denominator=4,known_roots=2,fresh_roots=0,
        errors=deepcopy(r['errors']),counts=r['counts'],bindings=bindings,producer_consume=deepcopy(p['producer_consume']),
        physical_report_sha256=expected,producer_report_sha256=p['producer_report_sha256'],artifact_authenticity_verified=True,
        equal_requested_execution_horizon=True,equal_actual_physical_execution_horizon=r['equal_actual_physical_execution_horizon'],
        all_requested_steps_completed=r['all_requested_steps_completed'],new_neural_calls=0,new_physics_calls=0,**FLAGS)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('consume',))
    parser.add_argument('--output',required=True,help='existing frozen physical v2 artifact directory; read-only')
    parser.add_argument('--report-sha',required=True,help='external actual report SHA256')
    args=parser.parse_args(argv);result=consume(args.output,args.report_sha)
    print(source.canonical(result))
    return 2 if result['status']=='partial_or_failed' else 0


if __name__=='__main__':raise SystemExit(main())

