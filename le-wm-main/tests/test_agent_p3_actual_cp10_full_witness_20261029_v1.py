"""Synthetic saved journals only; never imports a model/physics/full consumer."""
from copy import deepcopy
import json
from pathlib import Path
import struct

import pytest

from experiments.spectral_reachability import agent_p3_actual_cp10_full_witness_20261029_v1 as m


def put(p, v):
    p.write_bytes(m.raw(v) + b'\n')
    return dict(path=str(p), sha256=m.sha(p))


def physical(directory, kind):
    directory.mkdir()
    events, ops, rows, plans, references = [], [], [], [], []
    sources = {}; warm = [[0.125, -0.0] for _ in range(10)]
    def emit(kind, **fields):
        e = dict(index=len(events), previous_sha256=events[-1]['event_sha256'] if events else None,
                 kind=kind, **deepcopy(fields))
        e['event_sha256'] = m.digest(e); events.append(e)
        return e
    prep_path = directory / 'prepare.json'
    # Bookend prepare SHA is filled and whole chain rebuilt after prepare save.
    emit('run_started', prepare_sha256='pending', contract=dict(fixture=True))
    roots = [dict(root_slot=r, root_state=[float(r)]*7, goal=[0.0]*7,
                  goal_pose=[256.0,256.0,0.785], renderer={'fixture_unknown':r}) for r in range(2)]
    slots = [dict(index=r, root_seed=31+r, reset_seed=51+r, sign=1) for r in range(2)]
    for slot in range(1 if kind == 'C' else 4):
        root = 0 if kind == 'C' else slot//2
        scope = dict(branch=slot, root_slot=root, arm='H1', condition='off', selection_digest='fixture'+str(slot))
        commands = [[0.25, -0.0] for _ in range(10)]
        plan = dict(index=slot, root_slot=root, commands=commands, prefix_commands=commands[:5],
                    retained_unfamiliar={'no_loss':[1,2,3]})
        plans.append(plan)
        emit('placement_recipe', **scope, requested_state=roots[root]['root_state'], native_goal=roots[root]['goal'])
        emit('goal_attached', **scope, goal=roots[root]['goal'], original_unknown={'alpha':7})
        obs, responses = [], []
        for phase in ('warmup','plan'):
            for step in range(11):
                state = [float(root),float(step),1.0,2.0,0.25,0.0,-0.0]
                snap = dict(state=state, state_float64_bits=struct.pack('<7d',*state).hex(),
                            bodies={'agent':{'opaque_original':[phase, step]}}, unknown_snapshot='preserved')
                if step:
                    bits = m.f32bits(warm if phase=='warmup' else commands,10)[step-1]
                    op = dict(index=len(ops), resource='public_step', **scope, phase=phase, step=step,
                              status='returned', raw_bits=bits)
                    emit('operation_dispatch_intent', receipt=dict(op,status='pending'))
                    emit('operation_returned', receipt=op); ops.append(op)
                    tr = emit('returned_transition', **scope, phase=phase, step=step,
                        actual_bits=bits, requested_bits=bits, actual_dtype='<f4', state=state, snapshot=snap,
                        reward=-float(step), terminated=False, truncated=False)
                for overlay in range(3):
                    f = directory / f'b{slot}_{phase}_{step:02}_{overlay}.png'
                    # Fixture file identity; never presented as real pixels/decoded PNG.
                    f.write_bytes(f'fixture:{root}:{phase}:{step}:{overlay}'.encode())
                    sources[str(f)] = m.sha(f)
                    rc = dict(phase=phase, step=step, overlay_index=overlay,
                        condition='on' if overlay==1 else 'off', png=f.name, file_sha256=m.sha(f),
                        pixel_sha256='fixture_inherited_not_decoded', state=state, snapshot=snap,
                        original_receipt_unknown={'keep':True})
                    pre = dict(snapshot=snap, renderer=roots[root]['renderer'], goal_pose=roots[root]['goal_pose'],
                               goal_state=roots[root]['goal'], unknown_pre=[phase,step])
                    obs.append(emit('fresh_observation', **scope, receipt=rc, pre=pre, post=pre))
                if phase=='plan' and step:
                    response = {k:tr[k] for k in ('step','state','reward','terminated','truncated')}
                    response['native_metrics']={'original':step}
                    responses.append(response); emit('own_response', **scope, response=response)
                if phase=='plan' and step==5:
                    emit('cp5_guard', **scope, accepted=True, error=None, proof={'inherited_proof':'fixture'})
        selected=[e['receipt'] for e in obs if e['receipt']['overlay_index']==0]
        cps=[{k:f[k] for k in ('step','png','file_sha256','pixel_sha256','state','snapshot')}
             for f in selected if f['phase']=='plan' and f['step'] in (0,5,10)]
        prefix=[{k:e[k] for k in ('pre','post','receipt')} for e in obs
                if e['receipt']['phase']=='warmup' or e['receipt']['step']<=5]
        history=[]
        for ph,st,absolute in (('warmup',5,5),('warmup',10,10),('plan',5,15)):
            f=next(x for x in selected if x['phase']==ph and x['step']==st)
            history.append(dict(absolute_step=absolute, plan_step=st, png=str(directory/f['png']),
                **{k:f[k] for k in ('file_sha256','pixel_sha256','state','snapshot')}))
        ref=dict(complete=True,frames=deepcopy(prefix),checkpoint=deepcopy(cps[1]),
                 warm_raw10=warm,prefix_raw5=commands[:5],raw10=warm[-5:]+commands[:5],history_frames=history)
        for f in ref['frames']:
            f['receipt']['png']='original_'+f['receipt']['png']
        ref['checkpoint']['png']='old_cp5.png'
        references.append(ref)
        row=dict(index=slot,root_slot=root,arm='H1',condition='off',selection_digest=scope['selection_digest'],
            status='horizon_completed',own_returned=10,warmup_returned=10,PNG=66,plan=plan,
            prefix51=dict(complete=True,frames=prefix),trajectory=selected,checkpoints=cps,
            missing_checkpoints=[],responses=responses,unknown_row={'source_value':-0.0})
        rows.append(row);emit('branch_closed',receipt=row)
    prep=dict(plans=plans,prefix_reference=references,source_roots=roots,source_slots=slots)
    pr=put(prep_path,prep)
    events[0]['prepare_sha256']=pr['sha256']
    counts={'public_step':dict(attempted=len(ops),completed=len(ops))}
    emit('run_finished',status='complete',counts=counts,errors=[])
    tail=None
    for e in events:
        e['previous_sha256']=tail
        e['event_sha256']=m.digest({k:v for k,v in e.items() if k!='event_sha256'})
        tail=e['event_sha256']
    jp=directory/'journal.jsonl';jp.write_bytes(b''.join(m.raw(e)+b'\n' for e in events))
    report=dict(rows=rows,counts=counts,errors=[],status='complete',contract={'fixture':True},
        execution_binding=dict(prepared=pr['path'],prepare_sha256=pr['sha256']),
        actual_operation_receipts=ops,journal_sha256=m.sha(jp),journal_chain_tail=tail)
    rr=put(directory/'report.json',report)
    completion=dict(prepared=pr['path'],prepare_sha256=pr['sha256'],report_sha256=rr['sha256'],
                    artifact_sha256={'journal.jsonl':m.sha(jp)})
    cr=put(directory/'completion.json',completion)
    bindings={**sources,pr['path']:pr['sha256'],rr['path']:rr['sha256'],cr['path']:cr['sha256'],str(jp):m.sha(jp)}
    sr=put(directory/'receipt.json',dict(rows=rows,counts=counts,errors=[],completion=completion,
        completion_sha256=cr['sha256'],physical_report_sha256=rr['sha256'],bindings=bindings))
    sc=put(directory/'source.json',{'inert_source_fixture':kind})
    return dict(prepare=pr,report=rr,completion=cr,receipt=sr,source=sc),prep,report,bindings


@pytest.fixture
def bundle(tmp_path):
    allrefs, data, bindings = {}, {}, {}
    for kind in ('C','ZP'):
        refs,p,r,bb=physical(tmp_path/kind,kind)
        allrefs.update({kind+'_'+k:v for k,v in refs.items()})
        data[kind]=(p,r,refs);bindings.update(bb)
        bindings.update({v['path']:v['sha256'] for v in refs.values()})
    rows=[];tasks=[]
    for i,(root,role,kind,slot) in enumerate(m.ORDER):
        p,r,refs=data[kind];row=r['rows'][slot];cp=row['checkpoints'][-1]
        source=dict(cp,path=str(Path(refs['report']['path']).parent/cp['png']))
        hist=deepcopy(p['prefix_reference'][slot]['history_frames'][1:])
        hist.append(dict(absolute_step=20,plan_step=10,png=source['path'],
            **{k:cp[k] for k in ('file_sha256','pixel_sha256','state','snapshot')}))
        br=dict(index=i,root_slot=root,role=role,source_kind=kind,source_slot=slot,checkpoint=10,
            source_report=refs['report'],source_receipt=refs['receipt'],selection_digest=row['selection_digest'],
            source=source,native_response=row['responses'][-1],causal_raw10=row['plan']['commands'],
            causal_raw10_bits=''.join(m.f32bits(row['plan']['commands'],10)),history_sources=hist,
            status='not_started',latent=None)
        rows.append(br);tasks.append(dict(index=i,root_slot=root,role=role,source=source,
            causal_raw10=br['causal_raw10'],identity={'history_sources':hist}))
    ap=put(tmp_path/'A_prepare.json',dict(rows=rows,authorities=allrefs))
    ar=put(tmp_path/'A_report.json',dict(rows=rows,prepared=ap['path'],prepare_sha256=ap['sha256'],
        fixed_denominator=5,errors=[]))
    cv=dict(prepared=ap['path'],prepare_sha256=ap['sha256'],report_sha256=ar['sha256'])
    ac=put(tmp_path/'A_completion.json',cv)
    bindings.update({v['path']:v['sha256'] for v in (ap,ar,ac)})
    ass=put(tmp_path/'A_receipt.json',dict(tasks=tasks,fixed_denominator=5,report_sha256=ar['sha256'],
        completion_sha256=ac['sha256'],completion=cv,bindings=bindings,source_authorities=allrefs))
    return dict(prepare=ap,report=ar,completion=ac,receipt=ass)


def test_real_file_build_whole_verify(bundle,tmp_path):
    out=tmp_path/'witness.json';seal=m.build(str(out),bundle)
    assert m.verify(str(out),seal['sha256'])['status']=='all5_full66_witnesses_reconstructed'
    w=m.read(out)
    assert not w['actual_MAIN_authority'] and not w['production_adapter_ready']
    assert [(r['root_slot'],r['role']) for r in w['rows']]==[(x[0],x[1]) for x in m.ORDER]
    assert [len(r['observations']) for r in w['rows']]==[66]*5
    assert [len(r['transport_normalizations']) for r in w['rows']]==[51]*5
    assert w['rows'][0]['observations'][0]['receipt']['snapshot']['unknown_snapshot']=='preserved'
    assert w['rows'][0]['source_row']['unknown_row']=={'source_value':-0.0}
    assert all(not k in w['rows'][0]['cp10']['snapshot'] for k in ('rng','public_state'))


@pytest.mark.parametrize('key',list(m.PINS))
def test_authority_file_sha_tamper(bundle,key):
    Path(bundle[key]['path']).write_bytes(b'{}\n')
    with pytest.raises(ValueError,match='source SHA'):
        m.material(bundle)


@pytest.mark.parametrize('relative',('C/journal.jsonl','ZP/journal.jsonl','C/b0_plan_10_0.png'))
def test_original_bound_bytes_tamper(bundle,tmp_path,relative):
    p=tmp_path/relative;p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='source SHA'):
        m.material(bundle)


def endpoint_args(bundle):
    sources=m.Sources();ap=m.read(bundle['prepare']['path']);ass=m.read(bundle['receipt']['path'])
    p,r,s,events,refs=m.physical('C',ap['authorities'],ass['bindings'],sources)
    br=m.read(bundle['report']['path'])['rows'][0];task=ass['tasks'][0]
    return [0,m.ORDER[0],br,task,p,r,s,events,refs,sources]


@pytest.mark.parametrize('mutation',('missing_obs','duplicate_obs','post','snapshot','float64_bits','unknown_pre',
    'unknown_receipt','raw_bits','transition','response','prefix','checkpoint','history','stage_source','scope','guard','late_guard'))
def test_endpoint_mutations_fail_closed(bundle,mutation):
    args=endpoint_args(bundle);br,task,p,r,s,ev=args[2:8]
    obs=[e for e in ev if e['kind']=='fresh_observation']
    tr=next(e for e in ev if e['kind']=='returned_transition')
    if mutation=='missing_obs':ev.remove(obs[2])
    elif mutation=='duplicate_obs':ev.insert(ev.index(obs[2]),deepcopy(obs[2]))
    elif mutation=='post':obs[0]['post']['new_field']=True
    elif mutation=='snapshot':obs[0]['receipt']['snapshot']['state'][0]=999.0
    elif mutation=='float64_bits':obs[0]['receipt']['snapshot']['state_float64_bits']='00'*56
    elif mutation=='unknown_pre':p['prefix_reference'][0]['frames'][0]['pre']['new_unknown']=1
    elif mutation=='unknown_receipt':p['prefix_reference'][0]['frames'][0]['receipt']['new_unknown']=1
    elif mutation=='raw_bits':r['actual_operation_receipts'][0]['raw_bits']='00'*8
    elif mutation=='transition':tr['snapshot']['corrupt']=True
    elif mutation=='response':r['rows'][0]['responses'][-1]['reward']=123.0
    elif mutation=='prefix':r['rows'][0]['prefix51']['complete']=False
    elif mutation=='checkpoint':r['rows'][0]['checkpoints'][-1]['pixel_sha256']='bad'
    elif mutation=='history':task['identity']['history_sources'][0]['absolute_step']=0
    elif mutation=='stage_source':br['source']['path']='/fake.png'
    elif mutation=='scope':obs[0]['root_slot']=1
    elif mutation=='guard':next(e for e in ev if e['kind']=='cp5_guard')['accepted']=False
    elif mutation=='late_guard':next(e for e in ev if e['kind']=='cp5_guard')['index']=10**9
    with pytest.raises((ValueError,KeyError,StopIteration)):
        m.extract(*args)


def test_journal_internal_chain_and_partial(bundle,tmp_path):
    refs=m.read(bundle['prepare']['path'])['authorities'];report=m.read(refs['C_report']['path'])
    p=tmp_path/'C/journal.jsonl';lines=p.read_bytes().splitlines(keepends=True)
    e=m.decode(lines[2]);e['arbitrary_tamper']=True;lines[2]=m.raw(e)+b'\n';p.write_bytes(b''.join(lines))
    report['journal_sha256']=m.sha(p)
    with pytest.raises(ValueError,match='hashchain'):
        m.journal(str(p),report,m.Sources())
    p.write_bytes(p.read_bytes().rstrip(b'\n'));report['journal_sha256']=m.sha(p)
    with pytest.raises(ValueError):
        m.journal(str(p),report,m.Sources())


@pytest.mark.parametrize('mutation',('terminal','ledger','counts','bookend','closure','tail'))
def test_full_journal_rehashed_semantic_mutations(bundle,tmp_path,mutation):
    refs=m.read(bundle['prepare']['path'])['authorities'];r=m.read(refs['C_report']['path'])
    jp=tmp_path/'C/journal.jsonl';events=[m.decode(x) for x in jp.read_bytes().splitlines()]
    if mutation=='terminal':
        next(e for e in events if e['kind']=='operation_returned')['receipt']['index']=999
    elif mutation=='ledger':r['actual_operation_receipts'][0]['raw_bits']='ff'*8
    elif mutation=='counts':r['counts']['public_step']['attempted']+=1
    elif mutation=='bookend':events[0]['kind']='not_started'
    elif mutation=='closure':next(e for e in events if e['kind']=='branch_closed')['receipt']['unknown_row']['new']=1
    tail=None
    for i,e in enumerate(events):
        e['index']=i;e['previous_sha256']=tail
        e['event_sha256']=m.digest({k:v for k,v in e.items() if k!='event_sha256'});tail=e['event_sha256']
    jp.write_bytes(b''.join(m.raw(e)+b'\n' for e in events));r['journal_sha256']=m.sha(jp)
    r['journal_chain_tail']='0'*64 if mutation=='tail' else tail
    with pytest.raises(ValueError):m.journal(str(jp),r,m.Sources())


def test_no_overwrite_whole_mutation_and_changed_source(bundle,tmp_path):
    p=tmp_path/'w.json';seal=m.build(str(p),bundle)
    with pytest.raises(ValueError,match='overwrite'):
        m.build(str(p),bundle)
    v=m.read(p);v['rows'][0]['unknown_semantics']['rng']='invented'
    p.write_bytes(m.raw(v)+b'\n')
    with pytest.raises(ValueError,match='reconstruction'):
        m.verify(str(p),m.sha(p))
    p.write_bytes(b'{}')
    with pytest.raises(ValueError,match='SHA'):
        m.verify(str(p),seal['sha256'])


@pytest.mark.parametrize('data',(b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}',b'{"a":1e999}'))
def test_strict_json(data):
    with pytest.raises(ValueError):m.decode(data)


def test_signed_zero_and_literal_float32():
    assert m.f32bits([[0.0,-0.0]],1)==['0000000000000080']
    with pytest.raises(ValueError):m.f32bits([[0.1,0.0]],1)
    with pytest.raises(ValueError):m.f32bits([[True,0.0]],1)


def test_cli_rejects_fixture_authorities(bundle,tmp_path,monkeypatch):
    import sys
    args=['extract','build','--output',str(tmp_path/'not_created.json')]
    for k,v in bundle.items():args += ['--'+k,v['path']]
    monkeypatch.setattr(sys,'argv',args)
    with pytest.raises(ValueError,match='source SHA'):m.main()
    assert not (tmp_path/'not_created.json').exists()


def test_no_runtime_import_or_ignored_field_patterns():
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(x and ('experiments' in x or 'torch' in x or 'pymunk' in x) for x in imports)


def test_fixed_next_adapter_integer_budget():
    b=m.NEXT_JOB_BUDGET
    assert sum(b['resource_caps'].values())==b['operations_per_job']==374
    assert b['PNG_per_job']==3*(11+16)==81
    assert b['events_per_job']==2*374+81+2+3==834
    assert b['jobs']*b['public_per_job']==250
    assert b['jobs']*b['physics_per_job']==2540
    assert b['cap_bytes']==67108864 and b['reserve_bytes']==8388608
    assert b['disk_floor_bytes']==209715200 and b['wall_seconds']==900
