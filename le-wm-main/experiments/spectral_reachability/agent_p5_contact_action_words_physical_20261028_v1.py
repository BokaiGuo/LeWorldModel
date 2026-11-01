"""MAIN-only native four-word replay; saved-only prepare/consume.

All four fresh prefixes are byte-gated BEFORE any suffix. Observer profiles the
original collision callback, never replaces native physics or a callback.
OOD engineering calibration only: no task/fresh/D2P/model-mechanism admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np

TAG = 'agent_p5_contact_action_words_physical_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = 'agent_p5_contact_source_contract_20261028_v1'
PHYSICAL = 'agent_p5_action_words_physical_20261028_v1'
PANEL = 'p5_jepa_factual_panel_20261027_v3'
PINS = {
    CONTRACT:'7c64d0c5f5a665f87ef28464d48661873674bdb6317c34d085bc7593f3b44cc8',
    PHYSICAL:'6a28bef2cdd8141e7063b61144042b1401e0440dc6fedd7b8d92232b48d304fc',
    PANEL:'0d4f1d983ab46f78bc21b79161de1105ed92f9e6ca2c910ab014667bb30019ef',
}
ARMS = ('xy','yx','x_minus_x','zero_zero')
NAMES = ('rgb.npy','state.npy','proprio.npy','transition.json','full_public_state.json')
GOAL = dict(dtype='<f8',shape=[3],values=[256.,256.,float(np.pi/4)])
MAX_CALLBACKS = 8192


def require(ok,message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path,*path.parents)), 'regular nonsymlink file: '+str(path))
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):
            h.update(b)
    return h.hexdigest()


def read_json(path,expected=None):
    require(expected is None or sha(path)==expected,'actual JSON FILE SHA')
    sha(path)
    require(Path(path).stat().st_size<=8*1024**2,'JSON cap')
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out,'duplicate JSON key')
            out[k] = v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,
        parse_constant=lambda s:require(False,'nonfinite JSON '+s))


def load(name):
    path = ROOT/'experiments/spectral_reachability'/f'{name}.py'
    require(sha(path)==PINS[name],'sealed helper source: '+name)
    spec = importlib.util.spec_from_file_location(TAG+'_'+name,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    contract,physical,panel = load(CONTRACT),load(PHYSICAL),load(PANEL)
    return contract,contract.load(contract.BASE),panel,physical


def policy():
    return dict(arms=list(ARMS),denominator=4,frames=list(range(21)),
        phase_order='four independent prefix10, all44frame byte gate, four fixed suffix10, close_all',
        observer='unmodified _handle_collision Python call/return profile, not callback replacement',
        max_callbacks=MAX_CALLBACKS,source_witness_to_planner=False,
        contact_definition='positive points on actual agent-block callback, not static gap or all-shape collisions',
        response_definition='preserve all actual native agent/block states and callbacks; no fitted response gate',
        interpretation='all four fixed model words replay, no task/mechanism/generalization claim',
        expected_native_call_entries=EXPECTED,
        expected_public_intents=dict(construct=4,seed=4,reset=4,step=80,close=4),
        output_cap_bytes=128*1024**2,payload_cap_bytes=96*1024**2,final_reserve_bytes=8*1024**2,
        max_nodes=2048,free_floor_bytes=200*1024**2,wall_seconds=900,
        physics_accounting='Space.step Python entries and dt, not C solver completion; intents/returns separate',
        scientific_admission=False,task_success=None,action_witness_established=False)


NATIVE = 'agent_p5_contact_source_native_20261028_v1'
NUMERIC = 'agent_p5_contact_action_response_20261028_v1'
NATIVE_SHA = '78700888a70d5caf263d04301ca8a62f907364ba4b38cf9d3bdaf689d115fecd'
NATIVE_MANIFEST_SHA = 'b728eb810a78f976bc413d891984766b7743e6ae55a13be8dd5a191531bc3b10'
NUMERIC_SHA = 'bd5b85b308fcb621adda1f4452911f58e233b9ba0812e8d569d0dbd540940930'
MODEL_REPORT_SHA = '4fd032a4e5fb365f5a11eecbaba416c1ace7352e7afd092aa0adf6b972bded7f'
MODEL_MANIFEST_SHA = '572e2fded1c8927302c4ffe66921ce05e58f5954b43a9bc44bc022c19c6bbfb6'
MODEL_RECEIPT_SHA = '247a8a6328053f34b8367f39dca63f81fb009ca2a41734db0f618d1ea6eab2c3'
SOURCE_REPORT_SHA = 'ecd6afea8bcc9ee7f6fe32234c91a4fb4be62ecdd6e6bbc78351ed8e20d8a4ba'
SOURCE_PREFIX = ROOT/'results/agent_p5_contact_source_native_20261028_v1_physics_MAIN01/push_x'
EXPECTED = dict(env_construct=4,seed=8,reset=4,control_step=80,space_step=804,
    native_render=84,native_get_obs=84,close=4,setup=4,set_state=4,space_construct=4,
    native_goal_body=164,body_construct=176)


def upstreams():
    def get(name,h):
        path = ROOT/'experiments/spectral_reachability'/(name+'.py')
        require(sha(path)==h,'frozen upstream source')
        spec = importlib.util.spec_from_file_location(TAG+'_'+name,path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return get(NATIVE,NATIVE_SHA),get(NUMERIC,NUMERIC_SHA)


def prepare(native_path,native_sha,model_path,model_sha,report_path,report_sha,receipt_path,receipt_sha):
    # Privileged PHYSICAL prep: recipe and authenticated PREFIX inventory only.
    # Original report is parsed for inventory, not suffix labels/word selection.
    native,numeric = upstreams()
    contract,base,_,_ = helpers()
    require(native_sha==NATIVE_MANIFEST_SHA and report_sha==MODEL_REPORT_SHA and model_sha==MODEL_MANIFEST_SHA
            and receipt_sha==MODEL_RECEIPT_SHA,'MAIN actual four-word manifest/report/strict SHA')
    old = read_json(native_path,native_sha)
    native.verify(old)
    model = read_json(model_path,model_sha)
    actual = numeric.consume(model,report_path,report_sha)
    require(read_json(receipt_path,receipt_sha)==actual
            and actual['status']=='all4_saved_public_action_words_verified'
            and actual['scientific_admission'] is False,'actual MAIN persisted exact strict model receipt required')
    require(model['public_payload_sha256']==numeric.PUBLIC_SHA
            and model['protocol']['words']==list(ARMS),'same public context and all four words')
    private = old['source_contract']['generator_private']
    template = private['plans'][0]
    require(template['arm']=='push_x' and template['prefix_raw_u']==np.zeros((10,2),np.float64).tolist(),
            'original fixed source prefix')
    plans = []
    for word in ARMS:
        raw = np.asarray(model['plans'][word]['raw'],np.float64)
        require(raw.shape==(2,5,2) and np.array_equal(raw,numeric.owner.programs()[word]),'original sealed raw word')
        plans.append(dict(arm=word,seed=template['seed'],constructor=template['constructor'],
            prefix_raw_u=template['prefix_raw_u'],reference_suffix_raw_u=raw.reshape(10,2).tolist()))
    # Execution projection is a NEW object, never edits the sealed source contract.
    projection = dict(constructor_change=old['source_contract']['constructor_change'],
        generator_private=dict(plans=plans,expected_native_call_entries=EXPECTED))
    bindings = dict(old['bindings'])
    evidence = [(native_path,native_sha),(model_path,model_sha),(report_path,report_sha),(receipt_path,receipt_sha)]
    bindings.update({str(Path(p).resolve()):h for p,h in evidence})
    for name,h in ((NATIVE,NATIVE_SHA),(NUMERIC,NUMERIC_SHA),*PINS.items()):
        bindings[str(ROOT/'experiments/spectral_reachability'/(name+'.py'))]=h
    # Bind complete saved model trace via the already strict inventory, no model load.
    bindings.update(model['bindings'])
    bindings.update(model['model_spec']['loader_manifest']['seals'])
    for rec in model['inputs'].values():
        bindings[rec['path']]=rec['sha256']
    for rec in read_json(report_path,report_sha)['artifacts'].values():
        bindings[rec['path']]=rec['sha256']
    for p in Path(report_path).parent.glob('event_*.json'):
        bindings[str(p.resolve())]=sha(p)
    identity = []
    source_report=SOURCE_PREFIX.parent/'report.json'
    # Only authenticated prefix inventory used; no reference suffix/metrics chosen.
    source_inventory=read_json(source_report,SOURCE_REPORT_SHA)['artifacts']
    bindings[str(source_report)]=SOURCE_REPORT_SHA
    for frame in range(11):
        files = {}
        for name in NAMES:
            p=SOURCE_PREFIX/f'frame_{frame:02d}'/name
            h=sha(p)
            require(p.stat().st_size<=200000 and source_inventory[str(p.relative_to(SOURCE_PREFIX.parent))]
                    ==dict(sha256=h,bytes=p.stat().st_size),'authenticated saved prefix inventory')
            bindings[str(p)]=h
            files[name]=dict(path=str(p),sha256=h,bytes=p.stat().st_size)
        identity.append(dict(frame=frame,files=files))
    commands=[]
    for frame in range(1,11):
        p=SOURCE_PREFIX/f'command_{frame:02d}.json'
        require(source_inventory[str(p.relative_to(SOURCE_PREFIX.parent))]
                ==dict(sha256=sha(p),bytes=p.stat().st_size),'original saved command identity')
        require(read_json(p)['raw_env_argument']==template['prefix_raw_u'][frame-1],'saved prefix raw commands')
        bindings[str(p)]=sha(p)
        commands.append(dict(path=str(p),sha256=sha(p)))
    rng=SOURCE_PREFIX/'pre_reset_rng.json'
    require(source_inventory[str(rng.relative_to(SOURCE_PREFIX.parent))]
            ==dict(sha256=sha(rng),bytes=rng.stat().st_size),'authenticated explicit RNG')
    bindings[str(rng)]=sha(rng)
    for p in (Path(__file__).resolve(),ROOT/'tests'/f'test_{TAG}.py',ROOT/'innerdocs'/f'{TAG}.md'):
        bindings[str(p)]=sha(p)
    authority=dict(native_path=str(Path(native_path).resolve()),native_sha=native_sha,
        model_path=str(Path(model_path).resolve()),model_sha=model_sha,
        report_path=str(Path(report_path).resolve()),report_sha=report_sha,
        receipt_path=str(Path(receipt_path).resolve()),receipt_sha=receipt_sha)
    result = dict(schema=TAG,status='prepared_MAIN_native_words_not_executed',policy=policy(),
        source_contract=projection,prefix_identity=identity,prefix_commands=commands,
        seeded_rng=dict(path=str(rng),sha256=sha(rng)),
        public_inputs=model['inputs'],public_payload_sha256=model['public_payload_sha256'],
        bindings=bindings,runtime_manifest=old['runtime_manifest'],authority=authority,
        physics_executed=False,neural_executed=False,scientific_admission=False,task_success=None)
    bind_public_prefix(result)
    require(all(sha(p)==h for p,h in bindings.items()),'prepare complete input closure')
    return result


def verify(manifest):
    require(manifest==prepare(**manifest['authority']),'word physical manifest/source drift')


def bind_public_prefix(manifest):
    # Physical-only identity linkage, no evaluation/selection on object suffix.
    inputs=manifest['public_inputs']
    for frame in (0,5,10):
        require(manifest['prefix_identity'][frame]['files']['rgb.npy']['sha256']
                ==inputs[f'context_rgb_{frame:02d}']['sha256'],'same published prefix RGB')
    props=np.stack([np.load(manifest['prefix_identity'][i]['files']['proprio.npy']['path'],
                            allow_pickle=False,max_header_size=1024) for i in (0,5,10)])
    rec=inputs['context_proprio']
    require(sha(rec['path'])==rec['sha256'] and np.array_equal(props,np.load(rec['path'],allow_pickle=False)),
            'same published raw proprio')
    history=np.load(inputs['history_raw_u']['path'],allow_pickle=False)
    require(history.dtype==np.float64 and np.array_equal(history,
        np.asarray(manifest['source_contract']['generator_private']['plans'][0]['prefix_raw_u'],np.float64)),
        'same published actual raw prefix')
    _,_,_,physical=helpers()
    for i in range(11):
        physical.validate_frame(Path(manifest['prefix_identity'][i]['files']['rgb.npy']['path']).parent,i,GOAL)


def make_meter(base,cls,pymunk):
    class Meter(base.NativeCalls):
        def __init__(self):
            super().__init__(cls,pymunk)
            self.callback_code = cls._handle_collision.__code__
            self.codes[self.callback_code] = 'native_collision_callback'
            self.callbacks,self.errors,self.pending = [],[],{}

        def profile(self,frame,event,arg):
            # Never throw from a profiler: CPython disables it on an exception.
            # Any observer error blocks the next dispatch and strict consumption.
            try:
                super().profile(frame,event,arg)
                if frame.f_code is not self.callback_code:
                    return
                env = frame.f_locals['self']
                if event=='call':
                    require(len(self.callbacks)<MAX_CALLBACKS,'callback observation cap')
                    arbiter = frame.f_locals['arbiter']
                    roles = []
                    for shape in arbiter.shapes:
                        body = shape.body
                        role = next((name for name,b in [('agent',env.agent),('block',env.block),('static',env.space.static_body)] if body is b),None)
                        require(role is not None,'unmapped callback body')
                        roles.append(role)
                    contact = arbiter.contact_point_set
                    def xy(value):
                        a = np.asarray(tuple(value),np.float64)
                        require(a.shape==(2,) and np.isfinite(a).all(),'finite callback vector')
                        return a.tolist()
                    points = [dict(point_a=xy(p.point_a),point_b=xy(p.point_b),distance=float(p.distance)) for p in contact.points]
                    require(len(points)<=16 and all(np.isfinite(p['distance']) for p in points),'bounded native contact points')
                    entry = dict(index=len(self.callbacks),context=dict(self.context),space_step_index=len(self.space_steps),
                        substep_in_operation=sum(x['context']==self.context for x in self.space_steps),
                        body_roles=roles,points=points,normal=xy(contact.normal),total_impulse=xy(arbiter.total_impulse),
                        counter_before=int(env.n_contact_points),counter_after=None,returned=False)
                    self.pending[id(frame)] = entry
                    self.callbacks.append(entry)
                elif event=='return':
                    entry = self.pending.pop(id(frame))
                    entry.update(counter_after=int(env.n_contact_points),returned=True)
            except Exception as exc:
                if len(self.errors)<32:
                    self.errors.append(type(exc).__name__+': '+str(exc))

        def receipt(self):
            return dict(**super().receipt(),callbacks=[dict(x) for x in self.callbacks],
                        observer_errors=list(self.errors),pending_callbacks=len(self.pending))
    return Meter()


def reset_gate(public,contract,geometry):
    state = contract['constructor_change']['new']
    agent,block = public['bodies']['agent'],public['bodies']['block']
    require(agent['position']==state[:2] and agent['velocity']==[0.,0.]
            and block['position']==state[2:4] and block['angle']==state[4]
            and block['velocity']==[0.,0.] and block['angular_velocity']==0.,'actual post-reset pose/velocity')
    require(public['env']['goal_pose']==GOAL and public['env']['coverage_arr']==[], 'unmodified reset goal/history')
    polygons,radius = geometry.geometry(public)
    require(geometry.gap(agent['position'],polygons,radius)==1.,'actual reset native geometry gap')


def gate_prefix(out,rows,manifest):
    require([r['arm'] for r in rows]==list(ARMS)
            and all([r['frame'] for r in row['records'][:11]]==list(range(11)) for row in rows),
            'all four full prefixes before any suffix')
    checks=[]
    for frame in range(11):
        files=manifest['prefix_identity'][frame]['files']
        for name in NAMES:
            original=Path(files[name]['path'])
            require(sha(original)==files[name]['sha256'] and original.stat().st_size==files[name]['bytes'],
                    'saved source prefix drift')
            for arm in ARMS:
                actual=out/arm/f'frame_{frame:02d}'/name
                require(actual.stat().st_size==files[name]['bytes'] and sha(actual)==files[name]['sha256']
                        and actual.read_bytes()==original.read_bytes(),
                        'prefix byte gate '+arm+'/'+str(frame)+'/'+name)
        checks.append(dict(frame=frame,files={name:files[name]['sha256'] for name in NAMES}))
    for arm in ARMS:
        require(sha(out/arm/'pre_reset_rng.json')==manifest['seeded_rng']['sha256'],'saved explicit seeded RNG')
        for frame in range(1,11):
            require(sha(out/arm/f'command_{frame:02d}.json')==manifest['prefix_commands'][frame-1]['sha256'],
                    'saved prefix command bytes')
    bind_public_prefix(manifest)
    return dict(passed=True,frames=checks,public_payload_sha256=manifest['public_payload_sha256'],
                source_prefix_sha256=digest(manifest['prefix_identity']))


def execute(manifest,manifest_sha,out,contract,base,panel,physical,loader,stable):
    """Exact same control flow under record-only RAM fakes; not physical evidence."""
    require(manifest['policy']==policy(),'fixed execution policy')
    plans = manifest['source_contract']['generator_private']['plans']
    require([p['arm'] for p in plans]==list(ARMS),'four-word denominator/order')
    require(len(plans)==4 and all(np.asarray(p['prefix_raw_u']).shape==(10,2)
        and np.asarray(p['reference_suffix_raw_u']).shape==(10,2) for p in plans),'fixed prefix10/suffix10 budget')
    out = Path(out)
    require(out.is_absolute() and not out.exists() and out.parent.is_dir(),'exclusive new output directory')
    require(not any(p.is_symlink() for p in (out.parent,*out.parent.parents)),'output parent nonsymlink')
    writer = physical.Writer(base,out.parent)
    writer.mkdir(out)
    rows = [dict(arm=p['arm'],status='not_run',error=None,records=[],
        close=dict(status='not_constructed',attempted=False,returned=False,durable_return=False,error=None)) for p in plans]
    writer.save_json(out/'dispatch.json',dict(manifest_sha256=manifest_sha,policy=policy(),plans=plans))
    journal = writer.journal(out/'journal.jsonl')
    envs,calls,owned,error,prefix_gate = {},None,False,None,None
    def operation(arm,op,details,fn,final=False):
        if not final:
            require(not calls.errors,'callback observer failure')
            writer.guard()
        return base.operation(physical.FinalJournal(journal) if final else journal,calls,arm,op,details,fn)
    def capture(row,frame,obs,state,transition):
        record = panel.capture(base,writer,envs[row['arm']],obs,state,transition,out/row['arm']/f'frame_{frame:02d}')
        row['records'].append(record)
        callbacks = [x for x in calls.callbacks if x['context']==dict(role=row['arm'],operation='reset' if frame==0 else 'step',frame=None if frame==0 else frame)]
        writer.save_json(out/row['arm']/f'callbacks_{frame:02d}.json',dict(frame=frame,events=callbacks))
        public = read_json(out/row['arm']/f'frame_{frame:02d}/full_public_state.json')
        require(sum(len(x['points']) for x in callbacks)==public['env']['n_contact_points'],'actual callback/public counter')
        require(not calls.errors and not calls.pending,'callback return observer complete')
        return record
    def step(row,frame,u):
        env = envs[row['arm']]
        previous = row['records'][-1]
        command = physical.command_for(frame,u,np.array(env.agent.position,np.float64),previous['files']['full_public_state.json']['sha256'])
        writer.save_json(out/row['arm']/f'command_{frame:02d}.json',command)
        obs,reward,done,info = operation(row['arm'],'step',command,lambda:env.step(u.copy()))
        capture(row,frame,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info))
        require(done is False,'bound native done=False; no padding')
    try:
        stable()
        cls,calls,probe,geometry = loader()
        writer.save_json(out/'runtime_probe.json',probe)
        require(probe['status']=='native_import_ready_physics_not_executed' and cls is not None and calls is not None,
                'isolated native runtime probe rejected; no construction')
        calls.start()
        owned = True
        for p,row in zip(plans,rows):
            arm = p['arm']
            writer.mkdir(out/arm)
            row['status'] = 'constructing'
            def construct():
                row['close']['status'] = 'construction_attempted'
                envs[arm] = cls(**p['constructor'])
                row['close']['status'] = 'required'
                return envs[arm]
            env = operation(arm,'construct',dict(config=p['constructor']),construct)
            writer.save_json(out/arm/'constructor_rng.json',base.rng_state(env))
            operation(arm,'seed',dict(seed=p['seed']),lambda:env.seed(p['seed']))
            writer.save_json(out/arm/'pre_reset_rng.json',base.rng_state(env))
            obs,state = operation(arm,'reset',dict(seed=p['seed'],expected_space_steps=1,dt=.01),env.reset)
            capture(row,0,obs,state,dict(kind='reset'))
            reset_gate(read_json(out/arm/'frame_00/full_public_state.json'),manifest['source_contract'],geometry)
            for frame,u in enumerate(np.asarray(p['prefix_raw_u'],np.float64),1):
                step(row,frame,u)
            row['status'] = 'prefix_ready'
        require(all(len(row['records'])==11 for row in rows),'no suffix before four complete prefixes')
        prefix_gate = gate_prefix(out,rows,manifest)
        writer.save_json(out/'prefix_gate.json',prefix_gate)
        journal.append(dict(phase='all_prefixes_byte_gated',gate_sha256=sha(out/'prefix_gate.json')))
        for p,row in zip(plans,rows):
            require(prefix_gate['passed'],'four-word prefix gate before suffix')
            row['status'] = 'suffix_started'
            for frame,u in enumerate(np.asarray(p['reference_suffix_raw_u'],np.float64),11):
                step(row,frame,u)
            row['status'] = 'completed'
    except Exception as exc:
        error = type(exc).__name__+': '+str(exc)
        for row in rows:
            if row['status']!='completed':
                row['error'] = error
                row['status'] = 'not_run' if not row['records'] and row['arm'] not in envs else 'partial_failed'
    finally:
        for row in rows:
            arm,close = row['arm'],row['close']
            if arm in envs:
                try:
                    def close_call():
                        close['attempted'] = True
                        envs[arm].close()
                        close['returned'] = True
                    operation(arm,'close',{},close_call,final=True)
                    close.update(status='completed',durable_return=True)
                except Exception as exc:
                    close.update(status='failed',error=type(exc).__name__+': '+str(exc))
                    error = error or close['error']
                    row.update(status='partial_failed',error=row['error'] or close['error'])
        if owned:
            calls.stop()
            owned = False
        try:
            stable()
        except Exception as exc:
            error = error or ('source drift: '+str(exc))
    result = dict(schema=TAG+'.report',status='all4_native_word_traces_recorded' if error is None and all(r['status']=='completed' for r in rows) else 'native_source_incomplete',
        manifest_sha256=manifest_sha,manifest_digest=digest(manifest),policy=policy(),rows=rows,
        global_error=error,prefix_gate=prefix_gate,meter=None if calls is None else calls.receipt(),
        operation_meter=dict(durable_intents=dict(journal.attempted),durable_returns=dict(journal.returned)),
        profiler_released=not owned and not bool(calls and calls.started),wall_seconds=writer.clock()-writer.started,
        neural_executed=False,scientific_admission=False,task_success=None,action_witness_established=False)
    journal.append(dict(phase='terminal',summary_sha256=digest(result)),final=True)
    journal.close()
    result['artifacts'] = physical.inventory(out)
    writer.save_json(out/'report.json',result,final=True)
    physical.inventory(out)
    return result


def run(manifest_path,manifest_sha,out):
    manifest = read_json(manifest_path,manifest_sha)
    verify(manifest)
    contract,base,panel,physical = helpers()
    def no_neural():
        require(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'no neural-loaded physics process')
    def stable():
        for p,h in manifest['bindings'].items():
            require(sha(p)==h,'source/input drift')
    def loader():
        no_neural()
        runtime,parent = base.load_runtime(ROOT)
        probe = runtime.probe(ROOT,parent,Path(manifest['runtime_manifest']))
        if probe['status']!='native_import_ready_physics_not_executed':
            return None,None,probe,None
        path = ROOT/runtime.ENV_REL
        require(sha(path)==manifest['bindings'][str(path)],'native source identity')
        spec = importlib.util.spec_from_file_location(TAG+'_original_native',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        require(Path(module.PushTEnv.__init__.__code__.co_filename).resolve()==path,'native constructor provenance')
        require(Path(module.PushTEnv._handle_collision.__code__.co_filename).resolve()==path,'original native callback provenance')
        import pymunk
        no_neural()
        return module.PushTEnv,make_meter(base,module.PushTEnv,pymunk),probe,contract.load(contract.GEOMETRY)
    no_neural()
    require(signal.getsignal(signal.SIGALRM)==signal.SIG_DFL and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'preexisting alarm')
    def timeout(signum,frame):
        raise TimeoutError('900s native source wall cap')
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(900)
    try:
        return execute(manifest,manifest_sha,Path(out).absolute(),contract,base,panel,physical,loader,stable)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM,signal.SIG_DFL)


def validate_callbacks(receipt):
    events = receipt['callbacks']
    require(receipt['observer_errors']==[] and receipt['pending_callbacks']==0,'actual callback observer integrity')
    require(len(events)<=MAX_CALLBACKS and receipt['python_call_entries'].get('native_collision_callback',0)==len(events),'actual callback call census')
    for i,e in enumerate(events):
        require(e['index']==i and e['returned'] is True and 1<=e['space_step_index']<=len(receipt['space_step_events']),'actual callback return/space index')
        require(len(e['points'])<=16,'bounded callback points')
        require(receipt['space_step_events'][e['space_step_index']-1]['context']==e['context'],'callback physical substep context')
        require(e['substep_in_operation']==sum(x['context']==e['context'] for x in receipt['space_step_events'][:e['space_step_index']])
                and 1<=e['substep_in_operation']<=(1 if e['context']['operation']=='reset' else 10),'callback substep within native operation')
        require(len(e['body_roles'])==2 and set(e['body_roles'])<=set(['agent','block','static']),'callback actual body identities')
        require(type(e['counter_before']) is int and type(e['counter_after']) is int
                and e['counter_after']==e['counter_before']+len(e['points']),'original callback counter increment')
        for key in ('normal','total_impulse'):
            a = np.asarray(e[key],np.float64)
            require(a.shape==(2,) and np.isfinite(a).all(),'callback vector ABI')
        for point in e['points']:
            require(set(point)==set(['point_a','point_b','distance']) and np.isfinite(point['distance']),'contact point ABI')
            for k in ('point_a','point_b'):
                a = np.asarray(point[k],np.float64)
                require(a.shape==(2,) and np.isfinite(a).all(),'finite contact point')


def validate_saved(manifest,manifest_sha,report_path,report_sha,physical,geometry):
    report_path = Path(report_path)
    require(report_path.name=='report.json','fixed producer report name')
    r = read_json(report_path,report_sha)
    require(r['schema']==TAG+'.report' and r['manifest_sha256']==manifest_sha
            and r['manifest_digest']==digest(manifest) and r['policy']==policy(),'report contract binding')
    require(r['neural_executed'] is False and r['scientific_admission'] is False
            and r['task_success'] is None and r['action_witness_established'] is False,'no scientific promotion')
    require([x['arm'] for x in r['rows']]==list(ARMS),'complete four-word denominator')
    require(physical.inventory(report_path.parent)==r['artifacts'],'complete saved inventory')
    out = report_path.parent
    plans = manifest['source_contract']['generator_private']['plans']
    require(read_json(out/'dispatch.json')==dict(manifest_sha256=manifest_sha,policy=policy(),plans=plans),'predispatch authority')
    events = physical.read_journal(out/'journal.jsonl')
    terminal = {k:v for k,v in r.items() if k!='artifacts'}
    require(events[-1]==dict(phase='terminal',summary_sha256=digest(terminal)),'terminal report binding')
    attempts = [e for e in events if e['phase']=='intent']
    returns = [e for e in events if e['phase']=='returned']
    require(r['operation_meter']==dict(durable_intents=dict(Counter(e['operation'] for e in attempts)),
            durable_returns=dict(Counter(e['operation'] for e in returns))),'actual attempt/return meter')
    complete = r['status']=='all4_native_word_traces_recorded'
    if not complete:
        require(r['status']=='native_source_incomplete' and r['global_error'] is not None,'explicit partial error')
        return dict(schema=TAG+'.consume',status='partial_source_retained_not_verified',report_sha256=report_sha,
            manifest_sha256=manifest_sha,denominator=4,statuses={x['arm']:x['status'] for x in r['rows']},
            private_suffix_frame_rows=None,scientific_admission=False,task_success=None,physical_release=False)
    require(r['global_error'] is None and r['profiler_released'] is True and 0<=r['wall_seconds']<900,'complete runtime/cleanup')
    require(read_json(out/'runtime_probe.json')['status']=='native_import_ready_physics_not_executed','actual native runtime import receipt')
    require(all(x['status']=='completed' and x['error'] is None and x['close']==dict(status='completed',attempted=True,returned=True,durable_return=True,error=None) for x in r['rows']),'all four complete and closed')
    validate_callbacks(r['meter'])
    counts = dict(r['meter']['python_call_entries'])
    counts.pop('native_collision_callback',None)
    require(counts==manifest['source_contract']['generator_private']['expected_native_call_entries'],'native operation census')
    wanted = []
    for arm in ARMS:
        wanted += [(arm,'construct',None),(arm,'seed',None),(arm,'reset',None)]+[(arm,'step',i) for i in range(1,11)]
    for arm in ARMS:
        wanted += [(arm,'step',i) for i in range(11,21)]
    wanted += [(arm,'close',None) for arm in ARMS]
    ops = [e for e in events if e['phase'] in ('intent','returned')]
    require(len(ops)==2*len(wanted),'complete operation ledger')
    space = []
    for i,(arm,op,frame) in enumerate(wanted):
        before,after = ops[2*i:2*i+2]
        require(before['phase']=='intent' and after['phase']=='returned'
                and before['role']==after['role']==arm and before['operation']==after['operation']==op,'operation pairing/order')
        if i:
            require(before['counters_before']==ops[2*i-1]['counters_after'],'actual meter continuity')
        if op=='step':
            require(before['details']==read_json(out/arm/f'command_{frame:02d}.json'),'dispatch exact command')
        if op=='construct':
            require(before['details']==dict(config=plans[ARMS.index(arm)]['constructor']),'actual constructor contract')
        if op=='seed':
            require(before['details']==dict(seed=plans[ARMS.index(arm)]['seed']),'actual seed contract')
        if op in ('reset','step'):
            space += [dict(context=dict(role=arm,operation=op,frame=frame),dt=.01)]*(1 if op=='reset' else 10)
    require(ops[0]['counters_before']==dict(python_call_entries={},space_step_events=[],callbacks=[],observer_errors=[],pending_callbacks=0), 'fresh actual observer')
    require(ops[-1]['counters_after']==r['meter'] and r['meter']['space_step_events']==space,'final meter/physical substeps')
    gate = gate_prefix(out,r['rows'],manifest)
    require(r['prefix_gate']==read_json(out/'prefix_gate.json')==gate,'all11frame gate receipt')
    g = [i for i,e in enumerate(events) if e['phase']=='all_prefixes_byte_gated']
    require(len(g)==1 and events[g[0]]['gate_sha256']==sha(out/'prefix_gate.json'),'gate ledger')
    require(max(i for i,e in enumerate(events) if e['phase']=='returned' and e['operation']=='step' and e['counters_after']['space_step_events'][-1]['context']['frame']==10)<g[0]
            <min(i for i,e in enumerate(events) if e['phase']=='intent' and e['operation']=='step' and e['details']['frame']==11),'gate BEFORE any suffix')
    frame_rows,publics = [],{}
    for plan,row in zip(plans,r['rows']):
        arm = row['arm']
        require([x['frame'] for x in row['records']]==list(range(21)),'complete actual frames')
        for record in row['records']:
            frame = record['frame']
            path = out/arm/f'frame_{frame:02d}'
            require(set(record['files'])==set(NAMES),'five-file native capture')
            for name in NAMES:
                require(record['files'][name]==dict(sha256=sha(path/name),bytes=(path/name).stat().st_size),'native capture binding')
            public,transition = physical.validate_frame(path,frame,GOAL)
            require(record['state']==np.load(path/'state.npy',allow_pickle=False).tolist()
                    and record['proprio']==np.load(path/'proprio.npy',allow_pickle=False).tolist(),'frame summary binding')
            if frame==0:
                reset_gate(public,manifest['source_contract'],geometry)
            else:
                previous = publics[(arm,frame-1)]
                raw = np.asarray((plan['prefix_raw_u']+plan['reference_suffix_raw_u'])[frame-1],np.float64)
                command = physical.command_for(frame,raw,np.array(previous['bodies']['agent']['position'],np.float64),sha(out/arm/f'frame_{frame-1:02d}/full_public_state.json'))
                require(read_json(out/arm/f'command_{frame:02d}.json')==command and public['env']['latest_action']==command['absolute_target'],'native raw command/target chain')
            callbacks = [x for x in r['meter']['callbacks'] if x['context']==dict(role=arm,operation='reset' if frame==0 else 'step',frame=None if frame==0 else frame)]
            require(read_json(out/arm/f'callbacks_{frame:02d}.json')==dict(frame=frame,events=callbacks),'per-frame actual callbacks')
            total = 0
            for cb in callbacks:
                require(cb['counter_before']==total,'callback counter continuity within control step')
                total = cb['counter_after']
            require(total==public['env']['n_contact_points'],'callback sum/public points')
            if frame:
                require(transition['info']['n_contacts']==int(np.ceil(total/10)),'native averaged contact label')
            publics[(arm,frame)] = public
            if frame>10:
                frame_rows.append(dict(arm=arm,frame=frame,callback_points=total,
                    agent_block_callback_points=sum(len(x['points']) for x in callbacks if set(x['body_roles'])==set(['agent','block'])),
                    block=public['bodies']['block'],agent=public['bodies']['agent'],coverage=public['env']['coverage_arr'][-1]))
    return dict(schema=TAG+'.consume',status='all4_saved_native_words_verified',report_sha256=report_sha,
        manifest_sha256=manifest_sha,denominator=4,statuses={x['arm']:x['status'] for x in r['rows']},
        public_payload_sha256=manifest['public_payload_sha256'],private_suffix_frame_rows=frame_rows,
        scientific_admission=False,task_success=None,action_witness_established=False,physical_release=False,
        interpretation='constructed OOD full-four native word replay, no task/mechanism success or equal-total-compute claim')


def consume(manifest_path,manifest_sha,report_path,report_sha):
    manifest = read_json(manifest_path,manifest_sha)
    verify(manifest)
    contract,_,_,physical = helpers()
    return validate_saved(manifest,manifest_sha,report_path,report_sha,physical,contract.load(contract.GEOMETRY))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','verify','run','consume'])
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--manifest-sha256')
    for name in ('native','model','report','receipt'):
        p.add_argument('--'+name,type=Path)
        p.add_argument('--'+name+'-sha256')
    p.add_argument('--output',type=Path)
    p.add_argument('--confirm-physics',action='store_true')
    a=p.parse_args(argv)
    if a.command=='prepare':
        require(all(getattr(a,n) is not None and getattr(a,n+'_sha256') for n in ('native','model','report','receipt')),
                'actual native/model/report/strict paths and FILE SHAs required')
        value=prepare(a.native,a.native_sha256,a.model,a.model_sha256,a.report,a.report_sha256,a.receipt,a.receipt_sha256)
        helpers()[1].save_json(a.manifest,value)
        result=dict(status=value['status'],manifest_sha256=sha(a.manifest))
    else:
        require(a.manifest_sha256,'actual manifest FILE SHA')
        if a.command=='verify':
            verify(read_json(a.manifest,a.manifest_sha256))
            result=dict(status='saved_word_replay_prepare_verified')
        elif a.command=='run':
            require(a.confirm_physics and a.output is not None,'MAIN confirm-physics/new output required')
            result=run(a.manifest,a.manifest_sha256,a.output)
        else:
            require(a.report is not None and a.report_sha256 and a.output is not None,'actual physics report SHA/new output')
            result=consume(a.manifest,a.manifest_sha256,a.report,a.report_sha256)
            helpers()[1].save_json(a.output,result)
    print(json.dumps(result,sort_keys=True,allow_nan=False))
    return 2 if result['status'] in ('native_source_incomplete','partial_source_retained_not_verified') else 0


if __name__=='__main__':
    raise SystemExit(main())

