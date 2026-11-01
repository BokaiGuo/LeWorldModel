"""Fresh live native raster intervention; distinct from failed historical renderer."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

NAME = 'agent_p5_live_raster_intervention_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
HELPER = 'agent_p5_replay_target_renderer_20261028_v1'
HELPER_SHA = 'c7bbaaccd17b4734e6e133990ee4f22248efbaf328377f0c88f363eaa6bf3637'
OBSERVER = 'agent_p5_native_geometry_observer_20261028_v1'
OBSERVER_SHA = '5d4d95ffcf6c21d6053057e609eeaf7ef950c7643bf53b3b04d9864c94409b94'
OBSERVER_REPORT_SHA = 'e717c0eeb688d9b9b621b70a3caaa7496e6225f6733865b0357d13f3a30811fa'
FAILED_SHA = '1b02ed4a2ca136b4dc7f8549073ff030620df8587d654eb2a07ca86036cad0c6'


def bound(name, expected):
    path = ROOT/'experiments/spectral_reachability'/f'{name}.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('sealed helper source SHA '+name)
    return importlib.import_module('experiments.spectral_reachability.'+name)


h = bound(HELPER, HELPER_SHA)
require, canonical, digest, sha, read = h.require, h.canonical, h.digest, h.sha, h.read
Store = h.Store
STATES = ('xy', 'yx', 'x_minus_x', 'zero_zero', 'goal')
COLORS = ('visible', 'white')
SHOTS = ('baseline', 'visible', 'visible_restored', 'white', 'white_restored')
EXPECTED = dict(env_construct=5, seed=10, reset=5, control_step=60, space_step=605,
    native_get_obs=65, native_render=85, setup=5, set_state=5, space_construct=5,
    native_goal_body=145, body_construct=160, debug_draw=85, close=5)
DELTAS = {k:dict(h.DELTAS[k]) for k in ('construct','seed','reset','step','close')}
DELTAS['color'] = DELTAS['restore_render'] = dict(h.DELTAS['color'])


def protocol():
    return dict(schema=NAME, states=list(STATES), colors=list(COLORS), shots=list(SHOTS), cells=10,
        target_frames={s:0 if s=='goal' else 15 for s in STATES},
        order='four prefix10; all44 exact; four suffix5; original goal reset; all5 attained; five visible/off/white/off',
        native_entries=EXPECTED, physics_steps=605, public_controls=60,
        images=dict(observations=65, intervention=10, restoration=10, total=85),
        public_operations=dict(construct=5,seed=5,reset=5,step=60,color=10,restore_render=10,close=5),
        cap_bytes=h.CAP,final_reserve_bytes=h.RESERVE,floor_bytes=h.FLOOR,wall_seconds=h.WALL,
        gates='prefix all5 files exact; suffix RGB/state/proprio exact; full transition/public differences retained without tolerance',
        policy='no coverage/reward-dependent rendering/selection; snapshots retain those fields',
        old_renderer_release=False,historical_full_public_equivalence=None,scientific_admission=False,
        neural=0,retries=0,hydrate=False)


def build(observer_report, observer_sha, observer_receipt, receipt_sha):
    require(observer_sha == OBSERVER_REPORT_SHA, 'one fixed actual observer report')
    obs = bound(OBSERVER,OBSERVER_SHA)
    r = read(observer_report,observer_sha)
    op = obs.PREP.parent/'agent_p5_native_geometry_observer_20261028_v1_prepare_MAIN01.json'
    # The actual observer report binds its prepare; no filename-based identity acceptance.
    receipt = obs.consume(op,r['prepare_sha256'],observer_report,observer_sha)
    require(receipt == read(observer_receipt,receipt_sha), 'complete observer strict receipt equality')
    require(receipt['historical_transform_certification'] is None and receipt['renderer_release'] is False,
        'observation not historical repair')
    p = h.verify(obs.PREP,obs.PREP_SHA)
    require(sha(obs.FAILED) == FAILED_SHA, 'old failed renderer retained')
    bindings = dict(p['bindings'])
    observer_p = read(op,r['prepare_sha256']); bindings.update(observer_p['bindings'])
    for path,expected in ((op,r['prepare_sha256']),(Path(observer_report),observer_sha),(Path(observer_receipt),receipt_sha)):
        require(sha(path)==expected,'observer authority bytes'); bindings[str(Path(path).absolute())]=expected
    targets=[]
    for state in STATES:
        n=0 if state=='goal' else 15; f=p['frames'][state][n]
        targets.append(dict(state_id=state,frame=n,source_public=f['full_public_state.json'],source_rgb=f['rgb.npy']))
    for path in (Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md'):
        bindings[str(path)] = sha(path)
    return dict(schema=NAME+'.prepare',protocol=protocol(),bindings=bindings,plans=p['plans'],goal_protocol=p['goal_protocol'],
        frames={s:p['frames'][s][:1 if s=='goal' else 16] for s in STATES},targets=targets,
        observer_authority=dict(report=str(Path(observer_report).absolute()),report_sha256=observer_sha,
            receipt=str(Path(observer_receipt).absolute()),receipt_sha256=receipt_sha),
        old_failure=dict(path=str(obs.FAILED),sha256=FAILED_SHA),
        v3_prepare=p['v3_prepare'],v3_sha256=p['v3_sha256'])


def verify(path, expected):
    p=read(path,expected); a=p['observer_authority']
    require(p==build(a['report'],a['report_sha256'],a['receipt'],a['receipt_sha256']), 'deterministic full prepare')
    return p


def differences(expected, actual, pointer=''):
    """Exact JSON value/float-bit differences, no tolerance or missing-to-zero."""
    if canonical(expected)==canonical(actual): return []
    if isinstance(expected,dict) and isinstance(actual,dict):
        rows=[]
        for key in sorted(set(expected)|set(actual)):
            token=key.replace('~','~0').replace('/','~1'); path=pointer+'/'+token
            if key not in expected or key not in actual:
                rows.append(dict(pointer=path,expected_present=key in expected,actual_present=key in actual,
                    expected=expected.get(key),actual=actual.get(key)))
            else: rows.extend(differences(expected[key],actual[key],path))
        return rows
    if isinstance(expected,list) and isinstance(actual,list) and len(expected)==len(actual):
        return [r for i,(e,a) in enumerate(zip(expected,actual)) for r in differences(e,a,pointer+'/'+str(i))]
    return [dict(pointer=pointer,expected_present=True,actual_present=True,expected=expected,actual=actual)]


def snapshot(env,base):
    return dict(public_state=base.public_state(env),topology=listify(h.topology(env)),
        global_rng=base.numeric_json(dict(python=random.getstate(),numpy=np.random.get_state())),
        goal_color=dict(rgba=list(env.goal_color),object_id=id(env.goal_color)))


def listify(value):
    if isinstance(value,(tuple,list)): return [listify(x) for x in value]
    return value


def equal(a,b): return canonical(a)==canonical(b)


class Meter(h.Meter):
    def __init__(self,cls,pymunk,store,base):
        super().__init__(cls,pymunk,store)
        self.base=base

    def profile(self,frame,event,arg):
        if event not in ('call','return') or frame.f_code not in self.codes: return
        key=self.codes[frame.f_code]
        (self.entries if event=='call' else self.returns)[key]+=1
        try:
            if key=='native_render':
                env=frame.f_locals['self']
                if event=='call': self.render_views[id(env)]=dict(before=snapshot(env,self.base))
                else: self.render_views[id(env)]['after']=snapshot(env,self.base)
            self.store.event(dict(phase='native_'+event,name=key,context=self.context,
                dt=float(frame.f_locals['dt']) if key=='space_step' else None))
            require(self.entries[key]<=EXPECTED[key],'native entry cap '+key)
        except BaseException as exc:
            if len(self.errors)<32: self.errors.append(type(exc).__name__+': '+str(exc))

    def receipt(self):
        return dict(entries={k:self.entries[k] for k in EXPECTED},return_events={k:self.returns[k] for k in EXPECTED},errors=list(self.errors))


def capture_render(env,state,shot,store,meter,base,op):
    """Only actual native render; persist returned image and views before guards."""
    image=h.operation(store,meter,state,op,dict(shot=shot['shot']),lambda:env._render_frame('rgb_array'))
    stem=state+'/'+shot['shot']; shot['files']['rgb_image']=store.image(stem,image)
    view=meter.render_views[id(env)]
    shot['files']['before']=store.json(stem+'_before.json',view['before'])
    shot['files']['after']=store.json(stem+'_after.json',view['after'])
    require(equal(view['before'],view['after']) and equal(view['after'],snapshot(env,base)), 'render changed public state/RNG/topology/color')
    shot['status']='captured'
    return image


def intervene(env,row,store,meter,pygame,base,baseline_rgb):
    state=row['state_id']; old=env.goal_color; anchor=snapshot(env,base)
    require(list(old)==h.RGBA['visible'],'original native LightGreen goal color')
    for j,color in enumerate(COLORS):
        on=row['shots'][1+2*j]; off=row['shots'][2+2*j]
        cell=row['cells'][j]; cell['status']='started'
        cell['off_before_shot']='baseline' if j==0 else 'visible_restored'
        cell['original_rgba']=list(old); cell['original_color_id']=id(old)
        cell['anchor']=store.json(state+'/'+color+'_anchor.json',anchor)
        try:
            env.goal_color=pygame.Color('LightGreen' if color=='visible' else 'White')
            cell['requested_rgba']=list(env.goal_color)
            require(cell['requested_rgba']==h.RGBA[color],'requested native color')
            actual=capture_render(env,state,on,store,meter,base,'color')
            onstate=snapshot(env,base)
            require(equal(onstate['public_state'],anchor['public_state']) and equal(onstate['global_rng'],anchor['global_rng'])
                and equal(onstate['topology'],anchor['topology']),'color intervention changed physical state/RNG')
            if color=='visible': require(np.array_equal(actual,baseline_rgb),'native original-color pixels exact')
        finally:
            env.goal_color=old
            # Restoration receipt is attempted even if rendering raised internally;
            # never manufacture an after-frame from a previous observation.
            cell['restoration']=store.json(state+'/'+color+'_restoration.json',dict(snapshot=snapshot(env,base),
                original_color_id=id(old),same_color_object=env.goal_color is old),final=True)
        require(equal(snapshot(env,base),anchor),'state/RNG/color restoration')
        restored=capture_render(env,state,off,store,meter,base,'restore_render')
        require(np.array_equal(restored,baseline_rgb),'restored original pixels exact including contact marks')
        cell['status']='completed'
    row['status']='completed'


def execute(p,prepare_sha,out,loader,stable=lambda:None):
    store=Store(out); envs={}; meter=None; profiled=False; errors=[]; attained=False
    rows=[dict(state_id=s,status='not_run',records=[],baseline=None,
        shots=[dict(shot=k,status='not_run',files={}) for k in SHOTS],
        cells=[dict(color=c,status='not_run') for c in COLORS]) for s in STATES]
    store.json('execution_binding.json',dict(schema=NAME+'.execution',prepare_sha256=prepare_sha,prepare_digest=digest(p),protocol=protocol()))
    def capture(s,n,obs,state,transition):
        row=rows[STATES.index(s)];stem=s+f'/frame_{n:02d}'
        rec=dict(frame=n,files={},image=store.image(stem+'/rgb',obs['visual']),exact={})
        row['records'].append(rec); rec['files']['rgb.npy']=rec['image']['npy']
        for name,value in (('state',state),('proprio',obs['proprio'])):
            rec['files'][name+'.npy']=store.save(stem+'/'+name+'.npy',h.npy(np.asarray(value)))
        rec['files']['transition.json']=store.json(stem+'/transition.json',base.numeric_json(transition))
        rec['files']['full_public_state.json']=store.json(stem+'/full_public_state.json',base.public_state(envs[s]))
        delta={}
        for name in h.NAMES:
            source=p['frames'][s][n][name]
            rec['exact'][name]=(store.out/rec['files'][name]['path']).read_bytes()==Path(source['path']).read_bytes()
            if name.endswith('.json'): delta[name]=differences(read(source['path'],source['sha256']),read(store.out/rec['files'][name]['path']))
        rec['differences']=store.json(stem+'/source_differences.json',delta)
        if n==protocol()['target_frames'][s]:
            row['baseline']=rec;shot=row['shots'][0];view=meter.render_views[id(envs[s])]
            shot['files']=dict(rgb_image=rec['image'],before=store.json(s+'/baseline_before.json',view['before']),after=store.json(s+'/baseline_after.json',view['after']))
            shot['status']='captured'
            require(equal(view['before'],view['after']) and equal(view['after'],snapshot(envs[s],base)), 'fresh baseline native render invariance')
        store.event(dict(phase='frame_saved',state_id=s,frame=n,exact=rec['exact']))
        needed=h.NAMES if n<=10 else h.NAMES[:3]
        require(all(rec['exact'][k] for k in needed),'original prefix five-file or suffix RGB/state/proprio byte gate')
    def construct(s,config):
        def create(): envs[s]=cls(**config['constructor']);return envs[s]
        h.operation(store,meter,s,'construct',dict(constructor=config['constructor']),create)
        h.operation(store,meter,s,'seed',dict(seed=config['seed']),lambda:envs[s].seed(config['seed']))
        obs,state=h.operation(store,meter,s,'reset',{},envs[s].reset)
        capture(s,0,obs,state,dict(kind='independent_goal_scene_reset' if s=='goal' else 'reset'))
    def step(s,n,u):
        env=envs[s];pos=np.asarray(env.agent.position,np.float64);raw=np.asarray(u,np.float64)
        command=dict(frame=n,raw_env_argument=raw.tolist(),dtype=raw.dtype.str,agent_xy_before=pos.tolist(),absolute_target=(pos+100*raw).tolist(),dt=.01,expected_substeps=10)
        store.json(s+f'/command_{n:02d}.json',command)
        obs,reward,done,info=h.operation(store,meter,s,'step',command,lambda:env.step(raw.copy()))
        capture(s,n,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info))
        require(done is False,'native terminal; do not pad')
    try:
        cls,pymunk,pygame,base,probe=loader();store.json('runtime_probe.json',probe)
        meter=Meter(cls,pymunk,store,base)
        require(sys.getprofile() is None,'do not replace existing profiler')
        sys.setprofile(meter.profile);profiled=True
        for plan in p['plans']:
            s=plan['arm'];construct(s,plan)
            for n,u in enumerate(plan['prefix_raw_u'],1):step(s,n,u)
        store.event(dict(phase='all44_prefixes_exact'))
        for plan in p['plans']:
            for n,u in enumerate(plan['reference_suffix_raw_u'][:5],11):step(plan['arm'],n,u)
        construct('goal',p['goal_protocol'])
        require(all(row['baseline'] is not None for row in rows),'five fresh predetermined states')
        attained=True;store.event(dict(phase='all5_states_attained'))
        for row in rows:
            a=np.load(store.out/row['baseline']['image']['npy']['path'],allow_pickle=False)
            intervene(envs[row['state_id']],row,store,meter,pygame,base,a)
        stable()
    except BaseException as exc:
        errors.append(dict(phase='execute',error=type(exc).__name__+': '+str(exc)))
    finally:
        for s,env in envs.items():
            try:h.operation(store,meter,s,'close',{},env.close,cleanup=True)
            except BaseException as exc:errors.append(dict(phase='close',state_id=s,error=type(exc).__name__+': '+str(exc)))
        if profiled:sys.setprofile(None)
    receipt=meter.receipt() if meter else None
    if receipt!=dict(entries=EXPECTED,return_events=EXPECTED,errors=[]):errors.append(dict(phase='census',error='complete native budget not met'))
    try:store.event(dict(phase='final',errors=errors),final=True)
    except BaseException as exc:errors.append(dict(phase='final_journal',error=str(exc)))
    try:store.journal.close()
    except BaseException as exc:errors.append(dict(phase='journal_close',error=str(exc)))
    try:store.files['journal.jsonl']=dict(path='journal.jsonl',sha256=sha(store.out/'journal.jsonl'),bytes=(store.out/'journal.jsonl').stat().st_size)
    except BaseException as exc:errors.append(dict(phase='journal_hash',error=str(exc)))
    r=dict(schema=NAME+'.report',prepare_sha256=prepare_sha,prepare_digest=digest(p),protocol=protocol(),
        status='live_raster_recorded' if not errors else 'partial_engineering_failure',rows=rows,all_states_attained=attained,
        errors=errors,meter=receipt,artifacts=store.files.copy(),journal_events=store.sequence,journal_terminal_sha256=store.previous,
        wall_seconds=time.monotonic()-store.started,profiler_released=sys.getprofile() is None,
        old_failure=p['old_failure'],old_renderer_release=False,historical_full_public_equivalence=None,scientific_admission=False)
    try:store.json('report.json',r,final=True)
    except BaseException as exc:
        r['errors'].append(dict(phase='report_save',error=str(exc)));r['status']='partial_engineering_failure'
    return r


def run(path,expected,out,authorized):
    require(authorized,'MAIN explicit605-step physics authorization')
    p=verify(path,expected);out=Path(out).absolute()
    require(out.parent==ROOT/'results' and out.name.startswith(NAME+'_'),'own exclusive output child')
    require(signal.getsignal(signal.SIGALRM)==signal.SIG_DFL and signal.getitimer(signal.ITIMER_REAL)==(0.,0.),'no existing alarm')
    def timeout(*args):raise TimeoutError('900s live raster cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(h.WALL)
    try:
        v3=h.load_bound(h.V3,h.V3_SHA)
        return execute(p,expected,out,lambda:v3.load_runtime(read(p['v3_prepare'],p['v3_sha256'])),lambda:verify(path,expected))
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,signal.SIG_DFL)


def consume(path,expected,report_path,report_sha):
    """Saved-only independent byte/PNG/dispatch/state rederivation; no physics."""
    from PIL import Image
    p=verify(path,expected);r=read(report_path,report_sha);out=Path(report_path).absolute().parent
    require(r['schema']==NAME+'.report' and r['prepare_sha256']==expected and r['prepare_digest']==digest(p) and r['protocol']==protocol(),'execution identity')
    require(r['status']=='live_raster_recorded' and not r['errors'] and r['all_states_attained'] and r['profiler_released'],'complete fresh raster only')
    require(r['old_failure']==p['old_failure'] and r['old_renderer_release'] is False and r['historical_full_public_equivalence'] is None and r['scientific_admission'] is False,'old failure and claim boundaries')
    files={str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}
    require(all(not f.is_symlink() for f in out.rglob('*')) and files==set(r['artifacts'])|{'report.json'},'exclusive complete artifact census')
    require(sum((out/n).stat().st_size for n in files)<=h.CAP and 0<=r['wall_seconds']<=h.WALL,'actual job caps')
    for name,rec in r['artifacts'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and Path(name).as_posix()==name,'local artifact path')
        require(rec==dict(path=name,sha256=sha(out/name),bytes=(out/name).stat().st_size),'actual artifact SHA/size')
    require(read(out/'execution_binding.json')==dict(schema=NAME+'.execution',prepare_sha256=expected,prepare_digest=digest(p),protocol=protocol()),'saved execution edge')
    v3=h.load_bound(h.V3,h.V3_SHA);v=read(p['v3_prepare'],p['v3_sha256'])
    v3.check_origins(v,*[v['origins'][k] for k in ('environment','pymunk','pygame')],read(out/'runtime_probe.json'))
    events=[];previous=None
    for i,line in enumerate((out/'journal.jsonl').read_bytes().splitlines()):
        x=json.loads(line);hh=x.pop('entry_sha256')
        require(x['sequence']==i and x['previous_sha256']==previous and digest(x)==hh,'journal hashchain')
        previous=hh;events.append(x['event'])
    require(len(events)==r['journal_events'] and previous==r['journal_terminal_sha256'] and events[-1]==dict(phase='final',errors=[]),'journal terminal')
    allowed={'intent','returned','failed','native_call','native_return','frame_saved','all44_prefixes_exact','all5_states_attained','final'}
    require(all(e['phase'] in allowed for e in events),'known journal phases')
    for phase in ('native_call','native_return'):
        require(dict(Counter(e['name'] for e in events if e['phase']==phase))==EXPECTED,'native actual entry/return counts')
    require(r['meter']==dict(entries=EXPECTED,return_events=EXPECTED,errors=[]),'native receipt')
    require(all(e['dt']==.01 for e in events if e['phase']=='native_call' and e['name']=='space_step'),'605 dt exact')
    pending=None;ni=Counter();no=Counter();dispatch=[];frames=[];prefix=False;attained=False
    for e in events:
        phase=e['phase']
        if phase=='intent':
            require(pending is None,'nonoverlap public dispatch');pending=(e['state_id'],e['operation']);ni.clear();no.clear()
            s,op=pending;details=e['details'];dispatch.append((s,op,details.get('frame'),details.get('shot')))
            require(s in STATES,'known state id');config=p['goal_protocol'] if s=='goal' else p['plans'][STATES.index(s)]
            if op=='construct':require(details==dict(constructor=config['constructor']),'original constructor')
            elif op=='seed':require(details==dict(seed=config['seed']),'original explicit seed')
            elif op=='step':
                require(details==read(out/s/f"command_{details['frame']:02d}.json"),'durable command dispatch')
                require(details['frame']<=10 or prefix,'all prefix gate before suffix')
            elif op in ('color','restore_render'):require(attained,'all5 before color')
        elif phase in ('native_call','native_return'):
            require(pending==(e['context']['state_id'],e['context']['operation']),'native public operation context')
            (ni if phase=='native_call' else no)[e['name']]+=1
        elif phase=='returned':
            require(pending==(e['state_id'],e['operation']),'paired actual return')
            require(dict(ni)==DELTAS[pending[1]] and dict(no)==DELTAS[pending[1]],'per-dispatch native delta');pending=None
        elif phase=='failed':require(False,'failed public operation')
        elif phase=='frame_saved':frames.append((e['state_id'],e['frame'],e['exact']))
        elif phase=='all44_prefixes_exact':
            require(not prefix and [(s,n) for s,n,_ in frames]==[(s,n) for s in STATES[:4] for n in range(11)]
                and all(all(x.values()) for _,_,x in frames),'44 exact prefix frames');prefix=True
        elif phase=='all5_states_attained':
            require(not attained and len(frames)==65 and prefix,'all65 before intervention');attained=True
    wanted=[]
    for s in STATES[:4]:
        wanted.extend((s,k,None,None) for k in ('construct','seed','reset'));wanted.extend((s,'step',n,None) for n in range(1,11))
    wanted.extend((s,'step',n,None) for s in STATES[:4] for n in range(11,16))
    wanted.extend(('goal',k,None,None) for k in ('construct','seed','reset'))
    wanted.extend((s,k,None,shot) for s in STATES for k,shot in (('color','visible'),('restore_render','visible_restored'),('color','white'),('restore_render','white_restored')))
    wanted.extend((s,'close',None,None) for s in STATES)
    require(pending is None and dispatch==wanted and prefix and attained,'fixed entire operation chronology')
    require(Counter(x[1] for x in dispatch)==Counter(protocol()['public_operations']),'public attempt/return census')
    def file(rec):
        require(rec==r['artifacts'][rec['path']],'artifact edge')
        return out/rec['path']
    def image(rec):
        arr=np.load(file(rec['npy']),allow_pickle=False)
        require(arr.shape==(224,224,3) and arr.dtype==np.uint8,'native RGB224')
        with Image.open(file(rec['png'])) as im:require(im.mode=='RGB' and np.array_equal(np.asarray(im),arr),'PNG decoded pixels')
        require(rec['pixel_sha256']==hashlib.sha256(arr.tobytes()).hexdigest(),'pixel SHA')
        return arr
    require([x['state_id'] for x in r['rows']]==list(STATES),'five fixed rows')
    encoded=[];allframes=[];comparisons=[]
    for row,target in zip(r['rows'],p['targets']):
        s=row['state_id'];nmax=target['frame'];ns=list(range(nmax+1))
        require(row['status']=='completed' and [x['frame'] for x in row['records']]==ns,'full planned frames')
        for rec in row['records']:
            n=rec['frame'];image(rec['image']);require(rec['image']['npy']==rec['files']['rgb.npy'],'native image edge')
            delta={};exact={}
            for name in h.NAMES:
                fr=rec['files'][name];require(fr['path']==s+f'/frame_{n:02d}/'+name,'exact original frame identity')
                src=p['frames'][s][n][name];exact[name]=file(fr).read_bytes()==Path(src['path']).read_bytes()
                if name.endswith('.json'):delta[name]=differences(read(src['path'],src['sha256']),read(file(fr)))
            require(exact==rec['exact'] and equal(delta,read(file(rec['differences']))),'durable complete exact source differences')
            require(all(exact[k] for k in (h.NAMES if n<=10 else h.NAMES[:3])),'original prefix/suffix byte gates')
            allframes.append((s,n,exact))
            comparisons.append(dict(state_id=s,frame=n,exact=exact,differences=rec['differences']))
            if n:
                raw=np.asarray((p['plans'][STATES.index(s)]['prefix_raw_u']+p['plans'][STATES.index(s)]['reference_suffix_raw_u'])[n-1],np.float64)
                before=read(out/s/f'frame_{n-1:02d}/full_public_state.json');pos=np.asarray(before['bodies']['agent']['position'],np.float64)
                require(equal(read(out/s/f'command_{n:02d}.json'),dict(frame=n,raw_env_argument=raw.tolist(),dtype='<f8',agent_xy_before=pos.tolist(),absolute_target=(pos+100*raw).tolist(),dt=.01,expected_substeps=10)),'original raw commands and absolute targets')
        require(row['baseline']==row['records'][-1] and [x['shot'] for x in row['shots']]==list(SHOTS),'fresh predetermined baseline/shots')
        require([x['color'] for x in row['cells']]==list(COLORS),'two fixed colors')
        baseline=None;anchor=None
        for shot in row['shots']:
            require(shot['status']=='captured','all triplet slots present');files=shot['files']
            arr=image(files['rgb_image']);before=read(file(files['before']));after=read(file(files['after']))
            require(equal(before,after),'actual render before/after unchanged')
            if shot['shot']=='baseline':
                require(files['rgb_image']==row['baseline']['image'],'baseline native image reuse')
                anchor=after;baseline=arr
                require(equal(after['public_state'],read(file(row['baseline']['files']['full_public_state.json']))),'baseline actual public state')
                require(after['public_state']['space']['current_time_step']==.01,'actual postreset timestep')
            else:
                require(equal(after['public_state'],anchor['public_state']) and equal(after['global_rng'],anchor['global_rng'])
                    and equal(after['topology'],anchor['topology']),'color only invariant')
                if shot['shot'] in COLORS:
                    require(after['goal_color']['rgba']==h.RGBA[shot['shot']],'actual requested native color')
                    encoded.append(dict(image_key=s+'/'+shot['shot'],state_id=s,color=shot['shot'],
                        rgb=files['rgb_image']['npy'],png=files['rgb_image']['png'],pixel_sha256=files['rgb_image']['pixel_sha256'],
                        before=files['before'],after=files['after'],fresh_snapshot_digest=digest(after),
                        fresh_public_state_digest=digest(after['public_state'])))
                else:require(equal(after,anchor),'original color object fully restored')
            if shot['shot']!='white':require(np.array_equal(arr,baseline),'original-color/contact pixels byte exact')
        for j,cell in enumerate(row['cells']):
            color=COLORS[j];require(cell['status']=='completed' and cell['requested_rgba']==h.RGBA[color],'complete color cell')
            require(cell['off_before_shot']==('baseline' if j==0 else 'visible_restored'),'shared fresh off triplets')
            require(equal(read(file(cell['anchor'])),anchor),'unchanged intervention anchor')
            require(cell['original_rgba']==anchor['goal_color']['rgba'] and cell['original_color_id']==anchor['goal_color']['object_id'],'original color identity')
            require(equal(read(file(cell['restoration'])),dict(snapshot=anchor,original_color_id=anchor['goal_color']['object_id'],same_color_object=True)),'actual restoration receipt')
    # Frame order in journal differs from row-major saved layout (all prefixes first).
    expected_frames=[(s,n) for s in STATES[:4] for n in range(11)]+[(s,n) for s in STATES[:4] for n in range(11,16)]+[('goal',0)]
    by_frame={(s,n):x for s,n,x in allframes}
    require(frames==[(s,n,by_frame[(s,n)]) for s,n in expected_frames],'all65 journal/saved frame joins')
    require(len(encoded)==10 and len([x for x in files_set(r) if x.endswith('.png')])==85,'10 fixed encoder inputs/85 PNG')
    return dict(schema=NAME+'.receipt',status='strict_live_raster_complete',prepare_sha256=expected,report_sha256=report_sha,
        states=5,cells=10,physics_steps=605,native_entries=EXPECTED,encoder_inputs=encoded,source_comparisons=comparisons,
        old_failure=p['old_failure'],old_renderer_release=False,historical_full_public_equivalence=None,scientific_admission=False)


def files_set(r):return set(r['artifacts'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('prepare','verify','run','consume'))
    for key in ('prepare','prepare-sha','output','report','report-sha','observer-report','observer-sha','observer-receipt','receipt-sha'):ap.add_argument('--'+key)
    ap.add_argument('--main-authorized',action='store_true');a=ap.parse_args()
    if a.mode=='prepare':
        p=build(a.observer_report,a.observer_sha,a.observer_receipt,a.receipt_sha);data=canonical(p)+b'\n'
        require(len(data)<=2*1024**2,'prepare2MiB cap')
        import shutil
        require(shutil.disk_usage(Path(a.output).absolute().parent).free>=h.FLOOR+len(data),'prepare200MiB floor')
        h.new(a.output,data);result=dict(path=a.output,sha256=sha(a.output),physics_executed=False)
    elif a.mode=='verify':result=dict(status='verified',prepare_digest=digest(verify(a.prepare,a.prepare_sha)))
    elif a.mode=='run':result=run(a.prepare,a.prepare_sha,a.output,a.main_authorized)
    else:
        result=consume(a.prepare,a.prepare_sha,a.report,a.report_sha)
        if a.output:h.new(a.output,canonical(result)+b'\n')
    print(json.dumps(result,sort_keys=True));return 2 if result.get('errors') else 0


if __name__=='__main__':raise SystemExit(main())
