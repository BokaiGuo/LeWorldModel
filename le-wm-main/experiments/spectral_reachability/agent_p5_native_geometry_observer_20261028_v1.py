"""MAIN-only 434-substep observation; never repairs/releases the old renderer."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import io
import json
from pathlib import Path
import signal
import struct
import sys
import time

import numpy as np

NAME = 'agent_p5_native_geometry_observer_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
HELPER = 'agent_p5_replay_target_renderer_20261028_v1'
HELPER_SHA = 'c7bbaaccd17b4734e6e133990ee4f22248efbaf328377f0c88f363eaa6bf3637'
PREP = ROOT/'results/agent_p5_replay_target_renderer_20261028_v1_prepare_MAIN01.json'
PREP_SHA = 'c2ecf01381cf339bf8f7e89fa428f4f62ac46075b10458ee7cfc8deac661941d'
FAILED = ROOT/'results/agent_p5_replay_target_renderer_20261028_v1_render_MAIN01/report.json'
FAILED_SHA = '1b02ed4a2ca136b4dc7f8549073ff030620df8587d654eb2a07ca86036cad0c6'
ARMS = ('xy', 'yx', 'x_minus_x', 'zero_zero')
EXPECTED = dict(env_construct=4, seed=8, reset=4, control_step=43, space_step=434,
    native_get_obs=47, native_render=47, setup=4, set_state=4, space_construct=4,
    native_goal_body=90, body_construct=102, debug_draw=47, close=4, geometry=86)


def helper():
    # Importing this module does not load a physics runtime.
    import hashlib
    path = ROOT/'experiments/spectral_reachability'/f'{HELPER}.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError('sealed helper source SHA')
    return importlib.import_module('experiments.spectral_reachability.'+HELPER)


h = helper()
require, canonical, digest, sha, read = h.require, h.canonical, h.digest, h.sha, h.read


def protocol():
    return dict(schema=NAME, arms=list(ARMS), frames={a:list(range(14 if a == 'xy' else 11)) for a in ARMS},
        order='four reset/prefix10; xy suffix11..13 only; close four',
        target=dict(arm='xy', frame=13, geometry_returns=2), native_entries=EXPECTED,
        physics_steps=434, control_steps=43, resets=4, new_color_renders=0,
        cap_bytes=h.CAP, reserve_bytes=h.RESERVE, floor_bytes=h.FLOOR, wall_seconds=h.WALL,
        prefix_gate='all44 exact five-file bytes before suffix',
        suffix_gate='save exact comparisons; diagnostic may complete despite old full-public/transition divergence; never release renderer',
        observation='unmodified original pymunk_to_shapely return, child get_vertices and local_to_world returns',
        renderer_release=False, historical_transform_certification=None, scientific_admission=False,
        neural=0, retries=0)


def build():
    p = h.verify(PREP, PREP_SHA)
    failed = read(FAILED, FAILED_SHA)
    require(failed['prepare_sha256'] == PREP_SHA and not failed['all_baselines_passed'], 'bound failed renderer')
    bindings = dict(p['bindings'])
    for f in (PREP, FAILED, Path(__file__).resolve(), ROOT/f'tests/test_{NAME}.py', ROOT/f'innerdocs/{NAME}.md'):
        bindings[str(f)] = sha(f)
    require([x['arm'] for x in p['plans']] == list(ARMS), 'four fixed original plans')
    return dict(schema=NAME+'.prepare', protocol=protocol(), replay_prepare=str(PREP), replay_sha256=PREP_SHA,
        failed_report=str(FAILED), failed_sha256=FAILED_SHA, bindings=bindings,
        plans=p['plans'], frames={a:p['frames'][a][:14 if a == 'xy' else 11] for a in ARMS},
        v3_prepare=p['v3_prepare'], v3_sha256=p['v3_sha256'])


def verify(path, expected):
    p = read(path, expected)
    require(p == build(), 'deterministic complete observer prepare')
    return p


def bits(x):
    return struct.pack('>d', float(x)).hex()


def enumerate_geometry(records, threshold, transition):
    """Only observed native components; no rotation reconstruction/union/repair."""
    from shapely import wkb
    from shapely.geometry import MultiPolygon
    require(len(records) == 2 and [r['body_role'] for r in records] == ['goal', 'block'], 'two ordered native returns')
    components = []
    for r in records:
        g = wkb.loads(bytes.fromhex(r['wkb_hex']))
        require(g.geom_type == 'MultiPolygon' and len(g.geoms) == 2, 'two native polygons')
        require([list(map(list, q.exterior.coords)) for q in g.geoms] == r['world_vertices'], 'WKB/world coordinate edge')
        components.append(list(g.geoms))
    rows = []
    require(np.isfinite(threshold) and threshold > 0, 'native threshold')
    for go in ((0, 1), (1, 0)):
        for bo in ((0, 1), (1, 0)):
            goal = MultiPolygon([components[0][i] for i in go])
            block = MultiPolygon([components[1][i] for i in bo])
            area = goal.area; intersection = goal.intersection(block).area
            coverage = intersection/area
            reward = float(np.clip(coverage/threshold, 0, 1))
            values = dict(goal_area=area, intersection_area=intersection, coverage=coverage, reward=reward)
            rows.append(dict(goal_order=list(go), block_order=list(bo), values=values,
                bits={k:bits(v) for k,v in values.items()},
                matches_actual=bits(coverage) == bits(transition['info']['final_coverage']) and bits(reward) == bits(transition['reward'])))
    return dict(rows=rows, actual_order_matches=rows[0]['matches_actual'],
        historical_membership=None, historical_transform_certification=None, renderer_release=False)


class Observer(h.Meter):
    """Profile callbacks do not raise into native code; errors gate next dispatch."""
    def __init__(self, cls, pymunk, store, base, envs):
        super().__init__(cls, pymunk, store)
        fn = cls.step.__globals__['pymunk_to_shapely']
        self.geometry_code = fn.__code__
        self.codes[self.geometry_code] = 'geometry'
        self.vertices_code = pymunk.Poly.get_vertices.__code__
        self.world_code = pymunk.Body.local_to_world.__code__
        self.base = base; self.envs = envs; self.active = {}; self.observations = []

    def profile(self, frame, event, arg):
        if event not in ('call', 'return'):
            return
        try:
            code = frame.f_code
            target = self.context == dict(state_id='xy', operation='step', frame=13)
            if code == self.geometry_code and event == 'call' and target:
                self.active[id(frame)] = dict(shapes=[], shape_body_ids=[], local_vertices=[], world_calls=[])
            if event == 'return' and code in (self.vertices_code, self.world_code) and self.active:
                parent = frame.f_back
                while parent is not None and id(parent) not in self.active:
                    parent = parent.f_back
                if parent is not None:
                    trace = self.active[id(parent)]
                    if code == self.vertices_code:
                        trace['shapes'].append(id(frame.f_locals['self']))
                        trace['shape_body_ids'].append(id(frame.f_locals['self'].body))
                        trace['local_vertices'].append([list(v) for v in arg])
                    else:
                        trace['world_calls'].append(dict(body_id=id(frame.f_locals['self']),
                            local=list(frame.f_locals['v']), world=list(arg)))
            if code == self.geometry_code and event == 'return' and id(frame) in self.active:
                trace = self.active.pop(id(frame)); env = self.envs['xy']; body = frame.f_locals['body']
                # Observe the returned GEOS object, not newly recomputed transforms.
                rec = dict(ordinal=len(self.observations), body_id=id(body), env_block_id=id(env.block),
                    body_role='block' if body is env.block else 'goal',
                    body_public={k:self.base.numeric_json(getattr(body,k)) for k in self.base.BODY_FIELDS},
                    wkb_hex=arg.wkb_hex, component_wkb_hex=[g.wkb_hex for g in arg.geoms],
                    world_vertices=[list(map(list,g.exterior.coords)) for g in arg.geoms],
                    public_state=self.base.public_state(env), context=dict(self.context), **trace)
                # Persist BEFORE assertions, so a malformed/partial trace is retained.
                self.store.json(f'geometry/return_{len(self.observations)}.json', rec)
                self.observations.append(rec)
                validate_observation(rec)
            if code in self.codes:
                key = self.codes[code]
                (self.entries if event == 'call' else self.returns)[key] += 1
                self.store.event(dict(phase='native_'+event, name=key, context=self.context,
                    dt=float(frame.f_locals['dt']) if key == 'space_step' else None))
                require(self.entries[key] <= EXPECTED[key], 'native entry cap '+key)
        except BaseException as exc:
            if len(self.errors) < 32:
                self.errors.append(type(exc).__name__+': '+str(exc))

    def receipt(self):
        return dict(entries={k:self.entries[k] for k in EXPECTED},
            return_events={k:self.returns[k] for k in EXPECTED}, errors=list(self.errors))


def validate_observation(r):
    require(len(r['shapes']) == 2 and len(set(r['shapes'])) == 2 and len(r['local_vertices']) == 2, 'actual two shape visits')
    require(r['shape_body_ids'] == [r['env_block_id']]*2, 'native block shape ownership for both transforms')
    require(len(r['world_vertices']) == 2 and len(r['world_calls']) == sum(map(len,r['local_vertices'])), 'actual transform call census')
    n = 0
    for local, world in zip(r['local_vertices'], r['world_vertices']):
        calls = r['world_calls'][n:n+len(local)]; n += len(local)
        require(canonical([c['local'] for c in calls]) == canonical(local), 'shape visit/local call order')
        require(canonical([c['world'] for c in calls]+[calls[0]['world']]) == canonical(world), 'actual world returns exactly polygon coords')
        require(all(c['body_id'] == r['body_id'] for c in calls), 'actual transform body identity')
    require((r['body_id'] == r['env_block_id']) == (r['body_role'] == 'block'), 'body role identity')


def execute(p, prepared_sha, out, loader):
    store = h.Store(out); envs = {}; meter = None; profiled = False; errors = []
    rows = [dict(arm=a, status='not_run', records=[]) for a in ARMS]
    store.json('execution_binding.json', dict(prepare_sha256=prepared_sha, prepare_digest=digest(p), protocol=protocol()))
    def capture(a, n, obs, state, transition):
        stem = a+f'/frame_{n:02d}'; row = rows[ARMS.index(a)]
        rec = dict(frame=n, files={}, image=store.image(stem+'/rgb', obs['visual']), exact={})
        row['records'].append(rec); rec['files']['rgb.npy'] = rec['image']['npy']
        for name, value in (('state',state), ('proprio',obs['proprio'])):
            rec['files'][name+'.npy'] = store.save(stem+'/'+name+'.npy', h.npy(np.asarray(value)))
        rec['files']['transition.json'] = store.json(stem+'/transition.json', base.numeric_json(transition))
        rec['files']['full_public_state.json'] = store.json(stem+'/full_public_state.json', base.public_state(envs[a]))
        for name in h.NAMES:
            rec['exact'][name] = (store.out/rec['files'][name]['path']).read_bytes() == Path(p['frames'][a][n][name]['path']).read_bytes()
        store.event(dict(phase='frame_saved', arm=a, frame=n, exact=rec['exact']))
        # Prefix gate remains strict. Diagnostic suffix differences never release v1.
        require(all(rec['exact'][k] for k in (h.NAMES if n <= 10 else h.NAMES[:3])), 'prefix byte gate or suffix physical observation divergence')
    def step(a,n,u):
        command = dict(frame=n, raw_env_argument=np.asarray(u,np.float64).tolist(), dtype='<f8')
        store.json(a+f'/command_{n:02d}.json', command)
        obs, reward, done, info = h.operation(store,meter,a,'step',command,lambda:envs[a].step(np.asarray(u,np.float64).copy()))
        capture(a,n,obs,info['state'],dict(kind='step',reward=reward,done=done,info=info))
        require(done is False, 'native early stop; no padding')
    try:
        cls, pymunk, pygame, base, probe = loader()
        store.json('runtime_probe.json',probe)
        meter = Observer(cls,pymunk,store,base,envs)
        require(sys.getprofile() is None, 'existing profiler must not be replaced')
        sys.setprofile(meter.profile); profiled = True
        for plan,row in zip(p['plans'],rows):
            a = plan['arm']; row['status'] = 'started'
            def construct():
                envs[a] = cls(**plan['constructor']); return envs[a]
            h.operation(store,meter,a,'construct',dict(constructor=plan['constructor']),construct)
            h.operation(store,meter,a,'seed',dict(seed=plan['seed']),lambda:envs[a].seed(plan['seed']))
            obs,state = h.operation(store,meter,a,'reset',{},envs[a].reset)
            capture(a,0,obs,state,dict(kind='reset'))
            for n,u in enumerate(plan['prefix_raw_u'],1): step(a,n,u)
            row['status'] = 'prefix_completed'
        store.event(dict(phase='all44_prefixes_exact'))
        for n,u in enumerate(p['plans'][0]['reference_suffix_raw_u'][:3],11): step('xy',n,u)
        require(not meter.errors and len(meter.observations) == 2, 'two complete native observations')
        tr = read(store.out/'xy/frame_13/transition.json')
        threshold = meter.observations[0]['public_state']['env']['success_threshold']
        store.json('observed_permutations.json',enumerate_geometry(meter.observations,threshold,tr))
        rows[0]['status'] = 'diagnostic_completed'
    except BaseException as exc:
        errors.append(dict(phase='execute',error=type(exc).__name__+': '+str(exc)))
    finally:
        for a,env in envs.items():
            try: h.operation(store,meter,a,'close',{},env.close,cleanup=True)
            except BaseException as exc: errors.append(dict(phase='close',arm=a,error=str(exc)))
        if profiled: sys.setprofile(None)
    receipt = meter.receipt() if meter else None
    if receipt != dict(entries=EXPECTED,return_events=EXPECTED,errors=[]):
        errors.append(dict(phase='census',error='fixed434 census incomplete'))
    try: store.event(dict(phase='final',errors=errors),final=True)
    except BaseException as exc: errors.append(dict(phase='final_journal',error=str(exc)))
    store.journal.close()
    store.files['journal.jsonl'] = dict(path='journal.jsonl',sha256=sha(store.out/'journal.jsonl'),bytes=(store.out/'journal.jsonl').stat().st_size)
    r = dict(schema=NAME+'.report',prepare_sha256=prepared_sha,prepare_digest=digest(p),protocol=protocol(),
        status='native_geometry_observed' if not errors else 'partial_engineering_failure',rows=rows,errors=errors,
        meter=receipt,artifacts=store.files.copy(),journal_events=store.sequence,journal_terminal_sha256=store.previous,
        renderer_release=False,historical_transform_certification=None,scientific_admission=False,
        wall_seconds=time.monotonic()-store.started,profiler_released=sys.getprofile() is None)
    store.json('report.json',r,final=True)
    return r


def run(path, expected, out, authorized):
    require(authorized, 'MAIN explicit434 physics authorization')
    p = verify(path,expected); out = Path(out).absolute()
    require(out.parent == ROOT/'results' and out.name.startswith(NAME+'_'), 'exclusive own result child')
    require(signal.getsignal(signal.SIGALRM) == signal.SIG_DFL and signal.getitimer(signal.ITIMER_REAL) == (0.,0.), 'no existing alarm')
    def timeout(*args): raise TimeoutError('900s diagnostic cap')
    signal.signal(signal.SIGALRM,timeout); signal.alarm(h.WALL)
    try:
        v3 = h.load_bound(h.V3,h.V3_SHA)
        result = execute(p,expected,out,lambda:v3.load_runtime(read(p['v3_prepare'],p['v3_sha256'])))
        verify(path,expected)
        return result
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,signal.SIG_DFL)


def consume(path, expected, report_path, report_sha):
    from PIL import Image
    p = verify(path,expected); r = read(report_path,report_sha); out = Path(report_path).parent
    require(r['schema'] == NAME+'.report' and r['prepare_sha256'] == expected and r['prepare_digest'] == digest(p) and r['protocol'] == protocol(), 'report identity')
    require(r['status'] == 'native_geometry_observed' and not r['errors'] and r['profiler_released'], 'complete diagnostic only')
    require(r['renderer_release'] is False and r['scientific_admission'] is False and r['historical_transform_certification'] is None, 'no retrospective certification')
    files = {str(f.relative_to(out)) for f in out.rglob('*') if f.is_file()}
    require(files == set(r['artifacts'])|{'report.json'} and all(not f.is_symlink() for f in out.rglob('*')), 'exclusive full file inventory')
    require(sum((out/f).stat().st_size for f in files) <= h.CAP and r['wall_seconds'] <= h.WALL, 'actual engineering caps')
    for name,rec in r['artifacts'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'local artifact')
        require(rec == dict(path=name,sha256=sha(out/name),bytes=(out/name).stat().st_size), 'artifact SHA/bytes')
    require(read(out/'execution_binding.json') == dict(prepare_sha256=expected,prepare_digest=digest(p),protocol=protocol()), 'execution edge')
    v3 = h.load_bound(h.V3,h.V3_SHA); v = read(p['v3_prepare'],p['v3_sha256'])
    v3.check_origins(v,*[v['origins'][k] for k in ('environment','pymunk','pygame')],read(out/'runtime_probe.json'))
    events=[]; previous=None
    for i,line in enumerate((out/'journal.jsonl').read_bytes().splitlines()):
        x=json.loads(line); hh=x.pop('entry_sha256')
        require(x['sequence']==i and x['previous_sha256']==previous and digest(x)==hh,'journal hashchain')
        previous=hh; events.append(x['event'])
    require(len(events)==r['journal_events'] and previous==r['journal_terminal_sha256'] and events[-1]==dict(phase='final',errors=[]),'journal terminal')
    for phase in ('native_call','native_return'):
        require(dict(Counter(e['name'] for e in events if e['phase']==phase))==EXPECTED,'native exact434 counts')
    require(r['meter']==dict(entries=EXPECTED,return_events=EXPECTED,errors=[]),'meter receipts')
    require(all(e['dt']==.01 for e in events if e['phase']=='native_call' and e['name']=='space_step'),'dt exact')
    wanted=[]
    for plan in p['plans']:
        a=plan['arm']; wanted.extend((a,op,None) for op in ('construct','seed','reset'))
        wanted.extend((a,'step',n) for n in range(1,11))
    wanted.extend(('xy','step',n) for n in range(11,14)); wanted.extend((a,'close',None) for a in ARMS)
    dispatch=[]; pending=None; ni=Counter(); no=Counter()
    for e in events:
        if e['phase']=='intent':
            require(pending is None,'nonoverlap dispatch'); pending=(e['state_id'],e['operation'])
            dispatch.append((*pending,e['details'].get('frame'))); ni.clear();no.clear()
            a,op=pending; plan=p['plans'][ARMS.index(a)]
            if op=='construct': require(e['details']==dict(constructor=plan['constructor']),'constructor exact')
            if op=='seed': require(e['details']==dict(seed=plan['seed']),'seed exact')
            if op=='step': require(e['details']==read(out/a/f"command_{e['details']['frame']:02d}.json"),'command dispatch receipt')
        elif e['phase'] in ('native_call','native_return'):
            require(pending == (e['context']['state_id'],e['context']['operation']),'native context')
            (ni if e['phase']=='native_call' else no)[e['name']]+=1
        elif e['phase']=='returned':
            require(pending==(e['state_id'],e['operation']),'durable return pair')
            delta=dict(h.DELTAS[pending[1]])
            if pending[1]=='step': delta['geometry']=2
            require(dict(ni)==delta and dict(no)==delta,'per-operation native delta');pending=None
        elif e['phase']=='failed': require(False,'failed native dispatch')
    require(dispatch==wanted and pending is None,'fixed complete dispatch chronology')
    require([x['arm'] for x in r['rows']]==list(ARMS),'four fixed rows')
    for row in r['rows']:
        a=row['arm']; ns=protocol()['frames'][a]
        require([x['frame'] for x in row['records']]==ns,'all47 frames')
        for rec in row['records']:
            n=rec['frame']; exact={}
            for name in h.NAMES:
                fr=rec['files'][name]; require(fr==r['artifacts'][fr['path']] and fr['path']==a+f'/frame_{n:02d}/'+name,'frame edge')
                exact[name]=(out/fr['path']).read_bytes()==Path(p['frames'][a][n][name]['path']).read_bytes()
            require(exact==rec['exact'] and all(exact[k] for k in (h.NAMES if n<=10 else h.NAMES[:3])),'exact frame comparison')
            image=rec['image']; arr=np.load(out/image['npy']['path'],allow_pickle=False)
            require(image['npy']==rec['files']['rgb.npy'] and arr.shape==(224,224,3) and arr.dtype==np.uint8,'RGB identity')
            with Image.open(out/image['png']['path']) as im: require(im.mode=='RGB' and np.array_equal(np.asarray(im),arr),'PNG pixels')
            import hashlib
            require(image['pixel_sha256']==hashlib.sha256(arr.tobytes()).hexdigest(),'pixel hash')
            if n:
                plan=p['plans'][ARMS.index(a)]; u=(plan['prefix_raw_u']+plan['reference_suffix_raw_u'])[n-1]
                require(read(out/a/f'command_{n:02d}.json')==dict(frame=n,raw_env_argument=np.asarray(u,np.float64).tolist(),dtype='<f8'),'raw original action')
    observations=[read(out/f'geometry/return_{i}.json') for i in range(2)]
    for x in observations: validate_observation(x)
    require([x['ordinal'] for x in observations]==[0,1] and all(x['context']==dict(state_id='xy',operation='step',frame=13) for x in observations),'target only returns')
    tr=read(out/'xy/frame_13/transition.json'); threshold=observations[0]['public_state']['env']['success_threshold']
    result=enumerate_geometry(observations,threshold,tr)
    require(result==read(out/'observed_permutations.json') and result['actual_order_matches'],'actual ordered coverage/reward bit match')
    return dict(schema=NAME+'.consume',status='native_observation_verified',prepare_sha256=expected,report_sha256=report_sha,
        geometry_returns=2,physics_steps=434,frames=47,permutations=result,renderer_release=False,historical_transform_certification=None,scientific_admission=False)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['prepare','verify','run','consume'])
    ap.add_argument('--prepare');ap.add_argument('--prepare-sha');ap.add_argument('--output');ap.add_argument('--report');ap.add_argument('--report-sha')
    ap.add_argument('--authorize-main-physics',action='store_true');args=ap.parse_args()
    if args.mode=='prepare':
        p=build(); h.new(args.output,canonical(p)+b'\n'); result=dict(path=args.output,sha256=sha(args.output),physics_executed=False)
    elif args.mode=='verify': result=dict(status='verified',prepare_digest=digest(verify(args.prepare,args.prepare_sha)))
    elif args.mode=='run': result=run(args.prepare,args.prepare_sha,args.output,args.authorize_main_physics)
    else:
        result=consume(args.prepare,args.prepare_sha,args.report,args.report_sha)
        if args.output: h.new(args.output,canonical(result)+b'\n')
    print(json.dumps(result,sort_keys=True));return 2 if result.get('errors') else 0


if __name__=='__main__':
    raise SystemExit(main())
