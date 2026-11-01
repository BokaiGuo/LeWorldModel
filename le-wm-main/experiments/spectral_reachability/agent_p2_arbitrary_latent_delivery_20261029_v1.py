"""Pinned saved-input / numerical-only arbitrary FIRST delivery. No neural runtime."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from experiments.spectral_reachability import agent_p2_direct_latent_optimizer_20261029_v1 as core

NAME = 'agent_p2_arbitrary_latent_delivery_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT/'results/agent_p2_target_only_delivery_20261029_v1_prepare_MAIN01.json'
DELIVERY_SHA = '82b36fa3030f8ebb2da37bdc91ba3e8d582f129d6bc9e5c9f8e774b363ef3f62'
PROTOCOL = 'agent_p2_arbitrary_latent_waypoint_protocol_20261029_v1'
PINS = {
 'innerdocs/'+PROTOCOL+'.md': '179530bcfe842a678546daf66906d626d20765d1ba8e6f56c9abfd15ec3b540c',
 'experiments/spectral_reachability/'+core.NAME+'.py': '0f38e93e500ecfc75ec35581552d04a03ba2868dc81192659acdfc36c46a9b76',
 'tests/test_'+core.NAME+'.py': '76c28e8ab1617d3182adc2d5c028ce414e9f2200ee3279b051cc0e96663a93c4',
 'innerdocs/'+core.NAME+'.md': 'e2bb881d12d2698be42c6ef621a0ad86f43144c883ebb0a1d0eee452dad2bda7',
 'experiments/spectral_reachability/agent_p2_target_only_delivery_20261029_v1.py': 'e69b2d1e5fcd5d853004a0b3b2126c8506ffd7068a39ac867b60134d3358f4c7',
 'tests/test_agent_p2_target_only_delivery_20261029_v1.py': '6a3630ab6c67d7d1d590a856e594eb110f9cb4a8732adf56cf499ee8c5c44ae9',
 'innerdocs/agent_p2_target_only_delivery_20261029_v1.md': '722ca5aa5457c3f9e9708a7deffd7b79ff8a0b5506e6b4b8d64ff4a3deec740b',
 'experiments/spectral_reachability/'+core.PROTOCOL+'.py': core.PROTOCOL_SHA,
}
META, OUTPUT, RESERVE, FLOOR, WALL = 16*1024**2, 16*1024**2, 1024**2, 200*1024**2, 300
TASK_IDS = ['f5145aab0b44d21d0492d591590a73b4d4c9109903d5a97f18d01618f3bd5cc7',
            '697557899923163376df67cba615f13ebf1580166ca4de40a6759d12c927292b']
need, keys, digest = core.need, core.keys, core.digest


def raw(x): return core.raw(x)
def bh(b): return hashlib.sha256(b).hexdigest()
def ref(p): return dict(path=str(Path(p).absolute()), sha256=sha(p))


def regular(p):
    p = Path(p).absolute()
    need(p.is_absolute() and p.resolve() == p and p.is_file() and not p.is_symlink(), 'regular absolute file, no symlink')
    return p


def sha(p):
    p = regular(p); h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def read_bytes(p, cap=META):
    p = regular(p)
    need(p.stat().st_size <= cap, 'bounded saved file')
    with p.open('rb') as f: b = f.read(cap+1)
    need(len(b) <= cap, 'bounded saved read')
    return b


def decode(b):
    def pairs(xs):
        out = {}
        for k,v in xs: need(k not in out, 'duplicate JSON key'); out[k] = v
        return out
    return json.loads(b, object_pairs_hook=pairs, parse_constant=lambda _: need(False, 'nonfinite JSON'))


def read(p, h):
    core.hash_value(h); b = read_bytes(p)
    need(bh(b) == h, 'saved JSON SHA')
    return decode(b)


def fsync_dir(p):
    fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def durable(p, b):
    need(len(b) <= OUTPUT and shutil.disk_usage(Path(p).parent).free >= FLOOR+len(b), 'output cap/floor')
    with Path(p).open('xb') as f:
        f.write(b); f.flush(); os.fsync(f.fileno())
    fsync_dir(Path(p).parent)


def contract():
    return dict(schema=NAME+'.contract', roots=[0,1], condition='off', denominator=2,
        optimizer=core.contract(), selection='iterate64/FIRST only; one final float32 transport cast',
        optimization_only=True, neural_calls=0, controller_calls=0, physics_calls=0, fullstrict_calls=0,
        P1_admission=False, P2_admission=False, E1_admission=False, scientific_admission=False,
        time_alignment=False, same_generation_budget=False, equal_FLOPs=False, FLOPs=None,
        source_evidence='import pinned MAIN receipts and selected arrays; NOT upstream fullstrict rebuilt',
        metadata_cap=META, output_cap=OUTPUT, final_reserve=RESERVE, floor=FLOOR, wall_seconds=WALL)


def merge(dst, src):
    for p,h in src.items():
        core.hash_value(h)
        need(p not in dst or dst[p] == h, 'consistent binding SHA')
        dst[p] = h


def rehash(bindings):
    for p,h in bindings.items(): need(sha(p) == h, 'binding changed: '+p)


def source_seals():
    out = {str(ROOT/p):h for p,h in PINS.items()}
    for rel in ('experiments/spectral_reachability/'+NAME+'.py', 'tests/test_'+NAME+'.py', 'innerdocs/'+NAME+'.md'):
        p = ROOT/rel; out[str(p)] = sha(p)
    rehash(out)
    return out


def f32(value, shape):
    def scalar(x):
        if type(x) is list:
            for v in x: scalar(v)
        else: need(type(x) in (int,float) and math.isfinite(x), 'literal finite numeric, no bool')
    scalar(value)
    a = np.asarray(value, dtype=np.float64)
    need(a.shape == shape and np.isfinite(a).all(), 'exact finite shape')
    with np.errstate(over='ignore'): out = a.astype(np.float32)
    need(np.isfinite(out).all() and np.array_equal(out.astype(np.float64), a), 'literal original float32 values')
    return out


def same(a,b): return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
def exact(a,b): return raw(a) == raw(b)


def load_array(folder, name, desc):
    need(type(name) is str and len(name) == 15 and name.startswith('array_') and name[6:11].isdigit()
         and name.endswith('.npy'), 'original flat array name')
    keys(desc, ('data_sha256','dtype','file_bytes','file_sha256','shape'))
    need(type(desc['file_bytes']) is int and 0 <= desc['file_bytes'] <= 4*1024**2 and type(desc['shape']) is list
         and all(type(v) is int and v >= 0 for v in desc['shape']), 'bounded NPY descriptor')
    p = Path(folder)/name; b = read_bytes(p, 4*1024**2)
    need(len(b) == desc['file_bytes'] and bh(b) == desc['file_sha256'], 'NPY exact file SHA/bytes')
    s = io.BytesIO(b); need(np.lib.format.read_magic(s) == (1,0), 'original NPY v1')
    shape, order, dtype = np.lib.format.read_array_header_1_0(s)
    need(dtype.str == '<f4' and list(shape) == desc['shape'] and desc['dtype'] == '<f4' and order is False
         and math.prod(shape)*dtype.itemsize == len(b)-s.tell(), 'original f32 C-array header/EOF')
    a = np.frombuffer(b, dtype=dtype, offset=s.tell()).reshape(shape).copy()
    need(np.isfinite(a).all() and bh(a.tobytes()) == desc['data_sha256'], 'original finite data SHA')
    return a


def resolve_inputs(d, pm, pr, tp, get):
    """Selected-array join only. Never accept a prior/program target as the goal."""
    need(d['schema'] == 'agent_p2_target_only_delivery_20261029_v1.prepare', 'original delivery schema')
    need(exact(d['denominator'],dict(roots=2,arms=3,total=6,assembled=4,missing=2,dispatched=0)), 'preserve old six denominator')
    need([(x['root'],x['arm'],x['status']) for x in d['rows']] ==
         [(i,a,s) for i in (0,1) for a,s in (('prior_FIRST','assembled_not_dispatched'),
          ('program_first_postcondition','assembled_not_dispatched'),('arbitrary','missing'))], 'original all six slots')
    need([(s['root'],s['condition']) for s in pm['slots']] == [(0,'off'),(1,'off')]
         and len(pr['rows']) == len(tp['slots']) == 2, 'all known2/off')
    need(tp['protocol']['spec']['objective'] == 'identity192_terminal_meanMSE_float64'
         and len(tp['expected_state']) == 425, 'original frozen tracker identityQ/model inventory')
    out = []
    for i in (0,1):
        s, row, ts = pm['slots'][i], pr['rows'][i], tp['slots'][i]
        prior, program, missing = d['rows'][3*i:3*i+3]
        need(type(s['root']) is int and s['task_id'] == TASK_IDS[i] and row['root'] == i and row['status'] == 'returned', 'original root/task')
        need(missing['controller_payload'] is None, 'old missing unchanged')
        for x in (prior,program,missing):
            need(type(x['root']) is int and x['condition'] == 'off' and all(x[k] == s[k] for k in
                 ('task_id','task_sha256','blind_sha256')), 'all branch identity')
        for x in (prior,program): keys(x['controller_payload'], ('z','a','target'))
        z = f32(prior['controller_payload']['z'], (1,3,192)); a = f32(prior['controller_payload']['a'], (1,2,10))
        need(same(z,f32(program['controller_payload']['z'],(1,3,192))) and
             same(a,f32(program['controller_payload']['a'],(1,2,10))), 'all branches same causal history')
        need(same(z,f32([s['z']],(1,3,192))) and same(a,f32([s['a']],(1,2,10))), 'original slot exact bits')
        names = dict(z=row['routes']['hi_whole']['z'], a=row['routes']['hi_whole']['a'],
                     goal=row['pixels'][3]['paths']['hi']['output'])
        need(len(row['pixels']) == 4 and row['pixels'][3]['image'] == 3, 'original goal image index, NOT prior target')
        need(same(z,get(names['z'])) and same(a,get(names['a'])), 'actual observed arrays equal delivery')
        goal = get(names['goal']); need(goal.dtype == np.float32 and goal.shape == (1,192) and np.isfinite(goal).all(), 'original goal192')
        need(type(ts['root']) is int and ts['root'] == i and ts['condition'] == 'off' and all(ts[k] == s[k] for k in
             ('task_id','task_sha256','blind_sha256','pixel_sha256')), 'same tracker task/RGB')
        need(ts['pointers']['target'] == names['goal'] and ts['pointers']['history'] == names['z']
             and ts['pointers']['actions'] == names['a'], 'original tracker input recipe')
        for k,v in (('z',z),('a',a),('target',goal)):
            need(ts['input_sha256'][k] == bh(v.tobytes()), 'tracker actual input bytes')
        need(all(x['history_sha256'] == bh(z.tobytes()) and x['action_history_sha256'] == bh(a.tobytes())
                 for x in (prior,program,missing)), 'full history source digest')
        context = dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
                       latent_history=z[0].tolist(),action_history=a[0].tolist())
        out.append(dict(root=i,condition='off',task_id=s['task_id'],task_sha256=s['task_sha256'],
            blind_sha256=s['blind_sha256'],z=z.tolist(),a=a.tolist(),goal=goal[0].tolist(),context=context,
            input_sha256=dict(z=bh(z.tobytes()),a=bh(a.tobytes()),goal=bh(goal.tobytes())),
            goal_pointer=f'/rows/{i}/pixels/3/paths/hi/output',source_arrays=names))
    return out


def build(delivery=DELIVERY, delivery_sha=DELIVERY_SHA):
    need(delivery_sha == DELIVERY_SHA, 'pinned MAIN delivery only')
    bindings = source_seals(); d = read(delivery,delivery_sha); merge(bindings,{str(Path(delivery).absolute()):delivery_sha})
    merge(bindings,d['sources'])
    refs = d['authorities']; docs = {}
    for tag in ('parity_prepare','parity_report','parity_receipt','tracker_prepare','tracker_receipt'):
        r = refs[tag]; docs[tag] = read(r['path'],r['sha256']); merge(bindings,{r['path']:r['sha256']})
    pm,pr,pc,tp,tc = (docs[k] for k in ('parity_prepare','parity_report','parity_receipt','tracker_prepare','tracker_receipt'))
    need(pr['manifest_sha256'] == refs['parity_prepare']['sha256'] and
         Path(pr['manifest']) == Path(refs['parity_prepare']['path']) and pc == d['full_authority_receipts']['parity_receipt']
         and pc['report_sha256'] == refs['parity_report']['sha256'] and pc['all_values_exact'] is True, 'pinned parity full receipt imported')
    need(tp['upstream_receipt'] == pc and tc == d['full_authority_receipts']['tracker_receipt'] and
         tc['manifest_sha256'] == refs['tracker_prepare']['sha256'] and
         tc['report_sha256'] == refs['tracker_report']['sha256'] and
         tc['status'] == 'all_known2_normalized_tracker_saved_verified', 'tracker complete receipt imported')
    def get(name):
        desc = pr['arrays'][name]; p = Path(refs['parity_report']['path']).parent/name
        a = load_array(p.parent,name,desc); merge(bindings,{str(p):desc['file_sha256']}); return a
    slots = resolve_inputs(d,pm,pr,tp,get)
    coordinate_refs = {k:deepcopy(refs['tracker_prepare']) for k in core.COMMON}
    coordinate_refs['context'] = deepcopy(refs['parity_prepare']); coordinate_refs['task_goal'] = deepcopy(refs['parity_report'])
    for s in slots:
        s['task'] = dict(schema=core.NAME+'.task',root_slot=s['root'],condition='off',context=s['context'],
                         goal=s['goal'],coordinate_refs=deepcopy(coordinate_refs))
        core.task_check(s['task'])
    rehash(bindings)
    return dict(schema=NAME+'.prepare',contract=contract(),parent_delivery=ref(delivery),bindings=bindings,slots=slots,
        inherited_expected_state_sha256=digest(tp['expected_state']),original_controller=d['common_controller'],
        evidence='MAIN complete receipts inherited; all inherited binding bytes rehashed; selected array join only; no model deserialization; upstream fullstrict NOT rebuilt')


def prepare(output, delivery=DELIVERY, delivery_sha=DELIVERY_SHA):
    p = build(delivery,delivery_sha); durable(output,raw(p)+b'\n'); return ref(output)


def verify(prepared, h):
    p = read(prepared,h); r = p['parent_delivery']
    need(p == build(r['path'],r['sha256']), 'entire own prepare reconstruction')
    return p


class Store:
    """Own canonical-noLF + exactly one LF event transport, durable chain and files."""
    def __init__(self, path, clock=time.monotonic, start=None):
        self.path = Path(path); self.path.mkdir(exist_ok=False); fsync_dir(self.path.parent)
        self.clock = clock; self.start = clock() if start is None else start
        self.used = 0; self.files = {}; self.index = 0; self.tail = '0'*64
        self.last_intent = None
        self.journal = self.path/'events.jsonl'
        durable(self.journal,b'')

    def check(self): need(self.clock()-self.start <= WALL, 'whole numerical stage wall')

    def emit(self, kind, payload, final=False):
        if not final: self.check()
        e = dict(index=self.index,previous=self.tail,kind=kind,payload=payload)
        b = raw(dict(event=e,sha256=digest(e)))+b'\n'
        need(self.used+len(b) <= OUTPUT-(0 if final else RESERVE) and
             shutil.disk_usage(self.path).free >= FLOOR+len(b), 'journal cap/floor')
        with self.journal.open('ab') as f: f.write(b); f.flush(); os.fsync(f.fileno())
        self.used += len(b); self.tail = digest(e); self.index += 1
        if kind == 'operation_intent': self.last_intent = deepcopy(payload)
        elif kind == 'operation_return': self.last_intent = None

    def put(self, name, b, final=False):
        need(Path(name).name == name and name not in self.files and name not in ('events.jsonl','report.json'), 'own unique artifact')
        if not final: self.check()
        need(self.used+len(b) <= OUTPUT-(0 if final else RESERVE), 'whole output cap')
        durable(self.path/name,b); self.used += len(b)
        self.files[name] = dict(sha256=bh(b),bytes=len(b)); return name


def counters(): return dict(loss_attempted=0,loss_returned=0,gradient_attempted=0,gradient_returned=0,update_attempted=0,update_returned=0)


def initial(task):
    core.task_check(task)
    need(task['context']['layout'] == 'states_with_between_state_actions_v1', 'exact context layout')
    return [core.vector(task['context']['latent_history'][-1]) for _ in range(3)]


def operation(task, x, iteration, phase):
    if phase == 'loss':
        return core.objective(x,task['context']['latent_history'][-1],task['goal'],gradient=iteration<64)
    need(phase == 'update', 'fixed operations only')
    # Caller supplies the previously returned gradient; no second objective call.
    z,g = x
    out = [[v-core.ETA*dv for v,dv in zip(row,gr,strict=True)] for row,gr in zip(z,g,strict=True)]
    core.future(out); return out


def debit(c, phase, returned, iteration):
    suffix = 'returned' if returned else 'attempted'
    c[phase+'_'+suffix] += 1
    if phase == 'loss' and iteration < 64: c['gradient_'+suffix] += 1


def numerical(task, store):
    """Same objective/update order as frozen optimize; no old-global patch or opaque callback."""
    root = task['root_slot']; x = initial(task); history = []; c = counters(); failure = None
    for i in range(65):
        for phase in (('loss','update') if i<64 else ('loss',)):
            payload = dict(root=root,iteration=i,phase=phase)
            try:
                store.emit('operation_intent',payload); debit(c,phase,False,i)
                value = operation(task,x if phase == 'loss' else (x,g),i,phase)
                if phase == 'loss':
                    score,g = value
                    need(not history or score['loss'] <= history[-1]['score']['loss'], 'original fixed-step descent check')
                store.emit('operation_return',dict(**payload,value_sha256=digest(value)))
                debit(c,phase,True,i)
                if phase == 'loss': history.append(dict(iteration=i,targets=deepcopy(x),score=score,gradient=g))
                else: x = value
            except Exception as ex:
                failure = dict(**payload,error=type(ex).__name__+': '+str(ex))
                # If intent itself failed, there was no dispatched arithmetic and no debit.
                store.emit('operation_failed',dict(**failure,intent_recorded=exact(store.last_intent,payload)),final=True)
                break
        if failure is not None: break
    return dict(history=history,last_parameters=x,resources=c,failure=failure,
                delivered_targets=deepcopy(x) if failure is None else None)


def target_delivery(slot, result):
    if result['failure'] is not None: return dict(target64=None,target32=None,controller_payload=None)
    target64 = core.vector(result['delivered_targets'][0]); target32 = np.asarray(target64,np.float64).astype(np.float32)[None]
    need(np.isfinite(target32).all(), 'finite single transport cast, no repair')
    return dict(target64=target64,target32=target32.tolist(),controller_payload=dict(z=slot['z'],a=slot['a'],target=target32.tolist()))


def run(prepared, h, output, authorization, *, clock=time.monotonic):
    need(authorization == 'MAIN_NUMERICAL_ONLY', 'explicit MAIN numerical instantiation, no CUDA/physics')
    start = clock()  # Before all prepare/source/selected-array reconstruction I/O.
    p = verify(prepared,h); store = Store(output,clock=clock,start=start)
    store.emit('run_started',dict(prepare_sha256=h,task_sha256=[digest(s['task']) for s in p['slots']]),final=True)
    rows = []
    for s in p['slots']:
        store.emit('root_started',dict(root=s['root']),final=True)
        result = numerical(s['task'],store)
        try: delivery = target_delivery(s,result)
        except Exception as ex:
            result['failure'] = dict(root=s['root'],phase='transport',iteration=64,error=type(ex).__name__+': '+str(ex))
            result['delivered_targets'] = None; delivery = target_delivery(s,result)
            store.emit('transport_failed',result['failure'],final=True)
        name = store.put('root'+str(s['root'])+'_numerical.json',raw(result)+b'\n',final=True)
        row = dict(root=s['root'],condition='off',task_id=s['task_id'],status='delivered' if result['failure'] is None else 'partial',
            optimizer_result=name,optimizer_result_sha256=store.files[name]['sha256'],resources=result['resources'],
            target64=delivery['target64'],target32_sha256=bh(np.asarray(delivery['target32'],np.float32).tobytes()) if delivery['target32'] is not None else None,
            controller_payload=delivery['controller_payload'],history_sha256=s['input_sha256']['z'],
            action_history_sha256=s['input_sha256']['a'],goal_sha256=s['input_sha256']['goal'],source_time=None,failure=result['failure'])
        rows.append(row); store.emit('root_finished',dict(root=s['root'],row_sha256=digest(row)),final=True)
    store.emit('run_finished',dict(denominator=2,delivered=sum(r['status']=='delivered' for r in rows)),final=True)
    rehash(p['bindings']); need(sha(prepared) == h, 'prepare unchanged after stage')
    report = dict(schema=NAME+'.report',prepared=str(Path(prepared).absolute()),prepare_sha256=h,rows=rows,denominator=2,
        files=store.files,journal_tail=store.tail,event_count=store.index,whole_numerical_wall=store.clock()-store.start,
        contract=contract(),runtime=dict(numpy=np.__version__),evidence=p['evidence'])
    b = raw(report)+b'\n'; need(store.used+len(b) <= OUTPUT, 'final whole output cap'); durable(store.path/'report.json',b)
    store.used += len(b)
    report_ref = ref(store.path/'report.json')
    elapsed = clock()-start  # Includes verification, payloads, final report fsync and report hashing.
    complete = elapsed <= WALL and all(r['status']=='delivered' for r in rows)
    completion = dict(schema=NAME+'.completion',report_sha256=report_ref['sha256'],
        whole_stage_wall_through_report_fsync=elapsed,complete=complete,
        status='complete_numerical_within_budget' if complete else 'partial_stage_not_complete',
        boundary='before verify through durable report and its SHA; completion acknowledgement write excluded')
    cb = raw(completion)+b'\n'; need(store.used+len(cb) <= OUTPUT, 'completion whole output cap')
    durable(store.path/'completion.json',cb)
    return report_ref


def journal(path):
    b = read_bytes(path,OUTPUT); need(b.endswith(b'\n') and b'\n\n' not in b, 'exact single LF records')
    events = []; tail = '0'*64
    for i,line in enumerate(b[:-1].split(b'\n')):
        wrapper = decode(line); keys(wrapper,('event','sha256')); e = wrapper['event']; keys(e,('index','previous','kind','payload'))
        need(type(e['index']) is int and e['index'] == i and e['previous'] == tail and wrapper['sha256'] == digest(e)
             and line == raw(wrapper), 'whole native own event chain/canonical bytes')
        events.append(e); tail = wrapper['sha256']
    return events,tail


def replay(task, events, saved):
    """Recompute only actually returned operations; failed attempts never refunded."""
    x = initial(task); history = []; c = counters(); at = 0; failure = None
    for i in range(65):
        for phase in (('loss','update') if i<64 else ('loss',)):
            payload = dict(root=task['root_slot'],iteration=i,phase=phase)
            need(at < len(events), 'all operation lifecycle events present')
            e = events[at]; at += 1
            if e['kind'] == 'operation_failed':
                keys(e['payload'],('root','iteration','phase','error','intent_recorded'))
                need(e['payload']['intent_recorded'] is False, 'undispatched failure has no intent')
                failure = {k:v for k,v in e['payload'].items() if k != 'intent_recorded'}
            else:
                need(e['kind'] == 'operation_intent' and exact(e['payload'],payload), 'full loss/update chronology')
                debit(c,phase,False,i); need(at < len(events), 'operation return or failure')
                e = events[at]; at += 1
                if e['kind'] == 'operation_failed':
                    keys(e['payload'],('root','iteration','phase','error','intent_recorded'))
                    need(e['payload']['intent_recorded'] is True, 'dispatched failure retains intent')
                    failure = {k:v for k,v in e['payload'].items() if k != 'intent_recorded'}
                else:
                    value = operation(task,x if phase == 'loss' else (x,g),i,phase)
                    need(e['kind'] == 'operation_return' and exact(e['payload'],dict(**payload,value_sha256=digest(value))), 'returned exact arithmetic digest')
                    debit(c,phase,True,i)
                    if phase == 'loss':
                        score,g = value
                        need(not history or score['loss'] <= history[-1]['score']['loss'], 'original descent')
                        history.append(dict(iteration=i,targets=deepcopy(x),score=score,gradient=g))
                    else: x = value
            if failure is not None:
                need(exact({k:failure[k] for k in payload},payload) and type(failure['error']) is str and failure['error'], 'exact failure identity')
                break
        if failure is not None: break
    if failure is None and at < len(events):
        e = events[at]; at += 1
        keys(e['payload'],('root','phase','iteration','error'))
        need(e['kind'] == 'transport_failed' and type(e['payload']['root']) is int and e['payload']['root'] == task['root_slot'] and
             e['payload']['phase'] == 'transport' and type(e['payload']['iteration']) is int and e['payload']['iteration'] == 64
             and type(e['payload']['error']) is str and e['payload']['error'], 'transport-only failure')
        failure = e['payload']
    expected = dict(history=history,last_parameters=x,resources=c,failure=failure,delivered_targets=deepcopy(x) if failure is None else None)
    need(at == len(events) and raw(saved) == raw(expected), 'complete numerical payload and failed prefix independently replayed')
    return expected


def consume(prepared, ph, output, rh, receipt):
    p = verify(prepared,ph); folder = Path(output).absolute(); r = read(folder/'report.json',rh)
    keys(r,('schema','prepared','prepare_sha256','rows','denominator','files','journal_tail','event_count',
            'whole_numerical_wall','contract','runtime','evidence'))
    need(r['schema'] == NAME+'.report' and r['prepared'] == str(Path(prepared).absolute()) and r['prepare_sha256'] == ph
         and r['contract'] == contract() and r['evidence'] == p['evidence'] and type(r['denominator']) is int and r['denominator'] == 2
         and len(r['rows']) == 2, 'own authority and complete two denominator')
    need(type(r['whole_numerical_wall']) in (float,int) and math.isfinite(r['whole_numerical_wall']) and r['whole_numerical_wall'] >= 0, 'finite inclusive wall')
    completion = decode(read_bytes(folder/'completion.json'))
    keys(completion,('schema','report_sha256','whole_stage_wall_through_report_fsync','complete','status','boundary'))
    wall = completion['whole_stage_wall_through_report_fsync']
    need(completion['schema'] == NAME+'.completion' and completion['report_sha256'] == rh and
         type(wall) in (int,float) and math.isfinite(wall) and wall >= r['whole_numerical_wall'] and
         type(completion['complete']) is bool and completion['boundary'] ==
         'before verify through durable report and its SHA; completion acknowledgement write excluded', 'inclusive completion time/report identity')
    complete = wall <= WALL and r['whole_numerical_wall'] <= WALL and all(x['status']=='delivered' for x in r['rows'])
    need(completion['complete'] is complete and completion['status'] ==
         ('complete_numerical_within_budget' if complete else 'partial_stage_not_complete'), 'wall exceeded cannot certify complete')
    need(set(x.name for x in folder.iterdir()) == {'report.json','completion.json','events.jsonl',*r['files']}, 'exact output census')
    need(sum(x.stat().st_size for x in folder.iterdir()) <= OUTPUT, 'saved whole output cap')
    data = {}
    for name,desc in r['files'].items():
        need(Path(name).name == name, 'flat saved name'); keys(desc,('sha256','bytes'))
        b = read_bytes(folder/name); need(type(desc['bytes']) is int and len(b) == desc['bytes'] and bh(b) == desc['sha256'], 'artifact descriptor')
        data[name] = decode(b)
    events,tail = journal(folder/'events.jsonl')
    need(tail == r['journal_tail'] and type(r['event_count']) is int and len(events) == r['event_count'], 'full event census')
    at = 0
    def take(kind,payload):
        nonlocal at
        need(at < len(events) and events[at]['kind'] == kind and exact(events[at]['payload'],payload), 'whole run/root chronology: '+kind)
        at += 1
    take('run_started',dict(prepare_sha256=ph,task_sha256=[digest(s['task']) for s in p['slots']]))
    for s,row in zip(p['slots'],r['rows'],strict=True):
        i = s['root']; take('root_started',dict(root=i)); begin = at
        while at < len(events) and events[at]['kind'] != 'root_finished': at += 1
        name = 'root'+str(i)+'_numerical.json'; need(row['optimizer_result'] == name and name in data, 'one result per original root')
        result = replay(s['task'],events[begin:at],data[name]); delivered = target_delivery(s,result)
        expected = dict(root=i,condition='off',task_id=s['task_id'],status='delivered' if result['failure'] is None else 'partial',
            optimizer_result=name,optimizer_result_sha256=r['files'][name]['sha256'],resources=result['resources'],
            target64=delivered['target64'],target32_sha256=bh(np.asarray(delivered['target32'],np.float32).tobytes()) if delivered['target32'] is not None else None,
            controller_payload=delivered['controller_payload'],history_sha256=s['input_sha256']['z'],action_history_sha256=s['input_sha256']['a'],
            goal_sha256=s['input_sha256']['goal'],source_time=None,failure=result['failure'])
        need(raw(row) == raw(expected), 'exact FIRST delivery/transport/context identity, no extra fields')
        take('root_finished',dict(root=i,row_sha256=digest(row)))
    need(set(data) == {'root0_numerical.json','root1_numerical.json'}, 'all and only two numerical results')
    take('run_finished',dict(denominator=2,delivered=sum(x['status']=='delivered' for x in r['rows'])))
    need(at == len(events), 'unknown/tail events forbidden'); rehash(p['bindings']); need(sha(prepared) == ph, 'prepare stable after consume')
    out = dict(schema=NAME+'.receipt',status='all2_arbitrary_FIRST_numerically_verified_NOT_tracked' if complete else 'partial_arbitrary_numerical_prefix_verified_NOT_tracked',
        prepare_sha256=ph,report_sha256=rh,plans=r['rows'],denominator=2,upstream_fullstrict_rebuilt_here=False,
        parent_delivery=p['parent_delivery'],optimizer_source_sha256=PINS['experiments/spectral_reachability/'+core.NAME+'.py'],
        complete=complete,whole_stage_wall_through_report_fsync=wall,completion_sha256=sha(folder/'completion.json'),
        verification_cpu_calls={k:sum(row['resources'][axis] for row in r['rows']) for k,axis in
                                (('loss','loss_returned'),('analytic_gradient','gradient_returned'),('update','update_returned'))},
        actual_new_neural_calls=0,actual_controller_calls=0,verification_extra_cpu_arithmetic=True,scientific_admission=False)
    durable(receipt,raw(out)+b'\n'); return ref(receipt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('contract')
    q = sub.add_parser('prepare'); q.add_argument('--output',required=True); q.add_argument('--delivery',default=str(DELIVERY)); q.add_argument('--delivery-sha',default=DELIVERY_SHA)
    for name in ('verify','optimize','consume'):
        q = sub.add_parser(name); q.add_argument('--prepared',required=True); q.add_argument('--prepare-sha',required=True)
        if name != 'verify': q.add_argument('--output',required=True)
        if name == 'optimize': q.add_argument('--authorization',required=True)
        if name == 'consume': q.add_argument('--report-sha',required=True); q.add_argument('--receipt',required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'contract': out = contract()
    elif a.cmd == 'prepare': out = prepare(a.output,a.delivery,a.delivery_sha)
    elif a.cmd == 'verify': verify(a.prepared,a.prepare_sha); out = dict(verified=True,neural_calls=0,fullstrict_rebuilt=False)
    elif a.cmd == 'optimize': out = run(a.prepared,a.prepare_sha,a.output,a.authorization)
    else: out = consume(a.prepared,a.prepare_sha,a.output,a.report_sha,a.receipt)
    print(raw(out).decode())


if __name__ == '__main__': main()
