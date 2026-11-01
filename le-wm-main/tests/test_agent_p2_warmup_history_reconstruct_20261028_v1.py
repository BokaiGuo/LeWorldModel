"""RAM-only collector regression; no simulator, neural forward or download."""
from copy import deepcopy
import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_warmup_history_reconstruct_20261028_v1 as m
from experiments.spectral_reachability import pusht_d2p_packet_contract as packets


class Env:
    def __init__(self, stop=None, fail=None):
        self.unwrapped = self; self.s = np.zeros(7); self.n = 0
        self.stop, self.fail = stop, fail

    def _get_obs(self): return self.s.copy()

    def reset(self, *, seed=0, options=None):
        self.n = 0
        self.s = np.array(options['state'] if options else np.zeros(7),dtype=float)
        return {'state':self.s.copy()}, {'goal':np.full((224,224,3),99,np.uint8)}

    def step(self, action):
        self.n += 1
        if self.n == self.fail: raise RuntimeError('RAM step exception')
        self.s[0] += 1
        return {'state':self.s.copy()},0.,self.n == self.stop,False,{}

    def render(self): return np.full((224,224,3),int(self.s[0]),np.uint8)

    def close(self): return None


def fixture():
    roots = [dict(state=np.array([20.*i,0,0,0,0,0,0]),goal=np.full(7,100.+i)) for i in range(4)]
    p = dict(contract=m.contract(), roots=[])
    for i,r in enumerate(roots):
        ident = m.root_fingerprint(r)
        rgbs = [np.full((224,224,3),20*i+n,np.uint8) for n in (5,10,15)]
        expected = r['state'].copy(); expected[0] += 15
        p['roots'].append(dict(root_slot=i,root_sha256=ident,
            reference=dict(root_sha256=ident,history_rgb_sha256=hashlib.sha256(np.stack(rgbs).tobytes()).hexdigest(),
                           goal_rgb_sha256=m.inv.pixel_sha(np.full((224,224,3),99,np.uint8))),
            expected_context=dict(state=expected.tolist(),goal=r['goal'].tolist(),raw_history=[[0.,0.]]*3)))
    return p, roots


def collector(roots, count=32):
    def call(env, **kw):
        assert kw == dict(count=4,seed=202753010,oversample_factor=4)
        for _ in range(count): env.reset(seed=0)
        return deepcopy(roots)
    return call


def warmup(env, root, *, history_len):
    assert history_len == 3
    _,info = env.reset(seed=0,options={'state':root['state'],'goal_state':root['goal']})
    history = packets.BoundaryHistory(env,root_state=root['state'],goal=root['goal'])
    for _ in range(3): history.execute(packets.repeat_command(np.zeros(2)))
    # Same chronological boundary result as unchanged production root-anchored helper.
    current = env.render()
    return list(history.frames[1:3])+[current],np.zeros((3,2),np.float32),info['goal']


def ram_collect(tmp_path, env=None, mutate=None, helper=warmup, collected_count=32):
    p,roots = fixture()
    if mutate: mutate(p,roots)
    rec = m.Recorder(tmp_path,lambda:None); rows = m.initial_rows(p)
    env = Env() if env is None else env
    rec.call('constructor',lambda:env)
    try:
        m.collect(p,env,rec,collector(roots,collected_count),helper,rows)
        rec.call('close',env.close)
    finally: rec.stream.close()
    return p,rec,rows


def test_real_separate_ram_calls_and_complete_denominators(tmp_path):
    p,rec,rows = ram_collect(tmp_path)
    c = m.accounting(rec.events)
    assert c['observed_public_calls']['collector_reset'] == dict(attempts=32,returns=32)
    assert c['observed_public_calls']['root_reset'] == dict(attempts=3,returns=3)
    assert c['observed_public_calls']['warmup_step'] == dict(attempts=45,returns=45)
    assert c['unreturned_attempts'] == []
    assert len(rows) == 8 and [r['status'] for r in rows[:2]] == ['already_saved']*2
    for slot in (1,2,3):
        a,b = rows[2*slot:2*slot+2]
        assert a['history'] == b['history'] and a['history']['state_at_warmup10'][0] == 20*slot+10
        assert [f['warmup_step'] for f in a['history']['frames']] == [5,10,15]
        assert len({f['path'] for f in a['history']['frames']}) == 3
    assert len(list(tmp_path.glob('*.png'))) == 12
    assert p['contract']['planned_returns'] is None
    assert all(v is None for v in c['unmeasured_native'].values())


@pytest.mark.parametrize('stop',[1,5,10,15])
def test_native_stop_never_padded(tmp_path,stop):
    p,roots = fixture(); rec = m.Recorder(tmp_path,lambda:None); rows = m.initial_rows(p)
    with pytest.raises(ValueError,match='native stop'):
        m.collect(p,Env(stop=stop),rec,collector(roots),warmup,rows)
    rec.stream.close()
    c = m.accounting(rec.events)['observed_public_calls']['warmup_step']
    assert c == dict(attempts=stop,returns=stop)
    assert len(rows) == 8 and rows[-1]['status'] == 'not_started'
    assert all(r['history'] is None for r in rows)


def test_exception_is_attempt_not_return_and_no_retry(tmp_path):
    p,roots = fixture(); rec = m.Recorder(tmp_path,lambda:None); rows = m.initial_rows(p)
    with pytest.raises(RuntimeError,match='RAM step exception'):
        m.collect(p,Env(fail=4),rec,collector(roots),warmup,rows)
    rec.stream.close(); c = m.accounting(rec.events)
    assert c['observed_public_calls']['warmup_step'] == dict(attempts=4,returns=3)
    assert len(c['unreturned_attempts']) == 1 and rows[-1]['status'] == 'not_started'


@pytest.mark.parametrize('damage',['root','order','goal','history','state7','raw','missing_root'])
def test_identity_drift_fails_closed(tmp_path,damage):
    def mutate(p,roots):
        if damage == 'root': roots[1]['state'][0] += .1
        if damage == 'order': roots[1],roots[2] = roots[2],roots[1]
        if damage == 'goal': p['roots'][1]['reference']['goal_rgb_sha256'] = '0'*64
        if damage == 'history': p['roots'][1]['reference']['history_rgb_sha256'] = '0'*64
        if damage == 'state7': p['roots'][1]['expected_context']['state'].pop()
        if damage == 'missing_root': roots.pop()
    def helper(*a,**kw):
        frames,raw,goal = warmup(*a,**kw)
        if damage == 'raw': raw[0,0] = 1
        return frames,raw,goal
    with pytest.raises(ValueError): ram_collect(tmp_path,mutate=mutate,helper=helper)


@pytest.mark.parametrize('n',[31,33])
def test_exact_original_collector_budget_not_assumed(tmp_path,n):
    with pytest.raises(ValueError): ram_collect(tmp_path,collected_count=n)


def test_same_pixels_do_not_allow_aliasing_boundaries(tmp_path):
    def helper(*a,**kw):
        f,r,g = warmup(*a,**kw); f[1] = f[0]; return f,r,g
    with pytest.raises(ValueError): ram_collect(tmp_path,helper=helper)


def saved_report(tmp_path):
    p,rec,rows = ram_collect(tmp_path)
    binding = dict(prepare_sha256='a'*64,contract=m.contract(),authority=m.AUTH,initial_rows=m.initial_rows(p))
    m.write(tmp_path/'execution_binding.json',binding)
    r = dict(schema=m.NAME+'.report',prepare_sha256='a'*64,status='history_reconstructed',rows=rows,
             root_denominator=4,arm_denominator=8,errors=[],accounting=m.accounting(rec.events),wall_seconds=1.,
             artifact_sha256={f.name:m.sha(f) for f in tmp_path.iterdir()},**m.FLAGS)
    m.write(tmp_path/'report.json',r)
    return p,r


def test_strict_saved_ram_closure(tmp_path,monkeypatch):
    p,_ = saved_report(tmp_path)
    monkeypatch.setattr(m,'verify',lambda *args:p)
    path = tmp_path/'report.json'
    result = m.consume('RAM','a'*64,path,m.sha(path))
    assert result['status'] == 'history_reconstructed' and result['new_model_calls'] == 0
    assert result['scientific_admission'] is False and result['stageB_adapter_ready'] is False


@pytest.mark.parametrize('damage',['denominator','row','frame_state','attempt','command','pixel','roots','completion','flags','manifest','returns','status','reset','wall'])
def test_saved_mutations_rejected(tmp_path,monkeypatch,damage):
    import json
    p,r = saved_report(tmp_path)
    monkeypatch.setattr(m,'verify',lambda *args:p)
    if damage == 'denominator': r['arm_denominator'] = 6
    if damage == 'row': r['rows'].pop()
    if damage == 'frame_state': r['rows'][2]['history']['state_at_warmup10'] = [0.]*7
    if damage == 'pixel':
        name = 'root1.warmup10.png'
        (tmp_path/name).write_bytes(b'bad')
        r['artifact_sha256'][name] = m.sha(tmp_path/name)
    if damage == 'flags': r['P1_admission'] = True
    if damage == 'manifest': r['artifact_sha256'].pop('events.jsonl')
    if damage == 'returns': r['accounting']['observed_public_calls']['warmup_step']['returns'] = 44
    if damage == 'wall': r['wall_seconds'] = m.WALL+1
    if damage == 'status': r['rows'][2].update(status='not_started',history=None)
    if damage in ('attempt','command','roots','completion','reset'):
        path = tmp_path/'events.jsonl'
        events = [json.loads(line) for line in path.read_bytes().splitlines()]
        if damage == 'attempt': next(e for e in events if e['kind'] == 'return')['call_id'] = 999999
        if damage == 'command': next(e for e in events if e['kind'] == 'attempt' and e['operation'] == 'warmup_step')['command'] = [1.,0.]
        if damage == 'roots': next(e for e in events if e['kind'] == 'roots_collected')['roots'][0]['state'][0] = -123.
        if damage == 'completion': next(e for e in events if e['kind'] == 'root_completed')['root_slot'] = 0
        if damage == 'reset': next(e for e in events if e['kind'] == 'root_reset_intent')['seed'] = 123
        path.write_bytes(b''.join(m.encode(e)+b'\n' for e in events))
        r['artifact_sha256']['events.jsonl'] = m.sha(path)
    # Deliberately overwrite ONLY ephemeral RAM fixture files, never workspace evidence.
    path = tmp_path/'report.json'; path.write_bytes(m.encode(r))
    with pytest.raises((ValueError,KeyError)): m.consume('RAM','a'*64,path,m.sha(path))


def test_run_requires_explicit_authority_before_any_verification_or_import(tmp_path):
    with pytest.raises(ValueError,match='MAIN-only'):
        m.run('nonexistent','bad',tmp_path/'out','not-authorized')
    assert not (tmp_path/'out').exists()


def test_cold_import_and_contract_are_nonphysical():
    code = '''
import sys
class Guard:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','gym','gymnasium','pymunk','stable_worldmodel','stable_pretraining'}:
   raise AssertionError('forbidden runtime import '+fullname)
sys.meta_path.insert(0,Guard())
from experiments.spectral_reachability import agent_p2_warmup_history_reconstruct_20261028_v1 as m
assert m.contract()['planned_returns'] is None
assert m.contract()['planned_attempts']['warmup_step'] == 45
assert m.contract()['scientific_admission'] is False
'''
    p = subprocess.run([sys.executable,'-B','-c',code],cwd=m.ROOT,text=True,capture_output=True,timeout=60,
                       env={**os.environ,'OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'})
    assert p.returncode == 0,p.stdout+p.stderr


def test_exclusive_metadata_write(tmp_path):
    p = tmp_path/'once.json'; m.write(p,{'returns':None})
    with pytest.raises(FileExistsError): m.write(p,{'returns':32})


def test_cleanup_is_metered_even_after_dispatch_cap(tmp_path):
    def stopped(): raise ValueError('RAM cap')
    rec = m.Recorder(tmp_path,stopped); calls = []
    with pytest.raises(ValueError,match='RAM cap'): rec.call('constructor',lambda:calls.append('create'))
    rec.call('close',lambda:calls.append('close'),cleanup=True); rec.stream.close()
    assert calls == ['close']
    assert m.accounting(rec.events)['observed_public_calls'] == {'close':dict(attempts=1,returns=1)}
    with pytest.raises(ValueError,match='only cleanup'): rec.call('warmup_step',lambda:None,cleanup=True)
