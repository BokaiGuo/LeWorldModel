"""Numeric and fake-env fixtures only: no native imports/physics/neural runs."""
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import p5_jepa_factual_panel_20261027_v3 as p


@pytest.fixture
def bound():
    return p.bound(p.ROOT)


def test_fixed_remaining_runtime_seeds_and_source_binding(bound):
    v2, base, _ = bound
    _, runtime = base.runtime_manifest(p.ROOT)
    assert list(p.SEEDS) == runtime['fresh_calibration_proposal']['seeds'][1:]
    assert p.SEEDS == (2026102702,2026102703,2026102704)
    assert v2.SEED == 2026102701 and base.SEED == 2026102701
    assert p.sha(p.ROOT/f'experiments/spectral_reachability/{p.V2}.py') == p.V2_SOURCE_SHA


def test_exact_v2_schedule_and_native_normalization(bound):
    v2, _, _ = bound
    m = p.prepare(p.ROOT)
    assert m['protocol']['schedule'] == v2.protocol()['schedule']
    stats = v2.native_stats(p.ROOT)
    values = v2.action_inputs(stats)
    u = np.array(m['protocol']['schedule'],np.float64).reshape(15,2)
    pixels = (100*u).astype(np.float32)
    packets = ((pixels/np.float32(100)-stats['ACTION_MEAN'])/stats['ACTION_STD']).reshape(3,10)
    np.testing.assert_array_equal(packets,values['normalized_packets'])
    for i,arm in enumerate(p.ARMS):
        bank = values['controls_packets'][i]
        np.testing.assert_array_equal(bank[:2],packets[:2])
        if arm == 'normalized_zero':
            np.testing.assert_array_equal(bank[-1],np.zeros(10,np.float32))
        if arm == 'reverse_steps':
            np.testing.assert_array_equal(bank[-1].reshape(5,2),packets[-1].reshape(5,2)[::-1])


def test_prepare_no_physics_or_neural_import_and_tamper_rejected():
    before = set(sys.modules)
    m = p.prepare(p.ROOT)
    p.verify(p.ROOT,m)
    assert not {s.split('.')[0] for s in set(sys.modules)-before} & {'torch','pymunk','gym','pygame','skimage','transformers'}
    assert not m['physics_executed'] and not m['neural_executed']
    bad = copy.deepcopy(m)
    bad['protocol']['seeds'][0] = 2026102701
    with pytest.raises(ValueError,match='seal'):
        p.verify(p.ROOT,bad)


def test_protocol_counts_budget_and_noncontrol_scope(bound):
    v2, _, _ = bound
    q = p.protocol(v2)
    assert 'seed' not in q and 'expected' not in q
    assert q['per_root_expected'] == v2.protocol()['expected']
    assert q['panel_expected'] == {k:3*v for k,v in q['per_root_expected'].items()}
    assert q['panel_expected']['space_step'] == 906
    assert q['frames'] == [0,5,10,15] and q['context_frames'] == [0,5,10]
    assert q['budget']['maximum_accounted_bytes'] < 50_000_000
    assert not any(q[k] for k in ('neural_execution','training','prospective_control','action_fidelity_claim','control_benefit_claim','P5_admission'))


def test_dispatch_retains_failed_root_and_all_later_roots():
    rows = p.initial_rows()
    attempts, saved, snapshots = [], [], []
    def execute(seed):
        attempts.append(seed)
        return dict(status='fresh_native_panel_root_failed' if seed==p.SEEDS[0] else 'fresh_native_panel_root_recorded')
    p.dispatch(rows,execute,lambda row,result:saved.append((row['seed'],result['status'])),lambda i,r:snapshots.append(copy.deepcopy(r)),lambda:None)
    assert attempts == list(p.SEEDS)
    assert len(saved) == len(snapshots) == 3
    assert saved[0][1] == 'fresh_native_panel_root_failed'
    assert len(rows) == 3 and rows[0]['status'] == 'fresh_native_panel_root_failed'
    assert snapshots[0][1]['status'] == snapshots[0][2]['status'] == 'not_run'


def test_global_precondition_failure_all_notrun():
    rows, saved = p.initial_rows(), []
    def blocked():
        raise ValueError('source drift')
    def forbidden(seed):
        raise AssertionError('executor must not run')
    p.dispatch(rows,forbidden,lambda row,result:saved.append(result),lambda i,r:None,blocked)
    assert all(r['status']=='not_run' for r in rows)
    assert len(saved)==3 and all(not r['physics_executed'] for r in saved)
    assert all('source drift' in r['reason'] for r in saved)


def test_executor_escape_is_failed_not_falsely_notrun():
    rows, saved, attempted = p.initial_rows(), [], []
    def execute(seed):
        attempted.append(seed)
        raise RuntimeError('unknown partial native work')
    p.dispatch(rows,execute,lambda row,result:saved.append(result),lambda i,r:None,lambda:None)
    assert attempted == [p.SEEDS[0]]
    assert rows[0]['status']=='fresh_native_panel_root_failed'
    assert saved[0]['physics_executed'] is None
    assert all(r['status']=='not_run' for r in rows[1:])


def test_budget_writer_exclusive_exact_bytes_and_journal(tmp_path,bound):
    _, base, _ = bound
    w = p.BudgetIO(base)
    path = tmp_path/'root'
    w.mkdir(path)
    w.save_array(path/'a.npy',np.ones(30,np.float32))
    w.save_json(path/'a.json',dict(value=1))
    journal = w.journal(path/'journal.jsonl')
    journal.append(dict(phase='fixture_intent'))
    journal.append(dict(phase='terminal'),final=True)
    journal.close()
    assert w.payload == sum(f.stat().st_size for f in path.iterdir())
    assert w.nodes == 4
    with pytest.raises(FileExistsError):
        w.save_json(path/'a.json',dict(value=2))
    assert json.loads((path/'a.json').read_text()) == {'value':1}


def test_budget_reserves_failure_receipts_and_rejects_objects(bound,tmp_path):
    _, base, _ = bound
    w = p.BudgetIO(base)
    w.payload = p.MAX_PAYLOAD_BYTES-p.FINAL_RESERVE
    with pytest.raises(ValueError,match='budget'):
        w.save_json(tmp_path/'no.json',dict(x=1))
    assert not (tmp_path/'no.json').exists()
    w.save_json(tmp_path/'failure.json',dict(status='failed'),final=True)
    w.nodes = p.MAX_NODES-32
    with pytest.raises(ValueError,match='budget'):
        w.mkdir(tmp_path/'no_dir')
    w.save_json(tmp_path/'terminal.json',dict(status='not_run'),final=True)
    with pytest.raises(ValueError,match='numeric'):
        w.save_array(tmp_path/'object.npy',np.array([object()],object))


def test_no_outcome_reader_or_unsafe_imports():
    source = Path(p.__file__).read_text()
    tree = ast.parse(source)
    assert 'p5_jepa_fresh_fidelity_main_20261027_v11' not in source
    assert 'p5_jepa_fresh_fidelity_main_20261027_v10' not in source
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names = node.module if isinstance(node,ast.ImportFrom) else ' '.join(n.name for n in node.names)
            assert not any(x in names for x in ('torch','pickle','transformers'))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            if node.func.attr == 'read_text':
                assert ast.unparse(node.func.value) == 'Path(path)'
            assert node.func.attr not in ('load','add_safe_globals','safe_globals','set_state')
    assert 'v2.SEED =' not in source and 'base.SEED =' not in source


@pytest.mark.parametrize('seed',p.SEEDS)
def test_fake_seed_capture_export_exact_pair_and_intent_order(tmp_path,bound,monkeypatch,seed):
    v2, base, _ = bound
    monkeypatch.setattr(base,'public_state',lambda env:dict(fixture_seed=env._seed,frame=env.i))
    writer = p.BudgetIO(base)
    out = tmp_path/f'seed_{seed}'
    writer.mkdir(out)
    journal = writer.journal(out/'journal.jsonl')
    stats = v2.native_stats(p.ROOT)
    inputs = v2.action_inputs(stats)
    captured, constructed = {}, []
    class Calls:
        context = None
        def receipt(self):
            return dict(fixture_only=True)
    class FakeEnv:
        def __init__(self,**kwargs):
            constructed.append(self)
            self._seed = 123
            self.i = 0
            self.agent = SimpleNamespace(position=[10.,20.])
            self.np_random = np.random.default_rng(123)
            self.random_state = np.random.RandomState(123)
        def seed(self,value):
            assert value==seed
            self._seed = value
            self.np_random = np.random.default_rng(value)
            self.random_state = np.random.RandomState(value)
        def arrays(self):
            state = np.array([10,20,30,40,0,0,0],np.float32)
            return dict(visual=np.full((224,224,3),self.i,np.uint8),proprio=state[[0,1,5,6]]),state
        def reset(self):
            assert self._seed==seed
            return self.arrays()
        def step(self,u):
            np.testing.assert_array_equal(u,inputs['raw_env_commands'][self.i])
            self.i += 1
            last = json.loads((out/'journal.jsonl').read_text().splitlines()[-1])['event']
            assert last['phase']=='intent' and last['operation']=='step'
            role = last['role']
            cmd = json.loads((out/role/f'command_{self.i:02d}.json').read_text())
            np.testing.assert_array_equal(cmd['absolute_target'],np.array([10,20])+100*u)
            obs,state = self.arrays()
            return obs,0.,False,dict(state=state)
        def close(self):
            return None
    for role in ('a','b'):
        captured[role] = []
        p.collect_trajectory(base,writer,FakeEnv,seed,role,out,journal,Calls(),inputs,captured[role])
        receipt = p.export_inputs(v2,writer,out/role,seed,captured[role],inputs,stats)
        m = json.loads((out/receipt['path']).read_text())
        assert m['schema']==p.NAME and m['layout_schema']==p.V2 and m['seed']==seed
        assert m['frames'][3]['rgb']==f'{role}/frame_15/rgb.npy'
        assert m['frames'][2]['proprio']==f'{role}/frame_10/proprio.npy'
        assert m['normalized_packets_path']==f'{role}/consumer/normalized_packets.npy'
        np.testing.assert_array_equal(np.load(out/m['normalized_packets_path'],allow_pickle=False),inputs['normalized_packets'])
        assert np.load(out/m['context_proprio_path'],allow_pickle=False).shape==(3,4)
        assert len(m['file_hashes'])==9 and all(p.sha(out/role/'consumer'/k)==v['sha256'] for k,v in m['file_hashes'].items())
    journal.close()
    assert len(constructed)==2 and constructed[0] is not constructed[1]
    assert all(r['all_exact'] for r in base.compare_records(captured['a'],captured['b']))
    assert len(captured['a'])==len(captured['b'])==16
    assert v2.SEED==base.SEED==2026102701
    assert writer.payload==sum(f.stat().st_size for f in out.rglob('*') if f.is_file())
    assert writer.nodes==len(list(out.rglob('*')))+1


def test_wrong_manifest_sha_no_output_or_runtime(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}')
    with pytest.raises(ValueError,match='SHA'):
        p.run(p.ROOT,{},manifest,'0'*64,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


def test_symlink_manifest_rejected(tmp_path):
    actual = tmp_path/'manifest.json'
    actual.write_text('{}')
    link = tmp_path/'link.json'
    link.symlink_to(actual)
    with pytest.raises(ValueError,match='symlink'):
        p.sha(link)
