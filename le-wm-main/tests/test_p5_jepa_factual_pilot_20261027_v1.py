"""Contract/durable-I/O tests only; fixtures contain NO physics implementation."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import p5_jepa_factual_pilot_20261027_v1 as p


def test_protocol_exact_fresh_pilot_scope():
    q=p.protocol()
    assert q['seed']==2026102701 and q['roles']==['a','b']
    assert q['expected']['space_step']==302 and q['expected']['native_render']==32
    assert q['expected']['seed']==4 and q['controls_per_env']==15
    assert q['constructor']['reset_to_state'] is None
    assert not q['neural_execution'] and not q['action_fidelity_claim'] and not q['P5_admission']


def test_numeric_json_lossless_and_inf_explicit():
    x=np.array([np.nextafter(1.,2.),np.inf,-np.inf],np.float64)
    j=json.loads(p.canonical(p.numeric_json(x)))
    assert j['dtype']==x.dtype.str and j['values'][0]==x[0]
    assert j['values'][1:]==[{'nonfinite':'+inf'},{'nonfinite':'-inf'}]
    assert p.canonical(-0.)!=p.canonical(0.)


@pytest.mark.parametrize('x',[float('nan'),np.array([object()],object),object(),{1:'badkey'}])
def test_unsafe_or_unreviewed_values_rejected(x):
    with pytest.raises(ValueError):p.numeric_json(x)


def test_journal_hash_chain_and_durability(tmp_path):
    j=p.Journal(tmp_path/'journal.jsonl')
    j.append({'phase':'intent'});j.append({'phase':'returned'});j.close()
    previous=None
    for i,line in enumerate((tmp_path/'journal.jsonl').read_bytes().splitlines()):
        row=json.loads(line);digest=row.pop('entry_sha256')
        assert row['sequence']==i and row['previous_sha256']==previous
        assert hashlib.sha256(p.canonical(row)).hexdigest()==digest
        previous=digest


class FixtureCalls:
    context=None
    def receipt(self):return {'fixture_only':True}


def test_operation_intent_is_durable_before_callback(tmp_path):
    j=p.Journal(tmp_path/'j')
    def callback():
        rows=[json.loads(x) for x in (tmp_path/'j').read_text().splitlines()]
        assert rows[-1]['event']['phase']=='intent'
        return 42
    assert p.operation(j,FixtureCalls(),'a','fixture',{},callback)==42
    j.close()


def test_failed_callback_preserves_indeterminate_intent(tmp_path):
    j=p.Journal(tmp_path/'j')
    def callback():raise RuntimeError('fixture interruption')
    with pytest.raises(RuntimeError):p.operation(j,FixtureCalls(),'a','fixture',{},callback)
    j.close()
    assert len((tmp_path/'j').read_text().splitlines())==1
    assert json.loads((tmp_path/'j').read_text())['event']['phase']=='intent'


def test_npy_and_json_are_exclusive_and_pickle_free(tmp_path):
    a=np.arange(7,dtype=np.float32);path=tmp_path/'x.npy'
    p.save_array(path,a)
    np.testing.assert_array_equal(np.load(path,allow_pickle=False),a)
    with pytest.raises(FileExistsError):p.save_array(path,a)
    with pytest.raises(ValueError):p.save_array(tmp_path/'bad',np.array([object()],object))


def fixture_record(frame,digest='a'):
    return dict(frame=frame,files={n:{'sha256':digest} for n in ('rgb.npy','state.npy','proprio.npy','full_public_state.json','transition.json')})


def test_all16_rows_retained_on_mismatch():
    a=[fixture_record(i) for i in range(16)];b=[fixture_record(i) for i in range(16)]
    b[7]['files']['rgb.npy']['sha256']='different'
    rows=p.compare_records(a,b)
    assert len(rows)==16 and not rows[7]['all_exact']
    assert sum(x['all_exact'] for x in rows)==15


def test_comparison_refuses_missing_or_reordered_frames():
    a=[fixture_record(i) for i in range(16)]
    with pytest.raises(ValueError):p.compare_records(a,a[:-1])
    with pytest.raises(ValueError):p.compare_records(a,a[::-1])


def test_public_state_canonical_geometry_no_addresses():
    # Plain property fixtures, NOT pymunk objects or a simulator.
    def body():return SimpleNamespace(position=[1.,2.],velocity=[0.,0.],angle=0.,angular_velocity=0.,force=[0.,0.],torque=0.,center_of_gravity=[0.,0.],mass=1.,moment=1.,body_type=0,is_sleeping=False)
    a,b,s=body(),body(),body()
    Circle=type('Circle',(),{})
    def shape(x):
        c=Circle();c.body=a;c.radius=x;c.friction=.1;c.elasticity=0.;c.surface_velocity=[0.,0.];c.collision_type=0;c.sensor=False;c.color=[1,2,3,255];c.filter=SimpleNamespace(group=0,categories=1,mask=1);c.offset=[0.,0.];return c
    space=SimpleNamespace(bodies=[a,b],static_body=s,shapes=[shape(1.),shape(2.)],**{k:0. for k in p.SPACE_FIELDS})
    env=SimpleNamespace(agent=a,block=b,space=space,**{k:v for k,v in p.CTOR.items() if k!='reset_to_state'})
    env.np_random=np.random.default_rng(7);env.random_state=np.random.RandomState(7);env._seed=7
    env.goal_pose=[1,2,3];env.latest_action=None;env.n_contact_points=0;env.coverage_arr=[];env.sim_hz=100;env.control_hz=10;env.k_p=100;env.k_v=20;env.success_threshold=.95;env.teleop=False
    first=p.canonical(p.public_state(env));space.shapes.reverse()
    assert p.canonical(p.public_state(env))==first
    # Official _add_segment assigns LightGray to all static wall shapes.
    # Exercise static role, Segment geometry and color without importing pymunk.
    Segment=type('Segment',(),{})
    wall=Segment();wall.__dict__.update(shape(2.).__dict__)
    wall.body=s;wall.a=[5.,506.];wall.b=[5.,5.];wall.color=[211,211,211,255]
    space.shapes.append(wall)
    record=p.public_state(env)
    walls=[x for x in record['shapes'] if x['body']=='static']
    assert len(walls)==1 and walls[0]['type']=='Segment'
    assert walls[0]['color']==[211,211,211,255] and walls[0]['a']==[5.,506.]
    del wall.color
    with pytest.raises(AttributeError):p.public_state(env)
    space.shapes.remove(wall)
    b.velocity=[.001,0.]
    assert p.canonical(p.public_state(env))!=first


def test_fixture_collector_saves_all_native_returns_and_commands(tmp_path,monkeypatch):
    # No physics: constant ndarray-returning interface stub only.
    j=p.Journal(tmp_path/'j')
    monkeypatch.setattr(p,'public_state',lambda env:{'fixture_public':True})
    class FixtureEnv:
        def __init__(self,**kwargs):
            self.agent=SimpleNamespace(position=[10.,20.]);self._seed=123
            self.np_random=np.random.default_rng(123);self.random_state=np.random.RandomState(123)
        def seed(self,seed):
            self._seed=seed;self.np_random=np.random.default_rng(seed);self.random_state=np.random.RandomState(seed)
        def arrays(self):
            state=np.array([10,20,30,40,0,0,0],np.float32)
            return {'visual':np.zeros((224,224,3),np.uint8),'proprio':state[[0,1,5,6]]},state
        def reset(self):return self.arrays()
        def step(self,u):
            assert np.array_equal(u,np.zeros(2)) and u.dtype==np.float64
            last=json.loads((tmp_path/'j').read_text().splitlines()[-1])['event']
            assert last['phase']=='intent' and last['operation']=='step'
            obs,state=self.arrays();return obs,0.,False,{'state':state}
        def close(self):return None
    rows=p.collect_trajectory(FixtureEnv,'a',tmp_path,j,FixtureCalls());j.close()
    assert len(rows)==16
    assert len(list((tmp_path/'a').glob('frame_*/rgb.npy')))==16
    assert len(list((tmp_path/'a').glob('command_*.json')))==15
    assert json.loads((tmp_path/'a/command_01.json').read_text())['absolute_target']==[10.,20.]


def test_prepare_has_no_physics_or_neural_imports():
    before=set(__import__('sys').modules)
    m=p.prepare(p.ROOT)
    assert m['status']=='prepared_no_physics'
    assert not any(n.split('.')[0] in ('pymunk','gym','pygame','skimage','torch') for n in set(__import__('sys').modules)-before)
    p.verify(p.ROOT,m)


def test_run_rejects_wrong_sha_before_any_runtime_import(tmp_path):
    m=p.prepare(p.ROOT);f=tmp_path/'manifest';f.write_bytes(p.canonical(m))
    with pytest.raises(ValueError,match='SHA'):p.run(p.ROOT,m,f,'0'*64,tmp_path/'new')
    assert not (tmp_path/'new').exists()


def test_only_explicit_run_path_imports_physics_and_no_neural_code():
    tree=ast.parse(Path(p.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            name=n.module if isinstance(n,ast.ImportFrom) else ' '.join(a.name for a in n.names)
            assert not any(s in name for s in ('torch','pickle','transformers'))
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in ('add_safe_globals','safe_globals')
    for node in tree.body:
        assert not isinstance(node,ast.Import) or all(a.name!='pymunk' for a in node.names)
