"""Actual saved prepare plus Python-only metadata/hook mocks. NOT render evidence."""
from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p5_renderer_factorial_20261028_v2 as m

BASE=importlib.import_module('experiments.spectral_reachability.p5_jepa_factual_pilot_20261027_v1')


@pytest.fixture(scope='module')
def actual():
    root=m.ROOT/'results'
    return m.prepare(root/'agent_p5_terminal_goal_attribution_20261028_v5_prepare_MAIN01.json',
        root/'agent_p5_terminal_goal_attribution_20261028_v5_cuda_MAIN01/report.json',
        root/'agent_p5_terminal_goal_attribution_20261028_v5_consume_MAIN01.json')


def test_actual_target_correction_source_state_and_shapes(actual):
    assert all(s['source_state']['env']['settings']['with_target'] is True for s in actual['states'])
    assert all(s['original_target']=='visible' for s in actual['states'])
    assert all(s['original_target_evidence']['assignment_lines']==[704,705] for s in actual['states'])
    assert actual['protocol']['denominator']==10 and actual['protocol']['new_native_renders']==15
    assert len(actual['supported_design'])==10
    assert m.view_state(actual['states'][0]['source_state'])['space']['current_time_step']==0
    assert actual['states'][0]['source_state']['space']['current_time_step']==.01
    m.verify(actual)


def test_actual_cli_prepare_verify(actual,tmp_path):
    p=tmp_path/'prepare.json'; a=actual['authority']
    assert m.main(['prepare','--manifest',str(p),'--authority',a['manifest'],a['report'],a['receipt']])==0
    assert m.main(['verify','--manifest',str(p),'--manifest-sha256',m.sha(p)])==0
    assert m.read_json(p)==actual


def test_decoder_preserves_rng_arrays_and_infinities():
    x=m.decode(dict(dtype='<u4',shape=[2],values=[0,4294967295]))
    assert x.dtype==np.uint32 and x[-1]==4294967295
    assert m.decode({'nonfinite':'+inf'})==float('inf')
    with pytest.raises(ValueError): m.decode(dict(dtype='<f8',shape=[3],values=[1,2]))


class Body:
    def __init__(self,source=None):
        self.__dict__.update(m.decode(source or {'body_type':0,'position':[0.,0.]}))


class Shape:
    def __init__(self,record,body):
        for k,v in record.items():
            if k not in ('type','body','filter','vertices'): setattr(self,k,deepcopy(v))
        self.body=body; self.filter=SimpleNamespace(**record['filter'])
        self.vertices=deepcopy(record.get('vertices',[]))
    def get_vertices(self): return self.vertices


class Circle(Shape):
    """Metadata-only named type consumed by original public_state reader."""


class Poly(Shape):
    """Metadata-only named type consumed by original public_state reader."""


class Segment(Shape):
    """Metadata-only named type consumed by original public_state reader."""


class Space:
    def __init__(self,source):
        self.__dict__.update(m.decode(source['space'])); self.current_time_step=0.
        self.roles={k:Body(v) for k,v in source['bodies'].items()}
        self.static_body=self.roles['static']; self.bodies=[self.roles['agent'],self.roles['block']]
        self.shapes=[{'Circle':Circle,'Poly':Poly,'Segment':Segment}[s['type']](s,self.roles[s['body']]) for s in source['shapes']]
        self.reindexed=[]; self.steps=0; self.draws=0
    def reindex_shapes_for_body(self,body): self.reindexed.append(body)
    def step(self,dt): self.steps+=1
    def debug_draw(self,options=None): self.draws+=1


def color(name):
    return list({'LightGreen':(144,238,144,255),'White':(255,255,255,255)}[name])


def ram_runtime(actual,fault=None):
    """All 'RGB' values are fixed stub returns, never scientific observations."""
    class Env:
        made=[]
        def __init__(self,**config):
            self.index=len(self.made); self.made.append(self)
            self.rec=actual['states'][self.index]; self.source=deepcopy(self.rec['source_state'])
            self.__dict__.update(config); self.seed(None); self.window=self.clock=None
            self.sim_hz=100; self.control_hz=10; self.k_p=100; self.k_v=20
            self.success_threshold=.95; self.closed=False
        def seed(self,seed=None):
            self._seed=0 if seed is None else seed
            self.np_random=np.random.default_rng(self._seed); self.random_state=np.random.RandomState(self._seed)
        def _setup(self):
            self.space=Space(self.source); self.agent=self.space.roles['agent']; self.block=self.space.roles['block']
            self.goal_pose=np.array(m.parent.GOAL,np.float64); self.goal_color=color('LightGreen')
            self.coverage_arr=[]; self.latest_action=None; self.n_contact_points=0; self.teleop=False
        def _get_goal_pose_body(self,pose): return Body()
        def _render_frame(self,mode):
            self._get_goal_pose_body(self.goal_pose); self.space.debug_draw()
            if fault=='step' and self.index==0: self.space.step(.01)
            if fault=='state' and self.index==0: self.n_contact_points+=1
            pixels=np.load(self.rec['rgb']['path'],allow_pickle=False).copy()
            if fault=='baseline' and self.index==0: pixels[0,0,0]^=1
            if self.goal_color==color('White'): pixels=np.full_like(pixels,255)  # RAM-only arbitrary return, not native compositing.
            return pixels
        def reset(self): raise AssertionError('no reset permitted')
        def _set_state(self,state): raise AssertionError('no set_state permitted')
        def step(self,u): raise AssertionError('no control step permitted')
        def _get_obs(self): raise AssertionError('no extra observation permitted')
        def close(self): self.closed=True
    return Env,SimpleNamespace(Space=Space,Body=Body),SimpleNamespace(Color=color),BASE,dict(status='native_import_ready_physics_not_executed',RAM_only=True)


def test_hydrate_real_function_with_original_public_reader(actual):
    cls,_,_,_,_=ram_runtime(actual)
    rec=actual['states'][0]; env=cls(**rec['constructor']); env.seed(rec['source_state']['env']['seed']); env._setup()
    env.agent.position=[0.,0.]; env.block.angle=0.
    result=m.hydrate_view(env,rec['source_state'],BASE)
    assert result['render_view_equal'] is True and result['source_full_public_equal'] is False
    assert BASE.public_state(env)==m.view_state(rec['source_state'])
    assert len(env.space.reindexed)==3 and env.space.steps==0


@pytest.mark.parametrize('field',['shapes','settings','sleep','dt'])
def test_hydrate_does_not_ignore_unapproved_fields(actual,field):
    cls,_,_,_,_=ram_runtime(actual)
    rec=deepcopy(actual['states'][0]); env=cls(**rec['constructor']); env.seed(rec['source_state']['env']['seed']); env._setup()
    if field=='shapes': env.space.shapes[0].friction=.9
    elif field=='settings': env.action_scale=99
    elif field=='sleep': rec['source_state']['bodies']['agent']['is_sleeping']=True
    else: env.space.current_time_step=.02
    with pytest.raises(ValueError): m.hydrate_view(env,rec['source_state'],BASE)
    assert env.space.steps==0


@pytest.fixture(scope='module')
def ram_result(actual,tmp_path_factory):
    out=tmp_path_factory.mktemp('renderer_v2')/'RAM_output'
    runtime=ram_runtime(actual)
    result=m.execute(actual,'a'*64,out,lambda:runtime,lambda:None)
    assert result['global_error'] is None, result['global_error']
    return out,result,runtime[0]


def test_real_control_flow_profiler_complete_budget_all_baselines_first(actual,ram_result):
    out,r,cls=ram_result
    assert r['status']=='all10_static_target_views_recorded'
    assert len(r['rows'])==5 and len(r['cells'])==10 and all(c['status']=='completed' for c in r['cells'])
    assert r['meter']==dict(entries=m.EXPECTED,return_events=m.EXPECTED,pending=0,errors=[])
    assert all(e.closed and e.goal_color==color('LightGreen') and e.space.steps==0 for e in cls.made)
    assert m.sys.getprofile() is None
    receipt=m.consume(actual,'a'*64,out/'report.json',m.sha(out/'report.json'))
    assert receipt['status']=='all10_saved_static_target_views_verified'
    assert receipt['complete_dynamic_restore'] is receipt['agent_hidden_released'] is False


@pytest.mark.parametrize('fault',['step','state','baseline'])
def test_failures_no_toggles_before_all_baselines_no_retry_and_close(actual,tmp_path,fault):
    runtime=ram_runtime(actual,fault)
    out=tmp_path/'failure'; r=m.execute(actual,'a'*64,out,lambda:runtime,lambda:None)
    assert r['status']=='static_render_failed' and r['global_error']
    assert len(r['cells'])==10 and all(c['status']=='not_run' for c in r['cells'])
    assert all(e.closed for e in runtime[0].made) and m.sys.getprofile() is None
    assert all(e.space.steps==0 for e in runtime[0].made)
    if fault=='step':
        assert r['meter']['entries']['space_step']==1 and r['meter']['return_events']['space_step']==0
    with pytest.raises(ValueError): m.consume(actual,'a'*64,out/'report.json',m.sha(out/'report.json'))


@pytest.mark.parametrize('fault',['pixels','state','cells','readonly','meter','journal','extra','science'])
def test_saved_mutations_failclosed(actual,ram_result,tmp_path,fault):
    out,_,_=ram_result
    import shutil
    target=tmp_path/'mutated'; shutil.copytree(out,target)
    report=target/'report.json'; r=m.read_json(report)
    if fault=='pixels':
        p=target/'xy_white.npy'; a=np.load(p,allow_pickle=False); a[0,0,0]^=1
        with p.open('wb') as f: np.save(f,a,allow_pickle=False)
    elif fault=='state':
        p=target/'xy_white_after.json'; s=m.read_json(p); s['bodies']['agent']['position'][0]+=1
        p.write_text(m.json.dumps(s))
    elif fault=='cells': r['cells'].pop()
    elif fault=='readonly': r['rows'][0]['restore']['readonly_difference']['actual']=.01
    elif fault=='meter': r['meter']['entries']['view_restore']=4
    elif fault=='journal':
        p=target/'journal.jsonl'; p.write_text(p.read_text().replace('"sequence": 0','"sequence": 9',1))
    elif fault=='extra': (target/'extra.txt').write_text('retain mutation fixture')
    else: r['scientific_admission']=True
    report.write_text(m.json.dumps(r))
    with pytest.raises(ValueError): m.consume(actual,'a'*64,report,m.sha(report))


def test_authorization_and_manifest_mutation(actual,tmp_path):
    with pytest.raises(ValueError,match='authorization'): m.run(tmp_path/'missing','a'*64,tmp_path/'x')
    bad=deepcopy(actual); bad['states'][0]['original_target']='white'
    with pytest.raises(ValueError,match='recomputation'): m.verify(bad)


def test_writer_exclusive_and_cap(tmp_path,monkeypatch):
    writer=m.Writer(tmp_path/'out')
    try:
        writer.emit('kept.json',{'partial':True})
        with pytest.raises(FileExistsError): writer.emit('kept.json',{})
        monkeypatch.setattr(m,'CAP',1024)
        with pytest.raises(ValueError,match='cap'): writer.guard()
    finally: writer.file.close()
