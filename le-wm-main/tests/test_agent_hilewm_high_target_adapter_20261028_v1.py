"""Native source methods + NumPy RAM tensor shim, NO torch/models/physics."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import sys

import numpy as np
import pytest
from experiments.spectral_reachability import agent_hilewm_high_target_adapter_20261028_v1 as m


class Tensor:
    RAM_ONLY=True
    def __init__(self,a):self.a=np.array(a,copy=True)
    @property
    def shape(self):return self.a.shape
    @property
    def ndim(self):return self.a.ndim
    @property
    def device(self):return 'RAM'
    def size(self,i):return self.shape[i]
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.a
    def to(self,*args,**kwargs):return self
    def reshape(self,*shape):return Tensor(self.a.reshape(*shape))
    def unsqueeze(self,axis):return Tensor(np.expand_dims(self.a,axis))
    def squeeze(self,axis):return Tensor(np.squeeze(self.a,axis=axis))
    def expand(self,*shape):return Tensor(np.broadcast_to(self.a,tuple(self.shape[i] if n==-1 else n for i,n in enumerate(shape))))
    def __getitem__(self,key):return Tensor(self.a[key])
    def __sub__(self,other):return Tensor(self.a-other.a)
    def pow(self,n):return Tensor(self.a**n)
    def sum(self,dim):return Tensor(self.a.sum(axis=dim))


class TorchRAM:
    Tensor=Tensor
    is_tensor=staticmethod(lambda x:isinstance(x,Tensor))
    as_tensor=staticmethod(Tensor)
    stack=staticmethod(lambda xs,dim:Tensor(np.stack([x.a for x in xs],axis=dim)))
    cat=staticmethod(lambda xs,dim:Tensor(np.concatenate([x.a for x in xs],axis=dim)))


class Module:
    def __init__(self,fn):self.fn=fn;self.pre=[];self.post=[]
    def register_forward_pre_hook(self,fn):
        self.pre.append(fn);return NS(remove=lambda:self.pre.remove(fn))
    def register_forward_hook(self,fn):
        self.post.append(fn);return NS(remove=lambda:self.post.remove(fn))
    def __call__(self,*args):
        for h in self.pre:h(self,args)
        out=self.fn(*args)
        for h in self.post:h(self,args,out)
        return out


def rearrange(x,pattern,**kw):
    if pattern=='b t d -> (b t) d':return x.reshape(-1,x.shape[-1])
    assert pattern=='(b t) d -> b t d'
    return x.reshape(kw['b'],-1,x.shape[-1])


def native_method(source,cls,name):
    """Only the audited method body, no imports/decorators/module code executed.

    All tensor/module work below is NumPy arithmetic, not a CPU neural forward.
    """
    filename,h=m.SOURCES[source];p=m.TEXTS/filename;assert m.sha(p)==h
    tree=ast.parse(p.read_text());cl=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    node=deepcopy(next(x for x in cl.body if isinstance(x,ast.FunctionDef) and x.name==name));node.decorator_list=[]
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    mod=ast.fix_missing_locations(ast.Module(body=[future,node],type_ignores=[]))
    namespace=dict(torch=TorchRAM,rearrange=rearrange)
    exec(compile(mod,str(p),'exec'),namespace)
    fn=namespace[name];fn.RAM_ONLY=True;return fn


class FakeModel:
    RAM_ONLY=True
    def __init__(self):
        self.macro_to_condition=Module(lambda x:Tensor(np.tile(x.a,(1,1,2))))
        self.high_predictor=Module(lambda z,c:Tensor(z.a+c.a))
        self.high_pred_proj=Module(lambda x:Tensor(x.a*.75))
        self.latent_action_encoder=NS(quantize_latents=lambda x:Tensor(x.a+np.float32(.125)))
    def _device(self):return 'RAM'
    def _infer_high_context_len(self):return 2
    def _infer_latent_action_dim(self):return 2


for name in ('get_cost','get_cost_high','rollout_high','predict_high',
             'quantize_macro_actions_for_planning','project_macro_to_condition_space','_expand_sample_dim'):
    setattr(FakeModel,name,native_method('jepa','HiJEPA',name))
PLAN=native_method('policies','HierarchicalWorldModelPolicy','_plan_high')


class Solver:
    def __init__(self,model,cfg):self.model,self.cfg=model,cfg
    def __call__(self,info,init_action=None):
        b,d=info['z_init'].shape;h=self.cfg['horizon'];group=2*self.cfg['action_block'];s=3
        candidates=Tensor(np.arange(b*s*h*group,dtype=np.float32).reshape(b,s,h,group)/32)
        expanded=dict(planner_level='high',z_init=info['z_init'].unsqueeze(1).expand(b,s,d),
            z_goal=info['z_goal'].unsqueeze(1).expand(b,s,d))
        costs=self.model.get_cost(expanded,candidates)
        # Native CEM returns its updated mean, not necessarily an evaluated candidate.
        mean=Tensor(np.full((b,h,group),.2,np.float32))
        return dict(actions=mean,costs=costs,mean=[mean],var=[Tensor(np.ones((b,h,group),np.float32))])


def kwargs(**changes):
    cfg=dict(horizon=3,receding_horizon=2,action_block=1,warm_start=True)
    def factory(model):return Solver(model,cfg)
    factory.RAM_ONLY=True
    v=dict(native_plan=PLAN,model=FakeModel(),solver_factory=factory,cfg=cfg,
        z_init=Tensor(np.zeros((1,4),np.float32)),z_goal=Tensor(np.ones((1,4),np.float32)),
        next_high_init=Tensor(np.full((1,1,2),.3,np.float32)),limits=dict.fromkeys(m.AXES,1000))
    v.update(changes);return v


def capture(tmp_path,**changes):
    args=kwargs(**changes);out=tmp_path/'capture';r=m.capture_fixture(out,**args)
    return out,r,args


def test_complete_native_source_capture_and_saved_lineage(tmp_path):
    out,r,args=capture(tmp_path)
    assert r['errors']==[],r['errors']
    receipt=m.consume(out,m.sha(out/'report.json'))
    assert receipt['status']=='RAM_saved_high_target_verified_NOT_neural_evidence'
    assert receipt['production_executed'] is receipt['learned_prior_established'] is receipt['scientific_admission'] is False
    assert r['attempted']==r['completed']
    counts={k:r['completed'][k+'_calls'] for k in m.APIS}
    assert counts==dict(high_solver=1,get_cost_high=1,rollout_high=2,predict_high=5,
        quantize_macro_actions_for_planning=5,macro_to_condition=5,high_predictor=5,high_pred_proj=5)
    assert r['completed']['rollout_high_rows']==4  # 3 candidates + 1 selected, not API count
    assert r['completed']['predict_high_rows']==11  # 3*3 candidate horizon + selected2
    cost=receipt['diagnostics']['costs'][0]
    assert np.asarray(cost['native_L2sum']).shape==np.asarray(cost['saved_float64_L2sum']).shape==(1,3)
    assert receipt['diagnostics']['native_cost_vs_float64_bit_equality_claimed'] is False
    assert all(not getattr(args['model'],k).pre and not getattr(args['model'],k).post for k in m.MODULES)
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk','stable_worldmodel'))


@pytest.mark.parametrize('keep,block,warm',[(1,1,False),(3,1,True),(2,2,True)])
def test_native_receding_block_and_empty_warmstart(tmp_path,keep,block,warm):
    args=kwargs();args['cfg'].update(receding_horizon=keep,action_block=block,warm_start=warm)
    out=tmp_path/'capture';r=m.capture_fixture(out,**args)
    assert r['errors']==[],r['errors']
    c=m.consume(out,m.sha(out/'report.json'));assert c['diagnostics']['target_shape']==[1,4]
    assert r['completed']['predict_high_calls']==3*block+keep*block


@pytest.mark.parametrize('axis',['high_solver_calls','get_cost_high_rows','rollout_high_calls','predict_high_calls','macro_to_condition_calls'])
def test_predispatch_cap_retains_attempts_and_cleans_hooks(tmp_path,axis):
    args=kwargs();args['limits'][axis]=0
    out=tmp_path/'capture';r=m.capture_fixture(out,**args)
    assert r['errors'] and r['status']=='partial_or_failed'
    assert r['attempted'][axis]==r['completed'][axis]==0
    for key in m.MODULES:assert not getattr(args['model'],key).pre and not getattr(args['model'],key).post
    with pytest.raises(ValueError,match='complete capture'):m.consume(out,m.sha(out/'report.json'))


def test_forward_failure_no_refund_no_retry(tmp_path):
    model=FakeModel()
    def failed(z,c):raise RuntimeError('RAM callback failed')
    model.high_predictor.fn=failed
    out,r,_=capture(tmp_path,model=model)
    assert r['attempted']['high_predictor_calls']==1 and r['completed']['high_predictor_calls']==0
    assert r['attempted']['high_solver_calls']==1 and r['completed']['high_solver_calls']==0
    assert r['records'][0]['status']=='pending'
    assert all(not getattr(model,k).pre and not getattr(model,k).post for k in m.MODULES)


@pytest.mark.parametrize('fault',['report','array','journal','extra','truncated','dtype'])
def test_full_layout_mutations_fail(tmp_path,fault):
    out,r,_=capture(tmp_path);h=m.sha(out/'report.json')
    if fault=='report':(out/'report.json').write_bytes((out/'report.json').read_bytes()+b' ')
    elif fault=='array':
        p=next(out.glob('array_*.npy'));p.write_bytes(p.read_bytes()+b'x')
    elif fault=='journal':
        p=out/'journal.jsonl';p.write_bytes(p.read_bytes().replace(b'"returned"',b'"tampered"',1))
    elif fault=='extra':(out/'unexpected').write_bytes(b'bad')
    elif fault=='truncated':
        p=out/'journal.jsonl';p.write_bytes(p.read_bytes()[:-1])
    else:
        p=next(out.glob('array_*.npy'));np.save(p,np.zeros((1,4),np.float64),allow_pickle=False)
    with pytest.raises(ValueError):m.consume(out,h)


def rehash_events(out,edit):
    events=[m.decode(line) for line in (out/'journal.jsonl').read_bytes().splitlines()]
    report=m.decode((out/'report.json').read_bytes());edit(report,events)
    prev=None
    for i,e in enumerate(events):
        e['index']=i;e['previous']=prev;e['sha256']=m.digest({k:v for k,v in e.items() if k!='sha256'});prev=e['sha256']
    report['journal_tail']=prev
    (out/'journal.jsonl').write_bytes(b''.join(m.canonical(e)+b'\n' for e in events))
    (out/'report.json').write_bytes(m.canonical(report))


@pytest.mark.parametrize('fault',['last_target','goal_target','macro','source','warm','raw_quantized','native_mean_not_argmin'])
def test_independent_saved_math_binding_even_resealed_report(tmp_path,fault):
    out,r,_=capture(tmp_path)
    def edit(report,events):
        ret=[e['record'] for e in events if e['kind']=='return']
        selected=next(x for x in ret if x['kind']=='rollout_high' and x['parent'] is None)
        if fault in ('last_target','goal_target','warm'):
            if fault=='goal_target':report['output']['target']=report['inputs']['z_goal']
            elif fault=='last_target':
                pred=np.load(out/selected['output']['array'],allow_pickle=False)
                last=np.ascontiguousarray(pred[:,0,-1,:]);ref=report['output']['target'];path=out/ref['array']
                np.save(path,last,allow_pickle=False)
                ref.update(sha256=m.sha(path),bytes=path.stat().st_size,raw_sha256=m.hashlib.sha256(last.tobytes()).hexdigest())
            else:report['output']['next_high_init']=report['inputs']['next_high_init']
        else:
            i=selected['index']
            # Descriptor substitutions keep journal/report internally consistent.
            rec=deepcopy(report['records'][i])
            if fault=='source':rec['inputs']['z_init']=report['inputs']['z_goal']
            elif fault=='macro':rec['inputs']['macro']=report['inputs']['z_init']
            elif fault=='native_mean_not_argmin':
                solver=next(x for x in ret if x['kind']=='high_solver')
                rec['inputs']['macro']=solver['output']['actions']
            else:
                rec=deepcopy(next(x for x in ret if x['kind']=='predict_high'))
                q=next(x for x in ret if x['kind']=='quantize_macro_actions_for_planning')
                rec['inputs']['macro']=q['inputs']['macro'];i=rec['index']
            report['records'][i]=rec
            for e in events:
                if e.get('record',{}).get('index')==i:
                    e['record']=deepcopy(rec)
                    if e['kind']=='entry':e['record'].update(status='pending',output=None)
    rehash_events(out,edit)
    with pytest.raises(ValueError):m.consume(out,m.sha(out/'report.json'))


def test_production_gate_before_any_callbacks_or_outputs(tmp_path):
    marker=[]
    with pytest.raises(ValueError,match='production failclosed'):
        m.run_cuda(loader=lambda:marker.append('loader'),output=tmp_path/'no_create',MAIN=True)
    assert not marker and not (tmp_path/'no_create').exists()
    assert not any(x in sys.modules for x in ('torch','gymnasium','pymunk'))


def test_fixture_cannot_upgrade_to_real_via_report_flags(tmp_path):
    out,_,_=capture(tmp_path)
    def edit(r,e):r.update(fixture_only=False,production_executed=True,learned_prior_established=True)
    rehash_events(out,edit)
    with pytest.raises(ValueError,match='fixture only'):m.consume(out,m.sha(out/'report.json'))


def test_no_unknown_caps_zero_fill_or_fallback_model(tmp_path):
    args=kwargs();args['limits'].pop('high_solver_rows')
    with pytest.raises(ValueError):m.capture_fixture(tmp_path/'out',**args)
    args=kwargs();args['model']=object()
    with pytest.raises(ValueError,match='non-neural'):m.capture_fixture(tmp_path/'no_create',**args)
    assert not (tmp_path/'no_create').exists()


def test_no_global_or_original_model_method_replacement(tmp_path):
    model=FakeModel();original=dict(vars(model));cls=dict(vars(FakeModel));plan_code=PLAN.__code__
    out,r,_=capture(tmp_path,model=model)
    assert r['errors']==[] and dict(vars(model))==original and dict(vars(FakeModel))==cls and PLAN.__code__ is plan_code
    m.consume(out,m.sha(out/'report.json'))


def test_cannot_swallow_cap_then_continue(tmp_path):
    store=m.Store(tmp_path/'meter');caps=dict.fromkeys(m.AXES,10);caps['predict_high_calls']=0
    meter=m.Meter(store,caps);entered=[]
    with pytest.raises(ValueError,match='predispatch'):
        meter.call('predict_high',1,{},lambda:entered.append(1))
    with pytest.raises(ValueError,match='known actual'):
        meter.call('high_solver',1,{},lambda:entered.append(2))
    assert not entered and meter.poisoned and meter.attempted['predict_high_calls']==0


def test_return_file_failure_preserves_host_return_count(tmp_path):
    store=m.Store(tmp_path/'meter');meter=m.Meter(store,dict.fromkeys(m.AXES,10))
    original=store.pack;started=[]
    def pack(x):
        if isinstance(x,Tensor):raise OSError('return capture failed')
        return original(x)
    with patch.object(store,'pack',side_effect=pack),pytest.raises(OSError):
        meter.call('high_solver',1,{},lambda:(started.append(True),Tensor(np.ones((1,2),np.float32)))[1])
    assert started and meter.attempted['high_solver_calls']==meter.completed['high_solver_calls']==1
    assert meter.records[0]['status']=='returned_capture_pending' and meter.poisoned


@pytest.mark.parametrize('phase',['entry','return'])
def test_journal_failure_never_released_and_hooks_removed(tmp_path,phase):
    args=kwargs();original=m.Store.emit;faulted=False
    def emit(store,kind,**fields):
        nonlocal faulted
        if not faulted and kind==phase and fields.get('record',{}).get('kind')=='high_predictor':
            faulted=True;raise OSError('RAM journal fault')
        return original(store,kind,**fields)
    with patch.object(m.Store,'emit',emit):
        out=tmp_path/'capture';r=m.capture_fixture(out,**args)
    assert faulted and r['status']=='partial_or_failed'
    for key in m.MODULES:assert not getattr(args['model'],key).pre and not getattr(args['model'],key).post
    with pytest.raises(ValueError,match='complete capture'):m.consume(out,m.sha(out/'report.json'))


@pytest.mark.parametrize('axis',['high_solver_rows','get_cost_high_rows','predict_high_rows'])
def test_resealed_fake_counter_rejected_by_actual_shapes(tmp_path,axis):
    out,r,_=capture(tmp_path)
    def edit(report,events):
        kind=axis.removesuffix('_rows')
        rec=next(x for x in report['records'] if x['kind']==kind)
        rec['units'][axis]+=1;report['attempted'][axis]+=1;report['completed'][axis]+=1
        for e in events:
            if e.get('record',{}).get('index')==rec['index']:
                e['record']=deepcopy(rec)
                if e['kind']=='entry':e['record'].update(status='pending',output=None)
    rehash_events(out,edit)
    with pytest.raises(ValueError,match='recomputed native rows'):m.consume(out,m.sha(out/'report.json'))


def test_cli_actual_path_and_sha_roundtrip(tmp_path,capsys):
    out,_,_=capture(tmp_path)
    m.main(['consume','--output',str(out),'--report-sha256',m.sha(out/'report.json')])
    c=m.decode(capsys.readouterr().out)
    assert c['fixture_only'] is True and c['production_executed'] is False
    with pytest.raises(ValueError,match='external report SHA'):
        m.main(['consume','--output',str(out),'--report-sha256','0'*64])


def test_native_source_closure_drift_refused_before_callbacks(tmp_path):
    with patch.object(m,'sha',return_value='0'*64),pytest.raises(ValueError,match='native text'):
        m.capture_fixture(tmp_path/'no_output',**kwargs())
    assert not (tmp_path/'no_output').exists()
