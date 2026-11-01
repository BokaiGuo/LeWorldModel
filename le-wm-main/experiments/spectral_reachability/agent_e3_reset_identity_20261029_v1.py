"""Capture actual reset and placement-sampling inputs without patching globals.

Not a standalone launcher or full E3 semantic consumer.
"""
import base64
import ast
import hashlib
from pathlib import Path
import traceback
import numpy as np
from experiments.spectral_reachability import agent_e3_authorized_episode_20261029_v2 as previous

old=previous.old
native_meter=previous.native_meter
need=old.need


def vector_bytes(value):
    a=np.asarray(value)
    need(a.shape==(7,) and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'actual numeric native7')
    return np.asarray(a,dtype='<f8').tobytes().hex()


def raw_vector(value):
    a=np.asarray(value)
    need(a.shape==(7,) and a.dtype.kind in 'fi' and a.dtype.itemsize in (2,4,8)
         and np.isfinite(a).all(),'exact raw native vector')
    return dict(dtype=a.dtype.str,bits=np.ascontiguousarray(a).tobytes().hex())


def validate_raw(raw, normalized):
    need(type(raw) is dict and set(raw)=={'dtype','bits'} and type(raw['dtype']) is str
         and type(raw['bits']) is str,'raw native vector schema')
    dt=np.dtype(raw['dtype']);need(dt.kind in 'fi' and dt.itemsize in (2,4,8)
         and len(raw['bits'])==14*dt.itemsize,'raw native vector bound')
    a=np.frombuffer(bytes.fromhex(raw['bits']),dtype=dt)
    need(vector_bytes(a)==normalized,'raw/normalized native vector exact')


def log(store,kind,resource,token,detail):
    store.event('physical',kind,resource,token,detail)


def failure(exc):
    return dict(type=type(exc).__name__,message=str(exc)[:2048],traceback=traceback.format_exc()[-4096:])


class TracedEnv:
    def __init__(self,env,store,episode):self.env=env;self.store=store;self.episode=episode;self.count=0
    @property
    def unwrapped(self):return self.env.unwrapped
    def step(self,*args,**kwargs):return self.env.step(*args,**kwargs)
    def render(self,*args,**kwargs):return self.env.render(*args,**kwargs)
    def close(self):return self.env.close()
    def reset(self,*args,**kwargs):
        need(not args and self.count<2 and set(kwargs) in ({'seed'},{'seed','options'})
             and type(kwargs['seed']) is int,'original reset ABI')
        i=self.count;self.count+=1;options=kwargs.get('options')
        need((i==0 and options is None) or (i==1 and type(options) is dict
             and set(options)=={'state','goal_state'}),'fixed reset options')
        d=dict(episode_id=self.episode,ordinal=i,seed=kwargs['seed'],options=None if options is None else
               {k:vector_bytes(options[k]) for k in ('state','goal_state')},options_raw=None if options is None else
               {k:raw_vector(options[k]) for k in ('state','goal_state')})
        token=self.store.index;log(self.store,'intent','reset_identity',token,d)
        try:
            result=self.env.reset(*args,**kwargs)
            obs,info=result
            output=dict(episode_id=self.episode,ordinal=i,state=vector_bytes(obs['state']),
                        goal_state=vector_bytes(info['goal_state']),state_raw=raw_vector(obs['state']),
                        goal_state_raw=raw_vector(info['goal_state']))
        except BaseException as exc:
            try:log(self.store,'failure','reset_identity',token,dict(**d,error=failure(exc)))
            except BaseException as e:exc.add_note('reset identity journal: '+repr(e))
            raise
        log(self.store,'return','reset_identity',token,output)
        return result


class Shape:
    def __init__(self,shape,index,seen):self.shape=shape;self.index=index;self.seen=seen
    def get_vertices(self):
        vertices=self.shape.get_vertices()
        a=np.asarray([[float(v.x),float(v.y)] for v in vertices],dtype='<f8')
        need(a.ndim==2 and a.shape[1]==2 and 3<=len(a)<=128 and np.isfinite(a).all(), 'actual polygon vertices')
        self.seen.append(dict(index=self.index,count=len(a),bits=a.tobytes().hex()))
        return vertices


class Source:
    def __init__(self,source,store,episode):self.source=source;self.store=store;self.episode=episode
    def __getattr__(self,name):return getattr(self.source,name)
    def sample_state(self,goal,shapes,slot,proposal,attempt,plan):
        need(type(slot) is dict and set(slot)=={'root_seed'} and type(slot['root_seed']) is int
             and type(proposal) is int and type(attempt) is int and len(shapes)>0
             and len(shapes)<=128 and type(plan) is dict and type(plan['recipe']) is dict, 'sample ABI')
        inputs=dict(episode_id=self.episode,goal=vector_bytes(goal),goal_raw=raw_vector(goal),shape_count=len(shapes),slot=slot,
                    proposal=proposal,attempt=attempt,recipe=plan['recipe'])
        token=self.store.index;log(self.store,'intent','sample_identity',token,inputs)
        selected=[];wrapped=[Shape(shape,i,selected) for i,shape in enumerate(shapes)]
        try:
            observed=[]
            for shape in wrapped:
                t=self.store.index;detail=dict(episode_id=self.episode,index=shape.index)
                log(self.store,'intent','geometry_vertex_read',t,detail)
                try:shape.get_vertices()
                except BaseException as exc:
                    try:log(self.store,'failure','geometry_vertex_read',t,dict(**detail,error=failure(exc)))
                    except BaseException as e:exc.add_note('geometry vertex journal: '+repr(e))
                    raise
                log(self.store,'return','geometry_vertex_read',t,dict(**detail,polygon=selected[-1]))
            observed.extend(selected);selected.clear()
            result=self.source.sample_state(goal,wrapped,slot,proposal,attempt,plan)
            need(len(selected)==1,'actual selected polygon')
            need(selected[0]==observed[selected[0]['index']], 'sampled polygon unchanged after census')
            d=dict(episode_id=self.episode,polygons=observed,selected=selected[0],
                   state=vector_bytes(result),state_raw=raw_vector(result),
                   extra_vertex_calls_attempted=len(shapes),extra_vertex_calls_returned=len(shapes))
        except BaseException as exc:
            try:log(self.store,'failure','sample_identity',token,dict(episode_id=self.episode,error=failure(exc)))
            except BaseException as e:exc.add_note('sample identity journal: '+repr(e))
            raise
        log(self.store,'return','sample_identity',token,d);return result


class Ops:
    def __init__(self,ops,store,episode):self.ops=ops;self.source=Source(ops.source,store,episode)
    def __getattr__(self,name):return getattr(self.ops,name)


def reconstruct(inputs,output):
    """Independent f64 reproduction of source sampling from the recorded input."""
    def vec(s):
        need(type(s) is str and len(s)==112,'native7 raw bytes');a=np.frombuffer(bytes.fromhex(s),'<f8').copy()
        need(np.isfinite(a).all(),'finite native7 bytes');return a
    goal=vec(inputs['goal']);slot=inputs['slot'];recipe=inputs['recipe'];poly=output['selected']
    n=inputs['shape_count'];need(type(n) is int and 0<n<=128 and type(slot['root_seed']) is int
        and inputs['proposal']==inputs['attempt']==0,'fixed sampling inputs')
    rng=np.random.default_rng(np.random.SeedSequence([slot['root_seed'],0,0]));state=goal.copy()
    state[2:4]+=rng.uniform(-recipe['position_jitter'],recipe['position_jitter'],2)
    state[4]+=rng.uniform(*recipe['angle_offset_range']);state[5:]=0
    index=int(rng.integers(n));need(poly['index']==index and type(poly['count']) is int and 3<=poly['count']<=128
        and type(poly['bits']) is str and len(poly['bits'])==poly['count']*32,'selected polygon identity')
    vertices=np.frombuffer(bytes.fromhex(poly['bits']),'<f8').copy().reshape(-1,2)
    need(np.isfinite(vertices).all(),'finite polygon');k=int(rng.integers(len(vertices)))
    a=vertices[k];b=vertices[(k+1)%len(vertices)];edge=b-a;norm=float(np.linalg.norm(edge))
    signed=float(np.sum(vertices[:,0]*np.roll(vertices[:,1],-1)-vertices[:,1]*np.roll(vertices[:,0],-1)))
    need(norm>0 and signed!=0,'nondegenerate polygon')
    normal=np.array([edge[1],-edge[0]])/norm*(1 if signed>0 else -1)
    local=a+rng.uniform(*recipe['edge_fraction_range'])*edge+recipe['pusher_surface_offset']*normal
    c,s=np.cos(state[4]),np.sin(state[4]);state[:2]=state[2:4]+np.array([[c,-s],[s,c]])@local
    need(state.astype('<f8').tobytes().hex()==output['state'],'sampled placement exact bytes')
    return state


ENV_SHA='c6d2104adc368ff1c352edc0e35ac74da9f07fda3a748ddbef0ba6b7a2d1daa2'


def official_polygons(path, expected):
    """Parse only literal geometry from the pinned official environment source."""
    p=Path(path);need(p.is_absolute() and p.is_file() and not p.is_symlink() and p.stat().st_size<=2*1024**2 and expected==ENV_SHA,
                      'pinned official geometry source')
    with p.open('rb') as f:b=f.read(2*1024**2+1)
    need(len(b)<=2*1024**2 and hashlib.sha256(b).hexdigest()==expected,'official geometry source SHA')
    tree=ast.parse(b);classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PushT']
    need(len(classes)==1,'official PushT class');methods=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='add_tee']
    need(len(methods)==1,'official add_tee');values={}
    def expr(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):return n.value
        if isinstance(n,ast.Name) and n.id in values:return values[n.id]
        if isinstance(n,(ast.List,ast.Tuple)):return [expr(x) for x in n.elts]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -expr(n.operand)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Mult,ast.Div)):
            a=expr(n.left);b=expr(n.right);return a*b if isinstance(n.op,ast.Mult) else a/b
        raise ValueError('unsupported official polygon expression')
    for node in methods[0].body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in ('scale','length','vertices1','vertices2'):
                need(key not in values,'unique geometry definition');values[key]=expr(node.value)
    polygons=[values['vertices1'],values['vertices2']]
    need(polygons==[[[-60,30],[60,30],[60,0],[-60,0]],
                    [[-15,30],[-15,120],[15,120],[15,30]]],'reviewed official geometry')
    return polygons


def verify_geometry(events,bindings):
    """Match the complete actual polygon census to the pinned source geometry."""
    paths=[p for p in bindings if p.endswith('/stable_worldmodel/envs/pusht/env.py')]
    need(len(paths)==1,'one bound official environment source')
    expected=official_polygons(paths[0],bindings[paths[0]])
    ends=[e for e in events if e['origin']=='physical' and e['resource']=='sample_identity' and e['kind']=='return']
    starts=[e for e in events if e['origin']=='physical' and e['resource']=='sample_identity' and e['kind']=='intent']
    need(len(starts)==len(ends)==1 and starts[0]['token']==ends[0]['token'],'one actual geometry census')
    d=ends[0]['detail'];polys=d['polygons']
    need(type(polys) is list and len(polys)==2 and d['extra_vertex_calls_attempted']==2
         and d['extra_vertex_calls_returned']==2,'complete two-polygon read accounting')
    reads=[e for e in events if e['origin']=='physical' and e['resource']=='geometry_vertex_read']
    need(len(reads)==4 and [e['kind'] for e in reads]==['intent','return','intent','return'],
         'actual geometry reads lifecycle')
    actual=[]
    for i,part in enumerate(polys):
        need(type(part) is dict and set(part)=={'index','count','bits'} and part['index']==i
             and type(part['count']) is int and part['count']==4 and type(part['bits']) is str
             and len(part['bits'])==128,'actual ordered polygon bytes')
        a=np.frombuffer(bytes.fromhex(part['bits']),dtype='<f8').reshape(4,2)
        attempt,returned=reads[2*i:2*i+2]
        need(attempt['token']==returned['token'] and starts[0]['index']<attempt['index']<returned['index']<ends[0]['index']
             and attempt['detail']==dict(episode_id=d['episode_id'],index=i)
             and returned['detail']==dict(episode_id=d['episode_id'],index=i,polygon=part),
             'exact geometry read and sample linkage')
        need(np.isfinite(a).all(),'finite polygon geometry');actual.append(sorted(map(tuple,a.tolist())))
    need(sorted(actual)==sorted(sorted(map(tuple,p)) for p in expected),'complete original geometry correspondence')
    selected=d['selected'];need(type(selected.get('index')) is int and 0<=selected['index']<2
        and selected==polys[selected['index']],'selected original ordered polygon bits')
    return dict(polygon_count=2,vertex_read_attempted=2,vertex_read_returned=2,
                source_sha256=ENV_SHA,scientific_admission=False)


def verify_events(events,episode,setup):
    """Rebuild completed two-reset identity from an authenticated journal."""
    reset=[e for e in events if e['origin']=='physical' and e['resource']=='reset_identity']
    sample=[e for e in events if e['origin']=='physical' and e['resource']=='sample_identity']
    need(len(reset)==4 and len(sample)==2 and [e['kind'] for e in reset]==['intent','return','intent','return']
         and [e['kind'] for e in sample]==['intent','return'], 'complete reset/sample lifecycle')
    a,b,c,d=reset;x,y=sample
    need(a['index']<b['index']<x['index']<y['index']<c['index']<d['index']
         and a['token']==b['token'] and c['token']==d['token'] and x['token']==y['token'], 'actual reset/sample order')
    ai,ar,ci,cr,si,sr=(e['detail'] for e in (a,b,c,d,x,y))
    need(all(z['episode_id']==episode for z in (ai,ar,ci,cr,si,sr))
         and ai['ordinal']==ar['ordinal']==0 and ci['ordinal']==cr['ordinal']==1
         and ai['seed']==setup['ref']['slot']['reset_seed'] and ai['options'] is None
         and ci['seed']==0 and si['slot']==dict(root_seed=setup['ref']['slot']['root_seed'])
         and si['recipe']==setup['source_plan']['recipe'] and si['goal']==ar['goal_state'], 'source-bound inputs')
    need(ai['options_raw'] is None and type(ci['options_raw']) is dict
         and set(ci['options_raw'])=={'state','goal_state'},'actual reset raw inputs')
    for part in (ar,cr):
        validate_raw(part['state_raw'],part['state']);validate_raw(part['goal_state_raw'],part['goal_state'])
    for k in ('state','goal_state'):validate_raw(ci['options_raw'][k],ci['options'][k])
    validate_raw(si['goal_raw'],si['goal']);validate_raw(sr['state_raw'],sr['state'])
    need(si['goal_raw']['dtype']=='<f8' and sr['state_raw']['dtype']=='<f8'
         and ci['options_raw']['state']==sr['state_raw'] and ci['options_raw']['goal_state']==si['goal_raw'],
         'actual source-to-reset input dtype and bits')
    reconstruct(si,sr)
    need(ci['options']==dict(state=sr['state'],goal_state=ar['goal_state'])
         and cr['goal_state']==ar['goal_state'], 'actual placement and original goal input bytes')
    return dict(reset_completed=2,sample_completed=1,selected_polygon=sr['selected']['index'],
                reset_state=cr['state'],scientific_admission=False)


def inspect_partial(events):
    """Reconstruct attempted/returned/failed identity calls without inventing inputs."""
    calls=[];active=None;phase=0
    for e in events:
        if e['origin']!='physical' or e['resource'] not in ('reset_identity','sample_identity'):continue
        kind=e['kind'];resource=e['resource'];d=e['detail']
        if kind=='intent':
            need(active is None and phase<3 and resource==('sample_identity' if phase==1 else 'reset_identity'),
                 'partial reset/sample call order')
            if resource=='reset_identity':need(d['ordinal']==(0 if phase==0 else 1),'reset ordinal')
            active=dict(token=e['token'],resource=resource,intent=d,index=e['index']);phase+=1
        else:
            need(kind in ('return','failure') and active is not None and e['token']==active['token']
                 and resource==active['resource'] and e['index']>active['index']
                 and d.get('episode_id')==active['intent'].get('episode_id'),'partial matching return')
            if kind=='failure':
                error=d.get('error');need(type(error) is dict and set(error)=={'type','message','traceback'}
                    and all(type(v) is str for v in error.values()),'partial original exception')
            calls.append(dict(resource=resource,intent=active['intent'],status=kind,result=d))
            active=None
            if kind=='failure':phase=3
    return dict(calls=calls,pending=active,scientific_admission=False)


def run(authority_args,resolver_path,resolver_sha,model,torch,device,store,row,additional_sources,authorization):
    """Same guarded episode with own per-call reset/sample tracing."""
    need(authorization==previous.AUTH and type(authority_args) is tuple and len(authority_args)==9
         and type(additional_sources) is dict and type(row) is dict and type(row.get('episode_id')) is str
         and bool(row['episode_id']), 'authorized fixed inputs')
    prepared=old.authority.host.read(authority_args[0],authority_args[1]);bindings=dict(prepared['bindings_sha256'])
    need(prepared['actual_resolver']==dict(path=resolver_path,sha256=resolver_sha),'exact resolver authority')
    old.authority.host.source.merge(bindings,additional_sources)
    modules=(old,old.authority,old.guard,old.live,old.setup_authority,old.live.setup,old.live.region,
             old.live.capture,old.live.dispatcher,old.live.encoding,old.live.nodes,old.native,native_meter,previous)
    required=[str(Path(m.__file__).resolve()) for m in modules]+[str(Path(__file__).resolve())]
    need(all(p in bindings for p in required),'all direct sources pinned')
    meter=native_meter.Meter(store,row['episode_id'])
    def record(stage,value):
        data=old.authority.host.raw(value);need(0<len(data)<=16*1024**2,'bounded guard snapshot')
        sha=hashlib.sha256(data).hexdigest();parts=[data[i:i+6000] for i in range(0,len(data),6000)]
        for i,part in enumerate(parts):
            log(store,'record','guard_snapshot_chunk',store.index,
                dict(stage=stage,index=i,total=len(parts),sha256=sha,data=base64.b64encode(part).decode('ascii')))
        log(store,'record',stage,store.index,dict(sha256=sha,total=len(parts),bytes=len(data)))
    def load():
        result=old.authority.load(*authority_args)
        actual=old.setup_authority.resolve(resolver_path,resolver_sha,result['cell'],result['body'])
        old.authority.host.source.merge(bindings,actual.pop('source_bindings'))
        actual['ops']=Ops(old.native,store,row['episode_id']);result['setup_kwargs']=actual
        return result
    def snap():return old.guard.snapshot(model,torch,device,bindings,required)
    def validate(a,b):return old.guard.validate_pair(a,b,prepared['expected_model_hash'])
    def create():return TracedEnv(native_meter.native_factory(meter),store,row['episode_id'])
    def execute(env,binding):
        return old.live.execute(env,binding['setup_kwargs'],binding['body'],binding['first5'],model,torch,device,store,meter,row)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with sdpa_kernel(SDPBackend.MATH),torch.autocast(device_type='cuda',enabled=False):
        return old._sequence(load,snap,create,execute,validate,record)
