"""E3 per-cell 1800s native meter over the unified journal; not full E3 strict."""
from collections import defaultdict
from pathlib import Path
import traceback
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub
from experiments.spectral_reachability import p1_moving_live_feedback_physical_20261027_v1 as native

NAME='agent_e3_native_meter_20261029_v1'
CAPS=dict(constructor=1,close=1,reset=2,set_state=4,goal_setter=3,goal_render=2,
          goal_bind=1,public_step=30,physics_step=304,explicit_render=32)
REGIONS={'constructor':{'setup'},'close':{'cleanup'},'reset':{'setup'},'set_state':{'setup'},
         'goal_setter':{'setup','goal_bind'},'goal_render':{'setup'},'goal_bind':{'goal_bind'},
         'public_step':{'warmup','prefix','suffix'},'physics_step':{'setup','warmup','prefix','suffix'},
         'explicit_render':{'setup','warmup','goal_bind','prefix','suffix'}}
need=pub.base.need


class Meter:
    """Whole-job clock belongs to Store and must start before authority verification."""
    def __init__(self,store,episode):
        need(isinstance(store,pub.Store) and type(episode) is str and bool(episode),'unified store and episode')
        self.store=store;self.episode=episode;self.region='setup';self.secondary=[]
        self.counts={k:dict(attempted=0,completed=0,failed=0) for k in CAPS};self.pending={}
    def check(self):self.store.check('physical_events',0)
    def begin(self,resource,units=1):
        self.check();need(resource in CAPS and type(units) is int and units==1
                          and self.region in REGIONS[resource],'known native resource/region/unit')
        c=self.counts[resource];need(c['attempted']+units<=CAPS[resource],'native per-cell predispatch cap')
        token=self.store.index;detail=dict(episode_id=self.episode,region=self.region,units=units)
        self.store.event('physical','intent','native_meter/'+resource,token,detail)
        c['attempted']+=units;self.pending[token]=(resource,detail);return token
    def end(self,token):
        need(token in self.pending,'matching native return');resource,detail=self.pending[token]
        self.store.event('physical','return','native_meter/'+resource,token,detail)
        self.counts[resource]['completed']+=1;del self.pending[token]
    def fail(self,token,exc):
        need(token in self.pending,'matching native failure');resource,detail=self.pending[token]
        error=dict(type=type(exc).__name__,message=str(exc)[:2048],repr=repr(exc)[:4096],
                   traceback=traceback.format_exc()[-4096:])
        self.store.event('physical','failure','native_meter/'+resource,token,dict(**detail,error=error))
        self.counts[resource]['failed']+=1;del self.pending[token]
    def call(self,resource,fn):
        token=self.begin(resource)
        try:value=fn()
        except BaseException as exc:
            try:self.fail(token,exc)
            except BaseException as secondary:exc.add_note('native failure ledger: '+repr(secondary))
            raise
        self.end(token);return value
    def cleanup(self,fn):
        """Always attempt close, even if deadline/cap prevents a durable intent."""
        before=self.region;self.region='cleanup';token=None;record_error=None
        try:
            try:token=self.begin('close')
            except BaseException as exc:record_error=exc
            try:value=fn()
            except BaseException as exc:
                if token is not None:
                    try:self.fail(token,exc)
                    except BaseException as secondary:exc.add_note('close failure ledger: '+repr(secondary))
                if record_error is not None:exc.add_note('close intent failure: '+repr(record_error))
                raise
            if token is not None:self.end(token)
            if record_error is not None:raise record_error
            return value
        finally:self.region=before


class Env:
    """Close-only proxy; original gym env and its metered subclass are untouched."""
    def __init__(self,env,meter):self._env=env;self._meter=meter
    @property
    def unwrapped(self):return self._env.unwrapped
    def reset(self,*args,**kwargs):return self._env.reset(*args,**kwargs)
    def step(self,*args,**kwargs):return self._env.step(*args,**kwargs)
    def render(self,*args,**kwargs):return self._env.render(*args,**kwargs)
    def close(self):return self._meter.cleanup(self._env.close)


def native_factory(meter):
    need(isinstance(meter,Meter),'E3 meter required')
    return Env(meter.call('constructor',lambda:native.native_factory(meter)),meter)


def consume(output,report_sha,completion_sha,episode):
    """Independently rebuild native ledger; incomplete is not a full E3 strict."""
    p=Path(output);transport=pub.consume(p,report_sha,completion_sha)
    rb=pub.base.read_bytes(p/'report.json',pub.FINAL_CAP);need(pub.base.fsha(rb)==report_sha,'stable report')
    report=pub.base.decode(rb);data=pub.base.read_bytes(p/'events.jsonl',pub.base.CAP)
    need(pub.base.fsha(data)==report['journal_sha256'],'stable journal')
    events=[pub.base.decode(line) for line in data.splitlines()]
    counts={k:dict(attempted=0,completed=0,failed=0) for k in CAPS};stack=[];stopped=False;close_index=None
    for e in events:
        if e['origin']!='physical':continue
        resource=e['resource']
        if resource in ('native_control_return','native_warmup_return'):
            d=e['detail'];need(type(d.get('terminated')) is bool and type(d.get('truncated')) is bool,'native stop booleans')
            stopped=stopped or d['terminated'] or d['truncated']
        if resource=='native_initial_native_evaluation':
            need(type(e['detail'].get('success')) is bool,'native initial success boolean')
            stopped=stopped or e['detail']['success']
        if not resource.startswith('native_meter/'):continue
        r=resource.removeprefix('native_meter/');need(r in CAPS,'known native meter resource')
        d=e['detail'];kind=e['kind'];keys={'episode_id','region','units'}|({'error'} if kind=='failure' else set())
        need(kind in ('intent','return','failure') and set(d)==keys and d['episode_id']==episode
             and type(d['units']) is int and d['units']==1 and d['region'] in REGIONS[r],'native meter event ABI')
        if kind=='intent':
            need(not (stopped and r=='public_step') and close_index is None,'no dispatch after stop/close')
            counts[r]['attempted']+=1;need(counts[r]['attempted']<=CAPS[r],'independent native cap')
            stack.append((e['token'],r,{k:d[k] for k in ('episode_id','region','units')}))
        else:
            need(stack and stack[-1]==(e['token'],r,{k:d[k] for k in ('episode_id','region','units')}),
                 'exact nested native lifecycle')
            stack.pop();counts[r]['completed' if kind=='return' else 'failed']+=1
            if kind=='failure':
                err=d['error'];need(type(err) is dict and set(err)=={'type','message','repr','traceback'}
                    and all(type(v) is str for v in err.values()),'preserved native exception')
            if r=='close':close_index=e['index']
    complete=transport['complete'] and not stack and all(v['failed']==0 for v in counts.values())
    if complete:need(counts['constructor']==dict(attempted=1,completed=1,failed=0)
                     and counts['close']==dict(attempted=1,completed=1,failed=0),'complete constructor and cleanup')
    return dict(schema=NAME+'.consume',counts=counts,pending=[dict(token=t,resource=r,detail=d) for t,r,d in stack],
                native_stop_observed=stopped,transport_complete=transport['complete'],ledger_complete=complete,
                full_native_strict=False,scientific_admission=False)
