"""Pair-local immutable quota gate and real native entry accounting; no torch."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
from experiments.spectral_reachability import agent_p3_child_budget_host_cuda_20261028_v1 as host
from experiments.spectral_reachability import agent_p3_paired_quota_runtime_20261028_v1_compiler as compiler

core,contract,native,paid=host.core,host.contract,host.native,host.paid
require,digest,canonical,sha=host.require,host.digest,host.canonical,host.sha
AUTH='MAIN_APPROVED_PAIRED_QUOTA_CUDA'

def zero():return dict.fromkeys(core.AXES,0)
def add(a,b):return {k:a[k]+b[k] for k in core.AXES}

def envelope():
    # Common ceiling, not an assertion of equal actual non-predict resources.
    costs=[]
    for h,n in ((1,12),(3,4)):
        b=dict(status=compiler.READY,execution_quota=dict(horizon=h,candidates=n))
        costs.append(add(contract.signal_caps(1),compiler.caps(b)))
    return {k:max(c[k] for c in costs) for k in core.AXES}

def limits():
    node=envelope()
    return dict(schema='agent_p3_paired_quota_runtime_20261028_v1.limits',
        common_node_cap=node,common_method_cap={k:24*v for k,v in node.items()},
        pair_barrier='two independent complete paid signals before either compiler',
        quota={ 'H1':12,'H3':4 },search=24,prefix=1,signal=8,
        refusal='if either allocation is below original request or unsupported, refuse BOTH after payment')

def initial_rows(p):
    return [dict(child_id=c['child_id'],mode=m,
        status='not_reached' if c['status']=='not_reached' else 'not_started',
        signal=None,signal_resources=None,decision=None,execution_quota=None,
        pair_seal_sha256=None,compiler=None,error=None)
        for c in p['wiring']['children'] for m in contract.MODES]

def branch_for(child,mode,row):
    return dict(child_id=child['child_id'],root_slot=child['root_slot'],mode=mode,
        child_sha256=digest(child),parent_context_sha256=child['context_sha256'],status=compiler.READY,
        decision=deepcopy(row['decision']),execution_quota=deepcopy(row['execution_quota']),
        pair_seal_sha256=row['pair_seal_sha256'],
        response=deepcopy(next(r for r in row['signal']['responses'] if r['horizon']==row['decision']['horizon'])))

class Gate:
    def __init__(self,p,sink=None):
        require(p['limits']==limits(),'canonical immutable cap vector and quota')
        self.cap=deepcopy(p['limits'])
        self.signal_cap=contract.signal_caps(1)
        self.program_caps={h:compiler.caps(dict(status=compiler.READY,
            execution_quota=dict(horizon=h,candidates=n))) for h,n in ((1,12),(3,4))}
        self.children=deepcopy(p['children']);self.tasks=deepcopy(p['tasks'])
        require(len(self.children)==24 and len(self.tasks)==2 and
            [(c['episode_id'],c['boundary']) for c in self.children]==[(e,b) for e in contract.EPISODES for b in contract.STEPS],
            'complete original24 child order')
        for c in self.children:contract.validate_child(c,self.tasks)
        self.sink=sink;self.cursor=0;self.stage='closed';self.turn=0
        self.nodes={};self.pending={};self.events=[];self.failed=False;self.failure_ticket=None
        self.actual={m:dict(attempted=zero(),completed=zero()) for m in contract.MODES}
        self.seals=[]

    def emit(self,kind,**fields):
        e=dict(kind=kind,index=len(self.events),previous_sha256=self.events[-1]['event_sha256'] if self.events else None,**deepcopy(fields))
        e['event_sha256']=digest(e);self.events.append(e)
        if self.sink:
            try:self.sink(dict(kind='budget_gate',payload=deepcopy(e)))
            except BaseException as exc:
                self.failed=True
                self.failure_ticket=dict(journal_event=e['index'],error=str(exc))
                raise

    def key(self,mode):return self.children[self.cursor]['child_id']+':'+mode

    def open_pair(self):
        require(not self.failed and self.cursor<24 and self.stage=='closed','ordered fail-stop no pair reentry')
        c=self.children[self.cursor];self.turn=0
        self.stage='missing' if c['status']=='not_reached' else 'signal'
        for m in contract.MODES:
            self.nodes[self.key(m)]=dict(mode=m,phase='signal',attempted=zero(),completed=zero(),
                phase_attempted=zero(),phase_completed=zero(),decision=None,signals=None,quota=None,status=self.stage)
        self.emit('pair_opened',pair=self.cursor,child_sha256=digest(c))

    def begin(self,ticket,resource,shape=None,directions=None):
        require(not self.failed,'permanent fail-stop after failed/denied entry')
        try:return self._begin(ticket,resource,shape,directions)
        except BaseException as exc:
            if not self.failed:
                self.failed=True
                self.failure_ticket=dict(ticket=ticket,resource=resource,shape=shape,directions=directions,error=str(exc))
                self.emit('blocked_before_dispatch',failure_ticket=self.failure_ticket)
            raise

    def _begin(self,ticket,resource,shape=None,directions=None):
        require(not self.failed and self.stage in ('signal','compile') and ticket==self.key(contract.MODES[self.turn]),
            'same ordered method/stage; both signals BEFORE any compile; fail-stop')
        n=self.nodes[ticket];change=core.delta(resource,shape,directions)
        expected_shapes={'predict':[1,3,192],'action_encoder':[1,3,10],
            'predictor':[1,3,192],'pred_proj':[3,192]}
        require(resource not in expected_shapes or shape==expected_shapes[resource],'exact frozen native input shape')
        cap=self.signal_cap if self.stage=='signal' else self.program_caps[n['quota']['horizon']]
        require(all(n['phase_attempted'][k]+change[k]<=cap[k] and n['attempted'][k]+change[k]<=self.cap['common_node_cap'][k] and
            self.actual[n['mode']]['attempted'][k]+change[k]<=self.cap['common_method_cap'][k] for k in core.AXES),
            'predispatch exact stage/node/method cap')
        ident=len([e for e in self.events if e['kind']=='attempted'])
        entry=dict(operation_id=ident,ticket=ticket,phase=self.stage,resource=resource,shape=shape,directions=directions,delta=change)
        self.pending[ident]=entry
        for label in ('attempted','phase_attempted'):n[label]=add(n[label],change)
        self.actual[n['mode']]['attempted']=add(self.actual[n['mode']]['attempted'],change)
        self.emit('attempted',entry=entry);return ident

    def returned(self,ident):
        require(not self.failed and ident in self.pending,'actual unreturned entry only')
        e=self.pending.pop(ident);n=self.nodes[e['ticket']]
        for label in ('completed','phase_completed'):n[label]=add(n[label],e['delta'])
        self.actual[n['mode']]['completed']=add(self.actual[n['mode']]['completed'],e['delta'])
        self.emit('returned',operation_id=ident)

    def failed_call(self,ident,error):
        require(not self.failed and ident in self.pending,'retain actual failed ticket and cost')
        self.failed=True;self.failure_ticket=dict(operation_id=ident,error=str(error),entry=deepcopy(self.pending[ident]))
        self.emit('failed',operation_id=ident,error=str(error))

    def signal_done(self,ticket,decision,signals,signal_sha256,resource_sha256):
        require(not self.failed and self.stage=='signal' and ticket==self.key(contract.MODES[self.turn]) and not self.pending,
            'exact complete paid stage')
        n=self.nodes[ticket];c=self.children[self.cursor];task=self.tasks[c['root_slot']]
        require(n['phase_attempted']==n['phase_completed']==contract.signal_caps(1),'full independent8/2J/384VJP signal')
        require(decision==core.choose(task,dict(context=c['context'],depth=0),core.FrozenSignals(**signals),n['mode']),
            'unchanged original chooser')
        core.hash_value(signal_sha256);core.hash_value(resource_sha256)
        n.update(decision=deepcopy(decision),signals=deepcopy(signals),status='paid_signal',
            signal_sha256=signal_sha256,resource_sha256=resource_sha256)
        self.emit('signal_sealed',ticket=ticket,decision=decision,signals=signals,
            signal_sha256=signal_sha256,resource_sha256=resource_sha256)
        self.turn+=1
        if self.turn==2:self.stage='await_pair_seal'

    def pair_seal(self):
        require(not self.failed and self.stage=='await_pair_seal' and not self.pending,'both independently paid before pairseal')
        rows=[self.nodes[self.key(m)] for m in contract.MODES]
        qs=[compiler.quota(n['decision']) for n in rows]
        allowed=all(q['allowed'] for q in qs)
        seal=dict(pair=self.cursor,child_sha256=digest(self.children[self.cursor]),
            paid_signals_sha256=[digest(dict(decision=n['decision'],signals=n['signals'],
                signal_sha256=n['signal_sha256'],resource_sha256=n['resource_sha256'])) for n in rows],
            execution_quotas=qs,dispatch_allowed=allowed,
            reason=None if allowed else 'pair_refused_after_independent_paid_signals')
        self.seals.append(deepcopy(seal));self.turn=0;self.stage='compile' if allowed else 'terminal'
        for n,q in zip(rows,qs,strict=True):
            n.update(quota=q,phase='compile',phase_attempted=zero(),phase_completed=zero(),
                status='compile_ready' if allowed else 'paired_paid_refusal')
        self.emit('pair_sealed',seal=seal);return seal

    def finish(self,ticket):
        require(not self.failed and self.stage=='compile' and ticket==self.key(contract.MODES[self.turn]) and not self.pending,
            'one ordered full compiler return')
        n=self.nodes[ticket];cap=compiler.caps(dict(status=compiler.READY,execution_quota=n['quota']))
        require(n['phase_attempted']==n['phase_completed']==cap and n['attempted']['predict_API']==33 and
            n['attempted']['predict_samples']==33,'actual complete 12/4 compiler, not gate-only quota')
        n['status']='compiled';self.emit('finished',ticket=ticket);self.turn+=1
        if self.turn==2:self.stage='terminal'

    def close_pair(self):
        require(not self.failed and self.stage in ('terminal','missing') and not self.pending,'both terminal or observed missing')
        if self.stage=='terminal':
            a,b=(self.nodes[self.key(m)] for m in contract.MODES)
            require(all(a['attempted'][k]==b['attempted'][k] and a['completed'][k]==b['completed'][k]
                for k in ('predict_API','predict_samples')),'paired actual predict API and samples')
        self.emit('pair_closed',pair=self.cursor);self.cursor+=1;self.stage='closed';self.turn=0

    def receipt(self):
        return deepcopy(dict(limits=limits(),cap_vector_sha256=digest(limits()),cursor=self.cursor,stage=self.stage,
            turn=self.turn,nodes=self.nodes,pending=self.pending,events=self.events,failed=self.failed,
            failure_ticket=self.failure_ticket,actual=self.actual,seals=self.seals))

class Session:
    def __init__(self,p,sink):
        self.p=p;self.sink=sink;self.active_slot=0;self.stopped=False
        self.gate=Gate(p['wiring'],lambda e:self.emit(e))
        self.rows=initial_rows(p)

    def emit(self,e):self.sink(dict(kind='work',slot=self.active_slot,payload=e))

    def next_pair(self,factories,*,permit_ram=False,authorization=None):
        require(not self.stopped and ((permit_ram is True and factories.kind=='RAM_FIXTURE') or
            (permit_ram is False and type(factories) is native.NativeFactories and authorization==AUTH)),
            'MAIN native factory or explicit RAM; no retry after failure')
        i=self.gate.cursor;require(i<24,'complete24 no reentry');self.active_slot=2*i
        child=self.p['wiring']['children'][i];task=self.p['wiring']['tasks'][child['root_slot']]
        try:
            self.gate.open_pair()
            if child['status']!='not_reached':
                for j,mode in enumerate(contract.MODES):
                    self.active_slot=2*i+j;row=self.rows[self.active_slot];row['status']='signal_attempted'
                    sink=native.GateSink(self.gate,self.gate.key(mode),self.emit)
                    ledger=core.Ledger(contract.signal_caps(1),sink=sink.event)
                    try:
                        backend=factories.signal(task,child,sink);require(backend.kind==factories.kind,'literal backend')
                        node=dict(context=child['context'],depth=0)
                        ledger.scope=dict(child_id=child['child_id'],group=mode,phase='child_signal',parent_context_sha256=child['context_sha256'])
                        signals,receipt=core.local_signal(task,node,backend,ledger,core.policy(task,node))
                        require(ledger.attempted==ledger.completed==contract.signal_caps(),'full original signal payment')
                        require(all(r['producer_sha256']==sha(paid.__file__) for r in receipt['responses']),'child-local actual signal source')
                        ledger.scope=dict(child_id=child['child_id'],group=mode,mode=mode,phase='schedule',parent_context_sha256=child['context_sha256'])
                        decision=ledger.call('schedule',lambda:core.choose(task,node,signals,mode))
                        row.update(signal=receipt,decision=decision,status='paid_signal',signal_resources=ledger.receipt())
                        self.sink(dict(kind='signal_return',slot=self.active_slot,row=deepcopy(row)))
                        self.gate.signal_done(self.gate.key(mode),decision,asdict(signals),digest(receipt),digest(row['signal_resources']))
                    finally:
                        if row['signal_resources'] is None:row['signal_resources']=ledger.receipt()
                seal=self.gate.pair_seal()
                for j,mode in enumerate(contract.MODES):
                    self.active_slot=2*i+j;row=self.rows[self.active_slot]
                    row.update(execution_quota=deepcopy(seal['execution_quotas'][j]),pair_seal_sha256=digest(seal))
                    if not seal['dispatch_allowed']:
                        row['status']='paired_paid_refusal';continue
                    branch=branch_for(child,mode,row);row['compiler']=compiler.initial_rows([branch])[0]
                    sink=native.GateSink(self.gate,self.gate.key(mode),self.emit)
                    def factory(c,b,s):
                        require(c==child and b==branch and s is sink,'exact compiler child/branch/sink')
                        return factories.program(task,child,s)
                    compiler.execute_branch(task,child,branch,row['compiler'],factory,sink,permit_ram=permit_ram,
                        authorization='MAIN_APPROVED_HOST_CHILD_PROGRAM_CUDA' if not permit_ram else None)
                    row['status']='compiled'
                    self.sink(dict(kind='compile_return',slot=self.active_slot,row=deepcopy(row)))
                    self.gate.finish(self.gate.key(mode))
            self.gate.close_pair()
            self.sink(dict(kind='pair_return',pair=i,rows=deepcopy(self.rows[2*i:2*i+2])))
        except BaseException as exc:
            self.stopped=True
            self.rows[self.active_slot].update(status='engineering_failed',error=type(exc).__name__+': '+str(exc))
            raise

    def receipt(self):
        return dict(schema='agent_p3_paired_quota_runtime_20261028_v1.session',rows=deepcopy(self.rows),
            gate=self.gate.receipt(),stopped=self.stopped,fixed_denominator=48,child_denominator=24)
