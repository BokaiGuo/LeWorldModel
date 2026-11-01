"""C3 exact structural routing and static seal. RAM only, no production entry."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import struct
from collections.abc import MutableMapping

from experiments.spectral_reachability import agent_p3_c3_common_cap_compact_20261029_v1 as codec

NAME = 'agent_p3_c3_compact_routing_20261029_v1'
CODEC_SHA = '98b0039029f602537a873480a9d343efacd499debd9f3550b113e352fa60372e'
need, canonical, semantic = codec.need, codec.canonical, codec.semantic
LIMITS = dict(parent=512*1024, candidate=64*1024, slot=1024*1024)
# A schema is a finite tree. Reference targets are explicitly catalogued,
# never inferred from an arbitrary same-named string or a digest alone.
U = ('uint',)
F = ('float',)
B = ('bool',)
H = ('str',64)
S = ('str',256)
NONE = ('null',)
def ref(kind): return ('ref',kind)
def arr(*shape): return ('array',shape)
def seq(item, maximum): return ('list',item,maximum)
def opt(item): return ('optional',item)
def rec(**fields): return ('record',fields)
def variant(*items): return ('variant',items)
AXES = rec(**{a:U for a in codec.AXES})
FLAGS = ('P1_admission','P3_admission','equal_FLOPs','freshness','full_C3_demonstrated',
         'original_R_equivalence','physical_release','policy_tree_executed')
TREE_FLAGS = (*FLAGS,'scientific_admission','full_policy_executed','same_actual_resources','same_FLOPs')
BIND_NAMES = ('action_domain_sha256','blind_sha256','context_sha256','goal_sha256','model_sha256',
              'normalizer_sha256','projection_sha256','root_sha256','task_contract_sha256','task_id','task_sha256')
BIND = rec(**{k:H for k in BIND_NAMES})
RESPONSE = rec(binding_sha256=H,parent_context_sha256=H,horizon=U,
               nominal=ref('native_literal'),jacobian=ref('native_literal'),producer_sha256=H)
THRESHOLDS = rec(chi=F,gain=F)
DECISION = rec(program_type=S,extension_type=S,horizon=U,candidates=U,continue_expanding=B,
    program_depth_limit=U,inherited_engineering_thresholds=THRESHOLDS,
    no_optimality_claim=B,decision_source=S)
SIGNAL_REQUEST = rec(binding=BIND,binding_sha256=H,parent_context=ref('context'),
                     parent_context_sha256=H,operation=S,horizon=U)
EDGE = rec(kind=S,input_context_sha256=H,output_context_sha256=H,actions_sha256=H,
           producer_sha256=H,actions=variant(arr(5,2),arr(15,2)),path=seq(ref('context'),3))
NODE_BASE = dict(context=ref('context'),depth=U,indices=seq(U,3),
    actions=variant(arr(5,2),arr(10,2),arr(15,2),arr(20,2),arr(25,2),arr(30,2),arr(35,2),arr(40,2),arr(45,2),seq(F,0)),
    path=seq(ref('context'),9),edges=seq(EDGE,3))
NODE = variant(rec(**NODE_BASE),rec(**NODE_BASE,cost=F))
REQUEST = rec(binding=BIND,binding_sha256=H,parent_context=ref('context'),parent_context_sha256=H,
    operation=S,**DECISION[1],candidate_index=U,prefix_edges=seq(EDGE,3),
    prefix_actions=NODE_BASE['actions'],signal_sha256=H,local_response=seq(RESPONSE,2))
SOLVE = rec(params=variant(arr(10),arr(30)),singular_values=variant(arr(10),arr(30)),
    ball_multiplier=F,linear_terminal_cost=F,norm=F,own_reach_value=F,radius=F,ridge_penalty=F)
POINTER = rec(call_id=U,index=U,sha256=H)
RROW_BASE = dict(index=variant(U,('enum',('zero',))),status=S,actions=variant(arr(5,2),arr(15,2)),
    path=seq(ref('context'),3),postcontext=ref('context'),immediate=F,signal=RESPONSE,
    nominal_path=seq(ref('context'),1),nominal_pointer=POINTER,solve=SOLVE,reach=F,
    suffix_actions=arr(5,2),suffix_path=seq(ref('context'),1),bounded_suffix_cost=F)
_ROW_INDEX = variant(U,('enum',('zero',)))
_ROW_ATTEMPT = dict(index=_ROW_INDEX,status=S,actions=variant(arr(5,2),arr(15,2)))
_ROW_POST = dict(**_ROW_ATTEMPT,path=seq(ref('context'),3),postcontext=ref('context'),immediate=F)
RROW = variant(rec(**RROW_BASE,eligible=opt(B)),rec(**RROW_BASE),
    rec(**_ROW_POST,eligible=opt(B)),rec(**_ROW_POST),
    rec(**_ROW_ATTEMPT,eligible=opt(B)),rec(**_ROW_ATTEMPT),
    rec(index=variant(U,('enum',('zero',))),status=S,eligible=opt(B)),
    rec(index=variant(U,('enum',('zero',))),status=S))
PROPOSAL = variant(RROW,rec(index=U,actions=variant(arr(5,2),arr(15,2)),
                          path=seq(ref('context'),3),eligible=B,immediate=F))
JOB = rec(slot=U,status=S,request=opt(REQUEST),payload=opt(PROPOSAL),node=opt(NODE),error=opt(S))
SPECTRUM = rec(horizon=U,rank=U,singular_values=variant(arr(10),arr(30)),
               unreachable_fraction_same_reference_residual=F)
SIGNAL = rec(schema=S,responses=seq(RESPONSE,2),spectra=seq(SPECTRUM,2),reference=S,
    residual=arr(192),residual_energy=F,signed_horizon_coverage_gain=F,
    scheduling_signal=rec(source_seal=H,unreachable_fraction=F,horizon_gain=F),no_control_or_safety_guarantee=B)
MAPPING = rec(schema=S,root_task_sha256=H,root_binding=BIND,parent_task=ref('task'),
    outer_request=SIGNAL_REQUEST,native_request=SIGNAL_REQUEST,native_response=RESPONSE,
    outward_response=RESPONSE,mapping_sha256=H)
QUEUE = rec(parent=NODE,decision=DECISION,requests=seq(ref('request'),8),status=S,
            minimum=opt(AXES),verification_reserve=opt(AXES))
ENTRY = rec(index=U,scope=ref('scope'),resource=S,shape=opt(seq(U,3)),directions=opt(U),
    delta=AXES,status=S,started_ns=U,ended_ns=opt(U),error=opt(S))
EVIDENCE = rec(kind=S,artifact_path=S,artifact_sha256=H,call_id=U,native_call=ref('transcript'),
    origin_index=U,nominal_index=U,jacobian_index=U,signal_compile_entry=ENTRY)
SOURCE = rec(kind=S,native_verified=B,resources=opt(AXES),signal_compile_entry=ENTRY,
    artifact_path=S,artifact_sha256=H,call_id=U,origin_index=U,nominal_index=U,jacobian_index=U,
    native_call_sha256=H,source_artifact_bytes_read=B,scope=S)
AUTHORITY = rec(source_sha256=H,bank_sha256=H,zero_sha256=H,root_task_sha256=H,parent_context_sha256=H,
    goal_sha256=H,Q_sha256=H,normalizer_sha256=H,action_domain_sha256=H,
    bank_authenticated=B,authentication_sha256=opt(H))
MANIFEST = rec(schema=S,root_task_sha256=H,parent_task_sha256=H,parent_context_sha256=H,
    signal_mapping_sha256=H,native_response_sha256=H,evidence_sha256=H,protocol_sha256=H,
    bank_sha256=H,zero_sha256=H,params_sha256=H,bank_ledger_events_sha256=H,
    bank_resource_vector=AXES,numeric_environment=ref('static'),sources=ref('static'),
    generator_path=S,generator_sha256=H,source_native_verified=B,bank_artifact_authenticated=B)
BANK_EVENT = rec(index=U,bank=S,payload=variant(
    rec(kind=('enum',('bank_attempt',)),mapping_sha256=H,source=SOURCE),
    rec(kind=('enum',('bank_return',)),authority_manifest=MANIFEST)))
BANK = rec(schema=S,status=S,rule=ref('static'),horizon=U,root=ref('task'),parent_task=ref('task'),
    prefix_actions=NODE_BASE['actions'],prefix_path=seq(ref('context'),9),mapping=MAPPING,evidence=EVIDENCE,
    source=SOURCE,resources_before=AXES,completed_before=AXES,resources_expected=AXES,
    verification_reserve=AXES,params=variant(arr(8,10),arr(8,30)),
    actions=variant(arr(8,5,2),arr(8,15,2)),zero=variant(arr(5,2),arr(15,2)),authority=AUTHORITY,
    events=seq(BANK_EVENT,2),projected_J_float64=variant(arr(192,10),arr(192,30)),
    left_vectors=variant(arr(192,10),arr(192,30)),singular_values=variant(arr(10),arr(30)),
    right_vectors_original=variant(arr(10,10),arr(30,30)),right_vectors=variant(arr(10,10),arr(30,30)),
    kinds=seq(S,8),directions=seq(rec(index=U,radius=F,mode=U,sign=('signed',)),8),
    authority_manifest=MANIFEST,ledger_events=seq(ENTRY,2),resources=AXES,completed=AXES,
    **{k:B for k in FLAGS})
PROGRESS = rec(solve=SOLVE,divisor=F,fractions=variant(arr(4),arr(8)),
    actions=variant(arr(4,5,2),arr(4,15,2),arr(8,5,2),arr(8,15,2)),response_sha256=H,generator_sha256=H)
_GROUP_BASE = dict(bank=ref('bank'),control=ref('control'),rows=seq(ref('proposal'),8),eligible_indices=seq(U,8),status=S)
GROUP = variant(rec(**_GROUP_BASE,diagnostic_reach_winner=opt(U),resources=AXES),rec(**_GROUP_BASE))
CALL = rec(call_id=U,outer_index=U,transcript=ref('transcript'))
RECEIPT = rec(caps=AXES,attempted=AXES,completed=AXES,events=ref('ledger'),wall_ns=U,
    backward_engine_calls=NONE,cuda_ns=NONE,FLOPs=NONE,inclusive_operation_wall_ns=ref('inclusive'),
    wall_semantics=S,physics_steps=U,unknowns_are_not_zero=B)
TREE = rec(schema=S,mode=S,K=U,backend_kind=S,root_task_sha256=H,jobs=seq(ref('job'),144),
    signals=seq(ref('signal'),17),queues=seq(ref('queue'),17),R_groups=seq(ref('group'),17),
    replays=seq(rec(mode=S,status=S,path=opt(seq(ref('context'),9))),2),status=S,
    errors=seq(rec(type=S,message=S),1),model_plan=opt(ref('node')),diagnostic_best=opt(ref('node')),
    fixed_denominator=rec(arms=U,requests=U),resource_receipt=RECEIPT,
    signal_mappings=seq(ref('mapping'),34),native_calls=seq(CALL,639),realized_widths=seq(U,17),
    **{k:B for k in TREE_FLAGS})
SCHEMAS = dict(binding=BIND,response=RESPONSE,request=REQUEST,node=NODE,proposal=PROPOSAL,job=JOB,
    signal=SIGNAL,mapping=MAPPING,queue=QUEUE,bank=BANK,progress=PROGRESS,control=RROW,group=GROUP,tree=TREE)


def fields_of(spec):
    kind = spec[0]
    if kind == 'record': return set(spec[1]).union(*(fields_of(s) for s in spec[1].values()))
    if kind in ('list','optional'): return fields_of(spec[1])
    if kind == 'variant': return set().union(*(fields_of(s) for s in spec[1]))
    return set()


def bound(spec):
    """No value similarity/average/compression: charge every structural node."""
    k = spec[0]
    if k == 'ref': return 0
    if k in ('uint','signed','float'): return 57
    if k in ('bool','null'): return 49
    if k == 'str': return 49+spec[1]
    if k == 'enum': return 49+max(len(s.encode()) for s in spec[1])
    if k == 'array': return 128+8*math.prod(spec[1])
    if k == 'record': return 51+34*len(spec[1])+sum(bound(s) for s in spec[1].values())
    if k == 'list': return 53+32*spec[2]+spec[2]*bound(spec[1])
    if k == 'optional': return max(49,bound(spec[1]))
    if k == 'variant': return max(map(bound,spec[1]))
    raise ValueError('unclassified bound '+k)


def validate(value, spec, catalog=None):
    k = spec[0]
    if k == 'ref':
        need(catalog is not None and catalog.contains(spec[1],value), 'explicit '+spec[1]+' reference')
    elif k == 'record':
        codec.exact(value,spec[1]); [validate(value[n],s,catalog) for n,s in spec[1].items()]
    elif k == 'list':
        need(type(value) is list and len(value) <= spec[2], 'bounded list')
        for v in value: validate(v,spec[1],catalog)
    elif k == 'optional':
        if value is not None: validate(value,spec[1],catalog)
    elif k == 'variant':
        for s in spec[1]:
            try: validate(value,s,catalog); return
            except ValueError: pass
        raise ValueError('no exact structural variant')
    elif k == 'array':
        dense = codec.rectangle(value)
        need(dense is not None and dense[0] == spec[1], 'exact literal float array shape/type')
    elif k == 'uint': codec.uint(value)
    elif k == 'signed': need(type(value) is int and -2**63 <= value < 2**63,'literal signed integer')
    elif k == 'float': need(type(value) is float and math.isfinite(value),'literal finite float')
    elif k == 'bool': need(type(value) is bool,'literal bool')
    elif k == 'null': need(value is None,'literal null')
    elif k == 'str': need(type(value) is str and len(value.encode()) <= spec[1],'bounded UTF8 string')
    elif k == 'enum': need(type(value) is str and value in spec[1],'literal enum')
    else: raise ValueError('unknown schema')


class Catalog:
    def __init__(self, store): self.store,self.values = store,{}
    def bind(self, kind, value, key):
        need(canonical(self.store.get(key)) == canonical(value),'reference expanded exact bytes')
        identity = (kind,semantic(value))
        if identity in self.values: need(self.values[identity] == key,'one exact source reference')
        self.values[identity] = key
        return key
    def contains(self, kind, value): return (kind,semantic(value)) in self.values
    def get(self, kind, value):
        need(self.contains(kind,value),'missing explicit '+kind+' catalog entry')
        key = self.values[kind,semantic(value)]
        need(canonical(self.store.get(key)) == canonical(value),'same semantic digest must have exact bytes')
        return key


class Routing:
    """Structural encoder. No fallback generic encode of unclassified fields.

    Native transcript/ledger/context targets must already be real codec objects
    or views; this layer never fabricates their source or a production receipt.
    """
    def __init__(self, store, catalog):
        need(catalog.store is store,'one object store'); self.store,self.catalog=store,catalog
        self.owners = {}; self.bytes=Counter(); self.trace=[]

    def _intern(self, raw, owner):
        key = codec.h(codec.DOMAIN+raw)
        if key not in self.store.objects:
            bucket = owner[0]; need(bucket in LIMITS,'explicit parent/candidate/slot owner')
            charge=len(raw)+codec.INDEX.size
            need(self.bytes[owner]+charge <= LIMITS[bucket],'structural bucket overflow '+str(owner))
            self.store.intern(raw); self.owners[key]=owner; self.bytes[owner]+=charge
        return key

    def encode(self, value, spec, owner, path='$'):
        validate(value,spec,self.catalog); k=spec[0]
        if k == 'ref':
            key=self.catalog.get(spec[1],value)
            self.trace.append(dict(path=path,route=spec[1],object_sha256=key.hex(),owner=None))
            return key
        if k == 'optional': return self.encode(value,NONE if value is None else spec[1],owner,path)
        if k == 'variant':
            for item in spec[1]:
                try: validate(value,item,self.catalog)
                except ValueError: continue
                return self.encode(value,item,owner,path)
            raise ValueError('no exact variant')
        if k == 'record':
            # Iterate original dict order: no reordering its stored spine.
            raw=b'M'+struct.pack('<H',len(value))
            for n,v in value.items():
                need(n in self.store.codes,'sealed structural field code')
                raw+=struct.pack('<H',self.store.codes[n])+self.encode(v,spec[1][n],owner,path+'.'+n)
        elif k == 'list':
            raw=b'L'+struct.pack('<I',len(value))+b''.join(self.encode(v,spec[1],owner,f'{path}[{i}]') for i,v in enumerate(value))
        elif k == 'array':
            dims,data=codec.rectangle(value)
            raw=b'V'+bytes([len(dims)])+struct.pack('<'+'I'*len(dims),*dims)+data
        elif k == 'null': raw=b'N'
        elif k == 'bool': raw=b'T' if value else b'F'
        elif k == 'uint': raw=b'U'+codec.uint(value)
        elif k == 'signed': raw=b'I'+struct.pack('<q',value) if value<0 else b'U'+codec.uint(value)
        elif k == 'float': raw=b'D'+struct.pack('<d',value)
        else: raw=b'S'+value.encode()
        key=self._intern(raw,owner)
        self.trace.append(dict(path=path,route=k,object_sha256=key.hex(),owner=list(owner)))
        return key

    def record(self, kind, value, owner):
        need(kind in SCHEMAS,'known structural root kind')
        key=self.encode(value,SCHEMAS[kind],owner,'$.'+kind)
        need(canonical(self.store.get(key)) == canonical(value),'whole routed record byte-exact roundtrip')
        self.catalog.bind(kind,value,key)
        return key

    def census(self):
        need(all(len(self.store.objects[k])+48 <= self.bytes[o] for k,o in self.owners.items()),'charged object exists')
        recompute=Counter()
        for k,o in self.owners.items(): recompute[o]+=len(self.store.objects[k])+48
        need(recompute==self.bytes,'all structural bytes charged exactly once')
        return dict(schema=NAME+'.routing_census',buckets=[dict(owner=list(o),bytes=b,cap=LIMITS[o[0]]) for o,b in sorted(self.bytes.items())],
            unique_structural_objects=len(self.owners),unique_structural_bytes=sum(self.bytes.values()),
            production_ready=False,full_C3_demonstrated=False)

    def tree(self, slot, value, *, progress_banks=()):
        """Route EVERY saved tree field, with finite real parent/child owners.

        Only the explicit producer result is accepted. No observed cost is used
        to choose a cohort, a bank, or a different schema. Native targets must
        already have been installed from the journal by the caller.
        """
        codec.uint(slot); need(slot<12,'twelve fixed slots')
        need(type(value) is dict and len(value['jobs'])==144,'all144 child identities')
        need([j['slot'] for j in value['jobs']]==list(range(144)),'ordered atomic request identities')
        queues=value['queues']; mappings=value['signal_mappings']; signals=value['signals']
        progress_banks=list(progress_banks)
        indices=[i for i,_ in progress_banks]
        need(all(type(i) is int and 0<=i<len(queues) and queues[i]['status']=='queued' and
             queues[i]['decision']['extension_type']=='progress' for i in indices),
             'progress bank original exclusive queue branch')
        need(len(indices)==len(set(indices)),'one progress bank per queue ordinal')
        need(len(queues)<=len(signals)<=len(queues)+1 and len(signals)<=17 and len(mappings)<=34,
             'finite scheduled parent/partial signal counts')
        parents={}
        def owner(ordinal):
            need(type(ordinal) is int and 0<=ordinal<17,'at most17 parent queue ordinals')
            parents[ordinal]=('parent',slot,ordinal)
            return parents[ordinal]
        # A response array is a reference to actual saved native output, not a
        # newly rounded matrix or an arbitrary caller-created same-shape blob.
        for i,mapping in enumerate(mappings):
            ordinal=i//2
            if ordinal<len(queues):
                need(canonical(mapping['parent_task']['root_context'])==canonical(queues[ordinal]['parent']['context']),
                     'mapping pair belongs to actual ordinal parent')
            self.record('mapping',mapping,owner(ordinal))
        for job in value['jobs']:
            own=('candidate',slot,job['slot'])
            if job['request'] is not None: self.record('request',job['request'],own)
            if job['payload'] is not None:
                self.record('proposal',job['payload'],own)
                # R_item_return is emitted before eligibility is filled in.
                if 'eligible' in job['payload']:
                    before=dict(job['payload'],eligible=None)
                    if 'suffix_path' in before: self.record('proposal',before,own)
            if job['node'] is not None: self.record('node',job['node'],own)
            self.record('job',job,own)
        for i,signal in enumerate(signals): self.record('signal',signal,owner(i))
        unrequested=iter(j['slot'] for j in value['jobs'] if j['status']=='not_requested' and j['request'] is None)
        for i,queue in enumerate(queues):
            for request in queue['requests']:
                if not self.catalog.contains('request',request):
                    # Original queue admission creates width4/8 requests BEFORE
                    # cap/refusal; failed admission never assigns them to jobs.
                    # Retain them verbatim, charging unused144 budget identities
                    # without relabelling the actual not_requested job values.
                    need(queue['status']!='queued','queued request must match an assigned original job')
                    ordinal=next(unrequested,None)
                    need(ordinal is not None,'refused requests fit original144 identity allowance')
                    self.record('request',request,('candidate',slot,ordinal))
            self.record('queue',queue,owner(i))
        Rqueues=[i for i,q in enumerate(queues) if q['status']=='queued' and q['decision']['extension_type']=='reconfiguration']
        need(len(value['R_groups'])<=len(Rqueues),'R groups from actual queued R branches only')
        bank_kinds={}
        def bank_owner(i,kind):
            need(i<len(queues) and queues[i]['status']=='queued' and
                 queues[i]['decision']['extension_type']==kind,'bank matches original exclusive queue branch')
            need(i not in bank_kinds,'one bank per queue ordinal; no R/progress union hidden by max')
            bank_kinds[i]=kind; return owner(i)
        for i,group in zip(Rqueues,value['R_groups']):
            own=bank_owner(i,'reconfiguration')
            need(canonical(group['bank']['parent_task']['root_context'])==canonical(queues[i]['parent']['context']),
                 'R bank exact queue parent')
            self.record('bank',group['bank'],own)
            self.record('control',group['control'],own)
            for row in group['rows']:
                if not self.catalog.contains('proposal',row):
                    # A failed R group is retained before any jobs receive
                    # payloads. Its eight rows still belong to the *already
                    # assigned* exact queue requests; never drop null rows.
                    need(value['errors'] and group['status']=='partial_runtime_failure',
                         'complete R rows must be present in original jobs')
                    matches=[j for j in value['jobs'] if j['request'] is not None and
                        j['request']['candidate_index']==row['index'] and
                        any(canonical(j['request'])==canonical(q) for q in queues[i]['requests'])]
                    need(len(matches)==1,'partial R row exact assigned request identity')
                    self.record('proposal',row,('candidate',slot,matches[0]['slot']))
            self.record('group',group,own)
        for i,bank in progress_banks:
            self.record('progress',bank,bank_owner(i,'progress'))
        root=self.record('tree',value,('slot',slot,0))
        self.store.roots[f'slot_{slot:02}']=root
        self.census()
        return dict(object_sha256=root.hex(),expanded_canonical_sha256=semantic(value),
            parent_count=len(parents),candidate_identity_count=144,slot=slot,
            bank_kinds_by_queue_ordinal=bank_kinds)


def seal_static(tasks, sources, extra, *, read_sources=True):
    """Explicit-input source seal; never discovers tasks, weights, or results.

    Reading this helper is not production prepare. Caller supplies exact known
    metadata and SHA map; only those source paths are read, with length caps.
    """
    need(type(tasks) is list and len(tasks)==2,'two fixed known tasks')
    for task in tasks:
        codec.exact(task,('binding','root_context','goal','projection','action_mean','action_std'))
        validate(task['binding'],BIND)
        validate(task['goal'],arr(192)); validate(task['projection'],arr(192,192))
        validate(task['action_mean'],arr(2)); validate(task['action_std'],arr(2))
        need(all(x>0 for x in task['action_std']),'original positive action scales')
    need(type(sources) is dict and all(type(k) is str and type(v) is str for k,v in sources.items()),'explicit source map')
    entries=[]
    for name,expected in sorted(sources.items()):
        p=Path(name)
        need(p.is_absolute() and p.resolve()==p and not p.is_symlink() and p.is_file(),'exact regular source path')
        need(len(expected)==64 and all(c in '0123456789abcdef' for c in expected),'literal source SHA')
        n=p.stat().st_size; need(n<=codec.STATIC,'individual source bound before read')
        if read_sources: need(codec.file_sha(p)==expected,'source bytes exact SHA')
        entries.append(dict(path=str(p),sha256=expected,bytes=n))
    value=dict(schema=NAME+'.static_seal',tasks=tasks,sources=entries,extra=extra,
               source_bytes_checked=read_sources,materialized_capsules=False)
    raw=canonical(value)
    declared=extra.get('fields',[]) if type(extra) is dict else []
    need(type(declared) is list and all(type(x) is str for x in declared),'explicit extra field declarations')
    fields=sorted(set().union(*(fields_of(s) for s in SCHEMAS.values()),codec.keys_of(value),declared))
    need(len(canonical(dict(fields=fields,static=value)))<=codec.STATIC,'complete static/field seal32MiB')
    return dict(value=value,fields=fields,canonical_sha256=codec.h(raw).hex(),canonical_bytes=len(raw))


def install_static(seal):
    """Measure actual encoded table+pack overhead, not just JSON length.

    This reads no saved authority and is safe for literal RAM inputs. A seal
    made with read_sources=False cannot be promoted to a production authority.
    """
    codec.exact(seal,('value','fields','canonical_sha256','canonical_bytes'))
    raw=canonical(seal['value'])
    need(codec.h(raw).hex()==seal['canonical_sha256'] and len(raw)==seal['canonical_bytes'],'static exact bytes')
    declared=seal['value']['extra'].get('fields',[]) if type(seal['value']['extra']) is dict else []
    expected=sorted(set().union(*(fields_of(s) for s in SCHEMAS.values()),codec.keys_of(seal['value']),declared))
    need(seal['fields']==expected,'exact static field table derived before seal')
    need(codec.file_sha(Path(codec.__file__))==CODEC_SHA,'frozen codec source')
    store=codec.Codec(seal['fields'],seal['value']); catalog=Catalog(store)
    root=store.put(seal['value']); store.roots['static']=root
    # Bind all exact fixed subobjects. No type conversion, dynamic discovery,
    # or same-SHA-only aliasing is allowed on lookup.
    seen=set()
    def walk(value):
        identity=semantic(value)
        if identity in seen: return
        seen.add(identity); catalog.bind('static',value,store.put(value))
        if type(value) is dict:
            for v in value.values(): walk(v)
        elif type(value) is list:
            # Dense float arrays are one node; their elements need no static
            # reference IDs. Avoid creating unused per-float objects.
            if codec.rectangle(value) is None:
                for v in value: walk(v)
    walk(seal['value'])
    for t in seal['value']['tasks']:
        catalog.bind('task',t,store.put(t)); catalog.bind('context',t['root_context'],store.put(t['root_context']))
    measured=store.nodes_bytes+store.static_bytes
    need(measured<=codec.STATIC,'actual static typed pack+index+table seal32MiB')
    return store,catalog,dict(schema=NAME+'.static_budget',canonical_bytes=len(raw),
        object_and_index_bytes=store.nodes_bytes,field_and_metadata_bytes=store.static_bytes,
        total_static_bytes=measured,cap=codec.STATIC,source_bytes_checked=seal['value']['source_bytes_checked'],
        source_entries=seal['value']['sources'],production_ready=False)


class DiskCAS(MutableMapping):
    """Append-only compatible object mapping; no expanded capsule files.

    The index holds offsets only. Payload reads are demand-driven and verified
    on every access. Checkpoints fsync both files and directory, without an
    O(total-output) rehash at every native invocation. Final binding hashes are
    computed once by finish(). Failure never truncates/deletes existing bytes.
    """
    def __init__(self, directory, *, budget=codec.LIVE-codec.STATIC,
                 floor=200*1024**2):
        directory=Path(directory).absolute()
        need(directory.resolve()==directory and not directory.exists() and
             not directory.is_symlink(),'new exclusive diskCAS directory')
        need(type(budget) is int and 0<budget<=codec.LIVE-codec.STATIC,'bounded diskCAS allowance')
        need(type(floor) is int and floor>=200*1024**2,'disk floor200MiB unchanged')
        need(shutil.disk_usage(directory.parent).free>=budget+floor,'CAS allowance plus floor')
        directory.mkdir(); self.directory=directory; self.budget=budget; self.floor=floor
        self.pack=(directory/'objects.pack').open('x+b')
        self.index=(directory/'objects.idx').open('x+b')
        self.positions={}; self.offset=0; self.readonly=False; self.closed=False

    def __len__(self): return len(self.positions)
    def __iter__(self): return iter(self.positions)
    def __getitem__(self,key):
        offset,length=self.positions[key]
        self.pack.seek(offset); raw=self.pack.read(length)
        need(len(raw)==length and codec.h(codec.DOMAIN+raw)==key,'CAS exact content bytes')
        return raw
    def __setitem__(self,key,raw):
        need(not self.readonly and not self.closed,'writable open CAS')
        need(type(key) is bytes and len(key)==32 and type(raw) is bytes and 0<len(raw)<=codec.STATIC,
             'exact bounded CAS record')
        need(codec.h(codec.DOMAIN+raw)==key,'CAS content domain SHA')
        if key in self.positions:
            need(self[key]==raw,'immutable CAS collision'); return
        total=self.offset+len(raw)+(len(self)+1)*48
        need(total<=self.budget,'diskCAS byte allowance before append')
        need(shutil.disk_usage(self.directory).free>=self.floor+len(raw)+48,'disk floor before append')
        self.pack.seek(self.offset)
        need(self.pack.write(raw)==len(raw),'complete CAS raw append')
        self.index.seek(0,os.SEEK_END)
        need(self.index.write(codec.INDEX.pack(key,self.offset,len(raw)))==48,'complete CAS index append')
        self.positions[key]=(self.offset,len(raw)); self.offset+=len(raw)
    def __delitem__(self,key): raise ValueError('CAS immutable; no deletion or refund')
    def checkpoint(self):
        need(not self.closed,'open checkpoint')
        for f in (self.pack,self.index): f.flush(); os.fsync(f.fileno())
        fd=os.open(self.directory,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return dict(objects=len(self),pack_bytes=self.offset,index_bytes=len(self)*48,
                    boundary='pack_and_index_file_and_directory_fsync_returned')
    def finish(self):
        witness=self.checkpoint()
        return dict(schema=NAME+'.CAS',**witness,artifacts={name:dict(bytes=(self.directory/name).stat().st_size,
            sha256=codec.file_sha(self.directory/name)) for name in ('objects.pack','objects.idx')},
            materialized_capsules=False,production_ready=False)
    def close(self):
        if not self.closed:
            self.pack.close(); self.index.close(); self.closed=True
    @classmethod
    def open_saved(cls,directory,descriptor):
        directory=Path(directory).absolute()
        need(directory.resolve()==directory and not directory.is_symlink(),'exact saved CAS directory')
        codec.exact(descriptor,('schema','objects','pack_bytes','index_bytes','boundary','artifacts',
                                'materialized_capsules','production_ready'))
        need(descriptor['schema']==NAME+'.CAS' and descriptor['materialized_capsules'] is False
             and descriptor['production_ready'] is False,'own CAS descriptor, not material capsule')
        need(descriptor['boundary']=='pack_and_index_file_and_directory_fsync_returned','CAS durable boundary')
        codec.uint(descriptor['objects']); codec.uint(descriptor['pack_bytes']); codec.uint(descriptor['index_bytes'])
        need(descriptor['index_bytes']==48*descriptor['objects'] and
             descriptor['pack_bytes']+descriptor['index_bytes']<=codec.LIVE-codec.STATIC,'saved CAS byte bound')
        codec.exact(descriptor['artifacts'],('objects.pack','objects.idx'))
        need({p.name for p in directory.iterdir()}=={'objects.pack','objects.idx'},'exact isolated CAS file census')
        for name,entry in descriptor['artifacts'].items():
            codec.exact(entry,('bytes','sha256')); p=directory/name
            need(p.is_file() and not p.is_symlink() and entry['bytes']==p.stat().st_size
                 and entry['bytes']==descriptor['pack_bytes' if name=='objects.pack' else 'index_bytes']
                 and codec.file_sha(p)==entry['sha256'],'saved CAS complete file SHA/size')
        out=cls.__new__(cls); out.directory=directory; out.positions={}; out.offset=0
        out.readonly=True; out.closed=False; out.budget=codec.LIVE-codec.STATIC; out.floor=200*1024**2
        out.pack=(directory/'objects.pack').open('rb'); out.index=(directory/'objects.idx').open('rb')
        try:
            while raw:=out.index.read(48):
                need(len(raw)==48,'complete CAS index record')
                key,offset,length=codec.INDEX.unpack(raw)
                need(key not in out.positions and offset==out.offset and 0<length<=codec.STATIC
                     and offset+length<=descriptor['pack_bytes'],'unique contiguous CAS index')
                out.positions[key]=(offset,length); out.offset+=length
                out[key]  # verify every object, not just requested roots
            need(out.offset==descriptor['pack_bytes'] and len(out)==descriptor['objects'],'no unindexed CAS suffix')
            return out
        except BaseException:
            out.close(); raise


def attach_disk(store,directory,*,budget=codec.LIVE-codec.STATIC):
    """Move a bounded sealed RAM prefix to an exclusive CAS, no source patch.

    DiskCAS implements the frozen codec's explicit mapping boundary. We do not
    patch Codec globals, foreign readers, or old module os/path primitives.
    """
    disk=DiskCAS(directory,budget=budget)
    try:
        for key,raw in store.objects.items(): disk[key]=raw
        need(disk.offset+48*len(disk)==store.nodes_bytes,'same actual codec object/index accounting')
        disk.checkpoint()
    except BaseException:
        disk.close(); raise
    store.objects=disk
    return disk


def proof():
    # Explicit roots, not a generic per-event envelope. Each child reference
    # has zero *new* allocation only after Catalog.bind verifies its bytes.
    # Includes a second pre-eligibility R row emitted by R_item_return, plus
    # explicit catalog roots for request/node/proposal (shared objects can only
    # reduce this sum). Never assume the before/after dicts have same ID.
    candidate=bound(JOB)+bound(RROW)
    parent=(2*bound(MAPPING)+bound(SIGNAL)+bound(QUEUE)+max(bound(BANK),bound(PROGRESS))
            +bound(RROW)+bound(GROUP))
    slot=bound(TREE)
    b=dict(parent=parent,candidate=candidate,slot=slot)
    return dict(schema=NAME+'.schema_bounds',per_record={k:bound(s) for k,s in SCHEMAS.items()},
        per_owner_bytes_upper=b,limits=LIMITS,all_structural_buckets_fit=all(b[k]<=LIMITS[k] for k in b),
        external_reference_requirement='every ref must be exact catalogued object/view; no arbitrary ID',
        source_seal_required=True,production_control_flow_connected=False,production_ready=False,
        full_C3_demonstrated=False)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('command',choices=['proof'])
    ap.parse_args(argv); print(canonical(proof()).decode())


if __name__=='__main__': main()
