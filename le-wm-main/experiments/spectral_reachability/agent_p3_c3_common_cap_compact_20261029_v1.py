"""Lossless C3 typed codec / RAM evidence ONLY; no production runner or NN."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct

NAME = 'agent_p3_c3_common_cap_compact_20261029_v1'
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'innerdocs'/(NAME+'.md')
PROTOCOL_SHA = '30dda31424ce2e244314cfeb2c5c573d886ca05ff95bc3f656c29a5fe5d2c36e'
LIVE = 4*1024**3-512*1024**2
STATIC = 32*1024**2
DOMAIN = (NAME+'\0object\0').encode()
CHAIN = (NAME+'\0journal\0').encode()
INDEX = struct.Struct('<32sQQ')
JOURNAL = struct.Struct('<Q32s32s32s')
ZERO = bytes(32)
FLAGS = dict(production_ready=False, full_C3_demonstrated=False, physical_release=False,
             neural_executed=False, original_R_equivalence=False)
ARRAYS = ('io_input','io_return','packet','program_packet','boundary','program_boundary',
          'origin','nominal','AD_terminal','jacobian','raw_backward')
RESOURCES = ('resource_attempt','resource_return','resource_failure','resource_blocked')
NATIVE = ('native_call_attempt','native_call_return','native_call_failure')
MAP_KIND = 'agent_p3_parent_native_adapter_20261029_v1.signal_map'
AUX = {
    MAP_KIND: ('mapping',), 'queue': ('queue',), 'progress_bank': ('bank','parent_context_sha256'),
    'bank_event': ('event',), 'R_attempt': ('parent_context_sha256','bank_sha256'),
    'R_item_attempt': ('index','parent_context_sha256'), 'R_item_return': ('row',),
    'R_return': ('result',), 'signal_refusal': ('parent','minimum','remaining'),
    'denominator': ('rows','request_slots_per_arm'), 'arm_attempt': ('identity',),
    'arm_return': ('row',), 'source_before': ('sha256',), 'source_after': ('sha256',),
    'model_before': ('hashes','classes'), 'model_after': ('hashes','classes'),
    'precision_before': ('value',), 'precision_after': ('value',)}
KINDS = (*ARRAYS, *RESOURCES, 'direction', *NATIVE, *AUX)
PHASES = (None,'nominal','AD','whole','segmented')
OPERATIONS = ('predict','action_encoder','predictor','pred_proj','encoder','projector','jacobian',
    'grad','vjp','raw_backward','model_svd','array_svd','solve','signal_compile',
    'program_compile','replay','metric','schedule')
STATUSES = ('attempted','returned','failed','blocked_before_dispatch')
ENTRY = ('index','scope','resource','shape','directions','delta','status','started_ns','ended_ns','error')
AXES = ('predict_API','predict_samples','predict_context_tokens','action_encoder_calls',
    'action_encoder_samples','action_encoder_context_tokens','predictor_calls','predictor_samples',
    'predictor_context_tokens','pred_proj_calls','pred_proj_rows','encoder_calls','encoder_samples',
    'projector_calls','projector_samples','jacobian_API','VJP_directions','grad_API',
    'raw_backward_callbacks','model_SVD_calls','array_SVD_calls','solve_calls','signal_compile_calls',
    'program_compile_calls','replay_calls','metric_calls','schedule_decisions')


def need(ok, message):
    if not ok: raise ValueError(message)


def canonical(v): return json.dumps(v, sort_keys=True, separators=(',',':'), allow_nan=False).encode()
def h(raw): return hashlib.sha256(raw).digest()
def semantic(v): return h(canonical(v)).hex()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): digest.update(block)
    return digest.hexdigest()


def unique(pairs):
    d = {}
    for k, v in pairs:
        need(k not in d, 'duplicate JSON key'); d[k] = v
    return d


def exact(v, fields): need(type(v) is dict and set(v) == set(fields), 'exact typed fields: '+','.join(fields))


def uint(v):
    need(type(v) is int and 0 <= v < 2**64, 'literal u64, not bool/negative/overflow')
    return struct.pack('<Q',v)


def optional(v): return b'\0'+bytes(8) if v is None else b'\1'+uint(v)


def keys_of(v):
    """RAM fixtures/static discovery only, never an authorization or schema."""
    if type(v) is dict:
        need(all(type(k) is str for k in v), 'literal string keys')
        return set(v).union(*(keys_of(x) for x in v.values()))
    if type(v) is list: return set().union(*(keys_of(x) for x in v))
    return set()


class Cursor:
    def __init__(self, raw): self.raw, self.pos = raw, 0
    def take(self, n):
        need(type(n) is int and n >= 0 and self.pos+n <= len(self.raw), 'complete bounded record')
        v = self.raw[self.pos:self.pos+n]; self.pos += n; return v
    def number(self, fmt): return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]
    def option(self):
        present = self.number('<B'); value = self.number('<Q')
        need(present in (0,1) and (present or value == 0), 'canonical optional u64')
        return value if present else None
    def end(self): need(self.pos == len(self.raw), 'no trailing record bytes')


def rectangle(v, seen=frozenset()):
    if type(v) is float:
        need(math.isfinite(v), 'finite literal float'); return (), struct.pack('<d',v)
    if type(v) is not list or not v: return None
    need(id(v) not in seen and len(seen) < 128, 'bounded acyclic literal list')
    rows = [rectangle(x, seen | {id(v)}) for x in v]
    if any(x is None for x in rows) or any(x[0] != rows[0][0] for x in rows): return None
    dims = (len(v),)+rows[0][0]
    if len(dims)>3 or max(dims)>192: return None
    return dims, b''.join(x[1] for x in rows)


def spine_proof():
    # W has tag1+family1+count4+8*n bytes plus its48byte index. Slot/ledger
    # views have n>=1; native/capsule require distinct attempt+disposition,
    # hence n>=2. Views of the SAME family cannot overlap on any event.
    # Aliases in native_calls/bank evidence reference W once, not its members.
    overhead = 6+INDEX.size
    per_event = dict(slot=overhead+8,ledger=overhead+8,native=overhead//2+8,capsule=overhead//2+8)
    need(sum(per_event.values()) == 194 <= 256, 'worstcase actual view/index bytes')
    return dict(view_header_and_index=overhead,index_bytes_per_member=8,
        family_min_members=dict(slot=1,ledger=1,native=2,capsule=2),per_event_bounds=per_event,
        all_views_bytes_per_event_upper=194,approved_bytes_per_event=256,
        maximum_memberships_including_journal=5,approved_memberships=8,
        aliases='one W ID per native_calls/capsule/evidence reference; no new per-event membership')


class Codec:
    """Content-addressed raw records; exact Python JSON values, typed events.

    RAM-only archive today. Ordinary object coding is not a production bucket
    certificate: parent/candidate budget routing remains the next runner slice.
    """
    def __init__(self, fields, static=None):
        need(type(fields) in (list,tuple,set) and all(type(x) is str for x in fields), 'sealed field table')
        self.fields = sorted(set(fields)); need(len(self.fields) < 65536, 'u16 field table')
        self.codes = {k:i for i,k in enumerate(self.fields)}
        self.objects = {}; self.events = []; self.journal = []; self.tail = ZERO
        self.views = {}; self.capsules = {}; self.membership = []; self.roots = {}; self.active = set()
        self.static = json.loads(canonical({} if static is None else static), object_pairs_hook=unique)
        self.static_bytes = len(canonical(dict(fields=self.fields, static=self.static)))
        self.static_sha256 = semantic(dict(fields=self.fields,static=self.static))
        need(self.static_bytes <= STATIC, 'sealed static metadata32MiB')
        self.kind_counts = Counter(); self.row_max = {}; self.nodes_bytes = 0
        self.visited = None
        self.archive_binding = dict(kind='RAM_memory_only', materialized_capsules=False)

    def intern(self, raw):
        need(type(raw) is bytes and 0 < len(raw) <= STATIC, 'bounded raw record')
        key = h(DOMAIN+raw)
        if key in self.objects: need(self.objects[key] == raw, 'SHA collision')
        else:
            need(self.nodes_bytes+len(raw)+INDEX.size+self.static_bytes <= LIVE, 'live4GiB minus512MiB')
            self.objects[key] = raw; self.nodes_bytes += len(raw)+INDEX.size
        return key

    def _blob(self, v):
        exact(v, ('shape','dtype','sha256','data'))
        need(v['dtype'] in ('<f4','<f8') and type(v['shape']) is list
            and 1 <= len(v['shape']) <= 3 and all(type(x) is int and 1 <= x <= 192 for x in v['shape']), 'original tensor dtype/shape')
        need(type(v['data']) is str and type(v['sha256']) is str, 'literal tensor encoding')
        size = 4 if v['dtype'] == '<f4' else 8
        need(len(v['data']) == 4*((math.prod(v['shape'])*size+2)//3), 'exact base64 length before decode')
        raw = base64.b64decode(v['data'], validate=True)
        need(base64.b64encode(raw).decode() == v['data'], 'canonical original base64')
        need(len(raw) == math.prod(v['shape'])*size and h(raw).hex() == v['sha256'], 'tensor raw SHA/length')
        fmt = '<f' if size == 4 else '<d'
        need(all(math.isfinite(x[0]) for x in struct.iter_unpack(fmt, raw)), 'finite original tensor')
        prefix = b'B'+bytes([size,len(v['shape'])])+struct.pack('<'+'I'*len(v['shape']), *v['shape'])
        need(len(prefix)+INDEX.size <= 128, 'tensor header/index128')
        return self.intern(prefix+raw)

    def put(self, v, _depth=0):
        need(_depth < 128, 'bounded noncyclic object depth')
        t = type(v)
        if v is None: return self.intern(b'N')
        if t is bool: return self.intern(b'T' if v else b'F')
        if t is int:
            need(-2**63 <= v < 2**64, 'literal int64/u64 range')
            return self.intern(b'I'+struct.pack('<q',v) if v < 0 else b'U'+uint(v))
        if t is float:
            need(math.isfinite(v), 'finite literal float')
            return self.intern(b'D'+struct.pack('<d',v))
        if t is str:
            need(len(v.encode()) <= STATIC-1, 'bounded literal string')
            return self.intern(b'S'+v.encode())
        need(t in (list,dict), 'JSON types only; no foreign objects/tuples')
        identity = id(v); need(identity not in self.active, 'cyclic input')
        self.active.add(identity)
        try:
            if t is list:
                dense = rectangle(v)
                if dense is not None:
                    dims, raw = dense
                    return self.intern(b'V'+bytes([len(dims)])+struct.pack('<'+'I'*len(dims), *dims)+raw)
                return self.intern(b'L'+struct.pack('<I',len(v))+b''.join(self.put(x,_depth+1) for x in v))
            need(all(type(k) is str and k in self.codes for k in v), 'unknown or nonstring sealed field')
            if set(v) == {'shape','dtype','sha256','data'}: return self._blob(v)
            return self.intern(b'M'+struct.pack('<H',len(v))+b''.join(
                struct.pack('<H',self.codes[k])+self.put(x,_depth+1) for k,x in v.items()))
        finally: self.active.remove(identity)

    def _field(self, c):
        i = c.number('<H'); need(i < len(self.fields), 'valid field code'); return self.fields[i]

    def put_with_refs(self, value, refs):
        """Explicit equivalent view substitution, not schema/global patching."""
        need(type(value) is dict and type(refs) is dict and refs.keys() <= value.keys(), 'explicit record reference overrides')
        for name,key in refs.items():
            need(canonical(self.get(key)) == canonical(value[name]), 'reference expands to whole original field')
        need(all(name in self.codes for name in value), 'sealed record fields')
        raw = b'M'+struct.pack('<H',len(value))+b''.join(struct.pack('<H',self.codes[name])+
            (refs[name] if name in refs else self.put(v)) for name,v in value.items())
        return self.intern(raw)

    def get(self, key, _trail=frozenset()):
        need(type(key) is bytes and len(key) == 32 and key in self.objects and key not in _trail, 'existing acyclic SHA ref')
        raw = self.objects[key]; need(h(DOMAIN+raw) == key, 'object content SHA')
        if self.visited is not None: self.visited.add(key)
        need(len(_trail) < 128, 'bounded reference depth')
        trail = _trail | {key}; c = Cursor(raw); tag = c.take(1)
        sub = lambda: self.get(c.take(32), trail)
        if tag == b'N': v = None
        elif tag in (b'T',b'F'): v = tag == b'T'
        elif tag in (b'I',b'U'): v = c.number('<q' if tag == b'I' else '<Q')
        elif tag == b'D': v = c.number('<d'); need(math.isfinite(v), 'finite decoded literal')
        elif tag == b'S': v = c.take(len(raw)-1).decode('utf-8')
        elif tag == b'L': v = [sub() for _ in range(c.number('<I'))]
        elif tag == b'M':
            v = {}
            for _ in range(c.number('<H')):
                name = self._field(c); need(name not in v, 'no duplicate field code'); v[name] = sub()
        elif tag in (b'B',b'V'):
            size = c.number('<B') if tag == b'B' else 8; rank = c.number('<B')
            need(size in (4,8) and 1 <= rank <= 3, 'decoded dtype/rank')
            shape = [c.number('<I') for _ in range(rank)]
            need(all(1 <= d <= 192 for d in shape), 'decoded exact shape')
            data = c.take(math.prod(shape)*size)
            values = [x[0] for x in struct.iter_unpack('<f' if size == 4 else '<d',data)]
            need(all(math.isfinite(x) for x in values), 'finite decoded tensor/list')
            if tag == b'B':
                v = dict(shape=shape, dtype='<f4' if size == 4 else '<f8', sha256=h(data).hex(), data=base64.b64encode(data).decode())
            else:
                def nest(dims, values):
                    if len(dims) == 1: return values
                    n = math.prod(dims[1:]); return [nest(dims[1:],values[i*n:(i+1)*n]) for i in range(dims[0])]
                v = nest(shape,values)
        elif tag == b'E': v = self._decode_event(c, trail)
        elif tag == b'W': v = self._decode_view(c, trail)
        else: raise ValueError('unknown typed tag')
        c.end(); return v

    @staticmethod
    def peel(event):
        arm, io, slot, call = False, False, None, None
        if type(event) is dict and event.get('kind') == 'arm_event':
            exact(event, ('kind','slot','payload')); slot = event['slot']; uint(slot); need(slot < 12, 'fixed12 slot')
            arm = True; event = event['payload']
        if type(event) is dict and event.get('kind') == 'io':
            exact(event, ('kind','call','payload')); call = event['call']
            if call is not None: uint(call)
            io = True; event = event['payload']
        need(type(event) is dict and event.get('kind') in KINDS, 'known original event kind')
        return (arm,io,slot,call), event

    def _encode_event(self, event):
        wrapper, e = self.peel(event); arm, io, slot, call = wrapper; kind = e['kind']
        prefix = b'E'+bytes([KINDS.index(kind),int(arm)|int(io)<<1,slot if arm else 255])+optional(call)
        ref = lambda x: self.put(x)
        if kind in ARRAYS:
            exact(e, ('kind','scope','phase','step','operation_index','arrays'))
            need(e['phase'] in PHASES[1:] and type(e['scope']) is dict and type(e['arrays']) is dict
                and 1 <= len(e['arrays']) <= 7, 'native arrays variant')
            self._check_array_set(kind,e['arrays'])
            body = ref(e['scope'])+bytes([PHASES.index(e['phase'])])+optional(e['step'])+optional(e['operation_index'])
            items = []
            for name, value in e['arrays'].items():
                need(name in self.codes, 'sealed array name'); self._check_array(kind,name,value)
                items.append(struct.pack('<H',self.codes[name])+self._blob(value))
            body += bytes([len(items)])+b''.join(items); limit = 512
        elif kind in RESOURCES:
            exact(e, ('kind','entry')); entry = e['entry']; exact(entry,ENTRY)
            need(entry['resource'] in OPERATIONS and entry['status'] in STATUSES, 'original resource/status')
            shape = entry['shape']; need(shape is None or type(shape) is list and 1 <= len(shape) <= 3
                and all(type(x) is int and 1 <= x <= 192 for x in shape), 'exact resource shape')
            exact(entry['delta'],AXES); need(all(type(x) is int and x >= 0 for x in entry['delta'].values()), 'all27 literal resource axes')
            need(entry['error'] is None or type(entry['error']) is str, 'literal failure text')
            need(entry['status'] == dict(zip(RESOURCES,STATUSES))[kind], 'resource kind/disposition pairing')
            need(type(entry['scope']) is dict and (entry['ended_ns'] is None or type(entry['ended_ns']) is int
                and entry['started_ns'] <= entry['ended_ns']), 'scope and resource time order')
            body = uint(entry['index'])+ref(entry['scope'])+bytes([OPERATIONS.index(entry['resource'])])
            body += bytes([0 if shape is None else len(shape)])+(b'' if shape is None else struct.pack('<'+'I'*len(shape),*shape))
            body += optional(entry['directions'])+ref(entry['delta'])+bytes([STATUSES.index(entry['status'])])
            body += uint(entry['started_ns'])+optional(entry['ended_ns'])+ref(entry['error']); limit = 192
        elif kind == 'direction':
            exact(e, ('kind','scope','direction')); uint(e['direction']); need(e['direction'] < 192
                and type(e['scope']) is dict, 'direction0..191/scope')
            body = ref(e['scope'])+bytes([e['direction']]); limit = 64
        elif kind in NATIVE:
            fields = dict(native_call_attempt=('method','task','request'), native_call_return=('result',), native_call_failure=('error',))[kind]
            exact(e, ('kind',*fields))
            if kind == 'native_call_attempt': need(e['method'] in ('signal','replay'), 'native method')
            body = b''.join(ref(e[k]) for k in fields); limit = 128
        else:
            exact(e, ('kind',*AUX[kind])); body = b''.join(ref(e[k]) for k in AUX[kind]); limit = 256
        raw = prefix+body; need(len(raw) <= limit, 'actual typed row byte ceiling')
        return raw, limit

    @staticmethod
    def _check_array_set(kind, arrays):
        fixed = {'packet':('raw','token','outgoing'), 'program_packet':('latent','history','raw','mean','std','token','outgoing'),
            'boundary':('latent','history'), 'program_boundary':('latent','history'),
            'origin':('latent','history','raw_flat','mean','std')}
        if kind in fixed: need(set(arrays) == set(fixed[kind]), 'complete original array field set')
        elif kind in ('nominal','AD_terminal','jacobian','raw_backward'): exact(arrays,('value',))
        elif kind == 'io_input':
            variants = [('action_encoder:0',), ('predict:latent','predict:act_emb'), ('predictor:0','predictor:1'),
                ('pred_proj:0',), ('grad:output','grad:raw','grad:basis')]
            need(any(set(arrays) == set(v) for v in variants), 'complete original io_input variant')
        else:
            need(len(arrays) == 1 and next(iter(arrays)) in {x+':output' for x in
                ('action_encoder','predict','predictor','pred_proj','grad')}, 'complete original io_return variant')

    @staticmethod
    def _check_array(kind, name, v):
        # Exact original name/shape variants; neither shape-only acceptance nor
        # arbitrary field-name allowlisting may disguise a different ABI.
        exact(v, ('shape','dtype','sha256','data'))
        need(type(v['shape']) is list, 'literal native shape list')
        shapes = {'latent':[(1,3,192)], 'history':[(1,2,10)], 'raw':[(5,2)],
            'mean':[(2,)], 'std':[(2,)], 'token':[(10,)], 'outgoing':[(1,3,10)],
            'raw_flat':[(10,),(30,)]}
        family = {'packet':('raw','token','outgoing'), 'program_packet':('latent','history','raw','mean','std','token','outgoing'),
                  'boundary':('latent','history'), 'program_boundary':('latent','history'),
                  'origin':('latent','history','raw_flat','mean','std')}
        if kind in family: need(name in family[kind] and tuple(v['shape']) in shapes[name], 'original packet/origin array layout')
        elif kind in ('nominal','AD_terminal'): need(name == 'value' and v['shape'] == [192], 'terminal192')
        elif kind == 'jacobian': need(name == 'value' and v['shape'] in ([192,10],[192,30]), 'raw J192x10/30')
        elif kind == 'raw_backward': need(name == 'value' and v['shape'] in ([10],[30]), 'raw backward10/30')
        else:
            table = {'action_encoder:0':[(1,3,10)], 'action_encoder:output':[(1,3,192)],
                'predict:latent':[(1,3,192)], 'predict:act_emb':[(1,3,192)], 'predict:output':[(1,3,192)],
                'predictor:0':[(1,3,192)], 'predictor:1':[(1,3,192)], 'predictor:output':[(1,3,192)],
                'pred_proj:0':[(3,192)], 'pred_proj:output':[(3,192)], 'grad:output':[(192,)] if kind == 'io_input' else [(10,),(30,)],
                'grad:raw':[(10,),(30,)], 'grad:basis':[(192,)]}
            need(name in table and tuple(v['shape']) in table[name], 'original native IO name/shape')
            need((name.endswith(':output')) == (kind == 'io_return') or kind == 'io_input' and name == 'grad:output', 'input/output variant role')
        need(v['dtype'] == '<f4', 'original native precision f32')

    def append_event(self, event):
        raw, limit = self._encode_event(event); key = self.intern(raw)
        _, core = self.peel(event); kind = core['kind']
        self.row_max[kind] = max(len(raw),self.row_max.get(kind,0)); self.kind_counts[kind] += 1
        index = len(self.events); stem = struct.pack('<Q32s32s',index,self.tail,key); tail = h(CHAIN+stem)
        self.journal.append(stem+tail); self.events.append(key); self.membership.append({'journal'})
        self.tail = tail
        need(canonical(self.get(key)) == canonical(event), 'encode/decode whole event exact')
        return key

    def _decode_event(self, c, trail):
        code = c.number('<B'); need(code < len(KINDS), 'event kind code'); kind = KINDS[code]
        flags = c.number('<B'); slot = c.number('<B'); call = c.option()
        need(flags in (0,1,2,3) and (slot < 12 if flags&1 else slot == 255) and (flags&2 or call is None), 'exact wrapper variant')
        sub = lambda: self.get(c.take(32),trail)
        if kind in ARRAYS:
            scope = sub(); phase = c.number('<B'); need(phase < len(PHASES), 'phase code')
            e = dict(kind=kind,scope=scope,phase=PHASES[phase],step=c.option(),operation_index=c.option(),arrays={})
            for _ in range(c.number('<B')):
                name = self._field(c); need(name not in e['arrays'], 'unique array fields'); e['arrays'][name] = sub()
        elif kind in RESOURCES:
            entry = dict(index=c.number('<Q'),scope=sub()); op = c.number('<B'); rank = c.number('<B')
            need(op < len(OPERATIONS) and rank <= 3, 'resource codes')
            entry.update(resource=OPERATIONS[op],shape=[c.number('<I') for _ in range(rank)] if rank else None,
                         directions=c.option(),delta=sub())
            status = c.number('<B'); need(status < len(STATUSES), 'status code')
            entry.update(status=STATUSES[status],started_ns=c.number('<Q'),ended_ns=c.option(),error=sub()); e = dict(kind=kind,entry=entry)
        elif kind == 'direction': e = dict(kind=kind,scope=sub(),direction=c.number('<B'))
        else:
            fields = dict(native_call_attempt=('method','task','request'),native_call_return=('result',),native_call_failure=('error',)).get(kind,AUX.get(kind))
            e = dict(kind=kind, **{name:sub() for name in fields})
        if flags&2: e = dict(kind='io',call=call,payload=e)
        if flags&1: e = dict(kind='arm_event',slot=slot,payload=e)
        return e

    def event(self, index): return self.get(self.events[index])

    def add_view(self, family, indices):
        need(family in ('slot','native','capsule','ledger'), 'explicit finite view families')
        need(type(indices) is list and indices and indices == sorted(set(indices))
            and all(type(i) is int and 0 <= i < len(self.events) for i in indices), 'complete ordered view indices')
        self._validate_view(family,indices,[self.event(i) for i in indices])
        raw = b'W'+bytes([('slot','native','capsule','ledger').index(family)])+struct.pack('<I',len(indices))+b''.join(uint(i) for i in indices)
        key = h(DOMAIN+raw)
        if key in self.views: return key
        # These are ACTUAL memberships, not a claim inferred from a small trace.
        # native_calls/evidence/root references alias an existing view ID, not
        # an additional per-event list. Each family has at most one membership.
        need(all(family not in self.membership[i] for i in indices), 'overlapping duplicate view family')
        need(all(len(self.membership[i])+1 <= 8 for i in indices), 'spine at most8 memberships')
        key = self.intern(raw); self.views[key] = (family,indices)
        for i in indices: self.membership[i].add(family)
        return key

    def _decode_view(self, c, trail):
        code = c.number('<B'); need(code < 4, 'known view family'); family = ('slot','native','capsule','ledger')[code]
        indices = [c.number('<Q') for _ in range(c.number('<I'))]
        need(indices and indices == sorted(set(indices)) and all(i < len(self.events) for i in indices), 'bounded ordered view')
        values = [self.get(self.events[i],trail) for i in indices]
        self._validate_view(family,indices,values)
        if family == 'slot': return values
        cores = [self.peel(e)[1] for e in values]
        if family == 'ledger':
            need(all(e['kind'] in ('resource_return','resource_failure','resource_blocked') for e in cores), 'ledger dispositions, no attempts')
            entries = sorted((e['entry'] for e in cores),key=lambda e:e['index'])
            need(len({e['index'] for e in entries}) == len(entries), 'one final ledger disposition per index')
            return entries
        return cores

    @staticmethod
    def _validate_view(family, indices, values):
        peeled = [Codec.peel(v) for v in values]; wrappers = [x[0] for x in peeled]; cores = [x[1] for x in peeled]
        need(all(w[0] and w[2] == wrappers[0][2] for w in wrappers), 'one actual slot view')
        if family in ('native','capsule'):
            need(indices == list(range(indices[0],indices[-1]+1))
                and all(w == wrappers[0] and w[1] and w[3] is not None for w in wrappers), 'one contiguous explicit native call')
            need(cores[0]['kind'] == 'native_call_attempt' and cores[-1]['kind'] in ('native_call_return','native_call_failure')
                and all(e['kind'] not in NATIVE for e in cores[1:-1]), 'one complete native delimiter pair')
        elif family == 'ledger':
            need(all(e['kind'] in ('resource_return','resource_failure','resource_blocked') for e in cores)
                and len({e['entry']['index'] for e in cores}) == len(cores), 'unique ledger dispositions only')

    def virtual_capsule(self, slot, call, original, namespace='/__virtual__'):
        uint(slot); uint(call); need(slot < 12 and Path(namespace).is_absolute()
            and '__virtual__' in Path(namespace).parts and '..' not in Path(namespace).parts, 'virtual namespace/slot/call')
        indices = [i for i in range(len(self.events)) if self.peel(self.event(i))[0] == (True,True,slot,call)]
        need(indices and indices == list(range(indices[0],indices[-1]+1)), 'one contiguous actual native invocation')
        key = self.add_view('capsule',indices); expanded = self.get(key)
        need(canonical(expanded) == canonical(original) and expanded[0]['kind'] == 'native_call_attempt'
            and expanded[-1]['kind'] in ('native_call_return','native_call_failure'), 'whole original source transcript')
        path = str(Path(namespace)/f's{slot}_c{call}.json')
        entry = dict(schema=NAME+'.virtual_capsule',path=path,materialized=False,object_sha256=key.hex(),
            slot=slot,call=call,expanded_canonical_sha256=semantic(expanded),expanded_canonical_bytes=len(canonical(expanded)))
        if path in self.capsules: need(self.capsules[path] == entry, 'immutable virtual pointer')
        self.capsules[path] = entry; return entry

    def read_virtual(self, entry):
        exact(entry, ('schema','path','materialized','slot','call','object_sha256','expanded_canonical_sha256','expanded_canonical_bytes'))
        need(entry['schema'] == NAME+'.virtual_capsule' and entry['materialized'] is False
            and self.capsules.get(entry['path']) == entry, 'registered NONmaterial capsule, not filesystem authority')
        uint(entry['slot']); uint(entry['call']); need(entry['slot'] < 12, 'actual capsule slot')
        path = Path(entry['path']); need(path.is_absolute() and '__virtual__' in path.parts and '..' not in path.parts
            and path.name == f"s{entry['slot']}_c{entry['call']}.json", 'explicit virtual path identity')
        key = bytes.fromhex(entry['object_sha256']); need(key in self.views and self.views[key][0] == 'capsule', 'actual capsule view')
        indices = self.views[key][1]
        expected = [i for i in range(len(self.events)) if self.peel(self.event(i))[0] == (True,True,entry['slot'],entry['call'])]
        need(indices == expected and indices == list(range(indices[0],indices[-1]+1)), 'all and only explicit session identity')
        value = self.get(bytes.fromhex(entry['object_sha256']))
        need(semantic(value) == entry['expanded_canonical_sha256'] and len(canonical(value)) == entry['expanded_canonical_bytes'], 'expanded semantic SHA/length')
        need(value[0]['kind'] == 'native_call_attempt' and value[-1]['kind'] in ('native_call_return','native_call_failure'), 'complete source call disposition')
        return value

    def stats(self):
        views_bytes = sum(len(self.objects[k])+INDEX.size for k in self.views)
        tag_bytes = Counter(); tensor_headers = []; tensor_shapes = Counter()
        for raw in self.objects.values():
            tag_bytes[raw[:1].decode('ascii')] += len(raw)+INDEX.size
            if raw[:1] in (b'B',b'V'):
                c = Cursor(raw[1:]); size = c.number('<B') if raw[:1] == b'B' else 8
                rank = c.number('<B'); shape = tuple(c.number('<I') for _ in range(rank))
                tensor_headers.append(c.pos+1+INDEX.size)
                tensor_shapes[str((raw[:1].decode(),size,shape))] += 1
        return dict(schema=NAME+'.codec_stats',objects=len(self.objects),events=len(self.events),
            stored_object_and_index_bytes=self.nodes_bytes,journal_bytes=len(self.journal)*104,
            static_metadata_bytes=self.static_bytes,kind_counts=dict(self.kind_counts),row_max=self.row_max,
            actual_max_spine_memberships=max(map(len,self.membership),default=0),
            schema_max_spine_memberships=5, actual_view_bytes=views_bytes,
            view_budget_bytes=256*len(self.events), view_budget_fits=views_bytes <= 256*len(self.events),
            spine_schema_proof=spine_proof(),
            tensor_header_and_index_max=max(tensor_headers,default=0), tensor_shape_counts=dict(tensor_shapes),
            physical_object_and_index_bytes_by_tag=dict(tag_bytes), **FLAGS)

    def verify_events(self):
        need(self.static_sha256 == semantic(dict(fields=self.fields,static=self.static))
            and self.codes == {k:i for i,k in enumerate(self.fields)}, 'immutable sealed field/static bytes')
        before = set(self.objects)
        need(len(self.events) == len(self.journal), 'complete journal/event denominator')
        tail = ZERO
        for i,(key,line) in enumerate(zip(self.events,self.journal)):
            need(len(line) == 104, 'journal104')
            index, previous, ref, claimed = JOURNAL.unpack(line)
            need(index == i and previous == tail and ref == key and h(CHAIN+line[:72]) == claimed, 'exact journal index/chain/ref')
            value = self.get(key); raw,_ = self._encode_event(value)
            need(raw == self.objects[key], 'decoded event canonical typed row'); tail = claimed
        need(tail == self.tail, 'final journal tail')
        need(self.stats()['view_budget_fits'], 'actual spine bytes within256/event')
        need(set(self.objects) == before, 'decoder cannot add missing objects')
        return self.stats()

    def verify_closure(self):
        """Every physical node is reachable; unused tensors/hidden data refuse."""
        self.visited = set()
        try:
            for key in self.events: self.get(key)
            for key in self.roots.values(): self.get(key)
            for entry in self.capsules.values(): self.read_virtual(entry)
            need(self.visited == set(self.objects), 'no unused or hidden physical objects')
        finally: self.visited = None


class VirtualCapsuleReader:
    """Explicit reader adapter, NOT an os.path/read/old-global replacement."""
    def __init__(self, codec): self.codec = codec
    def descriptor(self, slot, call):
        found = [e for e in self.codec.capsules.values() if e['slot'] == slot and e['call'] == call]
        need(len(found) == 1, 'one explicit virtual descriptor'); return found[0]
    def callback(self, slot, call, original):
        entry = self.descriptor(slot,call); value = self.codec.read_virtual(entry)
        need(canonical(value) == canonical(original), 'original complete transcript supplied to reader')
        # Matches frozen source_evidence's injected callback ABI. sha256 here
        # is expanded canonical identity, NEVER a claim that a file exists.
        return dict(path=entry['path'],sha256=entry['expanded_canonical_sha256'])
    def provenance(self, slot, call):
        return dict(descriptor=self.descriptor(slot,call),physical_storage_binding=self.codec.archive_binding,
                    semantic_identity_is_not_material_file_SHA=True)


def proof():
    n,j,v = 6912,1188,228096; l = 4*n+3*v+2*j+7464; a = 10*n+3*v+5*j
    c = 2*(j+1968+1728); u = 12*4096; r = 2*l+12; e = r+a+v+c+u; t = 19*n+5*v+13*j
    parts = dict(native_tensor_bytes=4*(8192*n+474*v+8192*j), literal_f64_bytes=8*(5952*j+596*n),
        tensor_headers_and_indices=128*(t+2*j+2*n),resource_rows=r*192,array_rows=a*512,
        direction_rows=v*64,native_call_rows=c*128,auxiliary_rows=u*256,journal=e*104,
        event_object_indices=e*48,container_spines_and_secondary_views=e*256,
        parent_structures=204*524288,candidates=1728*65536,slot_structures=12*1048576,sealed_static_metadata=STATIC)
    need(sum(parts.values()) == 2930867648 < LIVE, 'approved integer proof')
    return dict(parts=parts,live_bytes_upper=sum(parts.values()),live_allowance=LIVE,
        total_bytes_upper=sum(parts.values())+512*1024**2,denominator=12,request_denominator=1728,
        spine_schema_proof=spine_proof(),
        proof_scope='conditional schema budget; production bucket routing/control-flow not implemented',**FLAGS)


def write_fixture(codec, directory):
    """Exclusive RAM artifact for codec tests, NEVER a production report."""
    codec.verify_events(); codec.verify_closure(); directory = Path(directory).absolute()
    need(codec.nodes_bytes+len(codec.journal)*104+STATIC <= LIVE, 'whole fixture including metadata bound')
    need(directory.resolve() == directory and not directory.exists() and not directory.is_symlink(), 'new exclusive fixture path')
    directory.mkdir(); offset = 0; idx = bytearray()
    with (directory/'objects.pack').open('xb') as f:
        for key,raw in codec.objects.items():
            need(f.write(raw) == len(raw), 'complete pack write'); idx += INDEX.pack(key,offset,len(raw)); offset += len(raw)
        f.flush(); os.fsync(f.fileno())
    for name,raw in [('objects.idx',bytes(idx)),('journal.bin',b''.join(codec.journal))]:
        with (directory/name).open('xb') as f: need(f.write(raw) == len(raw), 'complete fixture write'); f.flush(); os.fsync(f.fileno())
    meta = dict(schema=NAME+'.RAM_fixture',fields=codec.fields,static=codec.static,tail=codec.tail.hex(),
        codec_source_sha256=h(Path(__file__).read_bytes()).hex(),protocol_sha256=PROTOCOL_SHA,
        roots={k:v.hex() for k,v in codec.roots.items()},capsules=codec.capsules,
        artifacts={n:file_sha(directory/n) for n in ('objects.pack','objects.idx','journal.bin')},**FLAGS)
    raw = canonical(meta); need(len(raw) <= STATIC, 'fixture metadata32MiB')
    with (directory/'fixture.json').open('xb') as f: need(f.write(raw) == len(raw), 'fixture manifest'); f.flush(); os.fsync(f.fileno())
    fd = os.open(directory,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return dict(path=str(directory),fixture_sha256=h(raw).hex(),stats=codec.stats())


def read_fixture(directory, expected):
    directory = Path(directory).absolute()
    need(directory.resolve() == directory and not directory.is_symlink(), 'exact fixture directory')
    names = {'objects.pack','objects.idx','journal.bin','fixture.json'}
    need({x.name for x in directory.iterdir()} == names and all(x.is_file() and not x.is_symlink() for x in directory.iterdir()), 'exact fixture file census')
    need(sum(x.stat().st_size for x in directory.iterdir()) <= LIVE, 'bounded fixture total')
    need((directory/'fixture.json').stat().st_size <= STATIC, 'bounded fixture metadata before read')
    raw = (directory/'fixture.json').read_bytes(); need(h(raw).hex() == expected, 'pinned actual fixture SHA')
    m = json.loads(raw,object_pairs_hook=unique)
    exact(m, ('schema','fields','static','tail','roots','capsules','artifacts','codec_source_sha256','protocol_sha256',*FLAGS))
    need(m['schema'] == NAME+'.RAM_fixture' and all(m[k] is False for k in FLAGS), 'RAM only, no production certification')
    need(m['codec_source_sha256'] == h(Path(__file__).read_bytes()).hex() and m['protocol_sha256'] == PROTOCOL_SHA
        and h(PROTOCOL.read_bytes()).hex() == PROTOCOL_SHA, 'exact codec source/protocol closure')
    exact(m['artifacts'],('objects.pack','objects.idx','journal.bin'))
    for n,digest in m['artifacts'].items(): need(file_sha(directory/n) == digest, 'exact pack/index/journal SHA')
    need(type(m['fields']) is list and m['fields'] == sorted(set(m['fields'])), 'exact sealed field ordering')
    codec = Codec(m['fields'],m['static']); idx = (directory/'objects.idx').read_bytes()
    need(len(idx)%48 == 0, 'complete48byte index')
    offset = 0
    with (directory/'objects.pack').open('rb') as f:
        for i in range(0,len(idx),48):
            key,start,length = INDEX.unpack(idx[i:i+48]); need(start == offset and 0 < length <= STATIC and key not in codec.objects, 'contiguous unique index')
            data = f.read(length); need(len(data) == length and h(DOMAIN+data) == key, 'index exact record bytes')
            codec.objects[key] = data; codec.nodes_bytes += length+48; offset += length
        need(f.read(1) == b'', 'no unindexed trailing bytes')
    raw = (directory/'journal.bin').read_bytes(); need(len(raw)%104 == 0, 'complete104byte journal')
    codec.journal = [raw[i:i+104] for i in range(0,len(raw),104)]
    codec.events = [JOURNAL.unpack(x)[2] for x in codec.journal]; codec.tail = bytes.fromhex(m['tail'])
    codec.membership = [{'journal'} for _ in codec.events]; codec.roots = {k:bytes.fromhex(v) for k,v in m['roots'].items()}; codec.capsules = m['capsules']
    for key,data in list(codec.objects.items()):
        if data[:1] == b'W':
            c = Cursor(data[1:]); code = c.number('<B'); need(code < 4, 'view family code')
            indices = [c.number('<Q') for _ in range(c.number('<I'))]; c.end()
            family = ('slot','native','capsule','ledger')[code]
            need(indices and indices == sorted(set(indices)) and all(i < len(codec.events) for i in indices), 'complete bounded view index')
            need(all(family not in codec.membership[i] for i in indices), 'no duplicate view coverage')
            codec.views[key] = (family,indices)
            for i in indices: codec.membership[i].add(family)
    codec.verify_events()
    for ref in codec.roots.values(): codec.get(ref)
    for entry in codec.capsules.values(): codec.read_virtual(entry)
    codec.verify_closure()
    need(m['codec_source_sha256'] == h(Path(__file__).read_bytes()).hex(), 'source unchanged after decode')
    codec.archive_binding = dict(kind=NAME+'.RAM_fixture',fixture_sha256=expected,artifacts=m['artifacts'],materialized_capsules=False)
    return codec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('proof'); q = sub.add_parser('roundtrip-demo'); q.add_argument('--output',required=True)
    q = sub.add_parser('verify-fixture'); q.add_argument('--input',required=True); q.add_argument('--fixture-sha',required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'proof': result = proof()
    elif args.cmd == 'verify-fixture': result = read_fixture(args.input,args.fixture_sha).verify_events()
    else:
        event = dict(kind='direction',scope=dict(phase='RAM_FIXTURE'),direction=191)
        codec = Codec(keys_of(event)); codec.append_event(event); codec.append_event(event)
        result = write_fixture(codec,args.output)
    print(canonical(result).decode())


if __name__ == '__main__': main()
