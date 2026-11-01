"""C3 complete native evidence -> explicit Catalog. RAM integration slice only.

No build, production prepare, model import, or CUDA entry is exposed here.
The journal projection is NOT a replacement for the full producer journal.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import struct

from experiments.spectral_reachability import agent_p3_c3_compact_routing_20261029_v1 as routing
from experiments.spectral_reachability import agent_p3_c3_common_cap_cuda_20261029_v1 as original

codec = routing.codec
NAME = 'agent_p3_c3_compact_native_catalog_20261029_v1'
need, canonical, semantic = codec.need, codec.canonical, codec.semantic
FLAGS = dict(production_ready=False, neural_executed=False, physical_release=False,
             full_C3_demonstrated=False, original_R_equivalence=False)
NATIVE_KINDS = set(codec.ARRAYS + codec.RESOURCES + codec.NATIVE + ('direction',))
SOURCE_PINS = {
    original.__file__:'7419b72f12ea81f90703d2c075143aa46f07c8fbe47ba0313c488239797520a7',
    routing.__file__:'1ea53874b5f685c922fd3e6925252d890dc0f2206ef0c7f2bbaf811f69340592',
    codec.__file__:routing.CODEC_SHA,
}


def pointer(slot, call, transcript):
    """Original injected capsule ABI; semantic identity, never material SHA."""
    codec.uint(slot); codec.uint(call); need(slot < 12, 'fixed twelve slots')
    return dict(path=f'/__virtual__/{NAME}/s{slot}_c{call}.json', sha256=semantic(transcript))


def provenance_key(kind, value):
    return kind, semantic(value)


class NativeCatalog:
    """Register only outputs derived from original, complete native witnesses.

    Caller supplies the WHOLE ordered slot journal, not a selected call list.
    Auxiliary events remain in caller's journal. Only the native/resource
    projection is encoded here, with explicit original-position evidence.
    Full original saved reconstruction is a separate, mandatory boundary.
    """
    def __init__(self, store, catalog):
        need(catalog.store is store, 'one exact CAS instance')
        need(all(codec.file_sha(p)==sha for p,sha in SOURCE_PINS.items()),'frozen direct source pins')
        self.store, self.catalog = store, catalog
        self.origins = {}; self.slots = {}; self.native = {}; self.failed = set()
        self.accepting=True

    def bind(self, kind, value, origin, key=None):
        key = self.store.put(value) if key is None else key
        self.catalog.bind(kind, value, key)
        identity = provenance_key(kind, value)
        # Same byte-identical value may legitimately have multiple paid origins.
        self.origins.setdefault(identity, []).append(deepcopy(origin))
        return key

    def _context(self, task, context, origin):
        original.core.context(context)
        key = self.bind('context', context, origin)
        self.store.roots['context_' + semantic(context)] = key
        return key

    def _task(self, root, task, origin):
        original.core.validate_task(task)
        need(self.catalog.contains('context', task['root_context']),
             'parent context must have prior native/root provenance')
        expected = original.parent.parent_task(root, task['root_context'])
        need(canonical(expected) == canonical(task), 'complete immutable root/local task identity')
        return self.bind('task', task, origin)

    def ingest(self, slot, root, result, events):
        """Saved/RAM inputs only; never discovers production results or sources."""
        need(self.accepting,'failed ingestion instance cannot be reused')
        self.accepting=False  # only a complete successful ingestion reopens it
        codec.uint(slot); need(slot < 12 and slot not in self.slots, 'one fixed slot ingestion')
        original.core.validate_task(root)
        need(self.catalog.contains('task', root) and self.catalog.contains('context', root['root_context']),
             'root task must come from installed static seal')
        need(type(events) is list and events and all(type(e) is dict and
             set(e) == {'kind','call','payload'} and e['kind'] == 'io' for e in events),
             'whole original slot IO journal')
        # Do not let bool masquerade as a call ID or let foreign events hide in
        # the native projection. The original validator later checks all AUX.
        for e in events:
            cid = e['call']
            need(cid is None or type(cid) is int and 0 <= cid < len(result['native_calls']),
                 'literal existing call identity')
            codec.Codec.peel(dict(kind='arm_event', slot=slot, payload=e))
        original.old.resource_census(result['resource_receipt'], [e['payload'] for e in events])
        records = result['native_calls']
        need([r['call_id'] for r in records] == list(range(len(records))), 'complete ordered call IDs')
        projected = []; position_map = []; disposition = []; context_counts = Counter()
        call_positions = {}
        # First establish contexts strictly in invocation order. A child cannot
        # bootstrap its own provenance by appearing as an arbitrary task value.
        last = -1
        for record in records:
            cid = record['call_id']
            positions = [i for i,e in enumerate(events) if e['call'] == cid]
            need(positions and positions == list(range(positions[0], positions[-1]+1))
                 and positions[0] > last, 'whole contiguous ordered original invocation')
            last = positions[-1]
            call = [events[i]['payload'] for i in positions]
            need(canonical(call) == canonical(record['transcript']), 'every native event exactly once')
            need(call[0]['kind'] == 'native_call_attempt' and
                 call[-1]['kind'] in ('native_call_return','native_call_failure') and
                 all(e['kind'] not in codec.NATIVE for e in call[1:-1]), 'exact native delimiters')
            task = call[0]['task']; origin = dict(slot=slot, call=cid, original_positions=positions)
            self._task(root, task, origin)
            complete = original.old.audit_native(call)
            need(complete is (call[-1]['kind'] == 'native_call_return'), 'native disposition/closure distinction')
            if complete:
                out = call[-1]['result']
                if call[0]['method'] == 'signal':
                    for field in ('nominal','jacobian'):
                        self.bind('native_literal', out[field], dict(origin, field=field))
                    # Every nominal boundary is real full3/between2 native IO,
                    # including H3's intermediate boundaries; no shape proxy.
                    for i,e in enumerate(call):
                        if e['kind'] == 'boundary' and e['phase'] == 'nominal':
                            c = dict(task['root_context'],
                                latent_history=original.signal.unblob(e['arrays']['latent'])[0].tolist(),
                                action_history=original.signal.unblob(e['arrays']['history'])[0].tolist())
                            self._context(task, c, dict(origin, boundary_index=i))
                            context_counts['nominal_boundary'] += 1
                else:
                    for i,c in enumerate(out):
                        self._context(task, c, dict(origin, replay_boundary=i))
                        context_counts['replay_boundary'] += 1
            else:
                self.failed.add((slot,cid))
            disposition.append(dict(call=cid, complete_native_closure=complete,
                status='complete' if complete else 'failure_prefix_only'))
            call_positions[cid] = positions
        # Preserve original full-order indices, including auxiliary positions.
        # Holes are private, transient RAM state. They cannot be checkpointed
        # or advertised as a completed journal. finish_slot must fill them.
        start=len(self.store.events)
        need(len(self.store.journal)==start,'previous slot journal must be settled before next slot')
        self.store.events.extend([None]*len(events))
        self.store.membership.extend({'journal'} for _ in events)
        # Only now encode the original native/resource projection. Same shapes
        # or same field names alone never create a Catalog authority.
        for i,e in enumerate(events):
            if e['payload']['kind'] in NATIVE_KINDS:
                if e['call'] is not None:
                    need(i in call_positions[e['call']], 'native event belongs to actual invocation')
                else:
                    need(e['payload']['kind'] in codec.RESOURCES, 'no orphan native buffer/delimiter')
                projected.append(start+i); position_map.append(i)
                self._event_at(start+i,dict(kind='arm_event', slot=slot, payload=e))
        need(projected, 'nonempty native/resource projection')
        # Native and capsule W views refer to actual typed event rows; the
        # complete transcript is never encoded as a generic caller-provided L.
        by_original = dict(zip(position_map, projected, strict=True))
        for record, status in zip(records, disposition, strict=True):
            cid = record['call_id']; call = record['transcript']
            indices = [by_original[i] for i in call_positions[cid]]
            key = self.store.add_view('native', indices)
            self.bind('transcript', call, dict(slot=slot,call=cid,kind='native_view'), key)
            self.native[slot,cid] = dict(view=key.hex(), status=status,
                original_positions=call_positions[cid], typed_indices=indices)
            # A failed call may be read as a prefix, but is not a bank source.
            if status['complete_native_closure']:
                capsule_key = self.store.add_view('capsule',indices)
                p = pointer(slot,cid,call)
                entry = dict(schema=codec.NAME+'.virtual_capsule',path=p['path'],materialized=False,
                    object_sha256=capsule_key.hex(),slot=slot,call=cid,
                    expanded_canonical_sha256=semantic(call),expanded_canonical_bytes=len(canonical(call)))
                need(entry['path'] not in self.store.capsules,'one exact virtual capsule descriptor')
                self.store.capsules[entry['path']] = entry
                need(pointer(slot,cid,call) == dict(path=entry['path'],sha256=entry['expanded_canonical_sha256']),
                     'exact explicit semantic capsule ABI')
        finals = [by_original[i] for i,e in enumerate(events) if i in by_original and
                  e['payload']['kind'] in ('resource_return','resource_failure','resource_blocked')]
        need(finals, 'whole ledger disposition projection')
        key = self.store.add_view('ledger', finals)
        ledger = result['resource_receipt']['events']
        self.bind('ledger', ledger, dict(slot=slot,kind='ledger_disposition_view'), key)
        for entry in ledger:
            self.bind('scope', entry['scope'], dict(slot=slot,ledger_index=entry['index']))
        inclusive = result['resource_receipt']['inclusive_operation_wall_ns']
        # Derived from the original complete receipt, verified again by full
        # reconstruction. Not a new wall/FLOPs or completion certificate.
        self.bind('inclusive', inclusive, dict(slot=slot,kind='original_resource_receipt'))
        self.slots[slot] = dict(slot=slot, event_positions=position_map, typed_indices=projected,
            original_events=len(events), projected_events=len(projected), calls=disposition,
            context_counts=dict(context_counts),start=start,finished=False)
        self.accepting=True
        return deepcopy(self.slots[slot])

    def _event_at(self,index,event,encoder=None):
        raw,limit=(self.store if encoder is None else encoder)._encode_event(event)
        key=self.store.intern(raw)
        need(self.store.events[index] is None,'one event per original reserved position')
        self.store.events[index]=key
        _,e=codec.Codec.peel(event);kind=e['kind']
        self.store.row_max[kind]=max(len(raw),self.store.row_max.get(kind,0))
        self.store.kind_counts[kind]+=1
        need(canonical(self.store.get(key))==canonical(event),'whole original event exact roundtrip')

    def finish_slot(self,slot,result,events):
        """Actual whole-slot routing plus original-order auxiliary transport.

        Offline RAM slice: result/events must already exist. This does not
        claim durable online production capture of still-running callbacks.
        """
        self.verify_projection(slot,events)
        row=self.slots[slot];need(not row['finished'],'one finish per slot')
        ordinals=[i for i,q in enumerate(result['queues']) if q['status']=='queued' and
                  q['decision']['extension_type']=='progress']
        banks=[e['payload']['bank'] for e in events if e['payload']['kind']=='progress_bank']
        route=routing.Routing(self.store,self.catalog)
        ref=route.tree(slot,result,progress_banks=list(zip(ordinals,banks,strict=True)))
        encoder=CatalogEncoder(self.store,self.catalog)
        for i,e in enumerate(events):
            if self.store.events[row['start']+i] is None:
                self._event_at(row['start']+i,dict(kind='arm_event',slot=slot,payload=e),encoder)
        need(all(k is not None for k in self.store.events),'no unresolved event at commit')
        for index in range(len(self.store.journal),len(self.store.events)):
            key=self.store.events[index]
            stem=struct.pack('<Q32s32s',index,self.store.tail,key)
            tail=codec.h(codec.CHAIN+stem)
            self.store.journal.append(stem+tail);self.store.tail=tail
        row['finished']=True
        return dict(result=ref,routing=route.census(),original_order_journal_events=len(events),
                    auxiliary_events=len(events)-row['projected_events'],**FLAGS)

    def verify_full_journal(self):
        """Typed-row check with explicit reader, never frozen-global patch."""
        need(all(r['finished'] for r in self.slots.values()),'no pending auxiliary positions')
        need(len(self.store.journal)==len(self.store.events),'whole journal denominator')
        tail=codec.ZERO;encoder=CatalogEncoder(self.store,self.catalog)
        before=set(self.store.objects)
        for i,(key,line) in enumerate(zip(self.store.events,self.store.journal,strict=True)):
            index,previous,ref,claimed=codec.JOURNAL.unpack(line)
            need(index==i and previous==tail and ref==key and codec.h(codec.CHAIN+line[:72])==claimed,
                 'whole original-order journal hashchain')
            e=self.store.get(key);_,core=codec.Codec.peel(e)
            enc=self.store if core['kind'] in NATIVE_KINDS else encoder
            raw,_=enc._encode_event(e)
            need(raw==self.store.objects[key],'canonical original typed event')
            tail=claimed
        need(tail==self.store.tail and set(self.store.objects)==before,'no missing/created event object')
        return dict(events=len(self.store.events),tail=tail.hex(),complete_original_order=True,**FLAGS)

    def capsule(self, slot, cid, call):
        need((slot,cid) in self.native and (slot,cid) not in self.failed,
             'complete validated native call required for source capsule')
        # Explicit indexed reader, not a scan of every unrelated tensor once
        # per capsule. Same frozen W format, with complete per-call indices.
        p=pointer(slot,cid,call);entry=self.store.capsules[p['path']]
        record=self.native[slot,cid];key=bytes.fromhex(entry['object_sha256'])
        need(entry['materialized'] is False and entry['slot']==slot and entry['call']==cid
             and self.store.views[key]==('capsule',record['typed_indices']),
             'exact indexed nonmaterial source capsule')
        value=self.store.get(key)
        need(canonical(value)==canonical(call) and semantic(value)==entry['expanded_canonical_sha256']
             and len(canonical(value))==entry['expanded_canonical_bytes']
             and p['sha256']==entry['expanded_canonical_sha256'],'entire source capsule bytes')
        return p

    def verify_projection(self, slot, events):
        row = self.slots[slot]
        expected = [i for i,e in enumerate(events) if e['payload']['kind'] in NATIVE_KINDS]
        need(row['event_positions'] == expected, 'all and only native/resource original positions')
        for i,j in zip(expected,row['typed_indices'],strict=True):
            need(canonical(self.store.event(j)) == canonical(dict(kind='arm_event',slot=slot,payload=events[i])),
                 'whole projected event exact bytes')
        return True

    def reconstruct(self, slot, task, result, events, *, permit_fixture=False):
        """Original saved validator, not second independent native mathematics.

        Called only on RAM fixtures by this worker. MAIN may use this explicit
        reader in its later production consumer; no old global is patched.
        """
        row=self.slots[slot]
        need(row['finished'],'complete slot transport required before saved reconstruction')
        self.verify_full_journal()
        expanded=[]
        for i in range(row['start'],row['start']+row['original_events']):
            e=self.store.event(i)
            need(e['kind']=='arm_event' and e['slot']==slot,'whole saved slot wrapper identity')
            expanded.append(e['payload'])
        need(canonical(expanded)==canonical(events),'whole saved auxiliary/native event bytes')
        saved=self.store.get(self.store.roots[f'slot_{slot:02}'])
        need(canonical(saved)==canonical(result),'whole saved result DAG bytes')
        return original.validate_tree(task,saved,expanded,permit_fixture=permit_fixture,
            capsule_reader=lambda cid,call:self.capsule(slot,cid,call))


class RamExecution:
    """Actual original12/1728 loop with injected RAM callbacks; NO HOST entry.

    Complete journals/results are retained in RAM pending the online codec.
    This class deliberately cannot be promoted to a production Store.
    """
    def __init__(self): self.events=[]; self.results={}; self.pointers={}
    def event(self, value): self.events.append(deepcopy(value))
    def capsule(self, slot, cid, call):
        p=pointer(slot,cid,call); identity=(slot,cid)
        need(identity not in self.pointers or self.pointers[identity]==p,'immutable logical source pointer')
        self.pointers[identity]=p; return p
    def result(self, slot, value):
        need(slot not in self.results,'one result per original slot')
        self.results[slot]=deepcopy(value)
        return dict(slot=slot,expanded_canonical_sha256=semantic(value))
    def result_reader(self, slot, ref):
        need(ref==dict(slot=slot,expanded_canonical_sha256=semantic(self.results[slot])),
             'whole own RAM result reference')
        return deepcopy(self.results[slot])
    def run(self, tasks, factory):
        need(type(tasks) is list and len(tasks)==2,'fixed known2 literal tasks')
        p=dict(tasks=tasks,profile=original.profile())
        rows=original.execute(p,factory,self,permit_ram=True)
        need(len(rows)==12 and len(original.roster())*144==1728,'unchanged12/1728')
        return p,rows


class CatalogEncoder(codec.Codec):
    """Explicit event encoder adapter. It shares CAS, not old module globals.

    All original subvalues are restored; only exact previously registered
    Catalog values can replace large nested transcripts/trees with W/DAG refs.
    Unknown literal structures are not granted free budget by this adapter;
    the complete production budget/router remains a separate pending slice.
    """
    KINDS=('transcript','ledger','task','context','native_literal','static',*routing.SCHEMAS)
    def __init__(self,store,catalog):
        self.store,self.catalog=store,catalog;self.codes=store.codes
    def intern(self,raw):return self.store.intern(raw)
    def _blob(self,value):return self.store._blob(value)
    def get(self,key,*args):return self.store.get(key,*args)
    def put(self,value,_depth=0):
        need(_depth<128,'bounded explicit reference encoder')
        if type(value) in (dict,list):
            identity=semantic(value)
            for kind in self.KINDS:
                if (kind,identity) in self.catalog.values:return self.catalog.get(kind,value)
            if type(value) is dict:
                if set(value)=={'shape','dtype','sha256','data'}:return self.store._blob(value)
                need(all(k in self.codes for k in value),'sealed event field table')
                return self.intern(b'M'+struct.pack('<H',len(value))+b''.join(
                    struct.pack('<H',self.codes[k])+self.put(v,_depth+1) for k,v in value.items()))
            if codec.rectangle(value) is None:
                return self.intern(b'L'+struct.pack('<I',len(value))+b''.join(self.put(v,_depth+1) for v in value))
        return self.store.put(value)


def coverage():
    return dict(schema=NAME+'.coverage',
        implemented=['actual complete call -> original audit_native -> typed native W view',
                     'native output/boundary -> context/J Catalog with explicit origin',
                     'resource dispositions -> ledger W view',
                     'virtual capsule reader -> original full saved reconstruction',
                     'original twelve-slot control flow through injected RAM callbacks'],
        missing=['online auxiliary payload routing before append',
                 'durable full-order journal and pending-call recovery',
                 'complete transport/failure reserve integer proof',
                 'production source/model/class/precision and post-fsync completion wrapper',
                 'production DiskCAS runner and complete physical-file consumer'],
        complete_transport_fit_proved=False,whole_output_bytes=4*1024**3,
        reserve_bytes=512*1024**2,request_denominator=1728,slots=12,resource_axes=27,**FLAGS)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['coverage'])
    parser.parse_args(); print(canonical(coverage()).decode())


if __name__=='__main__': main()
