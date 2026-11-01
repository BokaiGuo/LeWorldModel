"""Chunk native model/callback events into the four-slot bounded journal.

Serialization/transport only; does not authenticate neural calls or physical IO.
"""
import base64
import hashlib
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as storage

MAX_EVENT = 8*1024**2
CHUNK = 6000
need = storage.need


class Bridge:
    def __init__(self, store, slot):
        need(type(slot) is int and 0 <= slot < 4, 'fixed slot')
        self.store = store; self.slot = slot; self.count = 0

    def event(self, payload):
        raw = storage.base.raw(payload)
        need(0 < len(raw) <= MAX_EVENT, 'bounded native event')
        digest = hashlib.sha256(raw).hexdigest()
        parts = [raw[i:i+CHUNK] for i in range(0,len(raw),CHUNK)]
        for index,part in enumerate(parts):
            detail = dict(slot=self.slot,sequence=self.count,index=index,total=len(parts),bytes=len(raw),
                          sha256=digest,data=base64.b64encode(part).decode('ascii'))
            self.store.event('numerical','record','matched_numeric_chunk',self.store.index,detail)
        self.count += 1


def reconstruct(events, slot):
    """Recover complete serialized events, refusing missing/interleaved chunks."""
    need(type(slot) is int and 0 <= slot < 4, 'fixed slot')
    chunks = [e['detail'] for e in events if e['origin']=='numerical' and
              e['kind']=='record' and e['resource']=='matched_numeric_chunk' and e['detail'].get('slot') == slot]
    output = []; i = 0
    while i < len(chunks):
        first = chunks[i]; n = first['total']; size = first['bytes']; digest = first['sha256']
        need(type(n) is int and n > 0 and type(size) is int and 0 < size <= MAX_EVENT and
             n == (size+CHUNK-1)//CHUNK and i+n <= len(chunks), 'complete native chunks')
        parts = []
        for j in range(n):
            d = chunks[i+j]
            need(set(d) == {'slot','sequence','index','total','bytes','sha256','data'} and
                 d['slot'] == slot and d['sequence'] == len(output) and d['index'] == j and
                 d['total'] == n and d['bytes'] == size and d['sha256'] == digest, 'chunk order and identity')
            part = base64.b64decode(d['data'],validate=True)
            need(len(part) == (CHUNK if j < n-1 else size-(n-1)*CHUNK), 'chunk byte count')
            parts.append(part)
        raw = b''.join(parts)
        need(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest, 'reassembled native hash')
        payload = storage.base.decode(raw)
        need(type(payload) is dict, 'native event dictionary')
        output.append(payload); i += n
    return output
