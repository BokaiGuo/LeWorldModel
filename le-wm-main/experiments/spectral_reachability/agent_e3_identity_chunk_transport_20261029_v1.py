"""Bounded identity binding chunks for v3 preflight; no model execution on import."""
import base64
import hashlib
import secrets
from pathlib import Path
from experiments.spectral_reachability import agent_e3_encoder_identity_20261029_v1 as old

base = old.launch.pub.base
need = old.need
CHUNK = 6000
MAX_RAW = 16 * 1024**2
MAX_CHUNKS = (MAX_RAW + CHUNK - 1) // CHUNK
MAX_EVENT = 16385


class Binding(old.Binding):
    def __init__(self, model, device, model_sha, prepare_sha, store):
        need(getattr(device, 'type', None) == 'cuda' and type(model_sha) is str and len(model_sha) == 64
             and type(prepare_sha) is str and len(prepare_sha) == 64, 'HOST identity inputs')
        self.model = model; self.device = device; self.store = store
        self.modules = {n: getattr(model, n) for n in ('encoder', 'projector')}
        need(self.modules['encoder'] is not self.modules['projector'], 'distinct module objects')
        self.params = old._inventory(model)
        need(all(p['device'] == str(device) for p in self.params), 'exact HOST parameter device')
        self.tag = secrets.token_hex(32)
        self.descriptor = dict(schema=old.NAME+'.binding', tag=self.tag, prepare_sha256=prepare_sha,
            model_sha256=model_sha, model_instance=hex(id(model)), device=str(device),
            modules={n: hex(id(v)) for n, v in self.modules.items()}, parameters=self.params)
        write_binding(store, self.descriptor)


def write_binding(store, descriptor):
    raw = base.raw(descriptor)
    need(0 < len(raw) <= MAX_RAW, 'identity binding raw bound')
    digest = hashlib.sha256(raw).hexdigest()
    parts = [raw[i:i+CHUNK] for i in range(0, len(raw), CHUNK)]
    # A conservative record ceiling, checked against the existing category and whole-job limits.
    store.check('physical_events', (len(parts)+1)*MAX_EVENT)
    for index, part in enumerate(parts):
        store.event('physical', 'record', 'preflight_binding_chunk', store.index,
                    dict(index=index, total=len(parts), sha256=digest, data=base64.b64encode(part).decode('ascii')))
    store.event('physical', 'record', 'preflight_binding_seal', store.index,
                dict(total=len(parts), bytes=len(raw), sha256=digest))


def load_bound(prepared, torch, device, store, prepare_sha):
    model, model_sha = old.launch.host.primitives.native.loader.loaded_model(
        prepared, torch, device, old.launch.loader_bridge.LoaderBridge(store), prepare_sha)
    need(model_sha == prepared['expected_model_hash'], 'original loaded model hash')
    return Binding(model, device, model_sha, prepare_sha, store)


def reconstruct(events, allow_partial=False):
    chunks = [e for e in events if e['origin']=='physical' and e['resource']=='preflight_binding_chunk']
    seals = [e for e in events if e['origin']=='physical' and e['resource']=='preflight_binding_seal']
    need(not any(e['origin']=='physical' and e['resource']=='encoder_identity_bind' for e in events),
         'no legacy bind in v3')
    need(len(seals) <= 1 and len(chunks) <= MAX_CHUNKS, 'binding chunk count')
    if not seals:
        need(allow_partial, 'complete binding seal required')
        for i,e in enumerate(chunks):
            x=e['detail']
            need(e['kind']=='record' and set(x)=={'index','total','sha256','data'}
                 and type(x['index']) is int and x['index']==i and type(x['total']) is int
                 and i<x['total']<=MAX_CHUNKS and type(x['sha256']) is str and len(x['sha256'])==64
                 and all(c in '0123456789abcdef' for c in x['sha256'])
                 and (i==0 or (e['index']==chunks[i-1]['index']+1
                      and x['total']==chunks[0]['detail']['total'] and x['sha256']==chunks[0]['detail']['sha256'])),
                 'partial binding chunk chronology')
            part=base64.b64decode(x['data'],validate=True)
            need(len(part)==CHUNK if i<x['total']-1 else 0<len(part)<=CHUNK, 'partial chunk length')
        return None
    seal = seals[0]; d = seal['detail']
    need(seal['kind']=='record' and set(d)=={'total','bytes','sha256'} and type(d['total']) is int
         and type(d['bytes']) is int and 0 < d['bytes'] <= MAX_RAW
         and d['total']==len(chunks)==(d['bytes']+CHUNK-1)//CHUNK, 'complete binding chunks')
    parts=[]
    for i,e in enumerate(chunks):
        x=e['detail']
        need(e['kind']=='record' and set(x)=={'index','total','sha256','data'} and type(x['index']) is int
             and x['index']==i and x['total']==d['total'] and x['sha256']==d['sha256']
             and e['index'] < seal['index'] and (i==0 or e['index']==chunks[i-1]['index']+1), 'binding chunk order')
        part=base64.b64decode(x['data'], validate=True)
        need(len(part)==CHUNK if i<len(chunks)-1 else 0<len(part)<=CHUNK, 'binding chunk length')
        parts.append(part)
    need(chunks[-1]['index']+1==seal['index'], 'binding seal chronology')
    raw=b''.join(parts)
    need(len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'], 'binding chunk hash')
    descriptor=base.decode(raw)
    need(base.raw(descriptor)==raw, 'canonical binding JSON')
    synthetic=dict(seal, resource='encoder_identity_bind', detail=descriptor)
    result=[e for e in events if e not in chunks and e is not seal]
    result.append(synthetic); result.sort(key=lambda e:e['index'])
    return result


def verify(events, prepared, prepare_sha):
    return old.verify(reconstruct(events), prepared, prepare_sha)
