"""Journal each preflight encoder array; partial transport is not scientific strict."""
import io
import numpy as np
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub

base = pub.base
RESOURCE = 'preflight_array_write'
MAX_EVENTS_BYTES = 28 * 16385


class Store(pub.Store):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array_attempts = 0

    def persist_array(self, name, data):
        self.write(name, data, 'encoding_arrays')

    def array(self, origin, role, value):
        if origin != 'encoding':
            return super().array(origin, role, value)
        base.need(self.array_attempts < 14, 'preflight array attempt budget')
        a = np.asarray(value)
        base.need((role, a.shape, a.dtype.str) in base.ENC_SHAPES and np.isfinite(a).all(), 'encoder array')
        buf = io.BytesIO()
        np.save(buf, a, allow_pickle=False)
        data = buf.getvalue()
        base.need(len(data) <= a.nbytes + 4096, 'bounded NPY')
        name = f'a{self.number:06}.npy'
        descriptor = dict(file=name, sha256=base.fsha(data), shape=list(a.shape), dtype=a.dtype.str,
                          origin=origin, role=role)
        plan = dict(planned_file=name, planned_sha256=descriptor['sha256'], shape=descriptor['shape'],
                    dtype=descriptor['dtype'], origin=origin, role=role)
        token = self.index
        self.event('physical', 'intent', RESOURCE, token, plan)
        self.array_attempts += 1
        try:
            self.persist_array(name, data)
            self.number += 1
            self.event('physical', 'return', RESOURCE, token, dict(descriptor=descriptor))
        except BaseException as exc:
            try:
                path = self.output / name
                saved = None
                if path.is_file() and not path.is_symlink():
                    actual = base.read_bytes(path, len(data))
                    if actual == data:
                        saved = descriptor
                        if self.number == int(name[1:7]):
                            self.number += 1
                self.event('physical', 'failure', RESOURCE, token,
                           dict(descriptor=saved, error_type=type(exc).__name__, message=str(exc)[:2048]))
            except BaseException as secondary:
                exc.add_note('array failure journal unavailable: ' + repr(secondary))
            raise
        return descriptor


def verify(events):
    pending = {}
    seen = set()
    attempted = returned = failed = 0
    saved = []
    for e in events:
        if e['origin'] != 'physical' or e['resource'] != RESOURCE:
            continue
        token, kind, detail = e['token'], e['kind'], e['detail']
        if kind == 'intent':
            base.need(type(token) is int and token == e['index'] and token not in seen and not pending
                      and set(detail) == {'planned_file','planned_sha256','shape','dtype','origin','role'}, 'array intent')
            seen.add(token)
            pending[token] = detail
            attempted += 1
        else:
            base.need(kind in ('return', 'failure') and token in pending, 'array terminal')
            plan = pending.pop(token)
            base.need(set(detail) == ({'descriptor'} if kind == 'return' else {'descriptor','error_type','message'}), 'array terminal schema')
            descriptor = detail['descriptor']
            if kind == 'return':
                base.need(descriptor is not None, 'returned array descriptor')
                returned += 1
            else:
                base.need(type(detail['error_type']) is str and type(detail['message']) is str, 'array failure')
                failed += 1
            if descriptor is not None:
                base.need(descriptor == dict(file=plan['planned_file'], sha256=plan['planned_sha256'],
                    shape=plan['shape'], dtype=plan['dtype'], origin=plan['origin'], role=plan['role']), 'array descriptor join')
                saved.append(descriptor)
    base.need(attempted <= 14 and len({d['file'] for d in saved}) == len(saved), 'bounded unique array inventory')
    indexed = {d['file']: d for d in saved}
    def check_refs(value):
        if type(value) is dict:
            if set(value) == {'file','sha256','shape','dtype','origin','role'} and value['origin'] == 'encoding':
                base.need(indexed.get(value['file']) == value, 'encoding descriptor without array ledger')
            else:
                for child in value.values():
                    check_refs(child)
        elif type(value) is list:
            for child in value:
                check_refs(child)
    for event in events:
        check_refs(event['detail'])
    return dict(attempted=attempted, returned=returned, failed=failed, pending=len(pending), saved_arrays=saved)
