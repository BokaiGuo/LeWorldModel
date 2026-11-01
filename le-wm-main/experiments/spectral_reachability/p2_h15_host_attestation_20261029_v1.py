"""P2 real NativeTree callback binding and pre-dispatch checks; HOST disabled.

This module never loads a model or chooses a device. A caller supplies an
already loaded model, but native neural calls remain unconditionally blocked.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json

from experiments.spectral_reachability import p2_h15_bridge_lifecycle_20261029_v1 as bridge
from experiments.spectral_reachability import p2_h15_sra_host_candidate_20261029_v1 as single
from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
from experiments.spectral_reachability import agent_p3_runtime_contract_20261028_v1 as core

NAME='p2_h15_host_attestation_20261029_v1'
model_hash=native.signal.model_hash


def need(ok, message):
    if not ok: raise ValueError(message)


def budget(): return single.caps()


def historical_inventory():
    host=bridge.source.prep.read(bridge.source.HOST, bridge.source.HOST_SHA)
    return deepcopy(host['bindings_sha256'])


def verify_sources(files):
    need(type(files) is dict and bool(files), 'nonempty source inventory')
    for name, expected in files.items():
        p=Path(name)
        core.hash_value(expected)
        need(p.is_absolute() and p.is_file() and not p.is_symlink() and bridge.sha(p)==expected,
             'frozen source mismatch: '+name)
    return core.digest(files)


def preflight():
    mapped=bridge.build_bridge()
    files=historical_inventory()
    source_digest=verify_sources(files)
    local={str(Path(m.__file__).resolve()):bridge.sha(m.__file__) for m in (bridge, single, native)}
    local[str(Path(__file__).resolve())]=bridge.sha(__file__)
    return dict(schema=NAME+'.preflight',bridge_sha256=core.digest(mapped),
                historical_inventory_sha256=source_digest,inventory_file_count=len(files),
                additional_sources=local,budget_per_root=budget(),fixed_denominator=2,
                runtime_model_verified=False,runtime_precision_verified=False,
                runtime_budget_verified=False,host_authorized=False,physical_release=False,
                limitation='disk source rehash is not runtime class, model, precision or call attestation')


def save_preflight(path):
    single._put(Path(path),preflight())
    return bridge.sha(path)


def consume_preflight(path,expected):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=1024*1024, 'bounded preflight')
    raw=path.read_bytes(); need(hashlib.sha256(raw).hexdigest()==expected, 'external preflight SHA')
    value=json.loads(raw)
    need(raw==core.canonical(value)+b'\n' and value==preflight(), 'preflight rebuild')
    return value


class Session:
    def __init__(self, mapped, model, device, torch, files, expected_model, *, fixture=False):
        self.mapped=deepcopy(mapped); self.model=model; self.device=device; self.torch=torch
        self.files=deepcopy(files); self.fixture=fixture; self.closed=False
        need(type(fixture) is bool and getattr(device,'type',None)=='cuda', 'native cuda device ABI')
        need(mapped==bridge.build_bridge() and mapped['host_authorized'] is False, 'two-root bridge rebuild')
        if not fixture:
            need(self.files==historical_inventory(), 'historical complete source inventory')
            need(expected_model['original_state_dict']==mapped['roots'][0]['task']['binding']['model_sha256'], 'frozen expected state dict')
        self.source_before=verify_sources(self.files)
        self.expected=deepcopy(expected_model)
        self.model_before=model_hash(model)
        need(self.model_before==self.expected, 'model hash/buffer mismatch')
        self.precision_before=native.checked_precision(torch)
        if not fixture:
            self.classes_before=native.model_sources(model,dict(sources=self.files))
        else: self.classes_before=None
        self.check()

    def check(self):
        need(not self.closed, 'session closed')
        need(verify_sources(self.files)==self.source_before, 'source changed')
        need(model_hash(self.model)==self.model_before, 'model or buffers changed')
        need(native.checked_precision(self.torch)==self.precision_before, 'precision changed')
        need(all(not x.training for x in self.model.modules()) and
             all(not x.requires_grad and x.device==self.device for x in self.model.parameters()) and
             all(x.device==self.device for x in self.model.buffers()), 'frozen eval model/device')
        if not self.fixture:
            need(native.model_sources(self.model,dict(sources=self.files))==self.classes_before, 'model class sources changed')

    def callbacks(self, root, sink):
        self.check()
        need(type(root) is int and root in (0,1), 'fixed root')
        task=self.mapped['roots'][root]['task']
        backend=native.NativeTree(self.model,self.device,task,sink)
        return Callbacks(self,root,backend)

    def close(self):
        self.check(); self.closed=True
        return dict(model_before=self.model_before,model_after=model_hash(self.model),
                    precision_before=self.precision_before,precision_after=native.checked_precision(self.torch),
                    sources_before=self.source_before,sources_after=verify_sources(self.files),
                    fixture_only=self.fixture,host_authorized=False,physical_release=False)


class Callbacks:
    def __init__(self, session, root, backend):
        self.session=session; self.root=root; self.backend=backend

    def _check(self, ledger):
        self.session.check()
        need(self.backend.task==self.session.mapped['roots'][self.root]['task'], 'callback root/task changed')
        need(isinstance(ledger,core.Ledger) and ledger.caps==budget(), 'exact fixed paid budget')
        # No flag, user argument or test double can authorize this path.
        raise ValueError('HOST dispatch not authorized: missing full lifecycle and runtime closure')

    def signal(self, request, ledger):
        self._check(ledger)
        result=self.backend.signal(request,ledger)
        self.session.check()
        return result

    def replay(self, parent_context, actions, segmented, ledger):
        self._check(ledger)
        result=self.backend.replay(parent_context,actions,segmented,ledger)
        self.session.check()
        return result


def bind_roots(session, sink_factory):
    """Bind both original ABI tasks to separate callbacks and nonshared ledgers."""
    result=[]
    for root in (0,1):
        sink=sink_factory(root)
        need(callable(getattr(sink,'event',None)), 'event sink')
        result.append(dict(root=root,callbacks=session.callbacks(root,sink),
                           ledger=core.Ledger(budget(),sink=sink.event)))
    return result


def host_dispatch(*args,**kwargs):
    raise ValueError('HOST dispatch not authorized')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sha256')
    a=p.parse_args(); need(a.output.name.startswith(NAME+'_'), 'exclusive namespace')
    if a.command=='prepare':
        need(a.sha256 is None,'unexpected SHA'); digest=save_preflight(a.output)
    else:
        need(a.sha256 is not None,'external SHA required'); consume_preflight(a.output,a.sha256); digest=a.sha256
    print(json.dumps(dict(path=str(a.output),sha256=digest,host_authorized=False)))


if __name__=='__main__': main()
