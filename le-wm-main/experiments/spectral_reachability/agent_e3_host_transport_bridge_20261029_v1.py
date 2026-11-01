"""Lossless HOST numeric array/event bridge onto unified transport.

This does not authenticate source/model/RGB or provide whole-experiment strict.
"""
from copy import deepcopy
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import agent_e3_unified_transport_20261029_v1 as transport

HOST_KEYS={'file','sha256','shape','dtype'}
UNIFIED_KEYS=HOST_KEYS|{'origin','role'}


def host_descriptor(d):
    transport.need(type(d) is dict and set(d)==UNIFIED_KEYS and d['origin']=='numerical'
                   and d['role']=='host_native','exact own numerical descriptor')
    return {k:deepcopy(d[k]) for k in ('file','sha256','shape','dtype')}


class Bridge:
    def __init__(self,store):self.store=store;self.output=store.output;self.descriptors={}
    def check(self):self.store.check('numerical_arrays',0)
    def array(self,value):
        full=self.store.array('numerical','host_native',value);short=host_descriptor(full)
        transport.need(short['file'] not in self.descriptors,'unique array file')
        self.descriptors[short['file']]=full;return short
    def expand(self,x):
        if type(x) is dict:
            if set(x)==HOST_KEYS:
                full=self.descriptors.get(x['file']);transport.need(full is not None and host_descriptor(full)==x,'owned exact HOST descriptor')
                return deepcopy(full)
            return {k:self.expand(v) for k,v in x.items()}
        if type(x) is list:return [self.expand(v) for v in x]
        return deepcopy(x)
    def event(self,value):
        transport.need(type(value) is dict and type(value.get('kind')) is str,'HOST native event')
        self.store.event('numerical','record','host_native',self.store.index,{'native':self.expand(value)})


def saved_native_events(events):
    """Recover exact HOST payload only from scoped own records, preserving order."""
    def reduce(x):
        if type(x) is dict:
            if set(x)==UNIFIED_KEYS:return host_descriptor(x)
            return {k:reduce(v) for k,v in x.items()}
        if type(x) is list:return [reduce(v) for v in x]
        return deepcopy(x)
    result=[]
    for e in events:
        if e['origin']!='numerical':continue
        transport.need(e['kind']=='record' and e['resource']=='host_native' and set(e['detail'])=={'native'},
                       'all numeric records are original HOST events')
        result.append(reduce(e['detail']['native']))
    return result
