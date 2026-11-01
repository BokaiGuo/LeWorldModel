"""Explicit spectral8 bank from parent-bound saved raw-zero J. No NN runner."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from experiments.spectral_reachability import agent_p3_parent_native_adapter_20261029_v1 as parent
from experiments.spectral_reachability import agent_p3_parent_r_bank_evaluator_20261029_v1 as kernel
from experiments.spectral_reachability import agent_p3_local_signal_cuda_20261028_v1 as signal

core = parent.core
require, digest = core.require, core.digest
NAME = 'agent_p3_r_spectral8_bank_20261029_v1'
ROOT = core.ROOT
PROTOCOL = ROOT/'innerdocs/main_p3_r_spectral8_variant_protocol_20261029_v1.md'
PROTOCOL_SHA = '3a9bddc658204ca837892cf17d69eafe205f7c3f3629c383df81bcadae38792e'
PINS = {
    Path(core.__file__): '72f076106fb8f6c2aa73386c06fecb912f033e81a5108aa1f36885bd0342c2bc',
    Path(parent.__file__): '150a1c4189a68f9b7c71479e5aa6980cea490fdeed0d5524b7bfad0cc78c2558',
    Path(signal.__file__): '6ce7722eca40cdd5f8958ba9590d8dc3d76d38bd10895d5ca494c1db3b6d100b',
    Path(kernel.__file__): '71e4e178eb180a0d9b740697c98538d6f9afd0ba50d35efb03aa23f91378e04d',
    Path(kernel.original.__file__): kernel.SOLVER_SHA256,
    ROOT/'experiments/spectral_reachability/pusht_full_packet_chart_20261027.py':
        '1a833505dbf811c8a59eddd2f272295990433629da7d8ae2299bd37e83894291',
    ROOT/'experiments/spectral_reachability/agent_p3_parent_native_cuda_20261029_v1.py':
        'eeea5710d4171b7c43f99d1f2b5ad2412e03a25ba14e8fc5589ea5ef73f57b3b',
    PROTOCOL: PROTOCOL_SHA,
}
FLAGS = dict(P1_admission=False, P3_admission=False, full_C3_demonstrated=False,
    freshness=False, physical_release=False, original_R_equivalence=False,
    policy_tree_executed=False, equal_FLOPs=False)


def contract():
    return dict(schema=NAME+'.contract', protocol=str(PROTOCOL), protocol_sha256=PROTOCOL_SHA,
        candidates=8, separate_zero_control=True, root_DLS_computed=False,
        horizons=[1,3], radii=list(kernel.original.RADII), order='radius,mode0/1,sign(-1,+1)',
        SVD='float64(Q@original_raw_J), full_matrices=False; one call',
        sign='first argmax(abs(right row)) must be nonnegative',
        map='float64 tanh -> one float32 cast -> time-major reshape(5h,2)',
        J_rescaled=False, raw_edge_normalized_here=False,
        h3='explicit30-dimensional extension; unchanged total radius, not sqrt(h) scaled',
        beam='all eligible to original modelQ beam; no_witness stops branch',
        diagnostic_suffix_horizon=1, suffix_is_executable_edge=False, **FLAGS)


def check_sources():
    for p, expected in PINS.items():
        require(core.sha(p)==expected, 'frozen protocol/source bytes: '+str(p))


def bank_cost():
    # Bank entry plus the only actual numerical heavy operation. No unused DLS.
    v = dict.fromkeys(core.AXES,0)
    v.update(program_compile_calls=1,array_SVD_calls=1)
    return v


def _mapping(root, actions, path, mapping):
    root = core.validate_task(deepcopy(root))
    require(root['root_context']['latent_dim']==192 and
        np.array_equal(root['projection'],np.eye(192)), 'identity192 original task')
    if actions:
        context = core.checked_path(root,root['root_context'],actions,path)
    else:
        require(path==[], 'empty root prefix')
        context = root['root_context']
    task = parent.parent_task(root,context)
    names = ('schema','root_task_sha256','root_binding','parent_task','outer_request',
        'native_request','native_response','outward_response','mapping_sha256')
    core.keys(mapping,names)
    base = {k:deepcopy(v) for k,v in mapping.items() if k!='mapping_sha256'}
    require(mapping['mapping_sha256']==digest(base) and mapping['schema']==parent.NAME+'.signal_map',
        'original full signal mapping digest/schema')
    require(mapping['root_task_sha256']==digest(root) and mapping['root_binding']==root['binding']
        and mapping['parent_task']==task, 'root/local-parent dual identity')
    native = mapping['native_response']
    core.keys(native,('binding_sha256','parent_context_sha256','horizon','nominal','jacobian','producer_sha256'))
    h = native['horizon']
    require(type(h) is int and h in (1,3), 'original H1/H3 only')
    nreq = core.request(task,dict(context=context),operation='compile_finite_horizon_J',horizon=h)
    oreq = core.request(root,dict(context=context),operation='compile_finite_horizon_J',horizon=h)
    require(mapping['native_request']==nreq and mapping['outer_request']==oreq,
        'original exact native/outer requests, not root J relabelled')
    require(native['binding_sha256']==digest(task['binding']) and
        native['parent_context_sha256']==digest(context) and native['producer_sha256']==core.sha(signal.__file__),
        'original native producer/postcontext identity')
    require(mapping['outward_response']==dict(native,binding_sha256=digest(root['binding']),
        producer_sha256=core.sha(parent.__file__)), 'outer response is actual original mapping')
    core.finite(native['nominal'],(192,),True)
    core.finite(native['jacobian'],(192,10*h),True)
    return root,task,h,native


def _evidence(task,h,response,mapping,evidence,permit_ram):
    core.keys(evidence,('kind','artifact_path','artifact_sha256','call_id','native_call',
        'origin_index','nominal_index','jacobian_index','signal_compile_entry'))
    kind = evidence['kind']
    require(kind=='HOST_CUDA' or (kind=='RAM_FIXTURE' and permit_ram is True), 'explicit saved HOST or RAM opt-in')
    require(type(evidence['artifact_path']) is str and Path(evidence['artifact_path']).is_absolute(),
        'explicit absolute artifact pointer; no guessed file access')
    core.hash_value(evidence['artifact_sha256']);core.integer(evidence['call_id'])
    call = evidence['native_call']
    require(type(call) is list and len(call)>=5, 'complete bounded native call')
    require(call[0]==dict(kind='native_call_attempt',method='signal',task=task,
        request=mapping['native_request']) and call[-1]==dict(kind='native_call_return',result=response),
        'actual source call identity/result')
    for index_name, event_kind in [('origin_index','origin'),('nominal_index','nominal'),('jacobian_index','jacobian')]:
        i = evidence[index_name];core.integer(i,1)
        require(i<len(call)-1 and call[i]['kind']==event_kind, 'in-call array origin pointer '+index_name)
    o,n,j = (call[evidence[k]] for k in ('origin_index','nominal_index','jacobian_index'))
    require(o.get('phase')=='nominal', 'raw zero original nominal origin')
    core.keys(o['arrays'],('latent','history','raw_flat','mean','std'))
    original = dict(latent=np.asarray(task['root_context']['latent_history'],np.float32)[None],
        history=np.asarray(task['root_context']['action_history'],np.float32)[None],
        raw_flat=np.zeros(10*h,np.float32),mean=np.asarray(task['action_mean'],np.float32),
        std=np.asarray(task['action_std'],np.float32))
    for k,v in original.items():
        a = signal.unblob(o['arrays'][k])
        require(a.dtype==np.float32 and a.shape==v.shape and a.tobytes()==v.tobytes(), 'original raw/normalizer/context buffer '+k)
    for event, value in [(n,response['nominal']),(j,response['jacobian'])]:
        core.keys(event['arrays'],('value',))
        a = signal.unblob(event['arrays']['value']);v = np.asarray(value,np.float32)
        require(a.dtype==np.float32 and a.shape==v.shape and a.tobytes()==v.tobytes(), 'response equals native array bits')
    entry = evidence['signal_compile_entry']
    core.keys(entry,('index','scope','resource','shape','directions','delta','status','started_ns','ended_ns','error'))
    core.integer(entry['index']);core.integer(entry['started_ns']);core.integer(entry['ended_ns'])
    require(entry['resource']=='signal_compile' and entry['shape'] is None and entry['directions'] is None and
        entry['status']=='returned' and entry['error'] is None and entry['scope'] is not None and
        entry['ended_ns']>=entry['started_ns'] and entry['delta']==core.delta('signal_compile'), 'paid outer signal compile entry')
    source_axes = dict.fromkeys(core.AXES,0)
    if kind=='HOST_CUDA':
        # Only this explicit saved invocation, NOT producer/fullstrict rerun.
        from experiments.spectral_reachability import agent_p3_parent_native_cuda_20261029_v1 as native
        require(native.audit_native(call) is True, 'complete native raw-zero signal IO/AD/resource closure')
        inner = [x['entry'] for x in call[1:-1] if x['kind']=='resource_attempt']
        require(inner and all(x['index']>entry['index'] and x['scope']==entry['scope'] and
            entry['started_ns']<=x['started_ns']<=entry['ended_ns'] for x in inner), 'paid source wrapper encloses all actual child calls')
        for x in inner:
            for k in core.AXES:source_axes[k]+=x['delta'][k]
        for k in core.AXES:source_axes[k]+=entry['delta'][k]
    else:
        # RAM does not claim native metering; these counts remain unknown.
        source_axes = None
    return dict(kind=kind,native_verified=kind=='HOST_CUDA',resources=source_axes,
        signal_compile_entry=deepcopy(entry),artifact_path=evidence['artifact_path'],
        artifact_sha256=evidence['artifact_sha256'],call_id=evidence['call_id'],
        origin_index=evidence['origin_index'],nominal_index=evidence['nominal_index'],jacobian_index=evidence['jacobian_index'],
        native_call_sha256=digest(call),source_artifact_bytes_read=False,
        scope='one supplied saved signal invocation; not root certificate, file authentication or fullstrict')


def build(*,root,prefix_actions,prefix_path,mapping,evidence,ledger,verification_reserve,sink,permit_ram=False):
    """Pure saved/RAM bank build, one actual NumPy SVD; caller owns persistence.

    HOST evidence is checked against one native call, never executes a model.
    artifact pointers are bound, not read or certified as on-disk bytes here.
    Return includes evaluator-compatible authority; production authentication
    remains false pending the future runner's artifact/source closure.
    """
    check_sources()
    require(type(ledger) is core.Ledger and ledger.scope is not None and not ledger.failed and callable(sink), 'scoped active original ledger/sink')
    reserve = core.counters(verification_reserve)
    root,task,h,response = _mapping(root,prefix_actions,prefix_path,mapping)
    source = _evidence(task,h,response,mapping,evidence,permit_ram)
    cost = bank_cost();before = deepcopy(ledger.attempted);completed_before=deepcopy(ledger.completed)
    result = dict(schema=NAME+'.bank',status='not_started',rule=contract(),horizon=h,
        root=deepcopy(root),parent_task=task,prefix_actions=deepcopy(prefix_actions),prefix_path=deepcopy(prefix_path),
        mapping=deepcopy(mapping),evidence=deepcopy(evidence),source=source,
        resources_before=before,completed_before=completed_before,resources_expected=cost,verification_reserve=reserve,
        params=None,actions=None,zero=None,authority=None,events=[],**FLAGS)
    def event(v):
        e=dict(index=len(result['events']),bank=NAME,payload=deepcopy(v))
        result['events'].append(e);sink(deepcopy(e))
    if any(before[k]+cost[k]+reserve[k]>ledger.caps[k] for k in core.AXES):
        result.update(status='resource_cap_before_bank',resources=dict.fromkeys(core.AXES,0),
            completed=dict.fromkeys(core.AXES,0),ledger_events=[])
        event(dict(kind='atomic_refusal',cost=cost,reserve=reserve))
        return result
    start = len(ledger.events)
    try:
        event(dict(kind='bank_attempt',mapping_sha256=mapping['mapping_sha256'],source=source))
        def generate():
            q = np.asarray(task['projection'],np.float64)
            raw_j = np.asarray(response['jacobian'],np.float64)
            projected = q@raw_j  # normalizer is ALREADY in raw J; no std scaling.
            result['projected_J_float64'] = projected.tolist()
            left,singular,right_raw = ledger.call('array_svd',lambda:np.linalg.svd(projected,full_matrices=False))
            result.update(left_vectors=left.tolist(),singular_values=singular.tolist(),right_vectors_original=right_raw.tolist())
            right = right_raw.copy()
            for row in right:
                if row[np.argmax(np.abs(row))]<0:row*=-1
            result['right_vectors'] = right.tolist()
            params=[];kinds=[];directions=[]
            for radius in kernel.original.RADII:
                for mode in range(2):
                    for sign in (-1,1):
                        params.append(sign*radius*right[mode])
                        kinds.append(f'spectral_mode_{mode}_radius_{radius:.8f}_sign_{sign}')
                        directions.append(dict(index=len(params)-1,radius=radius,mode=mode,sign=sign))
            p = np.stack(params)
            raw = np.tanh(p).astype(np.float32).reshape(8,5*h,2)
            result.update(params=p.tolist(),actions=raw.tolist(),zero=np.zeros((5*h,2),np.float32).tolist(),
                kinds=kinds,directions=directions)
        ledger.call('program_compile',generate)
        require({k:ledger.attempted[k]-before[k] for k in core.AXES}==cost, 'one actual SVD/compile, no DLS or hidden refund')
        require({k:ledger.completed[k]-completed_before[k] for k in core.AXES}==cost, 'all bank operations actually returned')
        manifest=dict(schema=NAME+'.authority',root_task_sha256=digest(root),parent_task_sha256=digest(task),
            parent_context_sha256=digest(task['root_context']),signal_mapping_sha256=mapping['mapping_sha256'],
            native_response_sha256=digest(response),evidence_sha256=digest(evidence),protocol_sha256=PROTOCOL_SHA,
            bank_sha256=digest(result['actions']),zero_sha256=digest(result['zero']),params_sha256=digest(result['params']),
            bank_ledger_events_sha256=digest(ledger.events[start:]),bank_resource_vector=cost,
            numeric_environment=dict(numpy_version=np.__version__,float64_dtype=np.dtype(np.float64).str,
                float32_dtype=np.dtype(np.float32).str),
            sources={str(p):core.sha(p) for p in PINS},generator_path=str(Path(__file__).resolve()),
            generator_sha256=core.sha(__file__),source_native_verified=source['native_verified'],
            bank_artifact_authenticated=False)
        result['authority_manifest']=manifest
        # Compatible explicit input for RAM evaluator; this never self-signs a
        # producer artifact just because the in-memory algebra is correct.
        result['authority']=dict(source_sha256=digest(manifest),bank_sha256=manifest['bank_sha256'],
            zero_sha256=manifest['zero_sha256'],root_task_sha256=digest(root),
            parent_context_sha256=digest(task['root_context']),goal_sha256=digest(task['goal']),
            Q_sha256=digest(task['projection']),normalizer_sha256=digest([task['action_mean'],task['action_std']]),
            action_domain_sha256=task['binding']['action_domain_sha256'],bank_authenticated=False,authentication_sha256=None)
        result['status']='complete'
        event(dict(kind='bank_return',authority_manifest=manifest))
    except Exception as exc:
        result.update(status='partial_failure',error=dict(type=type(exc).__name__,message=str(exc)))
        event(dict(kind='bank_failure',error=result['error']))
    result['ledger_events']=deepcopy(ledger.events[start:])
    result['resources']={k:ledger.attempted[k]-before[k] for k in core.AXES}
    result['completed']={k:ledger.completed[k]-completed_before[k] for k in core.AXES}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',action='store_true')
    if p.parse_args().contract:print(core.canonical(contract()).decode())


if __name__=='__main__':main()
