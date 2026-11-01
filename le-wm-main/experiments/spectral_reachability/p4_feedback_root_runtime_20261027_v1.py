"""Root-parameterized sealed-source adapter; no neural/physics driver.

All runtime actions are external: this module validates observations, produces
planning requests, seals selections via a caller's durable callback, and accepts
guarded execution receipts. CLI only validates sources; it never writes a run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import wraps
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.spectral_reachability.d2p_prerun_source_seal import verify_seal

TAG='p4_feedback_root_runtime_20261027_v1'
ARMS=('imagined','actual')
BOUNDARIES=(5,10,15)
MAX_FILE=16*1024*1024
MEAN=np.array([-.0087,.0068],np.float32)
STD=np.array([.2019,.2002],np.float32)


def require(ok,message):
    if not ok: raise ValueError(message)


def content_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def digest(path):
    path=Path(path).resolve(); require(path.is_file() and path.stat().st_size<=1024**3,'bounded source required')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def read(path):
    path=Path(path).resolve(); require(path.stat().st_size<=MAX_FILE,'small saved source required')
    def unique(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key'); result[k]=v
        return result
    value=json.loads(path.read_bytes(),object_pairs_hook=unique)
    content_sha(value); return value


def array(value,shape):
    a=np.asarray(value)
    require(a.shape==shape and a.dtype.kind in 'fiu' and np.isfinite(a).all(),'finite array '+str(shape))
    return a


def raw_packet(value):
    a=array(value,(5,2)); require(np.all(np.abs(a)<=1),'bounded raw packet'); return a


def normalize(raw):
    a=array(raw,(2,5,2)).astype(np.float32)
    require(np.all(np.abs(a)<=1),'bounded raw history')
    return (a.reshape(2,10)-np.tile(MEAN,5))/np.tile(STD,5)


def context(c):
    require(c['layout']=='states_with_between_state_actions_v1' and
        (c['context_length'],c['latent_dim'],c['action_dim'])==(3,192,10),'native context metadata')
    return array(c['latent_history'],(3,192)),array(c['action_history'],(2,10))


def rgb(value):
    a=np.asarray(value)
    require(a.shape==(224,224,3) and a.dtype==np.uint8,'exact actual uint8 RGB'); return a


def rgb_sha(value): return hashlib.sha256(np.ascontiguousarray(rgb(value)).tobytes()).hexdigest()


def load_png(path):
    require(Path(path).stat().st_size<=1024*1024,'PNG size cap')
    with Image.open(path) as im:
        require(im.mode=='RGB' and im.size==(224,224),'original RGB PNG')
        return rgb(np.asarray(im).copy())


def root_id(value):
    require(isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value),'root SHA256 required')
    return value


def ref(path,expected):
    path=Path(path).resolve(); require(digest(path)==expected,'pinned source SHA: '+str(path))
    return path


def bind_sources(*,root_sha256,capability_path,capability_sha256,response_path,response_sha256,
                 response_prepare_path,response_prepare_sha256,prior_root_sha256s,prior_batch_refs,excluded_root_sha256s=()):
    """Read-only source admission. Pinned SHA values must come from MAIN review.

    Supports root-specific v1 capability/response FORMAT, not old fixed-root
    producers as a generic generator. It does not manufacture a new source pack.
    """
    root=root_id(root_sha256)
    priors=[root_id(v) for v in prior_root_sha256s]; excluded=[root_id(v) for v in excluded_root_sha256s]
    require(priors and len(priors)==len(set(priors)),'explicit nonempty first/prior root exclusion roster')
    require(root not in set(priors+excluded),'same/prior/excluded root cannot supply second batch')
    require(prior_batch_refs,'pinned actual prior-batch reports required, not a caller-invented root list')
    prior_bindings={}; actual_prior_roots=set()
    for prior_ref in prior_batch_refs:
        path=ref(prior_ref['path'],prior_ref['sha256']); report=read(path)
        require(report['schema']=='p4_multiround_feedback_runtime_20261027_v1' and report['status']=='calibration_recorded' and
            not report['failures'] and len(report['rows'])==2 and {r['arm'] for r in report['rows']}==set(ARMS) and
            all(r['status']=='completed' and r['completed_packets']==4 and r['selected_returns']==20 for r in report['rows']),
            'complete real prior-batch schema/census')
        roots={root_id(r['root_sha256']) for r in report['rows']}
        require(len(roots)==1,'two prior arms are one root, not two batches')
        actual_prior_roots.update(roots); prior_bindings[str(path)]=prior_ref['sha256']
    require(actual_prior_roots==set(priors) and len(prior_bindings)==len(prior_batch_refs),'prior-root roster matches pinned batch identities')
    cp=ref(capability_path,capability_sha256); rp=ref(response_path,response_sha256)
    pp=ref(response_prepare_path,response_prepare_sha256)
    cap,response,prepared=read(cp),read(rp),read(pp)
    require(cp.parent!=rp.parent and pp.parent==rp.parent,'separate capability/response source chain')
    require(cap['schema']=='pusht_geometry_capability_20261027_v1' and cap['status']=='diagnostic_recorded' and
        cap['gates'] and all(v is True for v in cap['gates'].values()),'complete root-specific capability')
    require(response['schema']=='pusht_geometry_feedback_response_20261027_v1' and
        response['root_sha256']==root and Path(response['capability']).resolve()==cp and
        response['bindings'][str(cp)]==capability_sha256,'capability->response root/hash edge')
    require(root in cap['census']['qualified_roots'] and cap['census']['complete'] is True,'root-specific qualification census')
    cap_protocol=cp.parent/'protocol.json'
    require(digest(cap_protocol)==cap['protocol_sha256'],'capability protocol SHA')
    plans=read(cap_protocol)['plans']
    plans=[p for p in plans if p['root_sha256']==root]
    rows=[r for r in cap['rows'] if r['root_sha256']==root]
    require(len(plans)==len(rows)==123 and len({p['method'] for p in plans})==123 and
        {r['method'] for r in rows}=={p['method'] for p in plans},'complete same-root 3role41probe source pack')
    require(all(sum(p['role']==role for p in plans)==41 for role in ('selected','greedy','zero')),'three roles, forty-one probes each')
    for plan in plans:
        row=next(r for r in rows if r['method']==plan['method'])
        require(plan['selection_sha256']==content_sha({k:v for k,v in plan.items() if k!='selection_sha256'}) and
            row['selection_sha256']==plan['selection_sha256'] and row['status']=='completed' and
            row['completed_commands']==10 and row['full_horizon_completed'] is True and
            len(row['trace'])==10 and [v['command'] for v in row['trace']]==plan['commands'],'saved capability plan/execution witness')
        array(plan['commands'],(10,2))
    require(prepared['status']=='prepared_not_run' and prepared['protocol_sha256']==response_sha256 and
        prepared['source_seals'] and cap['source_seals'],'root-specific response prepare seal')
    tracked={**prior_bindings,str(cp):capability_sha256,str(rp):response_sha256,str(pp):response_prepare_sha256,str(cap_protocol):digest(cap_protocol)}
    seals=cap['source_seals']+prepared['source_seals']
    for seal in seals:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
        for name in ('manifest.json','args.json'):
            path=Path(seal['output_dir'])/name; tracked[str(path.resolve())]=digest(path)
    for path,h in response['bindings'].items():
        require(digest(path)==h,'response input/model drift: '+path); tracked[str(Path(path).resolve())]=h
    require(response['training'] is False and response['fit'] is False and response['heldout_selection'] is False,
        'no refit or outcome-selected substitute source')
    goal=array(response['goal'],(192,)); q=array(response['q'],(192,192))
    initial,actions=context(response['initial_context'])
    cases=[c for c in response['cases'] if c['role']=='selected']; require(len(cases)==1,'unique selected prefix role')
    case=cases[0]; raw_packet(case['prefix']); incoming=array(case['between_raw'],(2,5,2))
    require(np.array_equal(actions,normalize(incoming)),'original normalized action context')
    selected=next(p for p in plans if p['method']=='selected_zero')
    require(selected['commands'][:5]==case['prefix'],'root-specific selected prefix actions')
    array(case['saved_predicted_prefix'],(192,)); array(case['saved_actual_prefix'],(192,))
    require(selected['saved_candidate']['predicted_prefix']==case['saved_predicted_prefix'],'capability predicted prefix identity')
    stack_path=Path(response['root_rgb']).resolve()
    require(stack_path.stat().st_size<1024*1024,'root RGB archive cap')
    frames=np.load(io.BytesIO(stack_path.read_bytes()),allow_pickle=False)
    require(frames.shape==(3,224,224,3) and frames.dtype==np.uint8,'root RGB3 archive')
    prefix_path=Path(case['prefix_rgb']).resolve(); prefix_image=load_png(prefix_path)
    selected_row=next(r for r in rows if r['method']=='selected_zero')
    identity=selected_row['context_identity']
    require(hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest()==identity['history_rgb_sha256'] and
        np.array_equal(incoming,np.repeat(array(identity['history'],(3,2))[1:,None,:],5,axis=1)), 'capability actual root RGB/raw history identity')
    slots=[v for v in cap['endpoint_receipts'] if v['root_sha256']==root and v['method']=='selected_zero' and v['kind']=='prefix']
    enc=[v for v in cap['encodings'] if v['root_sha256']==root and v['method']=='selected_zero' and v['kind']=='prefix']
    require(len(slots)==len(enc)==1 and slots[0]['status']=='observed' and slots[0]['primitive_step']==5 and
        enc[0]['status']=='completed' and enc[0]['comparable'] is True and enc[0]['vector']==case['saved_actual_prefix'] and
        all(Path(v['rgb_path']).resolve()==prefix_path and v['rgb_sha256']==digest(prefix_path) for v in [slots[0],enc[0]]),
        'capability actual selected-prefix RGB/encoding identity')
    for path in (stack_path,prefix_path):
        require(str(path) in response['bindings'] and digest(path)==response['bindings'][str(path)],'bound actual RGB source')
    policy=Path(response['policy']).resolve()
    models={str(policy/n):response['bindings'][str(policy/n)] for n in ('weights.pt','config.json')}
    runtime_source=Path(__file__).resolve(); tracked[str(runtime_source)]=digest(runtime_source)
    binding=dict(schema=TAG+'.binding',root_sha256=root,prior_root_sha256s=priors,excluded_root_sha256s=excluded,
        prior_batch_refs=deepcopy(prior_batch_refs),
        source_refs=dict(capability_path=str(cp),capability_sha256=capability_sha256,response_path=str(rp),
            response_sha256=response_sha256,response_prepare_path=str(pp),response_prepare_sha256=response_prepare_sha256),
        tracked_sha256=tracked,source_seals=seals,model_bindings=models,root_rgb=str(stack_path),
        root_frame_pixel_sha256=[rgb_sha(f) for f in frames],prefix_pixel_sha256=rgb_sha(prefix_image),
        goal_q_sha256=content_sha([goal.tolist(),q.tolist()]),
        reset_identity_sha256=content_sha(identity),
        runtime_source_sha256=tracked[str(runtime_source)],boundaries=list(BOUNDARIES),
        role='selected',source_pack_validated=True,root_disjoint_from_prior=True,
        fresh_holdout_verified=False,P1_admission=False,P4_admission=False,
        new_neural_calls=0,new_physics_steps=0)
    # Never write protocol/manifest here; caller must externally retain this SHA
    # and seal the small binding together with this NEW adapter before use.
    return binding


def validate_binding(binding,expected_sha):
    require(content_sha(binding)==expected_sha,'runtime binding pinned SHA')
    require(binding['schema']==TAG+'.binding','runtime binding schema')
    rebuilt=bind_sources(root_sha256=binding['root_sha256'],prior_root_sha256s=binding['prior_root_sha256s'],
        prior_batch_refs=binding['prior_batch_refs'],excluded_root_sha256s=binding['excluded_root_sha256s'],**binding['source_refs'])
    require(rebuilt==binding,'capability->response->runtime binding reconstruction')
    return rebuilt


def fail_closed(fn):
    @wraps(fn)
    def wrapper(self,*args,**kwargs):
        require(self.phase!='failed','branch already failed; no retry/refund')
        try: return fn(self,*args,**kwargs)
        except BaseException as exc:
            self.phase='failed'; self.errors.append(dict(stage=fn.__name__,type=type(exc).__name__,message=str(exc))); raise
    return wrapper


class FeedbackAdapter:
    """Single-arm monotonic state machine. No environment/model callbacks.

    MAIN owns guarded execution, truthful encode receipts and durable seal IO.
    A successful mock cannot authorize scientific admission or prove CUDA work.
    """
    def __init__(self,binding,expected_sha,arm,runtime_source_seal):
        self.binding=validate_binding(binding,expected_sha)
        binding=self.binding
        verify_seal(output_dir=runtime_source_seal['output_dir'],expected_manifest_sha256=runtime_source_seal['manifest_sha256'])
        manifest=read(Path(runtime_source_seal['output_dir'])/'manifest.json')
        seal_args=read(Path(runtime_source_seal['output_dir'])/'args.json')
        require(seal_args['binding_sha256']==expected_sha and any(e['source']==str(Path(__file__).resolve()) and
            e['sha256']==binding['runtime_source_sha256'] for e in manifest['files']),'runtime seal binds this root/source protocol')
        self.runtime_source_seal=deepcopy(runtime_source_seal)
        require(arm in ARMS,'explicit feedback arm')
        self.arm=arm; self.root=binding['root_sha256']; self.binding_sha=expected_sha
        self.source=read(binding['source_refs']['response_path'])
        self.case=next(c for c in self.source['cases'] if c['role']=='selected')
        self.frames=np.load(self.source['root_rgb'],allow_pickle=False).copy()
        self.between=np.array(self.case['between_raw'],copy=True)
        self.predicted=deepcopy(self.source['initial_context']); self.last_observed=None
        self.phase='await_prefix'; self.boundary=0; self.pending=None; self.previous_selection=None
        self.errors=[]; self.decisions=[]; self.executions=[]; self.evaluation=None
        self.encoding_receipts=[]

    def identity(self,value,boundary):
        require(value['root_sha256']==self.root and value['arm']==self.arm and
            type(value['boundary']) is int and value['boundary']==boundary,'own root/arm/boundary')

    def execution(self,receipt,packet,start,terminal_rgb):
        self.identity(receipt,start)
        require(receipt['status']=='completed' and type(receipt['completed_commands']) is int and
            receipt['completed_commands']==5 and type(receipt['remaining_commands_not_executed']) is int and receipt['remaining_commands_not_executed']==0 and
            receipt['guard_authorized'] is True,'complete externally guarded five-command packet')
        trace=receipt['trace']
        require(len(trace)==5 and [t['primitive_step'] for t in trace]==list(range(start+1,start+6)) and
            [t['command'] for t in trace]==np.asarray(packet).tolist(),'exact executed packet trace')
        require(receipt['terminal_pixel_sha256']==rgb_sha(terminal_rgb)==trace[-1]['pixel_sha256'],'actual terminal RGB trace')

    def encode_receipt(self,receipt,value,frames,purpose):
        self.identity(receipt,self.boundary)
        require(receipt['purpose']==purpose and receipt['binding_sha256']==self.binding_sha and
            receipt['device']=='cuda:0' and receipt['model_bindings']==self.binding['model_bindings'] and
            receipt['frame_pixel_sha256']==frames and receipt['value_sha256']==content_sha(value) and
            receipt['errors']==[],'own frozen-model encoding provenance receipt')
        for key in ('encoder_attempted','encoder_completed','projector_attempted','projector_completed'):
            require(type(receipt[key]) is int and receipt[key]==1,'complete single encoding entry receipt')
        require(type(receipt['encoded_frames']) is int and receipt['encoded_frames']==len(frames),'exact encode frame census')

    @fail_closed
    def accept_prefix(self,receipt,terminal_rgb,predicted_context):
        require(self.phase=='await_prefix','prefix only once')
        packet=raw_packet(self.case['prefix']); self.execution(receipt,packet,0,terminal_rgb)
        require(receipt['reset_identity_sha256']==self.binding['reset_identity_sha256'],'root-specific reset provenance receipt')
        require(rgb_sha(terminal_rgb)==self.binding['prefix_pixel_sha256'],'bound real prefix pixels')
        before,_=context(self.predicted); after,actions=context(predicted_context)
        next_between=np.stack([self.between[1],packet])
        require(np.array_equal(after[:2],before[1:]) and after[-1].tolist()==self.case['saved_predicted_prefix'] and
            np.array_equal(actions,normalize(next_between)),'bound predicted prefix rolling context')
        self.frames=np.stack([self.frames[1],self.frames[2],rgb(terminal_rgb)]).copy()
        self.between=next_between; self.predicted=deepcopy(predicted_context)
        self.executions.append(deepcopy(receipt)); self.boundary=5; self.phase='await_observation'

    @fail_closed
    def planning_request(self,observation):
        require(self.phase=='await_observation' and self.boundary in BOUNDARIES,'feedback only before three decision boundaries')
        self.identity(observation,self.boundary)
        require(observation['purpose']=='real_feedback' and observation['frame_pixel_sha256']==[rgb_sha(f) for f in self.frames] and
            observation['between_raw']==self.between.tolist(),'own actual RGB/raw feedback provenance')
        observed=deepcopy(observation['context']); latents,actions=context(observed)
        require(np.array_equal(actions,normalize(self.between)),'observed action normalization')
        self.encode_receipt(observation['encoding_receipt'],observed,observation['frame_pixel_sha256'],'real_feedback')
        require(observation['encoding_receipt']['between_raw_sha256']==content_sha(self.between.tolist()),'encoding receipt raw action provenance')
        previous=context(self.source['initial_context'])[0] if self.last_observed is None else context(self.last_observed)[0]
        require(np.allclose(latents[:2],previous[1:],rtol=0,atol=2e-5),'observed history shift, no cross-arm substitution')
        if self.last_observed is None:
            require(np.allclose(latents[-1],self.case['saved_actual_prefix'],rtol=0,atol=2e-5),'actual prefix latent source witness')
        self.encoding_receipts.append(deepcopy(observation['encoding_receipt']))
        self.last_observed=observed
        chosen=observed if self.arm=='actual' else deepcopy(self.predicted)
        self.pending=dict(schema=TAG+'.planning_request',root_sha256=self.root,arm=self.arm,boundary=self.boundary,
            binding_sha256=self.binding_sha,chosen_context=chosen,observed_context=observed,
            goal_q_sha256=self.binding['goal_q_sha256'],previous_selection_sha256=self.previous_selection,
            feedback_used_for_selection=self.arm=='actual',purpose='selection')
        self.phase='await_selection'; return deepcopy(self.pending)

    @fail_closed
    def accept_selection(self,plan,persist):
        require(self.phase=='await_selection','one selection after an actual observation')
        validate_binding(self.binding,self.binding_sha)
        verify_seal(output_dir=self.runtime_source_seal['output_dir'],expected_manifest_sha256=self.runtime_source_seal['manifest_sha256'])
        self.identity(plan,self.boundary)
        require(plan['binding_sha256']==self.binding_sha and plan['context']==self.pending['chosen_context'] and
            plan['goal_q_sha256']==self.binding['goal_q_sha256'],'plan own root/source/context/metric')
        candidates=plan['candidates']; require(len(candidates)==10 and len({c['candidate_id'] for c in candidates})==10,'complete ten-candidate compile')
        goal=array(self.source['goal'],(192,)).astype(float); q=array(self.source['q'],(192,192)).astype(float)
        for c in candidates:
            array(c['predicted_terminal'],(192,)); raw_packet(c['commands'])
            value=float(np.mean((q@(goal-np.asarray(c['predicted_terminal'])))**2))
            require(np.isclose(value,c['Q_cost'],rtol=0,atol=1e-12),'independent saved-array candidate Q')
        winner=min(candidates,key=lambda c:c['Q_cost'])
        require(winner['candidate_id']==plan['selected_candidate_id'] and winner['commands']==plan['commands'],'stable complete-bank winner')
        successor=plan['successor']; before,_=context(plan['context']); after,actions=context(successor)
        require(np.array_equal(after[:2],before[1:]) and after[-1].tolist()==winner['predicted_terminal'] and
            np.array_equal(actions,normalize([self.between[1],plan['commands']])) and
            plan['verification_successor']==successor,'independent replay/successor full context')
        sealed=dict(planning_request=self.pending,plan=deepcopy(plan)); digest_value=content_sha(sealed)
        # Failure/incorrect acknowledgment leaves no dispatch authorization.
        acknowledgment=persist(deepcopy(sealed))
        require(self.phase=='await_selection' and acknowledgment==digest_value,'durable selection SHA acknowledgment before dispatch')
        self.pending=dict(selection_sha256=digest_value,commands=deepcopy(plan['commands']),successor=deepcopy(successor))
        self.decisions.append(dict(boundary=self.boundary,selection_sha256=digest_value))
        self.phase='await_execution'
        return dict(root_sha256=self.root,arm=self.arm,boundary=self.boundary,selection_sha256=digest_value,
                    commands=deepcopy(plan['commands']),purpose='single_five_step_dispatch')

    @fail_closed
    def record_execution(self,authorization_sha,receipt,terminal_rgb):
        require(self.phase=='await_execution' and authorization_sha==self.pending['selection_sha256'],'one-shot exact selection authorization')
        packet=raw_packet(self.pending['commands']); self.execution(receipt,packet,self.boundary,terminal_rgb)
        self.frames=np.stack([self.frames[1],self.frames[2],rgb(terminal_rgb)]).copy()
        self.between=np.stack([self.between[1],packet])
        self.predicted=deepcopy(self.pending['successor']); self.previous_selection=authorization_sha
        self.executions.append(deepcopy(receipt)); self.pending=None; self.boundary+=5
        self.phase='completed20' if self.boundary==20 else 'await_observation'

    @fail_closed
    def record_terminal_evaluation(self,evaluation):
        require(self.phase=='completed20' and self.evaluation is None,'terminal evaluation only after all decisions')
        self.identity(evaluation,20)
        require(evaluation['purpose']=='terminal_evaluation_only' and
            evaluation['pixel_sha256']==rgb_sha(self.frames[-1]),'evaluation bound to own real step20 RGB')
        native=array(evaluation['native192'],(192,)).astype(float)
        self.encode_receipt(evaluation['encoding_receipt'],evaluation['native192'],[evaluation['pixel_sha256']],'terminal_evaluation_only')
        goal=np.asarray(self.source['goal']); q=np.asarray(self.source['q'])
        self.evaluation=dict(evaluation,Q_cost=float(np.mean((q@(goal-native))**2)))
        return deepcopy(self.evaluation)

    def receipt(self):
        # This does not create/dispatch work. Final byte drift must fail closed.
        validate_binding(self.binding,self.binding_sha)
        return dict(schema=TAG+'.adapter_receipt',root_sha256=self.root,arm=self.arm,binding_sha256=self.binding_sha,
            phase=self.phase,boundary=self.boundary,decisions=deepcopy(self.decisions),errors=deepcopy(self.errors),
            executed_packets=len(self.executions),terminal_evaluation_saved=self.evaluation is not None,
            feedback_encode_receipts=len(self.encoding_receipts),
            P1_admission=False,P4_admission=False,scientific_independent_batch_verified=False,
            resource_scope='external encoding/compiler/guarded-executor calls are not measured by this interface',
            actual_neural_execution_verified=False,actual_physics_execution_verified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Read-only pinned source validation; no run/prepare/write command.')
    parser.add_argument('--root-sha256',required=True)
    for name in ('capability','response','response-prepare'):
        parser.add_argument('--'+name+'-path',type=Path,required=True)
        parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--prior-root-sha256s',nargs='+',required=True)
    parser.add_argument('--prior-batch-path',type=Path,action='append',required=True)
    parser.add_argument('--prior-batch-sha256',action='append',required=True)
    parser.add_argument('--excluded-root-sha256s',nargs='*',default=[])
    options=vars(parser.parse_args()); paths=options.pop('prior_batch_path'); hashes=options.pop('prior_batch_sha256')
    require(len(paths)==len(hashes),'one SHA per independently reviewed prior-batch path')
    options['prior_batch_refs']=[dict(path=str(p.resolve()),sha256=h) for p,h in zip(paths,hashes)]
    result=bind_sources(**options)
    print(json.dumps(dict(status='source_binding_validated_not_executed',binding_sha256=content_sha(result),binding=result),sort_keys=True))
