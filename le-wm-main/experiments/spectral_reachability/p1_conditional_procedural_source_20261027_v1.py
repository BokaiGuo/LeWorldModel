"""Bounded procedural source protocol. Import/CLI never reset an environment.

Only an explicit main-owned run_main_only call or --run-main with authorization
may reset/render. No model, hidden root factory, public step, or future labels.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

NAME = 'p1_conditional_procedural_source_20261027_v1'
PROJECT = Path(__file__).resolve().parents[2]
MAIN_AUTHORIZATION = 'MAIN_EXPLICIT_RESET_RENDER_ONLY_NO_PUBLIC_STEPS'
MINIMUM_FREE_BYTES = 200 * 1024**2
MODEL_METADATA = (
    ('pusht_d2p_p0_preflight_20261024_cuda', 'p0'),
    ('pusht_d2p_p0_preflight_packet_layout_repair_20261024_cuda', 'p0'),
    ('pusht_d2p_p0_packet_contract_v2_20261026_cuda', 'p0'),
    ('pusht_d2p_p1_reconfiguration_preflight_20261024_cuda', 'p1'),
    ('pusht_d2p_p1_root_anchored_v5_calibration_20261026_cuda', 'p1'),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fields(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names.split()), 'exact '+label+' fields required')


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 16), b''):
            h.update(block)
    return h.hexdigest()


def hash_value(value):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'SHA256 required')


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'literal bounded integer required')
    return value


def real(value, minimum=None):
    require(type(value) in (float, int) and math.isfinite(value), 'finite real required')
    if minimum is not None:
        require(value >= minimum, 'real below allowed bound')
    return value


def interval(value, integer_values=False):
    require(isinstance(value, list) and len(value) == 2, 'explicit interval required')
    for x in value:
        integer(x) if integer_values else real(x)
    require(value[1] > value[0], 'positive interval width required')


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def root_hash(state, goal):
    return hashlib.sha256(np.ascontiguousarray(np.concatenate((vector(state), vector(goal))), dtype=np.float64).tobytes()).hexdigest()


def vector(value):
    a = np.asarray(value)
    require(a.shape == (7,) and a.dtype.kind in 'fi' and np.isfinite(a).all(), 'finite state/goal7 required')
    return a.astype(np.float64)


def collect_current_exclusions(project=PROJECT):
    """Project whitelisted historical source metadata, never future values.

    Unlike the initial unsealed draft, directory names are not unknown-source
    evidence. Actual root/seed/input-binding evidence drives the resolver.
    """
    from experiments.spectral_reachability import p1_conditional_source_metadata_20261027_v1 as metadata
    return metadata.as_exclusions(metadata.resolve_history(project))


def validate_exclusions(e):
    fields(e, 'schema environment files root_hashes seed_families reset_spans unknown_sources coverage physical_future_labels_read', 'exclusions')
    require(e['schema'] == NAME+'.exclusions' and e['environment'] == 'swm/PushT-v1' and
            e['physical_future_labels_read'] is False, 'outcome-free PushT exclusions required')
    require(isinstance(e['files'], dict) and bool(e['files']), 'metadata bindings required')
    for path, h in e['files'].items():
        require(isinstance(path, str), 'metadata source name required'); hash_value(h)
    require(isinstance(e['root_hashes'], list) and len(set(e['root_hashes'])) == len(e['root_hashes']), 'unique root exclusions')
    for h in e['root_hashes']: hash_value(h)
    for rows, key in ((e['seed_families'], 'seed'), (e['reset_spans'], 'span')):
        require(isinstance(rows, list), 'source exclusion rows required')
        for row in rows:
            fields(row, key+' usage evidence', 'historical source')
            integer(row[key]) if key == 'seed' else interval(row[key], True)
            require(row['usage'] in ('used', 'reserved_not_opened', 'used_or_planned_input'), 'source usage required')
            hash_value(row['evidence'])
            require(row['evidence'] in e['files'].values(), 'exclusion must bind a metadata file')
    for row in e['unknown_sources']:
        fields(row, 'source reason', 'unknown provenance')
        require(all(isinstance(v, str) and bool(v) for v in row.values()), 'explicit unknown source/reason')
    fields(e['coverage'], 'status authority_sha256 scope', 'coverage')
    require(isinstance(e['coverage']['scope'], str) and bool(e['coverage']['scope']), 'coverage scope required')
    require(e['coverage']['status'] in ('unknown', 'externally_reviewed_complete'), 'explicit coverage status')
    if e['coverage']['status'] == 'externally_reviewed_complete':
        hash_value(e['coverage']['authority_sha256'])
        require(e['coverage']['authority_sha256'] in e['files'].values(), 'external coverage review must be metadata-bound, not just a ready flag')


def reset_recipe_id(p):
    return digest({k: p[k] for k in ('proposal_limit_per_seed', 'reset_attempts_per_proposal', 'placement_reset_seed', 'recipe', 'domain', 'history', 'rgb_shape')})


def derive_pusht_reset_semantics(path):
    """Independent source/AST receipt, not a fabricated zero-physics claim.

    Certifies the current direct call pattern ONLY; not native engine internals
    or an unknown historical factory. Refuses drift in the supported pattern.
    """
    path=Path(path).resolve();tree=ast.parse(path.read_text())
    body=next((x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PushT'),None)
    require(body is not None,'supported PushT environment source class required')
    methods={x.name:x for x in body.body if isinstance(x,ast.FunctionDef)}
    def call(node,name):return isinstance(node,ast.Call) and ast.unparse(node.func)==name
    reset=methods['reset'];setter=methods['_set_state']
    count=sum(call(n,'self._set_state') for n in ast.walk(reset))
    top=sum(isinstance(n,ast.Expr) and call(n.value,'self._set_state') for n in reset.body)
    physics=sum(call(n,'self.space.step') for n in ast.walk(setter))
    renders=sum(call(n,'self.render') for n in ast.walk(reset))
    require(count==top==2 and physics==1 and renders==1 and
            not any(isinstance(n,(ast.For,ast.While)) for n in ast.walk(setter)), 'unrecognized reset physics/render path: unknown, not zero')
    return dict(internal_physics_steps_per_reset=count*physics,internal_renders_per_reset=renders,authority_source_sha256=file_sha(path))


def prepare_calibration_request(config, project=PROJECT):
    """Metadata/source-file reads only; creates no environment or source root."""
    fields(config,'root_seeds reset_seed_start proposal_limit_per_seed reset_attempts_per_proposal placement_reset_seed recipe domain rgb_shape artifact_byte_limit environment_source_path source_closure_paths','calibration source configuration')
    require(isinstance(config['root_seeds'],list) and bool(config['root_seeds']), 'explicit finite parent seeds required')
    for seed in config['root_seeds']:integer(seed)
    P=integer(config['proposal_limit_per_seed'],1);A=integer(config['reset_attempts_per_proposal'],1)
    start=integer(config['reset_seed_start'])
    require(isinstance(config['source_closure_paths'],list),'explicit auxiliary source closure required')
    from experiments.spectral_reachability import p1_conditional_source_metadata_20261027_v1 as metadata
    environment=Path(config['environment_source_path']).resolve()
    paths={Path(__file__).resolve(),Path(metadata.__file__).resolve(),environment,
           *(Path(x).resolve() for x in config['source_closure_paths'])}
    closure={str(path):file_sha(path) for path in sorted(paths)}
    e=collect_current_exclusions(project)
    p=dict(schema=NAME+'.plan',environment='swm/PushT-v1',purpose='calibration_exploration',
           generator_sha256=closure[str(Path(__file__).resolve())],environment_sha256=closure[str(environment)],
           source_closure=closure,reset_semantics=derive_pusht_reset_semantics(environment),exclusion_id=digest(e),
           slots=[dict(candidate_id=i,root_seed=seed,reset_seed_span=[start+i*P*A,start+(i+1)*P*A],split='calibration')
                  for i,seed in enumerate(config['root_seeds'])],
           history=dict(mode='repeat_current_rgb_causal_padding_v1',frames=3,packet_steps=5,
                        zero_raw_actions_are_padding=True,public_warmup_steps=0),
           frozen_before_generation=True,physical_future_labels_read=False,
           **{key:deepcopy(config[key]) for key in ('proposal_limit_per_seed','reset_attempts_per_proposal','placement_reset_seed',
                                                   'recipe','domain','rgb_shape','artifact_byte_limit')})
    checked=validate_plan(p,e)
    require(checked['ready'],'known reused/reserved source collision; change PREDECLARED calibration source configuration before any generation')
    return dict(plan=p,exclusions=e,plan_sha256=digest(p),exclusion_sha256=digest(e))


def validate_plan(p, e):
    fields(p, 'schema environment purpose generator_sha256 environment_sha256 source_closure reset_semantics exclusion_id proposal_limit_per_seed reset_attempts_per_proposal placement_reset_seed slots recipe domain history rgb_shape artifact_byte_limit frozen_before_generation physical_future_labels_read', 'plan')
    validate_exclusions(e)
    require(p['schema'] == NAME+'.plan' and p['environment'] == e['environment'] and
            p['frozen_before_generation'] is True and p['physical_future_labels_read'] is False, 'own frozen source-only plan required')
    require(p['exclusion_id'] == digest(e), 'bound exclusion manifest required')
    require(p['purpose'] in ('calibration_exploration','fresh_confirmation'), 'explicit source purpose required')
    hash_value(p['generator_sha256']); hash_value(p['environment_sha256'])
    fields(p['reset_semantics'],'internal_physics_steps_per_reset internal_renders_per_reset authority_source_sha256','reset internal-operation semantics')
    for key in ('internal_physics_steps_per_reset','internal_renders_per_reset'):integer(p['reset_semantics'][key])
    require(p['reset_semantics']['authority_source_sha256']==p['environment_sha256'],'reset semantics must bind environment source, not claim zero physics')
    require(isinstance(p['source_closure'], dict) and bool(p['source_closure']), 'explicit source closure required')
    for name, h in p['source_closure'].items():
        require(isinstance(name, str) and Path(name).is_absolute(), 'absolute source closure path required'); hash_value(h)
    source = str(Path(__file__).resolve())
    require(p['source_closure'].get(source) == p['generator_sha256'] == file_sha(source), 'generator source closure binding')
    P = integer(p['proposal_limit_per_seed'], 1); A = integer(p['reset_attempts_per_proposal'], 1)
    integer(p['placement_reset_seed']); integer(p['artifact_byte_limit'], 1)
    fields(p['recipe'], 'position_jitter angle_offset_range edge_fraction_range pusher_surface_offset', 'sampler recipe')
    real(p['recipe']['position_jitter'], 0); real(p['recipe']['pusher_surface_offset'], 0)
    interval(p['recipe']['angle_offset_range']); interval(p['recipe']['edge_fraction_range'])
    require(0 <= p['recipe']['edge_fraction_range'][0] < p['recipe']['edge_fraction_range'][1] <= 1, 'edge fraction bounds')
    fields(p['domain'], 'canvas_xy contact_gap_range nonterminal_position_tolerance nonterminal_angle_tolerance', 'initial domain')
    interval(p['domain']['canvas_xy']); interval(p['domain']['contact_gap_range'])
    real(p['domain']['nonterminal_position_tolerance'], 0); real(p['domain']['nonterminal_angle_tolerance'], 0)
    fields(p['history'], 'mode frames packet_steps zero_raw_actions_are_padding public_warmup_steps', 'history policy')
    require(p['history'] == dict(mode='repeat_current_rgb_causal_padding_v1', frames=3, packet_steps=5,
                                zero_raw_actions_are_padding=True, public_warmup_steps=0),
            'only explicit causal padding; warmup public steps require a new execution contract')
    shape = p['rgb_shape']
    require(isinstance(shape, list) and len(shape) == 3 and shape[-1] == 3, 'explicit RGB shape required')
    for n in shape: integer(n, 1)
    require(isinstance(p['slots'], list) and bool(p['slots']), 'finite predeclared seed slots required')
    seeds=set(); ids=set(); spans=[]; blocked=[]
    for slot in p['slots']:
        fields(slot, 'candidate_id root_seed reset_seed_span split', 'seed slot')
        integer(slot['candidate_id']); integer(slot['root_seed']); interval(slot['reset_seed_span'], True)
        require(slot['split'] in ('calibration', 'confirmation') and slot['root_seed'] not in seeds and
                slot['candidate_id'] not in ids, 'unique seed IDs and predeclared splits')
        require(p['purpose']!='calibration_exploration' or slot['split']=='calibration', 'calibration exploration cannot generate confirmation roots')
        span=slot['reset_seed_span']
        require(span[1]-span[0] == P*A and not any(overlaps(span,s) for s in spans), 'exact disjoint finite reset spans required')
        seeds.add(slot['root_seed']); ids.add(slot['candidate_id']); spans.append(span)
        reasons=[]
        if any(x['seed']==slot['root_seed'] for x in e['seed_families']): reasons.append('used_or_reserved_seed_family')
        if any(overlaps(x['span'],span) for x in e['reset_spans']): reasons.append('used_or_reserved_random_reset_span')
        if reasons: blocked.append(dict(candidate_id=slot['candidate_id'],reasons=reasons))
    unknown = bool(e['unknown_sources']) or e['coverage']['status'] != 'externally_reviewed_complete'
    exploration=p['purpose']=='calibration_exploration'
    return dict(ready=not blocked and (exploration or not unknown),
                status='blocked' if blocked else 'calibration_exploration_only' if exploration else 'unknown' if unknown else 'source_protocol_ready_only',
                fresh_confirmation_ready=not exploration and not unknown and not blocked,
                blocked_slots=blocked, unknown_historical_provenance=unknown,
                maximum_random_resets=len(p['slots'])*P*A, maximum_total_resets=2*len(p['slots'])*P*A,
                maximum_reset_internal_physics_steps=2*len(p['slots'])*P*A*p['reset_semantics']['internal_physics_steps_per_reset'],
                maximum_render_calls_including_reset= len(p['slots'])*P*A*(1+2*p['reset_semantics']['internal_renders_per_reset']),
                maximum_public_steps=0, P1_admission=False)


class DurableJournal:
    """Hash-chained JSONL, flushed+fsynced before/after each external call."""
    def __init__(self, path, byte_limit):
        self.path=Path(path);self.stream=self.path.open('xb'); self.previous=None; self.count=0; self.bytes=0; self.limit=byte_limit
    def append(self, kind, **data):
        event=dict(index=self.count, previous_sha256=self.previous, kind=kind, **data)
        event['event_sha256']=digest(event)
        raw=(canonical(event)+'\n').encode()
        used=sum(x.stat().st_size for x in self.path.parent.iterdir() if x.is_file())
        require(used+len(raw)<=self.limit, 'total source/journal artifact byte cap exceeded')
        require(shutil.disk_usage(self.path.parent).free>=MINIMUM_FREE_BYTES+len(raw), '200MiB storage floor before journal write')
        self.stream.write(raw); self.stream.flush(); os.fsync(self.stream.fileno())
        self.bytes+=len(raw); self.count+=1; self.previous=event['event_sha256']
    def close(self): self.stream.close()


def read_journal(path):
    raw=Path(path).read_bytes()
    require(not raw or raw.endswith(b'\n'), 'torn journal suffix is incomplete, never silently accepted')
    events=[]; previous=None
    for i,line in enumerate(raw.splitlines()):
        event=json.loads(line); body={k:v for k,v in event.items() if k!='event_sha256'}
        require(event['index']==i and event['previous_sha256']==previous and digest(body)==event['event_sha256'], 'journal chain drift')
        previous=event['event_sha256']; events.append(event)
    return events


def rgb(value, p):
    a=np.asarray(value)
    require(a.dtype==np.uint8 and a.shape==tuple(p['rgb_shape']), 'exact uint8 source RGB shape')
    return a.copy()


def reset_summary(value, p):
    observation,info=value
    state=vector(observation['state']); goal=vector(info['goal_state']); image=rgb(info['goal'],p)
    return dict(state=state.tolist(),goal=goal.tolist(),goal_rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest())


def sample_state(goal, shapes, slot, proposal, attempt, p):
    """One proposal/attempt, no rejection loop or hidden factory."""
    rng=np.random.default_rng(np.random.SeedSequence([slot['root_seed'],proposal,attempt]))
    r=p['recipe']; state=goal.copy(); state[2:4]+=rng.uniform(-r['position_jitter'],r['position_jitter'],2)
    state[4]+=rng.uniform(*r['angle_offset_range']); state[5:]=0
    require(bool(shapes), 'environment block polygons required')
    shape=shapes[int(rng.integers(len(shapes)))]
    vertices=np.asarray([[float(v.x),float(v.y)] for v in shape.get_vertices()])
    require(vertices.ndim==2 and vertices.shape[1]==2 and len(vertices)>=3 and np.isfinite(vertices).all(), 'finite polygon required')
    k=int(rng.integers(len(vertices))); a=vertices[k]; b=vertices[(k+1)%len(vertices)]; edge=b-a
    norm=float(np.linalg.norm(edge)); require(norm>0, 'nondegenerate polygon edge')
    signed=float(np.sum(vertices[:,0]*np.roll(vertices[:,1],-1)-vertices[:,1]*np.roll(vertices[:,0],-1)))
    require(signed!=0, 'nondegenerate polygon area')
    normal=np.array([edge[1],-edge[0]])/norm*(1 if signed>0 else -1)
    local=a+rng.uniform(*r['edge_fraction_range'])*edge+r['pusher_surface_offset']*normal
    c,s=np.cos(state[4]),np.sin(state[4]); rotation=np.array([[c,-s],[s,c]])
    state[:2]=state[2:4]+rotation@local
    return state


def initial_domain(env, state, goal, p):
    """Initial geometry only; these are SOURCE DESIGN, never future labels."""
    body=env.unwrapped
    radii=[float(s.radius) for s in body.agent.shapes if hasattr(s,'radius')]
    require(bool(radii) and np.isfinite(radii).all(), 'agent radius required')
    gap=min(float(s.point_query(tuple(state[:2])).distance) for s in body.block.shapes)-max(radii)
    require(math.isfinite(gap), 'finite initial contact geometry')
    d=p['domain']; low,high=d['canvas_xy']; contact=d['contact_gap_range']
    position=float(np.linalg.norm(state[2:4]-goal[2:4])); angle=abs(float((state[4]-goal[4]+np.pi)%(2*np.pi)-np.pi))
    checks=dict(canvas=bool(((state[:4]>=low)&(state[:4]<=high)).all()),
                contact=contact[0]<=gap<=contact[1],
                nonterminal=position>d['nonterminal_position_tolerance'] or angle>d['nonterminal_angle_tolerance'])
    return dict(checks=checks,contact_gap=gap,position_error=position,angle_error=angle,
                accepted=all(checks.values()),source_design_not_future_label=True)


def source_identity(p, slot, proposal_index):
    return dict(kind='procedural_seed',generator_sha256=p['generator_sha256'],environment_sha256=p['environment_sha256'],
                reset_recipe_sha256=reset_recipe_id(p),root_seed=slot['root_seed'],
                reset_seed_span=deepcopy(slot['reset_seed_span']),proposal_index=proposal_index)


def write_raw_source(path, p, slot, proposal, attempt, state, goal, frame, goal_rgb):
    metadata=dict(schema=NAME+'.raw_source',plan_id=digest(p),root_sha256=root_hash(state,goal),
                  candidate_id=slot['candidate_id'],split=slot['split'],source=source_identity(p,slot,proposal),
                  selected_attempt=dict(attempt_index=attempt,random_reset_seed=slot['reset_seed_span'][0]+proposal*p['reset_attempts_per_proposal']+attempt,
                                        placement_reset_seed=p['placement_reset_seed']),
                  history_semantics='causal_padding_not_observed_transitions',actual_observed_frames=1,
                  new_public_steps=0,physical_future_labels_read=False,
                  normalizer_provenance='not_certified_or_inferred',threshold_authority='not_provided')
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, metadata_json=np.asarray(canonical(metadata)),
                            state=state,goal=goal,rgb_history=np.stack([frame]*3),goal_rgb=goal_rgb,
                            raw_action_history=np.zeros((10,2),np.float32))
        stream.flush(); os.fsync(stream.fileno())
    return metadata


def load_raw_source(path, p):
    with np.load(path,allow_pickle=False) as archive:
        require(set(archive.files)==set('metadata_json state goal rgb_history goal_rgb raw_action_history'.split()), 'exact raw source arrays')
        packet={k:archive[k].copy() for k in archive.files}
    meta=json.loads(str(packet.pop('metadata_json')))
    fields(meta,'schema plan_id root_sha256 candidate_id split source selected_attempt history_semantics actual_observed_frames new_public_steps physical_future_labels_read normalizer_provenance threshold_authority','raw source metadata')
    require(meta['schema']==NAME+'.raw_source' and meta['plan_id']==digest(p) and
            meta['root_sha256']==root_hash(packet['state'],packet['goal']), 'raw root/source plan binding')
    slot=next((s for s in p['slots'] if s['candidate_id']==meta['candidate_id']),None)
    require(slot is not None and meta['split']==slot['split'] and meta['source']==source_identity(p,slot,meta['source']['proposal_index']) and
            type(meta['source']['proposal_index']) is int and 0<=meta['source']['proposal_index']<p['proposal_limit_per_seed'], 'raw source identity/proposal binding')
    selected=meta['selected_attempt'];fields(selected,'attempt_index random_reset_seed placement_reset_seed','selected reset attempt')
    attempt=integer(selected['attempt_index'])
    require(attempt<p['reset_attempts_per_proposal'] and selected['random_reset_seed']==slot['reset_seed_span'][0]+
            meta['source']['proposal_index']*p['reset_attempts_per_proposal']+attempt and
            selected['placement_reset_seed']==p['placement_reset_seed'], 'selected root binds actual random reset attempt')
    history=packet['rgb_history']
    require(history.dtype==np.uint8 and history.shape==(3,*p['rgb_shape']) and
            np.array_equal(history[0],history[1]) and np.array_equal(history[1],history[2]), 'explicit repeated current frame, not imagined history')
    rgb(packet['goal_rgb'],p)
    require(packet['raw_action_history'].shape==(10,2) and packet['raw_action_history'].dtype==np.float32 and
            not np.count_nonzero(packet['raw_action_history']), 'zero raw padding actions required')
    require(meta['history_semantics']=='causal_padding_not_observed_transitions' and meta['actual_observed_frames']==1 and
            meta['new_public_steps']==0 and meta['physical_future_labels_read'] is False and
            meta['normalizer_provenance']=='not_certified_or_inferred' and meta['threshold_authority']=='not_provided', 'no invented history/statistics/threshold authority')
    packet['metadata']=meta
    return packet


def run_main_only(p, e, output, env, *, authorization, expected_plan_sha256, expected_exclusion_sha256):
    """EXPLICIT main entry. Tests use array fakes; THIS TURN NEVER calls real env.

    Caller owns env construction/close. Bootstrap only, no env.step anywhere.
    Unknown exclusions block before opening output or calling env. The token
    is an explicit-call guard, not OS isolation or proof of external authority.
    """
    require(authorization==MAIN_AUTHORIZATION, 'main explicit reset/render authorization required')
    checked=validate_plan(p,e)
    require(digest(p)==expected_plan_sha256 and digest(e)==expected_exclusion_sha256 and checked['ready'], 'sealed ready source policy required; unknown provenance is fail-closed')
    for name,h in p['source_closure'].items(): require(file_sha(name)==h, 'source closure drift: '+name)
    backend=inspect.getsourcefile(type(env.unwrapped))
    require(backend is not None and p['source_closure'].get(str(Path(backend).resolve()))==p['environment_sha256'], 'actual environment class source binding required')
    output=Path(output)
    require(shutil.disk_usage(output.parent).free>=MINIMUM_FREE_BYTES+p['artifact_byte_limit'],
            'preserve200MiB plus complete frozen artifact budget before source execution')
    output.mkdir(exist_ok=False)
    journal=DurableJournal(output/'journal.jsonl',p['artifact_byte_limit'])
    report=dict(schema=NAME+'.sourcepack',plan=deepcopy(p),exclusion_id=digest(e),status='running',
                slots=[dict(slot=deepcopy(s),status='not_started',proposals=[],raw_source_file=None,raw_source_sha256=None) for s in p['slots']],
                resources={k:dict(attempted=0,completed=0) for k in ('reset','render')},
                new_public_steps=0,history_semantics='causal_padding_not_observed_transitions',
                first_error=None,secondary_errors=[],durable_journal_complete=False,
                freshness_certified_globally=False,physical_future_labels_read=False,P1_admission=False,
                normalizer_provenance='not_certified_or_inferred',threshold_authority='not_provided')
    def error(exc): return dict(type=type(exc).__name__,message=str(exc))
    def retain(exc):
        item=error(exc)
        if report['first_error'] is None: report['first_error']=item
        else: report['secondary_errors'].append(item)
    def perform(operation, resource, fn, coordinates, summary):
        journal.append('operation_started',operation=operation,**coordinates)
        report['resources'][resource]['attempted']+=1
        value=fn(); report['resources'][resource]['completed']+=1
        journal.append('operation_completed',operation=operation,receipt=summary(value),**coordinates)
        return value
    seen=set(e['root_hashes'])
    try:
        journal.append('run_started',plan=deepcopy(p),exclusions=deepcopy(e),plan_id=digest(p))
        P=p['proposal_limit_per_seed']; A=p['reset_attempts_per_proposal']
        for row in report['slots']:
            slot=row['slot']; row['status']='running'; selected=None
            for proposal in range(P):
                entry=dict(proposal_index=proposal,attempts=[]); row['proposals'].append(entry)
                journal.append('proposal_started',candidate_id=slot['candidate_id'],proposal_index=proposal)
                for attempt in range(A):
                    seed=slot['reset_seed_span'][0]+proposal*A+attempt
                    coord=dict(candidate_id=slot['candidate_id'],proposal_index=proposal,attempt_index=attempt,
                               random_reset_seed=seed,placement_reset_seed=p['placement_reset_seed'])
                    receipt=dict(**coord,status='started'); entry['attempts'].append(receipt)
                    initial=perform('random_reset','reset',lambda:env.reset(seed=seed),coord,lambda v:reset_summary(v,p))
                    goal=vector(initial[1]['goal_state'])
                    state=sample_state(goal,list(env.unwrapped.block.shapes),slot,proposal,attempt,p)
                    placed=perform('placement_reset','reset',lambda:env.reset(seed=p['placement_reset_seed'],options=dict(state=state.copy(),goal_state=goal.copy())),coord,lambda v:reset_summary(v,p))
                    actual=vector(placed[0]['state']); actual_goal=vector(placed[1]['goal_state'])
                    require(np.array_equal(goal,actual_goal), 'placement reset silently changed goal')
                    domain=initial_domain(env,actual,actual_goal,p)
                    frame=rgb(perform('render','render',env.render,coord,lambda v:dict(rgb_sha256=hashlib.sha256(rgb(v,p).tobytes()).hexdigest())),p)
                    h=root_hash(actual,actual_goal)
                    receipt.update(status='completed',state=actual.tolist(),goal=actual_goal.tolist(),root_sha256=h,
                                   domain=domain,root_excluded_or_duplicate=h in seen)
                    if selected is None and domain['accepted'] and h not in seen:
                        selected=(proposal,attempt,actual.copy(),actual_goal.copy(),frame.copy(),rgb(placed[1]['goal'],p)); receipt['selected']=True
                    else: receipt['selected']=False
                    journal.append('attempt_finished',receipt=deepcopy(receipt))
                journal.append('proposal_finished',candidate_id=slot['candidate_id'],proposal_index=proposal,receipt=deepcopy(entry))
            if selected is None:
                row['status']='metadata_rejected'
            else:
                proposal,attempt,state,goal,frame,goal_image=selected
                path=output/f'source_{slot["candidate_id"]}.npz'
                # Conservative uncompressed source bound; no oversize write.
                size=frame.nbytes*3+goal_image.nbytes+2*7*8+10*2*4+16_384
                used=sum(x.stat().st_size for x in output.iterdir() if x.is_file())
                require(used+size<=p['artifact_byte_limit'], 'source artifact byte cap exceeded')
                require(shutil.disk_usage(output).free>=MINIMUM_FREE_BYTES+size, '200MiB storage floor before raw source write')
                journal.append('raw_source_write_started',candidate_id=slot['candidate_id'],root_sha256=root_hash(state,goal))
                metadata=write_raw_source(path,p,slot,proposal,attempt,state,goal,frame,goal_image)
                load_raw_source(path,p)
                row.update(status='completed',raw_source_file=path.name,raw_source_sha256=file_sha(path),metadata=metadata)
                seen.add(metadata['root_sha256'])
            journal.append('slot_finished',receipt=deepcopy(row))
        report['status']='completed_calibration_source_only' if p['purpose']=='calibration_exploration' else 'completed_source_only'
    except Exception as exc:
        retain(exc); report['status']='failed'
        for row in report['slots']:
            if row['status']=='running': row['status']='failed'
        try: journal.append('run_failed',first_error=report['first_error'])
        except Exception as secondary: retain(secondary)
    report['reset_internal_physics_steps']=dict(
        attempted_upper_bound=report['resources']['reset']['attempted']*p['reset_semantics']['internal_physics_steps_per_reset'],
        completed_reset_implied=report['resources']['reset']['completed']*p['reset_semantics']['internal_physics_steps_per_reset'],
        actual_observed=None,authority_source_sha256=p['reset_semantics']['authority_source_sha256'])
    report['reset_internal_render_calls_implied']=report['resources']['reset']['completed']*p['reset_semantics']['internal_renders_per_reset']
    try:
        journal.append('run_finished',status=report['status'],resources=report['resources'],first_error=report['first_error'])
        report['durable_journal_complete']=True
    except Exception as exc:
        retain(exc); report['status']='failed'
    try: journal.close()
    except Exception as exc: retain(exc); report['status']='failed'; report['durable_journal_complete']=False
    try: report['journal_sha256']=file_sha(output/'journal.jsonl')
    except Exception as exc: retain(exc); report['status']='failed'
    try:
        raw=(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        used=sum(x.stat().st_size for x in output.iterdir() if x.is_file())
        require(used+len(raw)<=p['artifact_byte_limit'], 'final report artifact byte cap exceeded')
        require(shutil.disk_usage(output).free>=MINIMUM_FREE_BYTES+len(raw), '200MiB storage floor before final report write')
        with (output/'sourcepack.json').open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    except Exception as exc: retain(exc); report['status']='failed'
    return report


def to_admission_scan(p, e, report, directory, task_ids):
    """Saved-array bridge AFTER main supplies encoded task IDs. No encoding here.

    Keep raw state/goal for provenance only: downstream neural planner inputs
    must be RGB/actions, never these privileged simulator vectors.
    """
    from experiments.spectral_reachability import p1_conditional_task_admission_20261027_v1 as admission
    require(validate_plan(p,e)['ready'], 'unknown historical source coverage cannot be promoted')
    expected_status='completed_calibration_source_only' if p['purpose']=='calibration_exploration' else 'completed_source_only'
    require(report['schema']==NAME+'.sourcepack' and report['plan']==p and report['exclusion_id']==digest(e) and
            report['status']==expected_status and report['first_error'] is None and not report['secondary_errors'] and
            report['durable_journal_complete'] is True and report['new_public_steps']==0 and
            report['physical_future_labels_read'] is False and report['P1_admission'] is False, 'complete source-only evidence required')
    directory=Path(directory)
    require(file_sha(directory/'journal.jsonl')==report['journal_sha256'], 'source journal hash drift')
    events=read_journal(directory/'journal.jsonl')
    require(events and events[0]['kind']=='run_started' and events[0]['plan']==p and events[0]['exclusions']==e and
            events[-1]['kind']=='run_finished' and events[-1]['status']==report['status'] and
            events[-1]['first_error'] is None, 'complete bound durable source journal required')
    slot_events=[x['receipt'] for x in events if x['kind']=='slot_finished']
    require(slot_events==report['slots'] and len(report['slots'])==len(p['slots']), 'complete ordered slot receipts, not edited report')
    counts={k:dict(attempted=0,completed=0) for k in ('reset','render')}; pending=None
    for event in events:
        if event['kind']=='operation_started':
            require(pending is None, 'overlapping external operations forbidden')
            pending=event
            counts['render' if event['operation']=='render' else 'reset']['attempted']+=1
        elif event['kind']=='operation_completed':
            keys=('operation','candidate_id','proposal_index','attempt_index','random_reset_seed','placement_reset_seed')
            require(pending is not None and all(pending[k]==event[k] for k in keys), 'external operation receipt mismatch')
            counts['render' if event['operation']=='render' else 'reset']['completed']+=1;pending=None
    N=len(p['slots'])*p['proposal_limit_per_seed']*p['reset_attempts_per_proposal']
    require(pending is None and counts==report['resources']==dict(reset=dict(attempted=2*N,completed=2*N),render=dict(attempted=N,completed=N)), 'exact nonrefundable finite reset/render budget')
    require(isinstance(task_ids,dict), 'main encoded task identity mapping required')
    success={r['metadata']['root_sha256'] for r in report['slots'] if r['status']=='completed'}
    require(set(task_ids)==success, 'exact encoded root/task mapping, no omitted/extra roots')
    for value in task_ids.values():hash_value(value)
    receipts=[];seen=set(e['root_hashes'])
    for slot,row in zip(p['slots'],report['slots']):
        require(row['slot']==slot and row['status'] in ('completed','metadata_rejected') and
                len(row['proposals'])==p['proposal_limit_per_seed'], 'complete finite proposal census')
        selected=[]
        for i,proposal in enumerate(row['proposals']):
            require(proposal['proposal_index']==i and len(proposal['attempts'])==p['reset_attempts_per_proposal'], 'every proposal/reset attempt required')
            for j,attempt in enumerate(proposal['attempts']):
                require(attempt['status']=='completed' and attempt['candidate_id']==slot['candidate_id'] and
                        attempt['proposal_index']==i and attempt['attempt_index']==j and
                        attempt['random_reset_seed']==slot['reset_seed_span'][0]+i*p['reset_attempts_per_proposal']+j and
                        attempt['placement_reset_seed']==p['placement_reset_seed'] and
                        attempt['root_sha256']==root_hash(attempt['state'],attempt['goal']), 'full deterministic reset schedule and root identity')
                eligible=attempt['domain']['accepted'] and attempt['root_sha256'] not in seen and not selected
                require(attempt['selected'] is bool(eligible), 'first legal fresh source selection required')
                if eligible:selected.append(attempt)
        identity=dict(candidate_id=slot['candidate_id'],root_sha256=None,task_id=None,split=slot['split'],
                      source=source_identity(p,slot,p['proposal_limit_per_seed']-1))
        if row['status']=='completed':
            require(len(selected)==1, 'one selected root per seed family')
            filename=row['raw_source_file']
            require(isinstance(filename,str) and filename==f'source_{slot["candidate_id"]}.npz', 'exact local source artifact path')
            path=directory/filename
            require(path.is_file() and not path.is_symlink() and file_sha(path)==row['raw_source_sha256'], 'raw source artifact SHA drift')
            packet=load_raw_source(path,p);meta=packet['metadata'];chosen=selected[0]
            require(meta==row['metadata'] and meta['root_sha256']==chosen['root_sha256'] and
                    meta['source']['proposal_index']==chosen['proposal_index'] and
                    meta['selected_attempt']['attempt_index']==chosen['attempt_index'], 'raw source/attempt/journal binding')
            seen.add(meta['root_sha256'])
            identity.update(root_sha256=meta['root_sha256'],task_id=task_ids[meta['root_sha256']],source=meta['source'])
        else:require(not selected and row['raw_source_file'] is None, 'rejected seed cannot hide selected root')
        receipts.append(dict(identity=identity,status=row['status'],error=None if selected else 'finite initial-domain scan exhausted; no replacement'))
    plan=dict(schema=admission.NAME+'.procedural_scan',generator_sha256=p['generator_sha256'],environment_sha256=p['environment_sha256'],
              reset_recipe_sha256=reset_recipe_id(p),proposal_limit_per_seed=p['proposal_limit_per_seed'],
              physical_future_labels_read=False,frozen_before_generation=True,slots=deepcopy(p['slots']))
    bundle=dict(plan=plan,plan_sha256=admission.digest(plan),receipts=receipts)
    admission.procedural_scan(**bundle)
    return bundle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--audit-exclusions',action='store_true')
    group.add_argument('--validate-plan',type=Path,help='JSON {plan,exclusions}; metadata only')
    group.add_argument('--run-main',type=Path,help='MAIN ONLY: sealed JSON {plan,exclusions,plan_sha256,exclusion_sha256}; creates environment ONLY after ready gate')
    group.add_argument('--prepare-calibration',type=Path,help='Metadata-only explicit source design JSON -> sealed calibration-only request; NO environment')
    parser.add_argument('--authorize-reset-render',action='store_true',help='Explicitly permit resets/rendering, NEVER public steps or warmup')
    parser.add_argument('--project-root',type=Path,default=PROJECT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.run_main:
        require(args.authorize_reset_render and args.output is not None, 'main reset/render requires explicit authorization and NEW output directory')
        payload=json.loads(args.run_main.read_text())
        fields(payload,'plan exclusions plan_sha256 exclusion_sha256','sealed main request')
        p,e=payload['plan'],payload['exclusions'];checked=validate_plan(p,e)
        require(digest(p)==payload['plan_sha256'] and digest(e)==payload['exclusion_sha256'], 'external plan/exclusion digest drift')
        if not checked['ready']:
            print(canonical(checked));raise SystemExit(2)
        require(not args.output.exists(), 'refuse existing main output before environment creation')
        require(shutil.disk_usage(args.output.parent).free>=MINIMUM_FREE_BYTES+p['artifact_byte_limit'], 'storage gate before environment creation')
        for name,h in p['source_closure'].items():require(file_sha(name)==h, 'source closure drift before environment import')
        # ONLY this explicit ready main branch imports/constructs an environment.
        # No old sampler or model module is imported by this implementation.
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401 -- registers the environment
        require(p['rgb_shape'][0]==p['rgb_shape'][1],'PushT renderer needs square source images')
        env=gym.make(p['environment'],render_mode='rgb_array',resolution=p['rgb_shape'][0])
        result=None
        try:
            result=run_main_only(p,e,args.output,env,authorization=MAIN_AUTHORIZATION,
                expected_plan_sha256=payload['plan_sha256'],expected_exclusion_sha256=payload['exclusion_sha256'])
        finally:
            try:env.close()
            except Exception as exc:
                cleanup=dict(schema=NAME+'.main_cleanup_error',message=str(exc),type=type(exc).__name__,
                             primary_error=None if result is None else result['first_error'])
                # New sidecar, NEVER overwrite the completed/failed sourcepack.
                if args.output.is_dir():
                    with (args.output/'main_cleanup_error.json').open('x') as stream:
                        stream.write(canonical(cleanup)+'\n');stream.flush();os.fsync(stream.fileno())
                if result is not None:result['main_cleanup_error']=cleanup;result['status']='failed'
        print(canonical(result))
        if result['status'] not in ('completed_source_only','completed_calibration_source_only'):raise SystemExit(2)
        return
    if args.prepare_calibration:
        result=prepare_calibration_request(json.loads(args.prepare_calibration.read_text()),args.project_root)
    elif args.audit_exclusions: result=collect_current_exclusions(args.project_root)
    else:
        payload=json.loads(args.validate_plan.read_text()); fields(payload,'plan exclusions','metadata validation input')
        result=validate_plan(payload['plan'],payload['exclusions'])
    if args.output:
        with args.output.open('x') as stream:
            stream.write(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    print(canonical(result))


if __name__=='__main__': main()
