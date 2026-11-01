"""Experiment1 saved-input/scoring contract. No model, planner or simulator.

Three arms stay distinct. Latent tracking deviation is not native task success.
Missing learned assets/observations are null, never synthetic priors or zeros.
CLI emits JSON to stdout only and never authorizes physical execution.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

NAME='agent_subgoal_gap_contract_20261028_v1'
ARMS=('arbitrary_latent_waypoint','existing_learned_prior','dynamics_compiled_witness')
SHARED=('root','history','task_goal','model','encoder','normalizer','q','controller',
        'observation','budget','native_success_evaluator')


def require(ok,message):
    if not ok:raise ValueError(message)


def encode(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(encode(v)).hexdigest()


def keys(v,names):require(type(v) is dict and set(v)==set(names),'exact contract fields: '+str(names))


def sha(v):require(type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v),'explicit SHA256')


def parse(data):
    def unique(pairs):
        out={}
        for k,v in pairs:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    v=json.loads(data,object_pairs_hook=unique,parse_constant=lambda s:require(False,'nonfinite JSON'))
    encode(v);return v


def bound(path,expected):
    sha(expected);p=Path(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size<=64*1024**2,'bounded regular saved input')
    raw=p.read_bytes();require(len(raw)<=64*1024**2 and hashlib.sha256(raw).hexdigest()==expected,'external byte SHA mismatch')
    return parse(raw)


def vector(v,d):
    require(type(v) is list and len(v)==d and all(type(x) in (int,float) and math.isfinite(x) for x in v),
            'complete finite latent vector, no bool or zero fill')


def deviations(target,observed,d):
    out=[]
    for t,o in zip(target,observed,strict=True):
        vector(t,d)
        if o is None:out.append(None);continue
        vector(o,d);s=sum((a-b)**2 for a,b in zip(t,o,strict=True))
        require(math.isfinite(s),'finite squared error')
        out.append(dict(l2=math.sqrt(s),mse=s/d))
    return out


def learned_pairing(receipt,row,shared,evidence):
    keys(receipt,('schema','method','prior_checkpoint_sha256','model_sha256','encoder_sha256',
        'normalizer_sha256','observation_sha256','history_sha256','task_goal_sha256',
        'targets_sha256','generator_attempted','generator_completed','fallback','lowlevel_targets_sha256'))
    require(receipt['schema']==NAME+'.learned_pairing' and receipt['method'] in ('HiLeWM_hierarchical_adaptation','SAGE_generator_adaptation'),
            'explicit learned method identity; neither is random proposal')
    require(receipt['prior_checkpoint_sha256'] in evidence,'real prior checkpoint evidence required')
    for key in ('model','encoder','normalizer','observation','history','task_goal'):
        require(receipt[key+'_sha256']==shared[key],'learned coordinate/history/goal pairing')
    require(receipt['targets_sha256']==receipt['lowlevel_targets_sha256']==digest(row['target']),
            'generated targets must actually be the low-level tracking targets')
    require(type(receipt['generator_attempted']) is int and receipt['generator_attempted']>=1
        and receipt['generator_completed']==receipt['generator_attempted'] and type(receipt['generator_completed']) is int
        and receipt['fallback'] is False,'real completed generator call, not final-goal fallback')


def compiled_pairing(witness,row,shared,steps):
    from experiments.spectral_reachability.d2p_composition_witness_v2 import compose
    keys(witness,('edges','receipt','transition_validation'))
    r=witness['receipt'];edges=witness['edges']
    require(r['provenance']['checkpoint_sha256']==shared['model']
        and r['provenance']['encoder_sha256']==shared['encoder']
        and r['provenance']['metric_sha256']==shared['q']
        and r['action_domain']['normalization_sha256']==shared['normalizer']
        and digest(r['root'])==shared['root'] and digest(r['root']['context'])==shared['history'],
        'compiled witness model/root/full-history/metric/normalizer pairing')
    def saved(context,actions):
        matches=[e for e in edges if e['input_context']==context and e['actions']==actions]
        require(matches and all(e['output_context']==matches[0]['output_context'] for e in matches),'unambiguous saved action transition')
        return matches[0]['output_context']
    rebuilt=compose(root=r['root'],provenance=r['provenance'],action_domain=r['action_domain'],
        edges=edges,replay=saved,max_edges=r['max_edges'],max_actions=r['max_actions'])
    require(rebuilt==r,'original compiled action/context receipt')
    ends=[];total=0
    for edge in edges:total+=len(edge['actions']);ends.append(total)
    require(ends==steps and [e['output_context']['latent_history'][-1] for e in edges]==row['target'],
            'compiled target must be action-generated checkpoint states, not arbitrary latent relabeling')


def score(spec,learned_receipts=None):
    """Pure arithmetic; external byte binding happens in CLI, not this RAM API."""
    keys(spec,('schema','latent_dim','checkpoints','trial_denominator','trials','evidence_sha256'))
    require(spec['schema']==NAME+'.input','input schema')
    d=spec['latent_dim'];steps=spec['checkpoints'];n=spec['trial_denominator']
    require(type(d) is int and d>0 and type(n) is int and n>0,'literal positive dimensions/denominator')
    require(type(steps) is list and steps and all(type(x) is int and x>0 for x in steps)
            and steps==sorted(set(steps)),'predeclared ordered public-step checkpoints')
    require(len(spec['trials'])==n,'complete fixed trial denominator')
    evidence=spec['evidence_sha256'];require(type(evidence) is list and len(set(evidence))==len(evidence),'unique evidence identities')
    for h in evidence:sha(h)
    rows=[];seen=set()
    for trial in spec['trials']:
        keys(trial,('trial_id','shared','arms'));shared=trial['shared'];keys(shared,SHARED)
        require(type(trial['trial_id']) is str and trial['trial_id'] and trial['trial_id'] not in seen,'unique fixed trial slot')
        seen.add(trial['trial_id'])
        for h in shared.values():sha(h);require(h in evidence,'shared identity requires bound evidence')
        require(len(trial['arms'])==3,'all three arm slots, including missing/failed')
        for arm,row in zip(ARMS,trial['arms'],strict=True):
            keys(row,('arm','shared_sha256','status','proposal','target','predicted','observed','native','resources','error'))
            require(row['arm']==arm and row['shared_sha256']==digest(shared),'same root/history/goal/controller/budget identities')
            require(row['status'] in ('unknown','failed','completed'),'explicit arm status')
            resources=row['resources'];require(type(resources) is dict,'explicit resource axes or empty unknown map')
            for value in resources.values():require(value is None or type(value) is int and value>=0,'resource unknown stays null; literal counts')
            if row['status']=='unknown':
                require(all(row[k] is None for k in ('proposal','target','predicted','observed','native'))
                        and all(v is None for v in resources.values()) and type(row['error']) is str and row['error'],
                        'unknown cannot invent proposal, execution or zero cost')
                rows.append(dict(trial_id=trial['trial_id'],arm=arm,status='unknown',deviation=None,
                    prediction_deviation=None,endpoint_deviation=None,endpoint_prediction_deviation=None,
                    native_success=None,resources=resources,error=row['error']));continue
            require((row['error'] is None) if row['status']=='completed' else bool(row['error']),'failure retains explicit error')
            proposal=row['proposal'];keys(proposal,('source_sha256','generator_receipt_sha256','action_witness_sha256'))
            require(proposal['source_sha256'] in evidence,'proposal source evidence binding')
            for field in ('generator_receipt_sha256','action_witness_sha256'):
                h=proposal[field]
                if h is not None:sha(h);require(h in evidence,'proposal receipt must be actual bound evidence')
            if arm==ARMS[1]:
                h=proposal['generator_receipt_sha256']
                require(h is not None and learned_receipts is not None and h in learned_receipts,
                        'learned arm needs real generator/paired asset receipt, no fallback substitution')
                learned_pairing(learned_receipts[h],row,shared,evidence)
            else:require(proposal['generator_receipt_sha256'] is None,'do not rename another generator as arbitrary or compiled')
            if arm==ARMS[2]:
                h=proposal['action_witness_sha256']
                require(h is not None and learned_receipts is not None and h in learned_receipts,'dynamics target needs complete action witness')
                compiled_pairing(learned_receipts[h],row,shared,steps)
            target=row['target'];observed=row['observed'];predicted=row['predicted']
            require(type(target) is list and type(observed) is list and len(target)==len(observed)==len(steps),'full checkpoint denominator')
            if row['status']=='completed':require(all(x is not None for x in observed),'completed low-level trace cannot have missing observations')
            actual=deviations(target,observed,d)
            pred=None
            if predicted is not None:
                require(type(predicted) is list and len(predicted)==len(steps),'prediction checkpoint denominator')
                pred=deviations(target,predicted,d)
            success=None
            if row['native'] is not None:
                native=row['native'];keys(native,('success','evaluator_sha256','evidence_sha256','task_goal_sha256'))
                require(type(native['success']) is bool and native['evaluator_sha256']==shared['native_success_evaluator']
                    and native['task_goal_sha256']==shared['task_goal'] and native['evidence_sha256'] in evidence,
                    'native evaluator/goal/outcome binding; never infer success from latent/object-only error')
                success=native['success']
            rows.append(dict(trial_id=trial['trial_id'],arm=arm,status=row['status'],deviation=actual,
                prediction_deviation=pred,endpoint_deviation=actual[-1],
                endpoint_prediction_deviation=None if pred is None else pred[-1],
                native_success=success,resources=resources,error=row['error']))
    summary={a:dict(denominator=n,unknown=sum(r['status']=='unknown' for r in rows if r['arm']==a),
        failed=sum(r['status']=='failed' for r in rows if r['arm']==a),
        native_known=sum(r['native_success'] is not None for r in rows if r['arm']==a),
        native_successes=sum(r['native_success'] is True for r in rows if r['arm']==a)) for a in ARMS}
    return dict(schema=NAME+'.scores',rows=rows,summary=summary,trial_denominator=n,arm_denominator=3*n,
        scientific_pass=False,physical_execution_authorized=False,P1_admission=False,
        scope='saved-input arithmetic, not independent evidence authenticity or model replay')


def score_file(path,expected,bindings):
    """MAIN supplies both input SHA and path->SHA evidence table in the input."""
    spec=bound(path,expected);table=bound(bindings[0],bindings[1]);require(type(table) is dict,'path/SHA binding table')
    def file_sha(p):
        h=hashlib.sha256()
        with Path(p).open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        return h.hexdigest()
    receipts={}
    needed={r['proposal'][field] for t in spec['trials'] for r in t['arms']
            if r.get('proposal') is not None for field in ('generator_receipt_sha256','action_witness_sha256')
            if r['proposal'].get(field) is not None}
    for p,h in table.items():
        sha(h);require(Path(p).is_absolute() and not Path(p).is_symlink(),'absolute non-symlink evidence')
        require(file_sha(p)==h,'bound evidence changed')
        if h in needed:receipts[h]=bound(p,h)
    require(set(spec['evidence_sha256'])==set(table.values()),'all declared evidence bytes must exist; no future SHA')
    result=score(spec,receipts);result['input_sha256']=expected;result['bindings_sha256']=bindings[1]
    for p,h in table.items():require(file_sha(p)==h,'evidence drift during scoring')
    return result


def legacy_waypoints(path,expected):
    """Reuse old strict causal analyzer; do NOT relabel it a paired Experiment1."""
    report=bound(path,expected)
    from experiments.spectral_reachability import analyze_reacher_d2p_waypoint_tracking as old
    summary=old.load(Path(path));rows=[]
    for row in report['rows']:
        for method,branch in row['methods'].items():
            rows.append(dict(seed=row['seed'],method=method,
                observed_to_frozen_waypoint_l2=[p['observed_waypoint_error_l2'] for p in branch['packets']],
                predicted_to_frozen_waypoint_l2=[p['predicted_waypoint_error_l2'] for p in branch['packets']],
                final_qpos_distance=branch['physical']['final_qpos_distance'],native_success=None))
    require(hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,'legacy input drift')
    return dict(schema=NAME+'.legacy_diagnostic',report_sha256=expected,root_count=8,branch_denominator=64,
        checkpoint_denominator=384,rows=rows,strict_original_summary=summary,
        arbitrary_latent_waypoint=None,existing_learned_prior=None,native_success=None,
        protocol_differences=['Reacher not PushT','6 packets x5 steps','four action-generated proposal families',
            'replay vs feedback have different controller/compute','no arbitrary-latent or learned-prior arm',
            'qpos distance is not a native success receipt'],
        physical_execution_authorized=False,scientific_pass=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('score','legacy-waypoints'))
    p.add_argument('--input',required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--bindings');p.add_argument('--bindings-sha256');a=p.parse_args(argv)
    if a.command=='score':
        require(a.bindings and a.bindings_sha256,'external evidence bindings file/SHA required')
        result=score_file(a.input,a.sha256,(a.bindings,a.bindings_sha256))
    else:result=legacy_waypoints(a.input,a.sha256)
    print(encode(result).decode());return 0


if __name__=='__main__':raise SystemExit(main())
