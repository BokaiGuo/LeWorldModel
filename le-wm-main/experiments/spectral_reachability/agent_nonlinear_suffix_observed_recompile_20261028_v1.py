"""Observed-cp5 suffix optimizer NUMERICAL CORE; production ABI not bound yet.

No model loading, physics, outcome input or guessed midpoint schema. Callback
interfaces match the frozen nonlinear H1 backend; MAIN binds real authority later.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p1_nonlinear_suffix_library_20261028_v2 as frozen

NAME='agent_nonlinear_suffix_observed_recompile_20261028_v1'
core,require,digest=frozen.core,frozen.require,frozen.digest
ORDER=tuple(frozen.source.ORDER)
UPDATES=8


def contract():
    return dict(schema=NAME+'.contract',stage='numerical_core_only_pending_frozen_midpoint_ABI',
        slots=4,known_roots=2,order=[list(v) for v in ORDER],prefix_steps=5,suffix_steps=5,
        updates=8,eta=1.,selection='stable(original terminal meanQ,iterate index), incumbent+ALL8 iterates',
        observed_context='actual own cp5 RGB latent3 and executed raw10, NOT imagined prefix context',
        prefix='parent selected raw5 preserved exactly; already executed, not replanned or re-predicted',
        incumbent='parent selected suffix and its saved logits; no inverse or iteration0 projection',
        raw_chart=frozen.contract()['chart'],cap=frozen.CAP,
        domain='original float64 ridge solver/cast witness OR original complete8 projected recurrence; never postcast incumbent cap rejection',
        normalizer='unchanged original float32 mean/std, subtract/divide once',
        per_slot=dict(score_forward=9,gradient_forward=8,predict=17,scalar_reverse=8),
        total=dict(score_forward=36,gradient_forward=32,predict=68,scalar_reverse=32),
        independent_replay=0,new_prefix_forward=0,new_visual=0,new_J=0,new_SVD=0,new_physics=0,
        cost_scope='NUMERICAL schedule only; actual native ledger supplied by future reviewed CUDA adapter, never RAM as executed receipt',
        inherited_compute=None,total_FLOPs=None,equal_FLOPs=False,training=False,outcome_input=False,
        production_ABI_bound=False,production_ready=False,production_run_authorized=False,
        P1_admission=False,freshness=False,physical_release=False,scientific_admission=False)


def selected_incumbent(parent_candidate,parent_result):
    """Numerical parent provenance; DOES NOT replace parent strict native receipt.

    Accept only the frozen library's complete9/8 recurrence and selected logits.
    Production still must call its actual strict consumer and match plan digest.
    """
    c,r=parent_candidate,parent_result
    require(r['index']==c['index'] and r['candidate_sha256']==digest(c)
        and len(r['iterates'])==9 and len(r['gradients'])==8,'complete original selected-prefix numerical provenance')
    d=c['incumbent_domain']
    frozen.incumbent_domain(dict(suffix_params=c['logits'],suffix_commands=c['incumbent_raw'],
        suffix_solve=dict(params=c['logits'],radius=frozen.CAP,norm=d['solver_float64_norm'])))
    x=np.asarray(c['logits'],float)
    for i,v in enumerate(r['iterates']):
        a=frozen.chart(x)
        require(v['iteration']==i and v['logits']==x.tolist() and v['commands']==a.tolist() and v['raw_bits']==frozen.bits(a)
            and type(v['Q']) in (int,float) and np.isfinite(v['Q']),'all original parent iterates literal chart/cost')
        if i<8:
            g=core.finite(r['gradients'][i]['raw_gradient'],(5,2),True)
            chain=1-a.astype(float)**2;dx=(g*chain).reshape(10)
            require(r['gradients'][i]['chain']==chain.tolist() and r['gradients'][i]['logit_gradient']==dx.tolist(),'original parent analytic derivative chain')
            x=frozen.proposal(x,dx)
    winner=min(range(9),key=lambda k:(r['iterates'][k]['Q'],k));v=r['iterates'][winner]
    require(r['winner_iteration']==winner and r['commands']==c['prefix']+v['commands'] and r['selected_Q']==v['Q']
        and r['selected_context']==v['context'] and frozen.bits(r['commands'][:5])==c['prefix_bits'],'exact original selected suffix/full10 identity')
    return dict(schema=NAME+'.incumbent',parent_candidate_sha256=digest(c),parent_result_sha256=digest(r),
        parent_iteration=winner,parent_prefix_index=c['index'],prefix=deepcopy(c['prefix']),prefix_bits=c['prefix_bits'],
        logits=deepcopy(v['logits']),commands=deepcopy(v['commands']),raw_bits=v['raw_bits'],
        original_ridge_domain=deepcopy(d),
        selected_domain='original_solver_float64_cast' if winner==0 else 'original_projected_iterate',
        production_authority_verified=False)


def request(slot,task,observed_context,root_raw10,observed_raw10,incumbent,parent_candidate,parent_result):
    """Our core input, NOT Bohr's external schema or a freshness certificate."""
    require(type(slot) is int and slot in range(4),'all original slot positions')
    t=core.validate_task(task);ctx=core.context(observed_context)
    require(len(t['goal'])==192 and np.array_equal(t['projection'],np.eye(192)),'original native192 identity meanQ')
    require(incumbent==selected_incumbent(parent_candidate,parent_result),'same immutable selected parent incumbent')
    root=core.finite(root_raw10,(10,2),True);actual=core.finite(observed_raw10,(10,2),True)
    prefix=core.finite(incumbent['prefix'],(5,2),True)
    require(np.max(abs(root))<=1 and np.max(abs(actual))<=1 and np.max(abs(prefix))<=1,'literal original legal raw domain')
    require(frozen.bits(actual)==frozen.bits(np.concatenate((root,prefix))[-10:]),'actual cp5 raw10 is root raw10 + exact executed prefix tail10')
    mean,std=(np.asarray(t[k],np.float32) for k in ('action_mean','action_std'))
    require(np.array_equal(t['root_context']['action_history'],((root.astype(np.float32)-mean)/std).reshape(2,10))
        and np.array_equal(ctx['action_history'],((actual.astype(np.float32)-mean)/std).reshape(2,10)),
        'root and observed history once-normalized from exact causal raw10')
    return dict(schema=NAME+'.core_request',slot=slot,root_slot=ORDER[slot][0],condition=ORDER[slot][1],
        task=deepcopy(t),observed_context=deepcopy(ctx),root_raw10=root.tolist(),observed_raw10=actual.tolist(),
        incumbent=deepcopy(incumbent),parent_candidate_sha256=digest(parent_candidate),parent_result_sha256=digest(parent_result),
        source_binding_status='numerical_only_not_external_authority',production_authority_verified=False)


def validate_request(q):
    core.keys(q,('schema','slot','root_slot','condition','task','observed_context','root_raw10','observed_raw10',
        'incumbent','parent_candidate_sha256','parent_result_sha256','source_binding_status','production_authority_verified'))
    require(q['schema']==NAME+'.core_request' and type(q['slot']) is int and q['slot'] in range(4)
        and (q['root_slot'],q['condition'])==ORDER[q['slot']] and q['production_authority_verified'] is False
        and q['source_binding_status']=='numerical_only_not_external_authority','exact core-only slot/no deployment certificate')
    t=core.validate_task(q['task']);ctx=core.context(q['observed_context']);inc=q['incumbent']
    require(np.array_equal(t['projection'],np.eye(192)) and len(t['goal'])==192,'same original identity Q')
    core.keys(inc,('schema','parent_candidate_sha256','parent_result_sha256','parent_iteration','parent_prefix_index',
        'prefix','prefix_bits','logits','commands','raw_bits','original_ridge_domain','selected_domain','production_authority_verified'))
    for k in ('parent_candidate_sha256','parent_result_sha256'):
        core.hash_value(q[k]);require(q[k]==inc[k],'exact inherited parent identity')
    require(inc['schema']==NAME+'.incumbent' and inc['production_authority_verified'] is False
        and type(inc['parent_iteration']) is int and 0<=inc['parent_iteration']<=8
        and type(inc['parent_prefix_index']) is int and 0<=inc['parent_prefix_index']<42,'original complete-library selection identity')
    x=core.finite(inc['logits'],(10,));a=core.finite(inc['commands'],(5,2),True);prefix=core.finite(inc['prefix'],(5,2),True)
    require(np.array_equal(a,frozen.chart(x)) and frozen.bits(a)==inc['raw_bits'] and frozen.bits(prefix)==inc['prefix_bits'], 'original exact prefix/suffix bits and saved logits')
    if inc['parent_iteration']==0:
        d=inc['original_ridge_domain']
        require(inc['selected_domain']=='original_solver_float64_cast' and d['solver_float64_norm']==float(np.linalg.norm(x))
            and d['solver_float64_norm']<=frozen.CAP and d['original_raw_bits']==inc['raw_bits']
            and d['cap']==frozen.CAP and d['postcast_norm_is_diagnostic_only'] is True
            and d['postcast_float64_norm']==float(np.linalg.norm(x.astype(np.float32).astype(float))),
            'original solver-domain incumbent; postcast norm never a new legality gate')
    else:
        require(inc['selected_domain']=='original_projected_iterate' and np.array_equal(x,x.astype(np.float32))
            and np.linalg.norm(x.astype(float))<=frozen.CAP,'selected original projected iterate domain')
    root=core.finite(q['root_raw10'],(10,2),True);actual=core.finite(q['observed_raw10'],(10,2),True)
    require(max(np.max(abs(v)) for v in (a,prefix,root,actual))<=1
        and frozen.bits(actual)==frozen.bits(np.concatenate((root,prefix))[-10:]),'same executed prefix and exact own cp5 raw history')
    mean,std=(np.asarray(t[k],np.float32) for k in ('action_mean','action_std'))
    require(np.array_equal(t['root_context']['action_history'],((root.astype(np.float32)-mean)/std).reshape(2,10))
        and np.array_equal(ctx['action_history'],((actual.astype(np.float32)-mean)/std).reshape(2,10)), 'literal actual history normalization')
    return t


def optimize(q,backend,sink):
    """Actual fixed8 arithmetic/control loop, delegated H1 callbacks, not manifest.

    Frozen Backend.step/gradient can be bound ONLY after real midpoint authority
    exists. This core does not authorize model execution or issue native receipts.
    """
    t=validate_request(q);ctx=q['observed_context'];inc=q['incumbent'];x=np.asarray(inc['logits'],float)
    sink(dict(kind='observed_suffix_intent',request_sha256=digest(q)))
    a=frozen.chart(x);after=backend.step(deepcopy(ctx),a,'observed_iterate_0')
    core.checked_path(t,ctx,a.tolist(),[after])
    its=[dict(iteration=0,logits=x.tolist(),commands=a.tolist(),raw_bits=frozen.bits(a),context=after,Q=frozen.cost(t['goal'],after))];grads=[]
    sink(dict(kind='observed_iterate',iterate=deepcopy(its[0])))
    for i in range(1,9):
        sink(dict(kind='observed_gradient_intent',iteration=i,context_sha256=digest(ctx),raw_bits=frozen.bits(a)))
        d=backend.gradient(deepcopy(ctx),a,f'observed_gradient_{i}')
        core.keys(d,('context','native_float32_loss','raw_gradient'))
        require(d['context']==its[-1]['context'],'same actual cp5 suffix AD forward at exact current raw')
        g=core.finite(d['raw_gradient'],(5,2),True);loss=d['native_float32_loss']
        require(type(loss) in (int,float) and np.isfinite(loss) and loss>=0,'native scalar loss witness')
        chain=1-a.astype(float)**2;dx=(g*chain).reshape(10)
        d.update(chain=chain.tolist(),logit_gradient=dx.tolist());grads.append(deepcopy(d))
        sink(dict(kind='observed_gradient_return',iteration=i,gradient=deepcopy(d)))
        x=frozen.proposal(x,dx);a=frozen.chart(x)
        after=backend.step(deepcopy(ctx),a,f'observed_iterate_{i}');core.checked_path(t,ctx,a.tolist(),[after])
        v=dict(iteration=i,logits=x.tolist(),commands=a.tolist(),raw_bits=frozen.bits(a),context=after,Q=frozen.cost(t['goal'],after))
        its.append(v);sink(dict(kind='observed_iterate',iterate=deepcopy(v)))
    k=min(range(9),key=lambda j:(its[j]['Q'],j));v=its[k]
    result=dict(schema=NAME+'.numeric_result',status='numerical_core_complete_NOT_deployable',slot=q['slot'],
        request_sha256=digest(q),parent_candidate_sha256=q['parent_candidate_sha256'],parent_result_sha256=q['parent_result_sha256'],
        actual_context_sha256=digest(ctx),goal_sha256=digest(t['goal']),incumbent=deepcopy(inc),iterates=its,gradients=grads,
        winner_iteration=k,commands=inc['prefix']+v['commands'],selected_context=v['context'],new_suffix_Q=v['Q'],
        observed_incumbent_Q=its[0]['Q'],parent_prediction_used_as_new_prediction=False,
        scheduled_cost=contract()['per_slot'],actual_native_resources=None,inherited_compute=None,
        production_authority_verified=False,physical_metrics=None,P1_admission=False,freshness=False,physical_release=False)
    sink(dict(kind='observed_suffix_selected',result_sha256=digest(result),iteration=k,commands=result['commands']))
    return result


def verify_numeric(q,r,events):
    t=validate_request(q);core.keys(r,('schema','status','slot','request_sha256','parent_candidate_sha256','parent_result_sha256',
        'actual_context_sha256','goal_sha256','incumbent','iterates','gradients','winner_iteration','commands','selected_context',
        'new_suffix_Q','observed_incumbent_Q','parent_prediction_used_as_new_prediction','scheduled_cost',
        'actual_native_resources','inherited_compute','production_authority_verified','physical_metrics','P1_admission','freshness','physical_release'))
    require(r['schema']==NAME+'.numeric_result' and r['status']=='numerical_core_complete_NOT_deployable' and r['slot']==q['slot']
        and r['request_sha256']==digest(q) and r['actual_context_sha256']==digest(q['observed_context'])
        and r['goal_sha256']==digest(t['goal']) and r['incumbent']==q['incumbent']
        and all(r[k]==q[k] for k in ('parent_candidate_sha256','parent_result_sha256')),'exact own observed-context/goal/parent numerical result')
    require(len(r['iterates'])==9 and len(r['gradients'])==8 and r['scheduled_cost']==contract()['per_slot']
        and all(r[k] is None for k in ('actual_native_resources','inherited_compute','physical_metrics'))
        and all(r[k] is False for k in ('production_authority_verified','parent_prediction_used_as_new_prediction','P1_admission','freshness','physical_release')),
        'full9/8 and unknown actual costs/outcomes, no forged deployment authority')
    x=np.asarray(q['incumbent']['logits'],float);want=[dict(kind='observed_suffix_intent',request_sha256=digest(q))]
    for i,v in enumerate(r['iterates']):
        core.keys(v,('iteration','logits','commands','raw_bits','context','Q'));a=frozen.chart(x)
        require(v['iteration']==i and v['logits']==x.tolist() and v['commands']==a.tolist() and v['raw_bits']==frozen.bits(a)
            and v['Q']==frozen.cost(t['goal'],v['context']),'every own observed candidate literal chart/cost')
        core.checked_path(t,q['observed_context'],v['commands'],[v['context']])
        want.append(dict(kind='observed_iterate',iterate=deepcopy(v)))
        if i<8:
            d=r['gradients'][i];core.keys(d,('context','native_float32_loss','raw_gradient','chain','logit_gradient'))
            g=core.finite(d['raw_gradient'],(5,2),True);chain=1-a.astype(float)**2;dx=(g*chain).reshape(10)
            require(d['context']==v['context'] and d['chain']==chain.tolist() and d['logit_gradient']==dx.tolist()
                and type(d['native_float32_loss']) in (int,float) and np.isfinite(d['native_float32_loss']) and d['native_float32_loss']>=0,
                'same observed suffix derivative point / analytic chain once')
            want.extend([dict(kind='observed_gradient_intent',iteration=i+1,context_sha256=digest(q['observed_context']),raw_bits=frozen.bits(a)),
                dict(kind='observed_gradient_return',iteration=i+1,gradient=deepcopy(d))])
            x=frozen.proposal(x,dx);require(np.linalg.norm(x)<=frozen.CAP,'new projected strict inward domain')
    k=min(range(9),key=lambda j:(r['iterates'][j]['Q'],j));v=r['iterates'][k]
    require(r['winner_iteration']==k and r['commands']==q['incumbent']['prefix']+v['commands'] and r['selected_context']==v['context']
        and r['new_suffix_Q']==v['Q']<=r['observed_incumbent_Q']==r['iterates'][0]['Q']
        and frozen.bits(r['commands'][:5])==q['incumbent']['prefix_bits'],'full10 exact fixed prefix, stable best9 observed-Q')
    want.append(dict(kind='observed_suffix_selected',result_sha256=digest(r),iteration=k,commands=r['commands']))
    require(events==want,'exact full numeric intent/return/selection chronology, no ignored events')
    return dict(numerical_verified=True,production_authority_verified=False,scheduled_cost=contract()['per_slot'])


def panel(requests,backend_factory,sink):
    require(len(requests)==4 and [q['slot'] for q in requests]==list(range(4)),'fixed all4 selected prefix denominator, no subset')
    for q in requests:validate_request(q)
    rows=[dict(slot=i,status='not_started',result=None,error=None) for i in range(4)]
    for q,row in zip(requests,rows,strict=True):
        events=[];row['status']='attempted'
        def emit(e):events.append(deepcopy(e));sink(dict(slot=q['slot'],event=deepcopy(e)))
        try:
            r=optimize(q,backend_factory(q),emit);verify_numeric(q,r,events)
            row.update(status='numeric_complete',result=r)
        except Exception as exc:
            row.update(status='failed_partial',error=repr(exc));break
        finally:row['events']=events
    return dict(schema=NAME+'.numeric_panel',fixed_denominator=4,known_roots=2,rows=rows,
        complete=all(r['status']=='numeric_complete' for r in rows),production_ready=False,physical_release=False,P1_admission=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('contract',));p.parse_args()
    print(core.canonical(contract()).decode(),flush=True)


if __name__=='__main__':main()
