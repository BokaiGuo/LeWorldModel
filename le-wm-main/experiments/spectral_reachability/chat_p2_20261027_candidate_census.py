"""Full predeclared slot census over native results, including stopped searches.

No search algorithm, model invocation, winner replacement or budget refund.
Actual model-entry counts must come from native instrumentation, not this ledger.
"""
from copy import deepcopy

from experiments.spectral_reachability import moving_p2_saved_program_graph_20261027_v1 as contract


def census(result, *, arm, expected_count):
    """Expected count comes from the externally sealed native plan, never len()."""
    contract.label(arm); contract.integer(expected_count,True)
    contract.require(result['arm']==arm,'native arm differs from sealed arm')
    attempts=result['attempts']; ledger=result['ledger']
    contract.require(type(attempts) is list and len(attempts)<=expected_count,'attempts exceed sealed denominator')
    slots=[]
    for i in range(expected_count):
        candidate_id=f'{arm}_{i:02d}'
        if i<len(attempts):
            attempt=attempts[i]
            contract.require(attempt['candidate_id']==candidate_id,'native candidate order/id drift')
            contract.require(type(attempt['completed']) is bool,'explicit completion required')
            status='completed' if attempt['completed'] else 'failed_or_interrupted'
            evidence=deepcopy(attempt)
        else:
            status='not_attempted_after_stop'; evidence=None
        slots.append(dict(index=i,candidate_id=candidate_id,status=status,native_attempt=evidence))
    totals=dict(predict_calls=0,predict_samples=0,action_encoder_calls=0,failed_packet_attempts=0)
    for entry in ledger:
        contract.require(type(entry['completed']) is bool,'explicit ledger completion required')
        for axis in ('predict_calls','predict_samples','action_encoder_calls'):
            contract.integer(entry[axis]);totals[axis]+=entry[axis]
        totals['failed_packet_attempts']+=int(not entry['completed'])
    # This adapter is specifically for the existing H2 producers: two packets
    # per candidate plus four independent winner-replay packets. Compilation
    # remains separately charged by the native operation journal, not here.
    expected_dispatch=2*expected_count+4
    all_completed=(all(s['status']=='completed' for s in slots) and bool(ledger)
                   and all(e['completed'] for e in ledger)
                   and all(totals[k]==expected_dispatch for k in ('predict_calls','predict_samples','action_encoder_calls')))
    native_complete=result.get('complete',result.get('candidate_and_replay_complete',result.get('prediction_budget_matched',False)))
    contract.require(type(native_complete) is bool,'native completion flag required')
    return dict(schema='chat_p2_20261027_candidate_census.v1',arm=arm,
                denominator=expected_count,attempted_count=len(attempts),slots=slots,
                completed_count=sum(s['status']=='completed' for s in slots),
                not_attempted_count=expected_count-len(attempts),
                native_result_sha256=contract.sha(contract.encode(result)),
                dispatch_ledger_totals=totals,dispatch_ledger_is_not_actual_model_entry_count=True,
                measured_world_model_api_calls=None,measured_world_model_batch_samples=None,
                failed_attempts_refunded=False,
                deployable_engineering_candidate=all_completed and native_complete,
                full_resource_budget_matched=False,P2_admission=False,execution_authorized=False)


def verify_census(saved, result, *, arm, expected_count):
    rebuilt=census(result,arm=arm,expected_count=expected_count)
    contract.require(contract.encode(saved)==contract.encode(rebuilt),'census drift')
    return True
