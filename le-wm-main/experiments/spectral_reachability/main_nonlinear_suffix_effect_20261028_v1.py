"""Fixed168 model-only suffix/selection decomposition after real strict consume."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_nonlinear_suffix_library_20261028_v2 as source

NAME='main_nonlinear_suffix_effect_20261028_v1'


def analyze(slot, rows):
    source.require(len(rows)==42 and [r['index'] for r in rows]==list(range(42)), 'whole ordered42')
    original=slot['original_generation_winner']['index']
    source.require(type(original) is int and 0<=original<42, 'original selector index')
    for c,r in zip(slot['candidates'],rows,strict=True):
        source.require(len(r['iterates'])==9 and len(r['gradients'])==8 and c['index']==r['index']
            and r['iterates'][0]['Q']==c['original_terminal_Q']
            and source.bits(r['commands'][:5])==c['prefix_bits']
            and r['selected_Q']==min(x['Q'] for x in r['iterates'])<=r['iterates'][0]['Q'], 'unchanged prefix/incumbent and nonlinear descent')
    terminal=min(rows,key=lambda r:(r['iterates'][0]['Q'],r['index']))
    new=min(rows,key=lambda r:(r['selected_Q'],r['index']))
    baseline=rows[original]['iterates'][0]['Q']
    selection=terminal['iterates'][0]['Q']-baseline
    optimization=new['selected_Q']-terminal['iterates'][0]['Q']
    root_cost=source.cost(slot['task']['goal'],slot['task']['root_context'])
    stage=[source.cost(slot['task']['goal'],r['mid']) for r in rows]
    uphill=[r for r in rows if stage[r['index']]>root_cost]
    best_uphill=min(uphill,key=lambda r:(r['selected_Q'],r['index'])) if uphill else None
    source.require(np.isclose(selection+optimization,new['selected_Q']-baseline,atol=1e-14,rtol=1e-14), 'exact telescoping model costs')
    return dict(index=slot['index'],root_slot=slot['root_slot'],condition=slot['condition'],
        original_index=original,terminal_only_index=terminal['index'],optimized_index=new['index'],
        original_Q=baseline,terminal_only_Q=terminal['iterates'][0]['Q'],optimized_Q=new['selected_Q'],
        terminal_reselection_delta=selection,nonlinear_suffix_envelope_delta=optimization,
        fixed_original_prefix_suffix_delta=rows[original]['selected_Q']-baseline,
        improved_prefixes=sum(r['selected_Q']<r['iterates'][0]['Q'] for r in rows),
        selected_iteration=new['winner_iteration'],root_Q=root_cost,selected_stage_Q=stage[new['index']],
        selected_model_uphill=stage[new['index']]>root_cost,model_uphill_prefixes=len(uphill),
        best_uphill=None if best_uphill is None else dict(index=best_uphill['index'],Q=best_uphill['selected_Q']),
        caveat='Descriptive model objective decomposition, not isolated physical causal effects or C2.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','producer-output','report-sha','receipt','receipt-sha'):
        p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output).absolute()
    source.require(out.parent==source.ROOT/'results' and out.name.startswith(NAME+'_') and not out.exists(), 'new own output')
    directory=Path(a.producer_output).resolve()
    strict=source.consume(directory,a.report_sha)
    source.require(source.sha(a.receipt)==a.receipt_sha and source.raw(strict)==Path(a.receipt).read_bytes()
        and strict['complete'] is True, 'entire actual persisted complete receipt')
    r=source.signal.read(directory/'report.json',source.contract()['final_reserve_bytes'])
    prep=source.signal.read(r['prepared'],16*1024**2)
    result=dict(schema=NAME,rows=[analyze(s,[x['result'] for x in r['rows'][42*i:42*i+42]])
        for i,s in enumerate(prep['slots'])],report_sha256=a.report_sha,receipt_sha256=a.receipt_sha,
        known_roots=2,root_conditions=4,candidate_denominator=168,new_neural_calls=0,new_physics_calls=0,
        full_upstream_rebuilt_here=True,P1_admission=False,scientific_admission=False)
    payload=json.dumps(result,sort_keys=True,allow_nan=False).encode()+b'\n'
    with out.open('xb') as f:f.write(payload)
    print(json.dumps(dict(output=str(out),sha256=hashlib.sha256(payload).hexdigest(),rows=result['rows'])))


if __name__=='__main__':main()
