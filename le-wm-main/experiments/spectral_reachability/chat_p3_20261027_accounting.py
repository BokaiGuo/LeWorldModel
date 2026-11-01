"""Post-execution saved-only paired accounting; no model/physics calls."""
import argparse
from pathlib import Path
import time

import numpy as np

from experiments.spectral_reachability import chat_p3_20261027_cuda as cuda
from experiments.spectral_reachability import chat_p3_20261027_physical as physical

ROOT=cuda.ROOT
CUDA_OUTPUT=physical.SCHEDULER_OUTPUT
CUDA_SHA=physical.SCHEDULER_SHA
PHYSICAL_OUTPUT=ROOT/'results/chat_p3_20261027_physical_run01'
PHYSICAL_SHA='d0f3b5b61dc66d1a87c2f940fe4035fa259f7f7d43c2161097500e047fa21cf5'


def resource_views(events):
    """Times are inclusive nested CPU+CUDA synchronization wall, not FLOPs."""
    attempts=[e['entry'] for e in events if e['kind']=='resource_attempt']
    returned={e['entry']['index']:e['entry'] for e in events if e['kind']=='resource_return'}
    out={}
    for group in ('shared',)+cuda.ARMS:
        es=[e for e in attempts if e['scope'][0]==group]
        module_inputs=[e for e in events if e['kind']=='module_input' and e['scope'][0]==group]
        pred=[cuda.old.unblob(e['arrays'][0]) for e in module_inputs if e['resource']=='predictor']
        ae=[cuda.old.unblob(e['arrays'][0]) for e in module_inputs if e['resource']=='action_encoder']
        proj=[cuda.old.unblob(e['arrays'][0]) for e in module_inputs if e['resource']=='pred_proj']
        calls={r:dict(attempted=sum(e['resource']==r for e in es),
                      completed=sum(e['resource']==r and e['index'] in returned for e in es)) for r in cuda.old.MAXIMA}
        axis_wall={r:sum(returned[e['index']]['ended_seconds']-e['started_seconds'] for e in es if e['resource']==r) for r in cuda.old.MAXIMA}
        out[group]=dict(calls=calls,predictor_batch_samples=sum(a.shape[0] for a in pred),
                        predictor_context_tokens=sum(a.shape[0]*a.shape[1] for a in pred),
                        action_encoder_batch_samples=sum(a.shape[0] for a in ae),
                        pred_proj_flattened_rows=sum(a.shape[0] for a in proj),
                        new_visual_encoder=0,new_visual_projector=0,
                        resource_inclusive_wall_seconds=axis_wall,
                        observed_scope_span_seconds=max(returned[e['index']]['ended_seconds'] for e in es)-min(e['started_seconds'] for e in es))
    return out


def manipulation(panel,physical_report):
    roots=[]
    for s in (0,1):
        rows={a:next(r for r in panel['rows'] if r['source_slot']==s and r['arm']==a) for a in cuda.ARMS}
        native={a:next(r for r in physical_report['rows'] if r['root_slot']==s and r['arm']==a) for a in cuda.ARMS}
        plans=[r['plan'] for r in rows.values()]
        for key in ('root_sha256','task_id','blind_sha256','task_export','task_sha256','task_contract_sha256','bank_sha256'):
            cuda.require(len({p[key] for p in plans})==1,'matched stable identity '+key)
        cuda.require(all(r['initial_exact'] for r in native.values()) and all(native[a]['goal']==native['U']['goal'] and native[a]['root_state']==native['U']['root_state'] for a in cuda.ARMS), 'actual paired same root/goal')
        winners={a:r['winner_index'] for a,r in rows.items()}
        commands={a:r['plan']['commands'] for a,r in rows.items()}
        prefixes={a:[c['index'] for c in r['candidates']] for a,r in rows.items()}
        root=dict(source_slot=s,root_sha256=plans[0]['root_sha256'],task_id=plans[0]['task_id'],
                  task_sha256=plans[0]['task_sha256'],bank_sha256=plans[0]['bank_sha256'],
                  searched_indices=prefixes,winner_indices=winners,
                  three_distinct_prefixes=len({tuple(prefixes[a]) for a in ('U','D_rank','S_rank')})==3,
                  all_winners_equal=len(set(winners.values()))==1,
                  all_literal_commands_equal=all(commands[a]==commands['U'] for a in cuda.ARMS),
                  all_actual_final_states_equal=all(native[a]['final_state']==native['U']['final_state'] for a in cuda.ARMS),
                  initial_metrics=native['U']['initial_metrics'],
                  actual_metrics={a:r['final_metrics'] for a,r in native.items()},
                  model_Q={a:r['plan']['model_cost'] for a,r in rows.items()})
        roots.append(root)
    return roots


def build():
    start=time.monotonic()
    cr=cuda.consume(CUDA_OUTPUT,CUDA_SHA)
    pr=physical.consume(PHYSICAL_OUTPUT,PHYSICAL_SHA)
    cp=cuda.read(CUDA_OUTPUT/'report.json'); pp=cuda.read(PHYSICAL_OUTPUT/'report.json')
    panel=cuda.read(CUDA_OUTPUT/'panel.json.gz',32*1024**2)
    plain,eof=cuda.old.blind.inflate((CUDA_OUTPUT/'events.jsonl.gz').read_bytes(),128*1024**2)
    cuda.require(eof,'full CUDA journal')
    events=[cuda.old.blind.decode(line) for line in plain.splitlines()]
    views=resource_views(events); roots=manipulation(panel,pp)
    cuda.require(all(r['three_distinct_prefixes'] for r in roots),'actual effective treatment, not quota-only receipt')
    cuda.require(all(views[a]['calls']['predict']==dict(attempted=128,completed=128) and views[a]['predictor_batch_samples']==128 for a in cuda.ARMS), 'equal paid arm API and actual samples')
    cpp=cuda.read(cp['prepared']); ppp=cuda.read(pp['prepared'])
    bindings=dict(cpp['bindings_sha256'])
    for f,h in ppp['bindings'].items():
        cuda.require(f not in bindings or bindings[f]==h,'nonconflicting paired input closure');bindings[f]=h
    for output,report in ((CUDA_OUTPUT,cp),(PHYSICAL_OUTPUT,pp)):
        bindings[str(output/'report.json')]=cuda.sha(output/'report.json')
        bindings[str(Path(report['prepared']).absolute())]=report['prepare_sha256']
        for f in output.iterdir():bindings[str(f)]=cuda.sha(f)
    files=[Path(__file__),ROOT/'tests/chat_p3_20261027_test_accounting.py']
    bindings.update({str(f.resolve()):cuda.sha(f) for f in files})
    for f,h in bindings.items():cuda.require(cuda.sha(f)==h,'final verified paired SHA '+f)
    return dict(schema='chat_p3_20261027.accounting.v1',status='effective_rank_contrast_executed_zero_gain_not_admitted',
                source_sha256=cuda.sha(Path(__file__)),bindings_sha256=bindings,
                queue_seal_sha256=cuda.SEAL_SHA,cuda_report_sha256=CUDA_SHA,physical_report_sha256=PHYSICAL_SHA,
                arms=list(cuda.ARMS),allocations=panel['allocations'],roots=roots,
                model_Q_comparison=cr['paired_summary'],native_comparison=pr['D_minus_control'],
                full_horizons=pr['full_horizons'],native_successes=pr['native_successes'],
                resource_views=views,shared_paid_once=views['shared'],
                standalone_logical_view={a:dict(predict_API=128+(0 if a=='N' else 8),predictor_batch_samples=128+(0 if a=='N' else 8),
                                                jacobian=0 if a=='N' else 2,VJP_directions=0 if a=='N' else 384,SVD=0 if a=='N' else 2,
                                                measured_wall_seconds=None) for a in cuda.ARMS},
                standalone_note='counterfactual logical deployment view, not additional execution; U pays shared per fixed protocol; no standalone timing measured',
                historical_resources=dict(encoder=8,projector=8,anchor_predict_API=288,
                                           full_encoder_receipt=cpp['historical_encoder'],full_anchor_receipts=cpp['historical_anchor_receipts'],
                                           other_history_cost='preserved upstream receipts; unknown never treated as zero'),
                cuda_wall_seconds=cp['limits']['elapsed_seconds'],physical_wall_seconds=pp['wall_seconds'],
                physical_wall_scope='physical dispatch lifecycle; saved closure/prepare and this accounting are separate overhead',
                physical_resources=pr['counts'],fresh_encoded_actual_Q=None,predicted_vs_actual_latent_MSE=None,
                missing_scientific_dependencies=['P1 admission receipt','certified fresh ordered root/goal manifest','fresh independent paired confirmation'],
                missing_optional_metrics=['fresh same-Q endpoint RGB encoding and its own consumer'],
                new_neural_in_accounting=0,new_physics_in_accounting=0,accounting_wall_seconds=time.monotonic()-start,
                equal_predict_API_and_sample_budget=True,equal_total_compute=False,equal_FLOPs=False,
                inference='two known roots; no CI/p-value/generalization; unchanged winners and all zero paired differences retained',**cuda.FLAGS)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    path=cuda.exclusive(a.output);value=build()
    import os
    with path.open('xb') as f:
        raw=cuda.rule.canonical(value);cuda.require(f.write(raw)==len(raw),'short accounting write');f.flush();os.fsync(f.fileno())
    print(cuda.rule.canonical(dict(output=str(path),sha256=cuda.sha(path),status=value['status'],
                                   cuda_wall_seconds=value['cuda_wall_seconds'],physical_wall_seconds=value['physical_wall_seconds'])).decode())


if __name__=='__main__':main()
