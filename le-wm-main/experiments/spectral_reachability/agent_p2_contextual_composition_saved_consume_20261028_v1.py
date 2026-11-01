"""Strict record+empty-line saved adapter; never edit/re-run the sealed producer.

All native consumption checks are retained except explicit framing and untagged
candidate terminal ABI seams. No arbitrary filtering or GPU/physics path.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np

from experiments.spectral_reachability import agent_p2_contextual_composition_runtime_20261028_v1 as producer
from experiments.spectral_reachability.agent_p2_contextual_composition_runtime_20261028_v1 import (
    require, sha, overlay, verify, NAME, protocol, ORDER, loader, EXPANDED,
    task_bindings, Gate, PREP_CAP, frontier, expanded_bank, digest,
    check_tensor_chain, check_module_event_order, ARMS, provenance, check_files)

STEM='agent_p2_contextual_composition_saved_consume_20261028_v1'
ROOT=producer.ROOT
PRODUCER_SHA='32db978cd943c571756b675be95e159020e2c4ffda1f926cdd4e8bd45896d205'
DIRECTORY=ROOT/'results/agent_p2_contextual_composition_runtime_20261028_v1_cuda_MAIN01'
REPORT_SHA='498b45ed9f4c580064d8f2a5dfd00ac8542b7ce9d0b86e6587d9ab6a9639d49c'
RECEIPT_CAP=32*1024**2


def framed_events(raw):
    """Exactly one canonical JSON line then exactly one empty LF line.

    Original canonical() already supplies LF. Store.event appended a second
    LF. CRLF, missing/extra separator, blank record and nonempty bad JSON fail.
    This creates an event view only; original compressed/plain bytes unchanged.
    """
    require(type(raw) is bytes and 0<len(raw)<=EXPANDED and raw.endswith(b'\n\n'),
        'complete bounded record+empty framing')
    pieces=raw.split(b'\n')
    require(pieces[-1]==b'' and len(pieces)%2==1 and len(pieces)>=3,
        'exact two-LF framing length')
    records=pieces[:-1:2]; separators=pieces[1:-1:2]
    require(len(records)==len(separators) and all(records) and all(x==b'' for x in separators),
        'strict alternating nonempty record and ONE empty separator')
    events=[]
    for line in records:
        value=overlay.blind.decode(line)
        require(type(value) is dict and producer.canonical(value)==line+b'\n',
            'exact original canonical record ABI, no repair/re-encoding')
        events.append(value)
    evidence=dict(framing='canonical_JSON_LF_then_exactly_one_empty_LF',
        decoded_bytes=len(raw),decoded_sha256=hashlib.sha256(raw).hexdigest(),
        splitline_count=2*len(records),record_count=len(records),empty_separator_count=len(separators),
        strictly_alternating_empty=True,original_bytes_preserved=True,
        arbitrary_blank_filter=False,nonempty_bad_lines_ignored=0)
    return events,evidence


def tagged_events(events,results):
    """Prove every untagged native record IS an original candidate terminal.

    The frozen order helper indexes kind directly, but run_arm emits candidate
    dictionaries without kind. Exclude only the exact complete candidate list
    from its typed hook view; Gate still receives every original native record.
    """
    expected=[c for result in results for c in result['attempts']]
    require(len(results)==2 and [len(r['attempts']) for r in results]==[80,40],
        'full120 candidate terminal denominator')
    actual=[]; tagged=[]
    kinds={'entry_attempt','entry_terminal','candidate_request','array_saved',
        'suffix_jacobian','suffix_solve','jacobian','bounded_ridge_svd','mode_svd','bank_frozen'}
    for e in events:
        if e['kind']=='native_event':
            x=e['event']; require(type(x) is dict,'native event dictionary')
            if 'kind' not in x:
                actual.append(x)
                continue
            require(x['kind'] in kinds,'known tagged native event kind')
        tagged.append(e)
    require(producer.canonical(actual)==producer.canonical(expected),
        'all120 untagged candidate terminals exactly match original attempts in order')
    return tagged


def consume_native(output, report_sha):
    # Keep the frozen producer.consume assertions/helper calls unchanged.
    # Only its three-line journal parser is replaced by framed_events().
    output=Path(output).absolute(); require(sha(output/'report.json')==report_sha, 'external report SHA')
    r=overlay.read(output/'report.json', 8*1024**2); p=verify(r['prepared'],r['prepare_sha256'])
    require(r['schema']==NAME+'.report' and r['status']=='known_contextual_plans_sealed' and r['errors']==[] and
        r['contract']==protocol() and r['source_verified'] is r['model_unchanged'] is True and
        r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'] and
        r['P1_admission'] is r['P2_admission'] is r['physical_release'] is False, 'terminal complete only, no admission')
    require(len(r['rows'])==4 and [(x['index'],x['root_slot'],x['condition'],x['status']) for x in r['rows']]==
        [(i,*ORDER[i],'complete') for i in range(4)], 'fixed four-slot terminal denominator')
    names={'binding.json','loader.json','events.jsonl.gz'}
    for i in range(4):
        names.update((f'slot{i}_result.json.gz',f'slot{i}_bank.json.gz'))
        names.update(f'slot{i}_{s}.json.gz' for s in ['shared_nominal','shared_jacobian']+
            [f'suffix{j}_{stage}' for j in range(40) for stage in ('nominal','jacobian')])
    require(set(r['artifact_sha256'])==names and {x.name for x in output.iterdir()}==names|{'report.json'}, 'exact complete artifacts')
    for name,h in r['artifact_sha256'].items():
        require(not (output/name).is_symlink() and sha(output/name)==h, 'artifact hash')
    require(overlay.read(output/'binding.json')==dict(prepared=r['prepared'],prepare_sha256=r['prepare_sha256']), 'execution binding')
    loader.validate_loader(overlay.read(output/'loader.json'),p,r['prepare_sha256'])
    raw,eof=overlay.blind.inflate((output/'events.jsonl.gz').read_bytes(),EXPANDED)
    require(eof and raw.endswith(b'\n'), 'complete bounded journal')
    events,framing=framed_events(raw)
    load=overlay.read(output/'loader.json')
    expected_load=[e for f in load['files'] for e in [dict(kind='load_intent',path=f['path'],expected_sha256=f['sha256']),
        dict(kind='consumed_buffer_frozen',receipt=f)]]
    require(events[:4]==expected_load, 'actual immutable loader buffers')
    plans=[]; offset=4
    for slot,rr in zip(p['slots'],r['rows'],strict=True):
        require(rr['result_file']==f'slot{slot["index"]}_result.json.gz' and rr['candidate_denominator']==[80,40],
            'fixed own result path and full failure denominator')
        row=overlay.read(output/rr['result_file'],EXPANDED); ready,compiled=task_bindings(slot)
        require(row['index']==slot['index'] and row['ready']==ready and row['compiled']==compiled and row['status']=='complete'
            and row['candidate_denominator']==[80,40] and row['faults']==dict(primary=None,secondary=[]), 'complete own task row')
        emitted=[]; gate=Gate(emitted.append); own=[]
        while offset<len(events) and events[offset].get('slot')==slot['index']:
            own.append(events[offset]['payload']); offset+=1
        require(own, 'all slot journals')
        for e in own:
            if e['kind']=='native_event': gate(e['event'])
        require([x for x in own if x['kind'] in ('native_event','reserve_intent')]==emitted and
            gate.entries==row['actual_model_entries'] and gate.operations==row['operations'] and
            gate.complete(row['results'])==row['resources']==rr['resources'], 'intent/entry/return/actual count reconciliation')
        requests=[x['receipt'] for x in own if x['kind']=='module_attempt']
        returns=[x['receipt'] for x in own if x['kind']=='module_return']
        require(returns==row['modules'] and requests==[{k:v for k,v in x.items() if k!='output'}|{'completed':False}
            for x in returns], 'real module request/return receipts')
        check_module_event_order(tagged_events(own,row['results']))
        values={}
        array_names=['shared_nominal','shared_jacobian']+[f'suffix{i}_{stage}' for i in range(40) for stage in ('nominal','jacobian')]
        require([x['file'] for x in row['arrays']]==[f'slot{slot["index"]}_{n}.json.gz' for n in array_names],
            'all82 derivative arrays, exact order and own flat paths')
        require(row['derivative_stages']==[dict(kind=kind,index=i,stage=stage,returned=True,converted=True,persisted=True)
            for kind,i in [('jacobian',None)]+[('suffix_jacobian',i) for i in range(40)]
            for stage in ('nominal','jacobian')], 'all derivative stage durable completions')
        require([e['event']['receipt'] for e in emitted if e['kind']=='native_event' and e['event'].get('kind')=='array_saved']==row['arrays'],
            'durable derivative receipt order')
        for name,receipt in zip(array_names,row['arrays'],strict=True):
            require(sha(output/receipt['file'])==receipt['sha256'], 'derivative file SHA')
            body=overlay.read(output/receipt['file'],PREP_CAP); a=np.asarray(body['value'],np.float64)
            require(body['name']==name and list(a.shape)==body['shape'] and body['dtype']=='float64' and np.isfinite(a).all() and
                hashlib.sha256(a.tobytes()).hexdigest()==body['array_sha256'], 'original derivative array bytes')
            values[body['name']]=body
        base=frontier.legacy().sra.spectral_bank(values['shared_jacobian']['value'],values['shared_nominal']['value'],
            compiled['goal_vectors'][0],compiled['scoring']['metric']['projection'])
        require(row['base']==base and row['bank']==expanded_bank(base,slot['index']) and
            overlay.read(output/f'slot{slot["index"]}_bank.json.gz',PREP_CAP)==dict(base=base,bank=row['bank']), 'root derivatives/bank/all80')
        require(all(values['shared_'+stage]['binding']==dict(task_id=slot['task_id']) for stage in ('nominal','jacobian')), 'root derivative identity')
        for i,candidate in enumerate(row['results'][1]['attempts']):
            binding=dict(task_id=slot['task_id'],context_sha256=digest(candidate['witness']['edges'][0]['output_context']),
                prefix_sha256=digest(row['bank']['logits'][i][:10]),index=i)
            for stage in ('nominal','jacobian'):
                body=values[f'suffix{i}_{stage}']
                require(body['binding']==binding and body['value']==candidate['suffix_compilation'][stage], 'own-context suffix derivative identity')
        check_tensor_chain(slot,row)
        require([x['arm'] for x in row['results']]==list(ARMS), 'both arms ordered')
        costs=[dict(index=j,arm=result['arm'],context_sha256=digest(c['witness']['receipt']['terminal_context']),
            completed=True,value=c['cost']) for j,(result,c) in enumerate(
            (result,c) for result in row['results'] for c in result['attempts'])]
        require(row['costs']==costs and [e for e in own if e['kind'] in ('cost_attempt','cost_return')]==
            [e for ticket in costs for e in [dict(kind='cost_attempt',receipt={k:v for k,v in ticket.items() if k!='value'}|{'completed':False}),
                dict(kind='cost_return',receipt=ticket)]], 'all120 fixed nonlinear cost callback entries/returns')
        for arm,result in zip(ARMS,row['results'],strict=True):
            prov=provenance(p,slot,row['bank'],arm)
            best=frontier.validate_result(result,arm=arm,bank=row['bank'],ready=ready,prov=prov,compiled=compiled)
            plan=dict(root_slot=slot['root_slot'],condition=slot['condition'],arm=arm,task_id=slot['task_id'],
                commands=best['actions'],commands_dtype='<f4',raw_sha256=hashlib.sha256(np.asarray(best['actions'],'<f4').tobytes()).hexdigest(),
                root_context=ready['root']['context'],goal=compiled['goal_vectors'][0],witness=best['witness'],
                original_task_file=slot['task_file'],original_task_sha256=slot['task_sha256'],
                original_blind_sha256=slot['blind_sha256'],candidate_bank_sha256=digest(row['bank']),
                predicted_Q=best['cost'],prepare_sha256=r['prepare_sha256'],planner_report_sha256=report_sha,
                engineering_executable=True,P1_admission=False,P2_admission=False,physical_release=False)
            plan['selection_digest']=digest(plan); plans.append(plan)
    require(offset==len(events) and len(plans)==8 and sha(output/'report.json')==report_sha, 'no unscoped/missing slots')
    check_files(p)
    native=dict(schema=NAME+'.consume',status='engineering_verified_science_blocked',plans=plans,
        candidate_denominator=480,known_roots=2,scientific_slots=4,total_predict=1328,total_FLOPs=None,
        P1_admission=False,P2_admission=False,physical_release=False,
        blockers=['independent P1 not admitted','no independent fresh cohort','no new physical efficacy evidence'])
    return native,framing


def own_files():
    return [Path(__file__),ROOT/'tests'/('test_'+STEM+'.py'),ROOT/'innerdocs'/(STEM+'.md')]


def analyze(output,report_sha):
    require(sha(producer.__file__)==PRODUCER_SHA, 'unchanged frozen runtime')
    native,framing=consume_native(output,report_sha)
    require(producer.canonical({'x':1})==b'{"x":1}\n', 'original canonical ABI includes one LF')
    bindings={str(p.resolve()):sha(p) for p in own_files()}
    compressed=Path(output).absolute()/'events.jsonl.gz'
    framing.update(compressed_sha256=sha(compressed),compressed_bytes=compressed.stat().st_size,
        original_journal_path=str(compressed),producer_source_sha256=PRODUCER_SHA)
    return dict(schema=STEM+'.receipt',status='strict_saved_verified_science_blocked',
        planner_output=str(Path(output).absolute()),planner_report_sha256=report_sha,
        framing_evidence=framing,native_strict_receipt=native,bindings_sha256=bindings,
        new_neural_calls=0,new_physics=0,P1_admission=False,P2_admission=False,physical_release=False)


def write_receipt(path,value):
    path=Path(path).absolute(); payload=producer.canonical(value)
    require(path.name.startswith(STEM+'_') and len(payload)<=RECEIPT_CAP and
        not path.exists() and not path.is_symlink() and shutil.disk_usage(path.parent).free>=producer.FLOOR+len(payload),
        'exclusive own32MiB receipt and200MiB floor')
    with path.open('xb') as f:
        require(f.write(payload)==len(payload),'short receipt');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(status=value['status'],receipt=str(path),receipt_sha256=sha(path),
        plans=len(value['native_strict_receipt']['plans']),candidate_denominator=480,
        framing_evidence=value['framing_evidence'],P2_admission=False,physical_release=False)


def verify_receipt(path,expected):
    require(sha(path)==expected,'external own strict receipt SHA')
    saved=overlay.read(path,RECEIPT_CAP)
    actual=analyze(saved['planner_output'],saved['planner_report_sha256'])
    require(producer.canonical(saved)==producer.canonical(actual) and sha(path)==expected,
        'entire strict saved receipt independently reconstructed')
    return actual


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('analyze','verify'))
    p.add_argument('--planner-output',type=Path,default=DIRECTORY)
    p.add_argument('--report-sha',default=REPORT_SHA)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--receipt-sha')
    a=p.parse_args()
    if a.command=='analyze':summary=write_receipt(a.receipt,analyze(a.planner_output,a.report_sha))
    else:
        value=verify_receipt(a.receipt,a.receipt_sha)
        summary=dict(status=value['status'],receipt=str(a.receipt.absolute()),receipt_sha256=sha(a.receipt),
            plans=len(value['native_strict_receipt']['plans']),candidate_denominator=480,P2_admission=False,physical_release=False)
    print(producer.canonical(summary).decode(),end='',flush=True)


if __name__=='__main__':main()
