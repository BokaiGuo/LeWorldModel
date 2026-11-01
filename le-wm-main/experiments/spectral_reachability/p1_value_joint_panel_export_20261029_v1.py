"""Full joint saved consumer to same-candidate value export; no model or physics."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import agent_p1_joint_vreach_host_cuda_20261029_v1 as host
from experiments.spectral_reachability import p1_value_source_export_20261029 as exporter
from experiments.spectral_reachability import p1_value_binding_bridge_20261029 as bridge

NAME = 'p1_value_joint_panel_export_20261029_v1'
core, F = host.core, host.F
require, sha, raw = host.require, host.sha, host.raw
ROOT = host.ROOT


def export_arm(q, arm, result, scalar_receipt, events, report_sha):
    """Existing per-arm audit plus explicit identity and raw-byte bindings."""
    host.method.original.request(q)
    prepared = {'authority': {'rows': [None]*q['slot']+[{'request':q}]}}
    bundle = dict(slot=q['slot'], arm=arm, result=result, scalar_receipt=scalar_receipt, events=events)
    out = exporter.export_joint_arm(prepared,bundle)
    require(len(out['rows']) == len(result['candidates']) == 20, 'candidate denominator')
    task=q['task']; binding=task['binding']
    context_sha=core.digest(task['root_context']); goal_sha=core.digest(task['goal'])
    require(context_sha == binding['context_sha256'] and goal_sha == binding['goal_sha256'], 'root task context/goal binding')
    rows=[]
    for i,(exported,c) in enumerate(zip(out['rows'],result['candidates'],strict=True)):
        require((c['start'],c['iteration']) == (i//5,i%5) and
            (exported['slot'],exported['arm'],exported['start'],exported['iteration']) == (q['slot'],arm,i//5,i%5), 'candidate order and arm')
        prefix=np.asarray(c['prefix'],dtype='<f4'); suffix=np.asarray(c['suffix'],dtype='<f4')
        commands=np.asarray(c['commands'],dtype='<f4')
        require(prefix.shape == suffix.shape == (5,2) and commands.shape == (10,2), 'action dimensions')
        phex=prefix.tobytes().hex(); shex=suffix.tobytes().hex()
        bridge.action(phex,'prefix'); bridge.action(shex,'suffix')
        require(commands.tobytes() == prefix.tobytes()+suffix.tobytes() and c['raw_float32_bits'] == phex+shex
            and exported['raw_float32_bits'] == phex+shex, 'full action byte identity')
        core.checked_path(task,task['root_context'],prefix.tolist(),[c['mid']])
        core.checked_path(task,c['mid'],suffix.tolist(),[c['end']])
        core.checked_path(task,c['mid'],[[0.,0.]]*5,[c['signal']['nominal_context']])
        parent=core.digest(c['mid'])
        require(exported['parent_context_digest'] == parent, 'parent context identity')
        identity=dict(slot=q['slot'],root_slot=q['root_slot'],condition=q['condition'],arm=arm,
            start=i//5,iteration=i%5,task_id=binding['task_id'],root_sha256=binding['root_sha256'],
            task_sha256=binding['task_sha256'],context_sha256=context_sha,goal_sha256=goal_sha,
            parent_context_sha256=parent,model_sha256=binding['model_sha256'],normalizer_sha256=binding['normalizer_sha256'],
            projection_sha256=binding['projection_sha256'],winner_authority=q['winner_authority'],
            prefix_hex=phex,suffix_hex=shex,full_action_sha256=hashlib.sha256(prefix.tobytes()+suffix.tobytes()).hexdigest())
        identity['candidate_id']=core.digest(identity)
        rows.append(dict(**exported, identity=identity, prefix_hex=phex,suffix_hex=shex,
            root_step=None,prefix_end_step=None,terminal_step=None,
            time_status='relative packet boundaries 0/5/10; absolute physical time not authenticated here',
            relative_steps=[0,5,10], source_report_sha256=report_sha,
            observed_terminal=exporter.missing('no authenticated physical execution or observed terminal for this candidate')))
    return rows


def build(output, report_sha, receipt_path, receipt_sha):
    core.hash_value(report_sha); core.hash_value(receipt_sha)
    output=Path(output); rp=Path(receipt_path)
    require(output.is_absolute() and output.resolve()==output and rp.is_absolute() and rp.resolve()==rp
        and not rp.is_symlink() and sha(rp)==receipt_sha, 'receipt path/SHA')
    actual=host.consume(str(output),report_sha)
    require(raw(actual)==rp.read_bytes() and sha(rp)==receipt_sha, 'receipt differs from full saved consumer')
    require(actual['status']=='verified_model_only' and actual['complete'] is True and
        actual['report_sha256']==report_sha and len(actual['rows'])==8 and len(actual['audits'])==8 and
        all(actual[k] is False for k in ('P1_admission','freshness','physical_release')), 'complete original saved consumer')
    report=host.signal.read(output/'report.json',256*1024**2)
    p=host.verify(report['prepared'],report['prepare_sha256'])
    args=p['arguments']; prepared=host.authority.verify(args['authority_prepared'],args['authority_sha'])
    require(len(p['requests'])==len(prepared['authority']['rows'])==4, 'four requests')
    for q, source in zip(p['requests'],prepared['authority']['rows'],strict=True):
        require(raw(q)==raw(source['request']), 'exact authority request')
    rows=[]
    for i,row in enumerate(actual['rows']):
        require((row['index'],row['slot'],row['arm'],row['status'])==(i,i//2,host.method.ARMS[i%2],'sealed'), 'eight arm order')
        events=F.load_journal(output,row)
        rows.extend(export_arm(p['requests'][i//2],row['arm'],row['result'],row['resources'],events,report_sha))
    require(len(rows)==160 and len({r['identity']['candidate_id'] for r in rows})==160, 'complete unique candidate roster')
    require(sha(output/'report.json')==report_sha and sha(rp)==receipt_sha, 'unchanged source artifacts')
    return dict(schema=NAME+'.export',source='joint_completed_panel', rows=rows,slot_denominator=4,
        arm_denominator=8,candidate_denominator=160,source_report_sha256=report_sha,source_receipt_sha256=receipt_sha,
        source_prepare_sha256=report['prepare_sha256'],source_authority_sha256=args['authority_sha'],
        source_consumer='full saved consumer rebuilt and byte-matched; each arm independently replay-audited',
        physical='missing for every candidate; no physical join attempted',scientific_admission='unchanged',
        P1_admission=False,freshness=False,physical_release=False,
        limitation='saved IO reconstruction is not hardware attestation or fresh-root/physical evidence')


def write(path,value):
    data=raw(value); path=Path(path)
    with path.open('xb') as f:
        require(f.write(data)==len(data),'complete write');f.flush();os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return sha(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('output-dir','report-sha','receipt','receipt-sha','export'):p.add_argument('--'+k,required=True)
    a=p.parse_args(); out=Path(a.export)
    require(out.is_absolute() and out.resolve()==out and out.parent==ROOT/'results' and
        out.name.startswith(NAME+'_') and not out.exists() and not out.is_symlink(),'exclusive output path')
    value=build(a.output_dir,a.report_sha,a.receipt,a.receipt_sha)
    print(json.dumps(dict(path=str(out),sha256=write(out,value),candidates=len(value['rows'])),sort_keys=True))


if __name__=='__main__': main()
