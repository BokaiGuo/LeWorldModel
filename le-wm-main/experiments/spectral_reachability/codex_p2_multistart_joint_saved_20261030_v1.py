"""Separate-process saved verifier; no model execution or CUDA initialization."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import codex_p2_multistart_joint_host_20261030_v1 as host
from experiments.spectral_reachability import codex_p2_program_library_saved_20261030_v1 as joint_saved

NAME='codex_p2_multistart_joint_saved_20261030_v1'
k=host.budget
need=k.old.require


def validate_branch(case,branch,events):
    arm=branch['arm'];base=case['base']; ctx=case['context']; goal=np.asarray(case['goal'],dtype=np.float64)
    need(arm in k.ARMS and branch['base']==base and branch['schema']==k.NAME+'.branch'
         and branch['context_sha256']==k.old.digest(ctx) and branch['goal_sha256']==k.old.digest(case['goal'])
         and branch['physical'] is False and branch['P2_admission'] is False,'branch source identity')
    s,v=k.old.canonical_basis(branch['root_jacobian'])
    need(np.array_equal(s,branch['root_singular_values']) and np.array_equal(v,branch['root_basis']),
         'saved root Jacobian and SVD')
    library,_=k.old.candidates(v)
    rows=branch['candidates']
    need(len(rows)==16,'sixteen fixed candidates')
    scoped=[e['event'] for e in events if e.get('base')==base and e.get('arm')==arm and 'event' in e]
    need(len(scoped)==len(events) and scoped[0]['kind']=='branch_start'
         and scoped[-1]['kind']=='branch_complete' and scoped[-1]['result_sha256']==k.old.digest(branch)
         and scoped[0]['context_sha256']==branch['context_sha256']
         and scoped[0]['goal_sha256']==branch['goal_sha256'] and scoped[0]['caps']==k.CAPS
         and scoped[-1]['resource_counts']==branch['resource_counts'],'journal branch bookends')
    attempts=dict.fromkeys(k.CAPS,0);returns=dict.fromkeys(k.CAPS,0);stack=[]
    permitted={'branch_start','branch_complete','packet_intent','packet_return','candidate_return',
               'prefix_return','joint_anchor_return','winner_replayed'}
    for e in scoped:
        kind=e['kind']
        if kind=='resource_attempt':
            axis=e['resource']
            need(axis in attempts and e['ordinal']==attempts[axis]+1 and e['depth']==len(stack),'resource intent')
            attempts[axis]+=1;stack.append((axis,e['ordinal']))
        elif kind=='resource_return':
            axis=e['resource']
            need(stack and stack[-1]==(axis,e['ordinal']) and e['depth']==len(stack)-1,'nested resource return')
            stack.pop();returns[axis]+=1
        else:
            need(kind in permitted,'unexpected failure or event')
    counts=branch['resource_counts']
    need(not stack and attempts==returns==counts['attempted']==counts['returned']
         and all(attempts[x]<=k.CAPS[x] for x in attempts) and counts['failed'] is False
         and 0<=counts['wall_seconds']<k.contract()['wall_seconds'],'actual per-axis caps')
    need(attempts['packet']>0 and all(attempts[x]==attempts['packet']
         for x in ('action_encoder','predict','predictor','pred_proj')),'actual model entry instrumentation')
    if arm=='joint_multistart':
        need(attempts['jacobian']==attempts['svd']==5 and attempts['packet']==44
             and attempts['score']==17 and len(branch['prefix_contexts'])==4,'joint actual census')
        anchors=[e for e in scoped if e['kind']=='joint_anchor_return']
        scored=[e for e in scoped if e['kind']=='candidate_return']
        need(len(anchors)==4 and len(scored)==16,'all joint anchor and candidate events')
        for p,item in enumerate(branch['prefix_contexts']):
            first=library[0][p]
            anchor=np.asarray(first['logits']+[0.0]*10,dtype=np.float64)
            singular,local=k.old.canonical_basis(item['jacobian'])
            need(item['prefix_index']==p and item['anchor_logits']==anchor.tolist()
                and np.array_equal(singular,item['singular_values']) and np.array_equal(local,item['basis'])
                and anchors[p]['prefix_index']==p and anchors[p]['anchor_logits']==anchor.tolist(),
                'four independent joint anchored SVDs')
            for n in range(4):
                row=rows[4*p+n];mode=n//2;sign=(-1,1)[n%2]
                logits=anchor+sign*(k.old.CAP/math.sqrt(2))*local[mode]
                norm=float(np.linalg.norm(logits))
                if norm>k.old.CAP:logits=logits*(k.old.CAP/norm)
                raw=np.tanh(logits).astype('<f4').reshape(10,2)
                mid=row['prefix_context'];final=row['terminal_context']
                z=np.asarray(final['latent_history'][-1],dtype=np.float64)
                need(row['index']==4*p+n and row['prefix_index']==p and row['mode']==mode and row['sign']==sign
                    and row['logits']==logits.tolist() and row['raw_float32_hex']==raw.tobytes().hex()
                    and row['prefix_context_sha256']==k.old.digest(mid)
                    and row['terminal_context_sha256']==k.old.digest(final)
                    and z.shape==(192,) and np.isfinite(z).all() and row['cost']==float(np.mean((z-goal)**2))
                    and scored[4*p+n]['candidate']==row,'joint exact full20 candidate and score')
        winner=min(rows,key=lambda x:(x['cost'],x['index']))
        need(branch['winner']==winner and branch['replay']==dict(
            prefix_context_sha256=winner['prefix_context_sha256'],
            terminal_context_sha256=winner['terminal_context_sha256'],cost=winner['cost']),
            'joint winner and replay binding')
        replay=[e for e in scoped if e['kind']=='winner_replayed']
        need(len(replay)==1 and replay[0]['index']==winner['index'] and replay[0]['replay']==branch['replay'],
            'joint replay event')
    else:
        need(attempts['jacobian']==attempts['svd']==5 and attempts['packet']==28
             and attempts['score']==17 and len(branch['prefix_contexts'])==4,'prefix arm actual census')
        pref=[e for e in scoped if e['kind']=='prefix_return']
        scored=[e for e in scoped if e['kind']=='candidate_return']
        need(len(pref)==4 and len(scored)==16,'all prefix and candidate events')
        for p,item in enumerate(branch['prefix_contexts']):
            first=library[0][p];mid=item['context']
            singular,local=k.local_basis(item['jacobian'])
            need(item['prefix_index']==p and item['context_sha256']==k.old.digest(mid)
                and np.array_equal(singular,item['singular_values']) and np.array_equal(local,item['basis'])
                and pref[p]['prefix_index']==p and pref[p]['program_id']==first['program_id']
                and pref[p]['raw_float32_hex']==first['raw_float32_hex']
                and pref[p]['context_sha256']==item['context_sha256'],'distinct-prefix local Jacobian')
            for n in range(4):
                row=rows[4*p+n]; mode=n//2;sign=(-1,1)[n%2]
                second=np.tanh(sign*(k.old.CAP/math.sqrt(2))*local[mode]).astype('<f4').reshape(5,2)
                final=row['terminal_context'];z=np.asarray(final['latent_history'][-1],dtype=np.float64)
                need(row['index']==4*p+n and row['prefix_index']==p and row['mode']==mode and row['sign']==sign
                    and row['first_raw_float32_hex']==first['raw_float32_hex']
                    and row['second_raw_float32_hex']==second.tobytes().hex()
                    and row['prefix_context_sha256']==item['context_sha256']
                    and row['terminal_context_sha256']==k.old.digest(final) and z.shape==(192,)
                    and np.isfinite(z).all() and row['cost']==float(np.mean((z-goal)**2))
                    and scored[4*p+n]['candidate']==row,'candidate exact raw/context/score')
        winner=min(rows,key=lambda x:(x['cost'],x['index']))
        need(branch['winner']==winner and branch['replay']==dict(
            prefix_context_sha256=winner['prefix_context_sha256'],
            terminal_context_sha256=winner['terminal_context_sha256'],cost=winner['cost']),
            'winner and actual replay binding')
        replay=[e for e in scoped if e['kind']=='winner_replayed']
        need(len(replay)==1 and replay[0]['index']==winner['index'] and replay[0]['replay']==branch['replay'],
            'durable winner replay event')
    return dict(base=base,arm=arm,status='saved_structure_and_resources_verified',
                counts=counts,winner=branch['winner'],P2_admission=False)


def read_events(path):
    raw=path.read_bytes()
    need(raw.endswith(b'\n') and len(raw)<=host.CAP,'bounded complete journal')
    return host.journal.read(path,json.loads(raw.splitlines()[-1])['sha256'],host.CAP)


def consume(directory,report_sha,supervisor_sha,prepared,prepare_sha):
    out=Path(directory)
    need(out.is_absolute() and out.is_dir() and not out.is_symlink(),'saved output')
    manifest=host.verify(prepared,prepare_sha)
    need(host.prep.original.previous.sha(out/'report.json')==report_sha and
         host.prep.original.previous.sha(out/'supervisor_events.jsonl')==supervisor_sha,'external hashes')
    report=json.loads((out/'report.json').read_bytes());sup=read_events(out/'supervisor_events.jsonl')
    need([e['kind'] for e in sup]==['lock_acquired','child_spawn_intent','child_spawned',
         'child_exited','supervisor_terminal'] and sup[-1]['returncode']==sup[-2]['returncode']==0,
         'complete supervisor lifecycle')
    proof=report['lock_proof'];need((proof['device'],proof['inode'])==(sup[0]['device'],sup[0]['inode']),
                                    'inherited cooperative inode')
    plan=manifest['parent'];panel=json.loads(Path(plan['panel_path']).read_bytes())
    need(report['schema']==host.NAME+'.report' and report['status']=='exploratory_complete'
         and report['prepare_sha256']==prepare_sha and report['error'] is None
         and report['device']['uuid']==plan['gpu_uuid'] and
         report['model_before']==report['model_after']==plan['expected_model_hash']
         and report['physical'] is False and report['P2_admission'] is False,'real model and device scope')
    host.identity.verify(report['loaded_sources'])
    events=host.journal.read(out/'child_events.jsonl',report['journal_tail'],host.CAP)
    need(events[0]['kind']=='child_start' and events[0]['lock_proof']==proof
         and events[0]['order']==k.contract()['order'] and events[-1]['kind']=='child_terminal'
         and events[-1]['status']=='exploratory_complete' and events[-1]['error'] is None,'child journal lifecycle')
    loaded=[e for e in events if e.get('kind')=='model_loaded']
    need(len(loaded)==1 and loaded[0]['device']==report['device']
         and loaded[0]['model_sha256']==report['model_before'],'model event')
    rows=report['rows'];need(len(rows)==8,'eight fixed slots')
    results=[];names={'report.json','child_events.jsonl','supervisor_events.jsonl'}
    for index,((base,arm),row) in enumerate(zip(k.contract()['order'],rows,strict=True)):
        name=f'branch_{index}.json'
        need((row['base'],row['arm'],row['status'],row['file'])==(base,arm,'complete',name)
             and host.prep.original.previous.sha(out/name)==row['sha256'],'saved branch hash')
        branch=json.loads((out/name).read_bytes())
        bound=[e for e in events if e.get('base')==base and e.get('arm')==arm and 'event' in e]
        checked=validate_branch(panel['cases'][base],branch,bound)
        file_event=[e for e in events if e.get('kind')=='branch_file' and e.get('index')==index]
        need(len(file_event)==1 and file_event[0]['file']==name and file_event[0]['sha256']==row['sha256']
             and checked['counts']==row['resource_counts'],'branch journal file binding')
        results.append(checked);names.add(name)
    need({p.name for p in out.iterdir()}==names and all(p.is_file() and not p.is_symlink() for p in out.iterdir()),
         'exact saved artifact inventory')
    return dict(schema=NAME+'.receipt',status='complete_saved_not_scientific',rows=results,
        report_sha256=report_sha,supervisor_sha256=supervisor_sha,prepare_sha256=prepare_sha,
        physical=False,P1_admission=False,P2_admission=False,independent_confirmation=False,
        full_source_closure='unknown',total_flops=None,energy=None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('output','report-sha','supervisor-sha','prepared','prepare-sha','receipt'):
        p.add_argument('--'+field,required=True)
    a=p.parse_args();path=Path(a.receipt)
    need(path.is_absolute() and path.parent==host.ROOT/'results' and path.name.startswith(NAME+'_RECEIPT_'),
         'exclusive receipt')
    value=consume(a.output,a.report_sha,a.supervisor_sha,a.prepared,a.prepare_sha)
    print(json.dumps(dict(path=str(path),sha256=host.prep.original.previous.write_new(path,value),
                          status=value['status'])))


if __name__=='__main__':
    main()
