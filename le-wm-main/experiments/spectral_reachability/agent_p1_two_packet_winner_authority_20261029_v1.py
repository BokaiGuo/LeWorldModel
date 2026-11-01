"""Authenticate original mixedmode winner and its literal two chart parameters.

This source-binding component is not a model runner or a complete host.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_two_packet_uphill_generator_20261029_v1 as method
from experiments.spectral_reachability import agent_mixedmode_prefix_20261028_v1 as mixed

NAME='agent_p1_two_packet_winner_authority_20261029_v1'
F=method.F
require,sha,raw=F.require,F.sha,F.raw
ROOT=F.ROOT


def join(p,r,accepted,tasks,report_sha):
    """Pure join; caller must first fully reconstruct mixed.consume(report)."""
    require(p['schema']==mixed.NAME+'.prepare' and p['contract']==mixed.contract()
            and r['schema']==mixed.NAME+'.report' and r['contract']==p['contract']
            and r['status']=='all4_mixedmode_plans_sealed_not_admitted' and not r['errors']
            and r['model_hash_before']==r['model_hash_after']==p['expected_model_hash'], 'original mixed source/model')
    require(accepted['schema']==mixed.NAME+'.consume' and accepted['status']=='all4_mixedmode_commands_verified_not_admitted'
            and accepted['errors']==[] and accepted['P1_admission'] is False and len(accepted['plans'])==4
            and len(p['slots'])==len(r['rows'])==len(tasks)==4,'full four original winners')
    result=[]
    for i,(slot,row,plan,task) in enumerate(zip(p['slots'],r['rows'],accepted['plans'],tasks,strict=True)):
        require(type(slot['index']) is type(row['index']) is int and slot['index']==row['index']==i
                and (slot['root_slot'],slot['condition'])==mixed.ORDER[i]
                and (row['root_slot'],row['condition'])==mixed.ORDER[i] and row['status']=='plan_sealed','fixed original slot')
        require(plan['whole_run_report_sha256']==report_sha and {k:v for k,v in plan.items() if k!='whole_run_report_sha256'}==row['plan'], 'exact original consumed plan')
        w=row['result']['winner'];idx=w['index'];require(type(idx) is int and 0<=idx<mixed.N
            and row['result']['candidates'][idx]==w and len(row['result']['candidates'])==mixed.N, 'exact complete original winner candidate')
        t=F.core.validate_task(task)
        require(t['root_context']==slot['packet']['root_context'] and t['goal']==slot['packet']['goal']
                and t['binding']['task_id']==slot['task_id'] and np.array_equal(t['projection'],np.eye(192)), 'same frozen model task/root/goal/Q')
        prefix=np.asarray(w['prefix_params'],dtype=np.float64);suffix=np.asarray(w['suffix_params'],dtype=np.float64)
        require(prefix.shape==suffix.shape==(10,) and np.isfinite(prefix).all() and np.isfinite(suffix).all()
            and float(np.linalg.norm(prefix))<=F.CAP and w['suffix_solve']['params']==w['suffix_params']
            and w['suffix_solve']['radius']==F.CAP and w['suffix_solve']['norm']==float(np.linalg.norm(suffix))
            and w['suffix_solve']['norm']<=F.CAP,'literal original prefix ball and suffix solver witness')
        a,b=F.chart(prefix),F.chart(suffix);commands=np.concatenate([a,b])
        require(F.bits(a)==F.bits(w['prefix_commands']) and F.bits(b)==F.bits(w['suffix_commands'])
            and F.bits(commands)==F.bits(w['commands'])==F.bits(plan['commands']),'literal original saved chart/raw bits')
        q=dict(slot=i,root_slot=slot['root_slot'],condition=slot['condition'],task=deepcopy(t),
            winner_prefix_logits=w['prefix_params'],winner_suffix_logits=w['suffix_params'],winner_commands=w['commands'],
            winner_raw_bits=F.bits(commands),winner_authority=F.digest(dict(report_sha256=report_sha,slot=i,winner=w)))
        method.request(q);result.append(dict(request=q,winner_index=idx,prefix_norm=float(np.linalg.norm(prefix)),
            suffix_norm=w['suffix_solve']['norm'],suffix_radius=w['suffix_solve']['radius'],source_plan=deepcopy(plan)))
    return result


def material(output,report_sha):
    """Full producer strict; MAIN only: worker must not call this on production."""
    mixed.core.hash_value(report_sha);output=Path(output).absolute()
    require(sha(output/'report.json')==report_sha,'external source report SHA')
    accepted=mixed.consume(output,report_sha)
    r=mixed.overlay.read(output/'report.json',mixed.CAP);p=mixed.overlay.read(r['prepared'],mixed.PREP_CAP)
    require(sha(r['prepared'])==r['prepare_sha256'],'actual producer preparation SHA')
    tasks=[]
    for slot in p['slots']:
        require(sha(slot['task_file'])==slot['task_sha256']==p['bindings'][slot['task_file']],'actual task file SHA')
        saved=mixed.overlay.read(slot['task_file'],mixed.PREP_CAP)
        require(saved['task_id']==slot['task_id'] and saved['task']['root_context']==slot['packet']['root_context']
                and saved['task']['goal']==slot['packet']['goal'],'source task packet identity')
        tasks.append(F.task_from(slot,p['expected_model_hash']))
    rows=join(p,r,accepted,tasks,report_sha);bindings=deepcopy(p['bindings'])
    for name,h in r['artifact_sha256'].items():
        require(Path(name).name==name,'producer flat artifact')
        bindings[str(output/name)]=h
    bindings[str(output/'report.json')]=report_sha;bindings[r['prepared']]=r['prepare_sha256']
    F.check_bindings(bindings)
    return dict(schema=NAME+'.material',source_report_sha256=report_sha,source_prepare_sha256=r['prepare_sha256'],
                whole_producer_strict_reconstructed=True,rows=rows,bindings=bindings,
                policy=p['policy'],expected_model_hash=p['expected_model_hash'],P1_admission=False,physical_release=False)
