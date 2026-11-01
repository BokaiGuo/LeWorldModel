"""Fixed40 saved Q-order errors; no new model/physics calls or winner selection."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import itertools
import math
import os
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_p1_teacher_library40_encode_20261028_v2 as rgb

NAME='agent_p1_teacher_library40_rank_error_20261028_v1'
ROOT=rgb.ROOT
REPORT=ROOT/'results/agent_p1_teacher_library40_encode_20261028_v2_cuda_MAIN01/report.json'
REPORT_SHA='cb08d580b3ee2e8d15d2cc83065a8dc68a74672df0e2b602479a836469cb15f3'
PREP=ROOT/'results/agent_p1_teacher_library40_encode_20261028_v2_prepare_MAIN01.json'
PREP_SHA='499f1d310b1844bb05ba78c88d3eecc4285c7041a6482fd7c68fc2fa39ce8676'
RECEIPT=ROOT/'results/agent_p1_teacher_library40_encode_20261028_v2_consume_MAIN01.json'
RECEIPT_SHA='9dfb1081711505cef281f1ff547b1dad46b59cb0ea207e985497ecbf9cecf0a9'
require,sha,canonical,digest=rgb.require,rgb.sha,rgb.canonical,rgb.digest
CAP=8*1024**2


def sign(x):
    require(math.isfinite(x),'finite exact delta, no tolerance/tie relaxation')
    return int(x>0)-int(x<0)


def scalar_terms(predicted,actual,goal):
    p,g=[np.asarray(x,np.float64) for x in (predicted,goal)]
    require(p.shape==g.shape==(192,) and np.isfinite(p).all() and np.isfinite(g).all(),'original native192 prediction and goal')
    qp=float(np.mean((p-g)**2))
    if actual is None:
        return dict(predicted_Q=qp,actual_Q=None,Q_error=None,latent_MSE=None,cross_term=None,quadratic_term=None,identity_residual=None)
    a=np.asarray(actual,np.float64);require(a.shape==(192,) and np.isfinite(a).all(),'actual native192, no padding')
    d=p-a;v=a-g;qa=float(np.mean(v**2))
    cross=float(2*np.mean(v*d));square=float(np.mean(d**2));error=qp-qa
    return dict(predicted_Q=qp,actual_Q=qa,Q_error=error,latent_MSE=square,cross_term=cross,quadratic_term=square,
                identity_residual=error-(cross+square))


def midranks(values):
    if any(v is None for v in values):return [None]*len(values)
    for v in values:sign(v)
    # 1-based ascending midranks; output never reordered and no argmin/winner.
    return [1+sum(w<v for w in values)+(sum(w==v for w in values)-1)/2 for v in values]


def matching(a,b,kind):
    same_task=(a['root_slot'],a['condition'])==(b['root_slot'],b['condition'])
    require(same_task and a['goal192']==b['goal192'] and a['goal_binding']==b['goal_binding'],
            'legal matching requires exact same task/condition/native goal; no off/on or cross-root pool')
    if kind=='within_role_library':
        require(a['role']==b['role'] and a['anchor_id']==b['anchor_id'] and a['library_index']<b['library_index'],
                'within-role same anchor, fixed ascending library pairs')
    elif kind=='same_word_ZS':
        require(a['role']=='Z' and b['role']=='S' and a['library_index']==b['library_index'] and
            a['library_label']==b['library_label'] and a['suffix_raw_bits']==b['suffix_raw_bits'],
            'ZS matches only original identical suffix word; prefixes/contexts remain different')
    else:raise ValueError('unregistered matching stratum')


def pair(a,b,kind):
    matching(a,b,kind)
    dp=a['predicted_Q']-b['predicted_Q'];sp=sign(dp)
    out=dict(kind=kind,i=a['index'],j=b['index'],root_slot=a['root_slot'],condition=a['condition'],
        roles=[a['role'],b['role']],words=[a['library_label'],b['library_label']],
        predicted_delta=dp,predicted_sign=sp,actual_delta=None,actual_sign=None,error_delta=None,
        cross_delta=None,quadratic_delta=None,identity_residual=None,reversal=None,sign_agreement=None,
        classification='unknown_missing_terminal',available=False,missing_indices=[])
    missing=[r['index'] for r in (a,b) if r['actual_Q'] is None]
    if missing:out['missing_indices']=missing;return out
    da=a['actual_Q']-b['actual_Q'];sa=sign(da);error=dp-da
    cross=a['cross_term']-b['cross_term'];square=a['quadratic_term']-b['quadratic_term']
    classification=('both_tie' if sp==sa==0 else 'predicted_tie_only' if sp==0 else 'actual_tie_only' if sa==0 else 'concordant' if sp==sa else 'discordant')
    out.update(actual_delta=da,actual_sign=sa,error_delta=error,cross_delta=cross,quadratic_delta=square,
        identity_residual=error-(cross+square),reversal=sp*sa<0,sign_agreement=sp==sa,
        classification=classification,available=True)
    return out


def pair_summary(pairs):
    names=('concordant','discordant','both_tie','predicted_tie_only','actual_tie_only','unknown_missing_terminal')
    counts={k:sum(p['classification']==k for p in pairs) for k in names}
    known=sum(p['available'] for p in pairs);unknown=len(pairs)-known
    c,d=counts['concordant'],counts['discordant'];strict=c+d
    denominator=math.sqrt((strict+counts['predicted_tie_only'])*(strict+counts['actual_tie_only']))
    tau=(c-d)/denominator if denominator else None
    complete=unknown==0
    return dict(fixed_pair_denominator=len(pairs),available_pairs=known,unknown_pairs=unknown,counts=counts,
        strict_nonzero_comparable_pairs=strict,observed_reversals=d,observed_reversal_rate=d/strict if strict else None,
        observed_rate_denominator=strict,observed_rate_is_full_fixed_library=complete,
        full_fixed_library_reversal_rate=d/strict if complete and strict else None,
        full_rate_unknown_reason='missing_terminal' if unknown else 'no_strict_nonzero_pairs' if not strict else None,
        observed_tau_b=tau,full_fixed_library_tau_b=tau if complete else None,
        tau_b_unknown_reason='missing_terminal' if unknown else 'zero_rank_variation' if denominator==0 else None)


def assemble(rows):
    require(len(rows)==40 and [r['index'] for r in rows]==list(range(40)),'all40 literal slots, no outcome filtering')
    for i,r in enumerate(rows):
        task,rem=divmod(i,10);role,word=divmod(rem,5);slot,condition=((0,'off'),(0,'on'),(1,'off'),(1,'on'))[task]
        require((r['root_slot'],r['condition'],r['role'],r['library_index'],r['library_label'])==
            (slot,condition,('Z','S')[role],word,rgb.teacher.LIBRARY[word]),'fixed 4task x 2role x 5word order')
        require((r['actual_Q'] is None)==(i==12),'only original missing12; never reduce full40')
    groups=[];all_pairs=[]
    for anchor in range(8):
        members=rows[anchor*5:anchor*5+5]
        predicted_ranks=midranks([r['predicted_Q'] for r in members]);actual_ranks=midranks([r['actual_Q'] for r in members])
        for r,pr,ar in zip(members,predicted_ranks,actual_ranks,strict=True):
            r.update(predicted_rank_full5=pr,actual_rank_full5=ar,rank_error_full5=None if ar is None else pr-ar,
                     rank_status='unknown_incomplete_fixed_library' if ar is None else 'known_full5_midranks')
        pairs=[pair(a,b,'within_role_library') for a,b in itertools.combinations(members,2)]
        groups.append(dict(anchor_index=anchor,root_slot=members[0]['root_slot'],condition=members[0]['condition'],role=members[0]['role'],
            fixed_words=5,available_words=sum(r['actual_Q'] is not None for r in members),indices=[r['index'] for r in members],
            missing_indices=[r['index'] for r in members if r['actual_Q'] is None],summary=pair_summary(pairs),pairs=pairs))
        all_pairs+=pairs
    zs=[]
    for task in range(4):
        matches=[pair(rows[task*10+w],rows[task*10+5+w],'same_word_ZS') for w in range(5)]
        zs.append(dict(task=task,interpretation='same suffix word / same goal; different prefixes and teacher contexts, not isolated causality',
                       summary=pair_summary(matches),pairs=matches))
    return dict(rows=rows,groups=groups,within_role_summary=pair_summary(all_pairs),same_word_ZS=zs,
        same_word_ZS_summary=pair_summary([p for g in zs for p in g['pairs']]),
        rank_rows=dict(fixed=40,known=sum(r['rank_error_full5'] is not None for r in rows),unknown=sum(r['rank_error_full5'] is None for r in rows)))


def build():
    r=rgb.read(REPORT,REPORT_SHA);p=rgb.read(PREP,PREP_SHA);c=rgb.read(RECEIPT,RECEIPT_SHA)
    require(r['prepared']==str(PREP) and r['prepare_sha256']==PREP_SHA and c['report_sha256']==REPORT_SHA,'actual RGB report/prepare/MAIN receipt binding')
    # Reuse frozen full native validator; this entry is saved-only, never run().
    require(c==rgb.consume(REPORT.parent,REPORT_SHA),'complete native saved receipt, not a partial file/hash-only green light')
    require(p['schema']==rgb.NAME+'.prepare' and r['rows']==c['rows'] and len(r['rows'])==40,'all40 accepted rows exact')
    compiler=Path(p['compiler_output']);cw=rgb.read(ROOT/rgb.INPUTS['compiler_receipt'][0],rgb.INPUTS['compiler_receipt'][1])
    bindings={str(REPORT):REPORT_SHA,str(PREP):PREP_SHA,str(RECEIPT):RECEIPT_SHA}
    for path,h in rgb.INPUTS.values():bindings[str(ROOT/path)]=h
    arrays={};rows=[]
    def load(base,receipt):
        a=rgb.teacher.load_array(base,receipt);path=rgb.teacher.flat(base,receipt['file'])
        require(str(path) not in arrays or arrays[str(path)]==receipt,'no conflicting original array receipt')
        arrays[str(path)]=deepcopy(receipt);bindings[str(path)]=receipt['file_sha256'];return a
    for s,row,plan in zip(p['rows'],r['rows'],cw['plans'],strict=True):
        require(all(row[k]==s[k]==plan[k] for k in rgb.IDENTITY if k!='job_digest'),'original prediction/actual/plan identity')
        goal=load(compiler,s['goal_array']);pred=load(compiler,s['prediction_array'])
        require(goal.dtype==pred.dtype==np.float32 and goal.shape==(192,) and pred.shape==(3,192) and
            np.array_equal(goal,np.asarray(s['goal192'],np.float32)) and np.array_equal(pred[-1].astype(float),np.asarray(s['predicted192'],float)),
            'original actual saved arrays/native precision/goal/context, no new prediction')
        actual=None
        if row['eligible_for_encoding']:
            actual=load(REPORT.parent,row['latent']);require(actual.dtype==np.float32 and actual.shape==(192,),'saved original actual192')
        else:require(row['index']==12 and row['actual_latent'] is row['actual_Q'] is row['actual_model_error'] is None,'missing12 strictly null')
        terms=scalar_terms(pred[-1],actual,s['goal192'])
        require(terms['predicted_Q']==row['predicted_Q']==plan['model_Q'] and terms['actual_Q']==row['actual_Q'] and
                terms['latent_MSE']==row['actual_model_error'],'independent exact original Q and model MSE recomputation')
        raw=np.asarray(plan['commands'],dtype='<f4')
        require(raw.shape==(10,2) and hashlib.sha256(raw.tobytes()).hexdigest()==s['raw10_digest'],'literal full10 action witness')
        item={k:deepcopy(row[k]) for k in rgb.IDENTITY}
        item.update(goal192=deepcopy(s['goal192']),goal_binding=deepcopy(s['goal_binding']),suffix_raw_bits=raw[5:].tobytes().hex(),
            status='unknown_missing_condition_terminal' if actual is None else 'observed_saved_native192',
            missing_reason=row['missing_reason'],native_success=row['native_success'],
            array_receipts=dict(goal=deepcopy(s['goal_array']),prediction=deepcopy(s['prediction_array']),actual=None if actual is None else deepcopy(row['latent'])),**terms)
        rows.append(item)
    analysis=assemble(rows)
    own=[Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')]
    for path in own:bindings[str(path)]=sha(path)
    rgb.verify_bindings(bindings)
    return dict(schema=NAME+'.analysis',status='fixed40_saved_rank_error_attribution_not_admitted',
        bindings=bindings,array_receipts=arrays,prepare_closure_digest=digest(p['bindings']),
        source_files={str(path):bindings[str(path)] for path in own},fixed_slots=40,available_Q=39,missing=[12],
        known_roots=2,independent_roots=0,P1_admission=False,full40_released=False,physical_benefit_claim=False,
        winner_selection=False,goal_or_Q_changed=False,new_neural_calls=0,new_physics_calls=0,
        arithmetic='float64 Q=mean((z-goal)^2); error=predQ-actualQ; 2mean((actual-goal)*(pred-actual))+mean((pred-actual)^2); exact sign ties',
        rank_contract='fixed ascending midranks only; incomplete group all actual full5 ranks unknown; no ordering/winner output',
        limitations=['known2 not40 independent roots','observed-only pair rates do not complete the missing library',
                    'ZS suffix matching does not isolate prefix/state/velocity/contact mediation','latent Q is not native success or physical benefit'],
        **analysis)


def summary(a):
    return {k:a[k] for k in ('status','fixed_slots','available_Q','missing','known_roots','P1_admission','rank_rows','within_role_summary','same_word_ZS_summary')}


def write(path,a):
    p=Path(path).absolute();raw=canonical(a)
    require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix=='.json' and not p.exists() and len(raw)<=CAP,
        'bounded exclusive own-stem analysis file only; never overwrite MAIN')
    with p.open('xb') as stream:require(stream.write(raw)==len(raw),'short saved report write');stream.flush();os.fsync(stream.fileno())
    rgb.sync_dir(p.parent);return dict(output=str(p),analysis_sha256=sha(p),summary=summary(a))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    ap.add_argument('--output');ap.add_argument('--analysis');ap.add_argument('--analysis-sha');ap.add_argument('--summary',action='store_true');args=ap.parse_args()
    a=build()
    if args.command=='verify':
        require(rgb.read(args.analysis,args.analysis_sha,CAP)==a,'entire deterministic saved analysis rebuild')
        out=dict(verified=True,analysis_sha256=args.analysis_sha,summary=summary(a))
    elif args.output:out=write(args.output,a)
    else:out=summary(a) if args.summary else a
    print(canonical(out).decode())


if __name__=='__main__':main()
