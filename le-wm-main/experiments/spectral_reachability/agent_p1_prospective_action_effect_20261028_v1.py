"""Preparation + injected scheduling kernel only. No production CUDA/physics run CLI."""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

NAME='agent_p1_prospective_action_effect_20261028_v1'
ROOT=Path(__file__).resolve().parents[2]
WORDS=('zero','uZ','-uZ','uS','-uS')
CAP=32*1024**2
INPUTS={
    'selector':('results/agent_p1_model_library_selector_20261028_v1_MAIN01.json','f807d2ffe2d7ffb4d012852302f199b9edb64afc5a23cc36ac264c6adea6c1bb'),
    'margin':('results/agent_p1_pair_margin_decomposition_20261028_v1_MAIN01.json','f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143'),
    'T3':('results/agent_p1_t3_structure_diagnostic_20261028_v1_MAIN01.json','03eb182af71d4cfd4acfe3ee25ef1ad8e253ce3cd80e82f37a2eeba2abcf888e'),
    'teacher_prepare':('results/agent_p1_teacher_cross_library_20261028_v1_prepare_MAIN01.json','5140f9c6baeadc1c9889e65797ebdb4bec9b68db530f06467556c5fb743058e1'),
    'context_history':('results/agent_p1_teacher_library40_history_localization_20261028_v1_analysis_NASH01.json','7d171fea5ad553541dfd6393e4bf2530992ef916c92fcc39295590de7337d3a2'),
    'already_H1H2':('results/p1_moving_teacher_horizon_localization_cuda_20261027_v1_cuda_MAIN01/report.json','e319035d9e13a6ae2dc4cbfa5cefb9eb2e6ccd49dbcf35408fda1704f5f34253'),
    'already_API':('results/agent_p1_teacher_library40_api_equivalence_20261028_v1_cuda_MAIN01/report.json','38ad7134ba7f13b7244a9bc9ad311d476ccd67a0df53daf9fa7f314e7876ae3e'),
    'already_rolling':('results/consume_zero_and_rolling_MAIN_20261028_v1_rolling_MAIN01.json','182d5ce14a6363b287607c511d3bc4dde85769618e8c8082029f090fb19c1232'),
}
SEAMS={
    'agent_p1_teacher_library40_history_localization_20261028_v1.py':'0f59ee8ad57a2e6bea619cdca07edb4cbc3eb9a84652d97c019dd3a6ef4fcb33',
    'p1_moving_live_feedback_physical_20261027_v1.py':'6742a1b35102e433d7772e4a208b4f97b37b76f815b991cfc0e472763754ace9',
    'agent_p4_moving_multiboundary_20261028_v1.py':'2776029cc98f8e7d80bb566183ecdd5b44d8e4a0cd25e3f85fff23056b4bb68a',
    'p1_moving_live_feedback_teacher_fidelity_cuda_20261027_v2.py':'492ff318c83a312be5f5f6efd85e49ad06eb56457d5c0e92c3b92ee617a38a1b',
    'agent_p1_teacher_cross_library_physical_20261028_v1.py':'5367287a375cf89e45af2675997aa682ec63464d24b2b01ab413c5863882d8a4',
}


def require(ok,msg):
    if not ok:raise ValueError(msg)


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path,expected):
    p=Path(path).absolute()
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,'bounded regular input')
    raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'input SHA')
    def unique(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    value=json.loads(raw,object_pairs_hook=unique);canonical(value);return value


def literal(value,shape):
    x=np.asarray(value,dtype='<f8');y=np.asarray(value,dtype='<f4')
    require(x.shape==y.shape==shape and np.isfinite(y).all() and
            x.tobytes()==y.astype('<f8').tobytes(),'literal finite float32 including signed zeros')
    return y


def rawbits(bits):
    require(type(bits) is str and len(bits)==80,'literal five raw steps bits')
    try:a=np.frombuffer(bytes.fromhex(bits),dtype='<f4').reshape(5,2)
    except ValueError as exc:raise ValueError('literal raw bytes') from exc
    require(a.tobytes().hex()==bits and np.isfinite(a).all() and (abs(a)<=1).all(),'original raw domain; no clipping')
    return a


def contract():
    return dict(known_roots=2,fresh_roots=0,anchors=8,words=list(WORDS),new_candidate_slots=40,new_pairs=80,
        cohort='all original root0/1 x off/on x Z/S; no outcome filtering',
        origin='replay warmup10 + original prefix5 + sealed stage15 model-selected suffix5',
        feedback_absolute_step=20,terminal_absolute_step=25,
        selection='stable argmin original identity192 float64 MSE; original word index breaks exact ties',
        new_forward_slots=40,new_image_encoding_cap=88,replay_public_step_cap=1000,
        terminal_renderer_intervention='both original/alternate target overlay at same full physical state25; no extra dynamics',
        P2_evidence_used_to_explain_P1=False,teacher15_all_actual_proof_required=True,
        forward='actual [z10,z15,z20], actual action packets11:15 and16:20 plus candidate21:25',
        physics_mode='five independent exact replay20 starts, not state7 teleport/clone',
        reuse_leader_for_word0=True,early_stop='unknown labels, no fallback/retry/drop',
        old_panel=dict(comparable=76,unknown=4,fixed=80,never_replaced=True),
        new_unknown_count='determined by actual returns/guards; not forced to old4 or silently zero',
        P1_admission=False,training=False,fitted_Q=False,fitted_residual_model=False,
        residual_correction_deployed=False,oracle_selection=False,science_gate_created=False,
        run_cli_implemented=False,production_adapter_implemented=False,
        full_upstream_closure_rebuilt_by_preparer=False,
        real_dispatch_blocked_until='MAIN native adapter + loadedbytes/reset ledger + strict consume frozen',
        resource_caps_pending=dict(internal_physics_steps=None,reset=None,constructor=None,goal_setter=None,
            total_FLOPs=None,wall_seconds=None,output_bytes=None),
        no_extra_predict=True,encoder_device='CUDA only',autograd=0,Jacobian=0,SVD=0)


def jobs(sealed):
    cs,ss=sealed['candidates'],sealed['selections']
    require(len(cs)==40 and len(ss)==8,'fixed all40/all8 selector inputs')
    out=[]
    for a in range(8):
        group=cs[a*5:a*5+5];first=group[0]['original_plan'];prefix=group[0]['full10_raw_bits'][:80]
        words=[]
        for w,c in enumerate(group):
            p=c['original_plan'];suffix=c['suffix_raw_bits'];rawbits(suffix);rawbits(prefix)
            require((p['index'],p['root_slot'],p['condition'],p['role'],p['library_index'],p['library_label'])==
                (a*5+w,a//4,('off','on')[(a//2)%2],('Z','S')[a%2],w,WORDS[w]),'original fixed roster')
            require(all(p[k]==first[k] for k in ('anchor_id','task_id','blind_sha256')) and
                    c['full10_raw_bits']==prefix+suffix,'same original prefix/anchor and literal full10')
            if w==0:require((rawbits(suffix)==0).all(),'fixed zero reference, not outcome-chosen reference')
            words.append(dict(word=w,label=WORDS[w],raw5_bits=suffix,original_plan_digest=p['plan_digest']))
        s=ss[a];chosen=s['selected_original_index']-a*5
        require(type(chosen) is int and 0<=chosen<5 and s['anchor_index']==a and
                s['commands_raw_bits']==words[chosen]['raw5_bits'],'original sealed model selection')
        out.append(dict(anchor_index=a,source_slot=a//4,condition=first['condition'],role=first['role'],
            task_id=first['task_id'],anchor_id=first['anchor_id'],prefix5_bits=prefix,
            first_suffix5_bits=words[chosen]['raw5_bits'],first_selected_original_index=a*5+chosen,
            first_selection_digest=s['selection_digest'],words=words))
    return out


def attest_context(saved,rebuilt,js):
    require(canonical(saved)==canonical(rebuilt),'complete saved teacher15 history rebuild, signed bytes')
    require(saved['fixed_slots']==40 and len(saved['rows'])==40,'full40 teacher context proof')
    for i,r in enumerate(saved['rows']):
        j=js[i//5];w=i%5;h=r['history'];frames=r['RGB_history_frames']
        require(r['index']==i and r['input_binding_status']=='saved_exact' and
            all(r[k]==j[k] for k in ('anchor_id','condition','role')) and r['root_slot']==j['source_slot'] and
            r['plan_digest']==j['words'][w]['original_plan_digest'],'same actual teacher slot identity')
        require(h['teacher_latent_absolute_steps']==[5,10,15] and
            h['predictor_action_absolute_steps']==[list(range(6,11)),list(range(11,16)),list(range(16,21))],
            'actual three frames and all actual action packet time indices')
        require(h['prefix_raw_bits']==j['prefix5_bits'] and h['suffix_raw_bits']==j['words'][w]['raw5_bits'] and
            h['full_public20_raw_bits'].endswith(j['prefix5_bits']+j['words'][w]['raw5_bits']) and
            h['teacher_raw_bits']==h['full_public20_raw_bits'][80:160]+j['prefix5_bits'],
            'actual raw6..15 history and original action witnesses')
        require([(f['phase'],f['step'],f['condition']) for f in frames]==
            [('warmup',5,j['condition']),('warmup',10,j['condition']),('prefix',5,j['condition'])],
            'all same-condition actual RGB5/10/15, never imagined frame')
    return dict(all_three_actual=True,actual_latent_absolute_steps=[5,10,15],checked_slots=40,
        actual_history_absolute_steps=list(range(6,16)),full_saved_history_rebuilt=True,
        proof_sha256=INPUTS['context_history'][1],P2_evidence_used=False,
        true_mid_replacement_experiment=False,remaining_internal_cause_identified=False)


def context_proof(saved,js):
    # Only original saved validators. No forward/encoder/env execution.
    from experiments.spectral_reachability import agent_p1_teacher_library40_history_localization_20261028_v1 as history
    path=ROOT/'experiments/spectral_reachability'/Path(history.__file__).name
    require(Path(history.__file__).resolve()==path and sha(path)==SEAMS[path.name],'frozen actual-context validator')
    return attest_context(saved,history.build(),js)


def validate_forward(r):
    require(set(r)=={'schema','anchor_index','word','latents','history','raw5_bits'} and
        r['schema']==NAME+'.forward' and type(r['anchor_index']) is int and 0<=r['anchor_index']<8 and
        type(r['word']) is int and 0<=r['word']<5,'forward whitelist and identity; no evaluator labels')
    literal(r['latents'],(3,192));literal(r['history'],(2,10));rawbits(r['raw5_bits'])


def forward20(teacher15_latents,actual20,actual_raw11_to20_bits,mean,std):
    """Pure packet adapter: shift actual15 context, actual20 feedback, actual past."""
    z=literal(teacher15_latents,(3,192));current=literal(actual20,(192,))
    require(type(actual_raw11_to20_bits) is str and len(actual_raw11_to20_bits)==160,'two actual raw packets')
    raw=np.concatenate((rawbits(actual_raw11_to20_bits[:80]),rawbits(actual_raw11_to20_bits[80:])))
    mu,sd=literal(mean,(2,)),literal(std,(2,));require((sd>0).all(),'positive original std')
    return dict(latents=np.concatenate((z[1:],current[None])).tolist(),history=((raw-mu)/sd).reshape(2,10).tolist())


def validate_start(start,j):
    require(set(start)=={'fingerprint','forward','handle'} and set(start['forward'])=={'latents','history'},'split physical/forward domains')
    f=start['fingerprint'];require(set(f)=={'raw20_bits','state7','snapshot_sha256','png_sha256','pixel_sha256'},'complete replay20 fingerprint fields')
    state=np.asarray(f['state7'],dtype=float);require(state.shape==(7,) and np.isfinite(state).all(),'full native state7')
    bits=f['raw20_bits'];require(type(bits) is str and len(bits)==320 and bits.endswith(j['prefix5_bits']+j['first_suffix5_bits']),
                               'actual warm10/prefix5/selected5 chronology')
    for key in ('snapshot_sha256','png_sha256','pixel_sha256'):
        require(type(f[key]) is str and len(f[key])==64 and all(c in '0123456789abcdef' for c in f[key]),'literal replay fingerprint SHA')


def pack(x):return dict(numerator=str(x.numerator),denominator=str(x.denominator))


def vec(x):return tuple(Fraction.from_float(float(v)) for v in literal(x,(192,)))
def sub(x,y):return tuple(a-b for a,b in zip(x,y,strict=True))
def dot(x,y):return sum((a*b for a,b in zip(x,y,strict=True)),Fraction())/192


def effects(predicted,actual,goal):
    """Exact zero-referenced do(action) effect errors; never a correction/selector."""
    require(len(predicted)==len(actual)==5,'all five words including unknowns')
    p=list(map(vec,predicted));a=[None if x is None else vec(x) for x in actual];g=vec(goal)
    e0=None if a[0] is None else sub(a[0],p[0])
    h=[None if e0 is None or a[w] is None else sub(sub(a[w],a[0]),sub(p[w],p[0])) for w in range(5)]
    rows=[dict(word=w,available=a[w] is not None,response_error_mse=None if h[w] is None else pack(dot(h[w],h[w]))) for w in range(5)]
    pairs=[]
    for i,j in itertools.combinations(range(5),2):
        rp,rq=sub(p[i],g),sub(p[j],g);mp=dot(rp,rp)-dot(rq,rq)
        ma=None if a[i] is None or a[j] is None else dot(sub(a[i],g),sub(a[i],g))-dot(sub(a[j],g),sub(a[j],g))
        B=R=residual=None
        if ma is not None and e0 is not None:
            B=2*dot(sub(p[i],p[j]),e0)
            R=2*(dot(rp,h[i])-dot(rq,h[j]))+2*dot(e0,sub(h[i],h[j]))+dot(h[i],h[i])-dot(h[j],h[j])
            residual=ma-mp-B-R;require(residual==0,'zero-reference exact margin identity')
        pairs.append(dict(i=i,j=j,available=ma is not None,predicted_margin=pack(mp),
            observed_margin=None if ma is None else pack(ma),strict_reversal=None if ma is None else mp*ma<0,
            shared_zero_projection=None if B is None else pack(B),response_remainder=None if R is None else pack(R),
            identity_residual=None if residual is None else pack(residual)))
    return dict(zero_residual_mse=None if e0 is None else pack(dot(e0,e0)),words=rows,pairs=pairs)


def renderer_effects(original,alternate):
    require(len(original)==len(alternate)==5,'both renderer views retain all five words')
    a=[None if x is None else vec(x) for x in original]
    b=[None if x is None else vec(x) for x in alternate]
    rows=[]
    for w in range(5):
        d=None if any(x is None for x in (a[0],a[w],b[0],b[w])) else sub(sub(b[w],b[0]),sub(a[w],a[0]))
        rows.append(dict(word=w,effect_change_mse=None if d is None else pack(dot(d,d))))
    return dict(words=rows,interpretation='do(renderer) change in encoded action effects at fixed native endpoints; not physical task change')


def drive_anchor(j,goal,adapter):
    """Inject replay20/predict/seal/advance5. RAM-tested kernel, NOT a native runtime.

    Adapter owns CUDA guards, native validation, all actual ledgers and durable
    partial/final artifacts. This function cannot certify these by itself.
    advance5 includes guarded factual terminal25 encoding, not a future input.
    """
    g=literal(goal,(192,)).astype(np.float64)
    rows=[dict(word=w,endpoint=None,alternate_endpoint=None,status='not_started') for w in range(5)]
    out=dict(anchor_index=j['anchor_index'],selected_word=None,policy_endpoint=None,branches=rows,
             complete_pairs=0,unknown_pairs=10,status='not_started',errors=[],predictions=[None]*5)
    try:
        leader=adapter.replay20(j,0);validate_start(leader,j)
        for w in range(5):
            r=dict(schema=NAME+'.forward',anchor_index=j['anchor_index'],word=w,
                   **deepcopy(leader['forward']),raw5_bits=j['words'][w]['raw5_bits'])
            validate_forward(r);out['predictions'][w]=literal(adapter.predict(r),(192,)).tolist()
        costs=[float(np.mean((np.asarray(p,dtype=np.float64)-g)**2)) for p in out['predictions']]
        chosen=min(range(5),key=lambda w:(costs[w],w))
        seal=dict(anchor_index=j['anchor_index'],fingerprint=deepcopy(leader['fingerprint']),
                  predictions=deepcopy(out['predictions']),predicted_Q=costs,selected_word=chosen,
                  selected_raw5_bits=j['words'][chosen]['raw5_bits'])
        out['seal_sha256']=adapter.seal(deepcopy(seal))
        require(type(out['seal_sha256']) is str and len(out['seal_sha256'])==64,'durable prediction/selection seal receipt')
        out['selected_word']=chosen
    except Exception as exc:
        out.update(status='prediction_or_seal_failed');out['errors'].append(repr(exc));return out
    for w in range(5):
        try:
            start=leader if w==0 else adapter.replay20(j,w);validate_start(start,j)
            require(canonical(start['fingerprint'])==canonical(leader['fingerprint']),'same full native/pixel/replayed past at20')
            result=adapter.advance5(start,j['words'][w]['raw5_bits'])
            if not result['complete']:
                rows[w].update(status='native_stop_or_guard_unknown');continue
            require(result['actual_raw5_bits']==j['words'][w]['raw5_bits'] and
                    result['terminal_absolute_step']==25,'actual intervention packet and exact terminal25')
            state=np.asarray(result['terminal_state7'],dtype=float)
            require(state.shape==(7,) and np.isfinite(state).all(),'complete terminal state7')
            rows[w].update(endpoint=literal(result['endpoint'],(192,)).tolist(),status='observed25')
            if w==chosen:out['policy_endpoint']=deepcopy(rows[w]['endpoint'])
            audit=result.get('renderer_audit',{})
            if (audit.get('restored') is True and type(audit.get('before_sha256')) is str and
                len(audit['before_sha256'])==64 and audit['before_sha256']==audit.get('after_sha256')):
                rows[w]['alternate_endpoint']=literal(result['alternate_endpoint'],(192,)).tolist()
            else:rows[w]['renderer_status']='unknown_missing_or_native_drift'
        except Exception as exc:rows[w].update(status='failed_unknown',error=repr(exc))
    n=sum(r['endpoint'] is not None for r in rows)
    out.update(status='recorded_not_admitted',complete_pairs=n*(n-1)//2,unknown_pairs=10-n*(n-1)//2,
        effects=effects(out['predictions'],[r['endpoint'] for r in rows],goal),
        renderer_effects=renderer_effects([r['endpoint'] for r in rows],[r['alternate_endpoint'] for r in rows]))
    return out


def build():
    inputs={k:read(ROOT/p,h) for k,(p,h) in INPUTS.items()}
    js=jobs(inputs['selector']);prior=inputs['margin'];prep=inputs['teacher_prepare']
    proof=context_proof(inputs['context_history'],js)  # Before any new feedback/action experiment can be prepared.
    require(prior['fixed_pairs']==80 and prior['summary']['model_to_observed']['available']==76 and
            prior['summary']['model_to_observed']['unknown']==4,'historical fixed80/76/4 unmodified')
    require(len(prep['anchors'])==8,'original fixed8 teacher anchors')
    for j,anchor in zip(js,prep['anchors'],strict=True):
        require(all(j[k]==anchor[k] for k in ('anchor_id','task_id','condition','role')) and
                j['source_slot']==anchor['root_slot'],'same original teacher task, no fresh reserved')
        j['original_task']=dict(path=anchor['task_file'],sha256=anchor['task_sha256'])
        j['original_goal_ref']=dict(snapshot=INPUTS['margin'][0],slot_index=j['anchor_index']*5)
    seams={str(ROOT/'experiments/spectral_reachability'/p):h for p,h in SEAMS.items()}
    for p,h in seams.items():require(sha(p)==h,'frozen runner seam SHA')
    own=[Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']
    return dict(schema=NAME+'.prepare',status='contract_and_injected_kernel_only_not_dispatch_authority',
        contract=contract(),jobs=js,inputs={k:dict(path=str(ROOT/p),sha256=h) for k,(p,h) in INPUTS.items()},
        frozen_seams=seams,source_files={str(p):sha(p) for p in own},teacher15_actual_context_proof=proof,
        model=dict(policy=prep['policy'],expected_model_hash=prep['expected_model_hash']),
        pending=['native20/25 adapter and explicit visual ABI','full task/model/source/loadedbytes closure',
                 'complete reset/internal physics/renderer caps and durable attempt ledger','strict new saved consumer'],
        P1_admission=False,fresh_confirmation=False,run_authorized=False)


def write(path,value):
    p=Path(path).absolute();raw=canonical(value)
    require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and
            p.suffix=='.json' and len(raw)<=CAP,'exclusive bounded own output')
    with p.open('xb') as f:
        require(f.write(raw)==len(raw),'complete write');f.flush();os.fsync(f.fileno())
    return dict(output=str(p),sha256=sha(p),run_authorized=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','verify'))
    parser.add_argument('--output');parser.add_argument('--prepared');parser.add_argument('--prepare-sha')
    args=parser.parse_args();value=build()
    if args.command=='prepare':
        require(args.output,'prepare needs --output');out=write(args.output,value)
    else:
        require(args.prepared and args.prepare_sha,'verify needs prepared and SHA')
        require(canonical(read(args.prepared,args.prepare_sha))==canonical(value),'whole prepare rebuild')
        out=dict(verified=True,run_authorized=False)
    print(canonical(out).decode())


if __name__=='__main__':main()
