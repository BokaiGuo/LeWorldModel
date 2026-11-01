"""Saved-only T3/T4: fixed80 pair margins, exact algebra, no model/physics run."""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
import gzip
import hashlib
from io import BytesIO
import itertools
import json
import math
import os
from pathlib import Path
import struct

import numpy as np

NAME = 'agent_p1_pair_margin_decomposition_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT/'results/agent_p1_model_library_selector_evaluation_20261028_v1_MAIN01.json'
EVALUATION_SHA = 'd8b3bedd38e9f9a56d0156f2452d5a20f363acbdad42ea5e3ee4d91f9626ff67'
DIM = 192
CAP = 4*1024**2
AUTHORITY_CAP = 32*1024**2
ARRAY_CAP = 2*1024**2
LIBRARY = ('zero', 'uZ', '-uZ', 'uS', '-uS')
IDENTITY = ('index', 'root_slot', 'condition', 'role', 'library_index', 'library_label',
            'anchor_id', 'task_id', 'blind_sha256', 'plan_digest')
METRICS = dict(task_success=-1, native_distance=1, all_xy_error=1, agent_xy_error=1,
               object_xy_error=1, object_wrapped_angle_error=1, agent_velocity_error=1,
               native_coverage=-1, object_only_coverage=-1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def same(a, b, label):
    require(canonical(a) == canonical(b), 'exact signed JSON binding: '+label)


def read(path, expected, cap=CAP):
    p = Path(path).absolute()
    require(p.resolve() == p and p.is_file() and not p.is_symlink() and p.stat().st_size <= cap,
            'bounded canonical regular input')
    raw = p.read_bytes()
    require(len(raw) <= cap and hashlib.sha256(raw).hexdigest() == expected, 'input SHA')
    def unique(items):
        out = {}
        for k, v in items:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    out = json.loads(raw, object_pairs_hook=unique)
    canonical(out)
    return out


def native(a, shape, label):
    require(isinstance(a, np.ndarray) and a.dtype.str == '<f4' and a.shape == shape and np.isfinite(a).all(),
            'original finite little-endian float32 '+label)
    return a


def literal(value, shape):
    original = np.asarray(value, dtype='<f8')
    a = np.asarray(value, dtype='<f4')
    require(original.shape == a.shape == shape and np.isfinite(a).all() and
            original.tobytes() == a.astype('<f8').tobytes(), 'literal float32, including signed zeros; no silent rounding')
    return a


def pack(x):
    return dict(numerator=str(x.numerator), denominator=str(x.denominator))


def unpack(x):
    return Fraction(int(x['numerator']), int(x['denominator']))


def fractions(a):
    return tuple(Fraction.from_float(float(x)) for x in a)


def dot(a, b):
    return sum((x*y for x,y in zip(a,b,strict=True)), Fraction()) / DIM


def sub(a, b):
    return tuple(x-y for x,y in zip(a,b,strict=True))


def add(a, b):
    return tuple(x+y for x,y in zip(a,b,strict=True))


def sign(x):
    require(math.isfinite(x), 'finite exact-sign input')
    return int(x > 0)-int(x < 0)


def classify(a, b):
    if a is None or b is None:
        return dict(available=False, classification='unknown', delta_a=a, delta_b=b,
                    sign_a=None, sign_b=None, reversal=None)
    sa, sb = sign(a), sign(b)
    label = ('both_tie' if sa == sb == 0 else 'left_tie_only' if sa == 0 else
             'right_tie_only' if sb == 0 else 'concordant' if sa == sb else 'discordant')
    return dict(available=True, classification=label, delta_a=a, delta_b=b,
                sign_a=sa, sign_b=sb, reversal=sa*sb < 0)


def decompose(predicted, actual, goal):
    """Exact polynomial on original binary32 values; float64 is a separate diagnostic."""
    p, g = native(predicted, (DIM,), 'prediction'), native(goal, (DIM,), 'goal')
    rp = sub(fractions(p), fractions(g))
    qp = dot(rp,rp)
    fp, fg = p.astype(np.float64), g.astype(np.float64)
    v = fp-fg
    floating = dict(predicted_Q=float(np.mean(v**2)), observed_Q=None, observed_minus_predicted=None,
                    cross=None, quadratic=None, identity_residual=None, rounding_vs_exact={})
    exact = dict(predicted_Q=pack(qp), observed_Q=None, observed_minus_predicted=None,
                 cross=None, quadratic=None, identity_residual=None)
    data = dict(rp=rp, e=None, predicted_Q=qp, observed_Q=None, cross=None, quadratic=None)
    if actual is not None:
        a = native(actual,(DIM,),'actual')
        e = sub(fractions(a), fractions(p))
        ra = add(rp,e)
        qa, cross, square = dot(ra,ra), 2*dot(rp,e), dot(e,e)
        correction = qa-qp
        require(correction == cross+square, 'exact per-slot polynomial identity')
        data.update(e=e,observed_Q=qa,cross=cross,quadratic=square)
        exact.update({k:pack(x) for k,x in dict(observed_Q=qa,observed_minus_predicted=correction,
                     cross=cross,quadratic=square,identity_residual=correction-cross-square).items()})
        fa = a.astype(np.float64); fe = fa-fp
        fqa = float(np.mean((fa-fg)**2))
        fc, fq = float(2*np.mean(v*fe)), float(np.mean(fe**2))
        fd = fqa-floating['predicted_Q']
        floating.update(observed_Q=fqa,observed_minus_predicted=fd,cross=fc,quadratic=fq,
                        identity_residual=fd-(fc+fq))
    floating['rounding_vs_exact'] = {k:float(x)-float(unpack(exact[k])) for k,x in floating.items()
                                    if k != 'rounding_vs_exact' and x is not None}
    return dict(exact=exact,float64=floating,_math=data)


def same_float(a, b, label):
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and
            struct.pack('<d',float(a)) == struct.pack('<d',float(b)), 'original float64 scalar bits: '+label)


def validate_slot(row, join, candidate, blind, goal, prediction, actual):
    """Additional local checks after the full upstream strict authority is rebuilt."""
    i = row['index']; anchor, word = divmod(i,5); task, role = divmod(anchor,2)
    p, w = candidate['original_plan'], candidate['prediction_witness']
    require(0 <= i < 40 and type(i) is int, 'original slot index')
    require(tuple(row[k] for k in ('root_slot','condition','role','library_index','library_label')) ==
            (task//2,('off','on')[task%2],('Z','S')[role],word,LIBRARY[word]), 'fixed original slot roster')
    same({k:row[k] for k in IDENTITY}, {k:p[k] for k in IDENTITY}, 'row/plan identity')
    require(p['plan_digest'] == digest({k:v for k,v in p.items() if k != 'plan_digest'}) and
            candidate['prediction_digest'] == digest(w), 'original plan and prediction digests')
    for k in ('index','plan_digest','job_digest'):
        same(row[k],join[k], 'native reuse '+k)
    require(join['prediction_digest'] == candidate['prediction_digest'] and
            digest(blind['context']) == join['teacher_context_sha256'] == p['teacher_context_sha256'],
            'original complete signed context, never just shape or endpoint')
    require(join['actual20_verified'] is join['exact_existing_execution_reuse'] is True and
            join['newly_executed'] is False and row['actual_return_absolute_step'] == 20,
            'same original completed native return20, not a different frame')
    for t,k in ((10,'actual_root10'),(15,'actual_start15')):
        same(row['boundary_trace'][str(t)]['state'],join[k], 'actual boundary state'+str(t))
    same(row['boundary_trace']['20']['state'],row['returned_state'],'terminal native state')
    require(row['actual_return_event_sha256'] == join['actual_return_event_sha256'], 'same returned native event')
    raw = literal(p['commands'],(10,2)); prefix = literal(p['prefix_commands'],(5,2)); suffix = literal(p['suffix_commands'],(5,2))
    require(np.max(np.abs(raw)) <= 1 and raw.tobytes() == np.concatenate((prefix,suffix)).tobytes() and
            candidate['full10_raw_bits'] == raw.tobytes().hex() and
            candidate['suffix_raw_bits'] == row['suffix_raw_bits'] == suffix.tobytes().hex() and
            array_sha(raw) == join['full10_sha256'] and array_sha(prefix) == join['prefix_sha256'] and
            array_sha(suffix) == join['suffix_sha256'], 'exact signed raw full10/prefix/suffix witness')
    g = native(goal,(DIM,),'goal'); pred = native(prediction,(3,DIM),'shifted context')
    require(g.tobytes() == literal(row['goal192'],(DIM,)).tobytes() == literal(blind['goal'],(DIM,)).tobytes() and
            array_sha(g) == join['latent_goal_sha256'] == w['goal_sha256'], 'same signed latent goal')
    q = np.asarray(blind['q'],dtype='<f8')
    require(q.shape == (DIM,DIM) and q.tobytes() == np.eye(DIM,dtype='<f8').tobytes(),
            'original identity192 operator only, no fitted or alternative Q')
    same(row['array_receipts']['prediction'],w['shifted_context'],'original shifted array receipt')
    same(row['physical_goal'],join['native_goal'],'native task goal')
    same(row['physical_goal'],row['goal_binding']['state'],'rendered goal/task authority')
    same(row['outline_pose'],row['goal_binding']['pose'],'outline authority distinct from task goal')
    require((actual is None) == (row['actual_Q'] is None) == (row['array_receipts']['actual'] is None) == (i == 12),
            'only original missing12, no imputation or extra missing slot')
    terms = decompose(pred[-1],actual,g)
    same_float(terms['float64']['predicted_Q'],row['predicted_Q'],'original predicted MSE')
    same_float(row['predicted_Q'],p['model_Q'],'original plan Q')
    if actual is not None:
        same_float(terms['float64']['observed_Q'],row['actual_Q'],'original observed MSE')
    metrics = {k:deepcopy(row['physical_metrics'][k]) for k in METRICS}
    require(type(metrics['task_success']) is bool and metrics['task_success'] is row['native_success'], 'original returned success')
    for k,v in metrics.items():
        if v is not None:
            require(type(v) in (bool,int,float) and math.isfinite(v), 'finite native metric '+k)
    return dict(index=i,anchor_index=anchor,root_slot=row['root_slot'],condition=row['condition'],role=row['role'],
                word=LIBRARY[word],task_id=row['task_id'],anchor_id=row['anchor_id'],
                plan_digest=p['plan_digest'],job_digest=row['job_digest'],context_sha256=p['teacher_context_sha256'],
                prefix_sha256=array_sha(prefix),suffix_sha256=array_sha(suffix),full10_sha256=array_sha(raw),
                goal_sha256=array_sha(g),physical_goal_sha256=array_sha(np.asarray(row['physical_goal'],dtype='<f8')),
                goal_binding_digest=digest(row['goal_binding']),terminal_absolute_step=20,
                predicted_Q=row['predicted_Q'],actual_Q=row['actual_Q'],physical_metrics=metrics,
                missing_reason=row['missing_reason'],actual_return_event_sha256=row['actual_return_event_sha256'],
                arrays=deepcopy(row['array_receipts']),endpoint_sha256=array_sha(pred[-1]),
                actual_sha256=None if actual is None else array_sha(actual),**terms)


def pair(a, b):
    require(a['index'] < b['index'], 'fixed ascending i-j order')
    for key in ('anchor_index','root_slot','condition','role','task_id','anchor_id','context_sha256',
                'prefix_sha256','goal_sha256','physical_goal_sha256','goal_binding_digest','terminal_absolute_step'):
        same(a[key],b[key],'within-role same task/context/time '+key)
    dp = a['predicted_Q']-b['predicted_Q']
    missing = [x['index'] for x in (a,b) if x['actual_Q'] is None]
    da = None if missing else a['actual_Q']-b['actual_Q']
    physical, mo, op = {}, {}, {}
    for k,orientation in METRICS.items():
        va,vb = a['physical_metrics'][k],b['physical_metrics'][k]
        delta = None if va is None or vb is None else orientation*(float(va)-float(vb))
        physical[k] = delta
        mo[k],op[k] = classify(dp,delta),classify(da,delta)
    out = dict(i=a['index'],j=b['index'],anchor_index=a['anchor_index'],missing_indices=missing,
               model_to_observed=classify(dp,da),model_to_physical=mo,observed_to_physical=op,
               physical_lower_is_better_deltas=physical,exact=None,float64=None)
    if missing:
        return out
    ma,mb = a['_math'],b['_math']
    ep,ea = ma['predicted_Q']-mb['predicted_Q'],ma['observed_Q']-mb['observed_Q']
    cross_i,cross_j = ma['cross'],mb['cross']
    square_i,square_j = ma['quadratic'],mb['quadratic']
    s,delta = sub(ma['rp'],mb['rp']),sub(ma['e'],mb['e'])
    common = 2*dot(s,mb['e'])
    relative = 2*dot(add(ma['rp'],mb['e']),delta)
    relative_square = dot(delta,delta)
    correction = ea-ep
    ir = correction-(cross_i-cross_j+square_i-square_j)
    fr = correction-(common+relative+relative_square)
    require(ir == fr == 0, 'exact pair identity and common/relative response factorization')
    exact = dict(predicted_margin=ep,observed_margin=ea,observed_minus_predicted_margin=correction,
                 cross_i=cross_i,cross_j=cross_j,quadratic_i=square_i,quadratic_j=square_j,
                 common_bias_projection=common,relative_response_projection=relative,
                 relative_response_quadratic=relative_square,identity_residual=ir,factorization_residual=fr)
    fa,fb = a['float64'],b['float64']
    fc,fq = fa['cross']-fb['cross'],fa['quadratic']-fb['quadratic']
    out['exact'] = {k:pack(v) for k,v in exact.items()}
    out['float64'] = dict(predicted_margin=dp,observed_margin=da,observed_minus_predicted_margin=da-dp,
                         cross_delta=fc,quadratic_delta=fq,identity_residual=(da-dp)-(fc+fq),
                         rounding_vs_exact=dict(predicted_margin=dp-float(ep),observed_margin=da-float(ea),
                                                correction=(da-dp)-float(correction)),
                         exact_rational_predicted_sign=sign(ep),exact_rational_observed_sign=sign(ea),
                         original_float64_signs_equal_exact=(sign(dp)==sign(ep) and sign(da)==sign(ea)))
    return out


def tally(items):
    names = ('concordant','discordant','both_tie','left_tie_only','right_tie_only','unknown')
    counts = {k:sum(x['classification']==k for x in items) for k in names}
    require(sum(counts.values()) == len(items), 'all pair categories counted')
    return dict(fixed=len(items),available=len(items)-counts['unknown'],unknown=counts['unknown'],
                strict_reversals=counts['discordant'],classifications=counts,
                independent_sample_count=None,full_library_rate=None)


def assemble(slots):
    require(len(slots) == 40 and [s['index'] for s in slots] == list(range(40)), 'fixed ordered40; never filter/reorder')
    for i,s in enumerate(slots):
        require((s['actual_Q'] is None) == (i==12) and s['anchor_index'] == i//5 and
                (s['_math']['e'] is None) == (i==12), 'original unknown12 and anchor roster')
    pairs = [pair(a,b) for anchor in range(8) for a,b in itertools.combinations(slots[anchor*5:anchor*5+5],2)]
    require(len(pairs)==80 and sum(bool(p['missing_indices']) for p in pairs)==4, 'fixed80/76available/4unknown')
    return dict(slots=[{k:v for k,v in s.items() if k!='_math'} for s in slots],pairs=pairs,
                summary=dict(model_to_observed=tally([p['model_to_observed'] for p in pairs]),
                    model_to_physical={k:tally([p['model_to_physical'][k] for p in pairs]) for k in METRICS},
                    observed_to_physical={k:tally([p['observed_to_physical'][k] for p in pairs]) for k in METRICS}),
                fixed_slots=40,observed_latent_slots=39,missing_indices=[12],fixed_pairs=80,
                known_roots=2,fresh_roots=0,P1_admission=False,C2_claim=False,fresh_confirmation=False,
                fitted_Q=False,winner_selection=False,new_neural_calls=0,new_physics_calls=0,
                mechanism_identified=False,physical_release=False)


def load_array(base, meta):
    """Bound compressed file, decompressed NPY, dtype/shape, exact signed array bytes."""
    base = Path(base).absolute(); name = meta['file']
    require(isinstance(name,str) and name == Path(name).name and name not in ('','.', '..'), 'flat array path')
    path = base/name
    require(path.resolve()==path and not path.is_symlink() and path.is_file() and
            path.stat().st_size<=ARRAY_CAP and sha(path)==meta['file_sha256'], 'compressed array authority')
    with gzip.GzipFile(fileobj=BytesIO(path.read_bytes())) as stream:
        raw = stream.read(ARRAY_CAP+1)
    require(len(raw)<=ARRAY_CAP and hashlib.sha256(raw).hexdigest()==meta['npy_sha256'], 'bounded NPY byte authority')
    stream = BytesIO(raw); a = np.load(stream,allow_pickle=False)
    require(isinstance(a,np.ndarray) and a.dtype.str=='<f4' and str(a.dtype)==meta['dtype'] and
            list(a.shape)==meta['shape'] and stream.tell()==len(raw) and np.isfinite(a).all() and
            array_sha(a)==meta['array_sha256'], 'complete finite native NPY shape/dtype/signed bytes, no trailing data')
    return a


def upstream():
    # Lazy: only the CLI build path imports existing *saved* validators, never their run/execute functions.
    from experiments.spectral_reachability import agent_p1_model_library_selector_evaluation_20261028_v1 as evaluation
    return evaluation


def load_authority():
    old = upstream()
    value = old.teacher.read(EVALUATION,EVALUATION_SHA,AUTHORITY_CAP)
    rebuilt = old.build()  # Full native source/arrays/journal closure, not trusting affirmative fields.
    same(value,rebuilt,'entire pinned MAIN evaluation reconstructed, including signed JSON bits')
    return value,old


def build():
    authority,old = load_authority()
    require(authority['schema']=='agent_p1_model_library_selector_evaluation_20261028_v1.evaluation' and
            authority['status']=='exact_seen_execution_reuse_exploratory_not_admitted' and
            authority['fixed_slots']==40 and authority['actual_latent_available']==39 and
            authority['missing_actual_latent']==[12] and authority['known_roots']==2 and
            authority['fresh_roots']==0 and authority['fresh_confirmation'] is authority['P1_admission'] is False,
            'original fixed MAIN authority, not a fresh/scientific admission')
    # Every source below is already inside the complete rebuilt evaluation closure.
    sealed = old.teacher.read(old.SELECTOR,old.SELECTOR_SHA,AUTHORITY_CAP)
    model = old.teacher.read(old.selector.REPORT,old.selector.REPORT_SHA,AUTHORITY_CAP)
    rank = old.panel.base.rank
    compiler_folder = old.selector.REPORT.parent
    actual_folder = rank.REPORT.parent
    require(len(sealed['candidates'])==len(authority['rows'])==len(authority['exact_reuse_witnesses'])==40 and
            len(model['rows'])==8, 'all original candidate and array slots')
    rows = []
    for i,(row,join,c) in enumerate(zip(authority['rows'],authority['exact_reuse_witnesses'],sealed['candidates'],strict=True)):
        require(row['index']==i and join['index']==i, 'same fixed slot ordering')
        blind = model['rows'][i//5]['blind_input']
        receipts = row['array_receipts']
        goal = load_array(compiler_folder,receipts['goal'])
        prediction = load_array(compiler_folder,receipts['prediction'])
        actual = None if receipts['actual'] is None else load_array(actual_folder,receipts['actual'])
        rows.append(validate_slot(row,join,c,blind,goal,prediction,actual))
    value = assemble(rows)
    own = [Path(__file__).resolve(),ROOT/f'tests/test_{NAME}.py',ROOT/f'innerdocs/{NAME}.md']
    # One compact authority reference replaces 49k repeated source paths/large old report trees.
    value.update(schema=NAME+'.analysis',status='saved_T3_T4_exact_pair_algebra_not_admitted',
        authority=dict(evaluation=str(EVALUATION),evaluation_sha256=EVALUATION_SHA,
            full_saved_native_rebuilt=True,closure_entries=len(authority['bindings']),
            closure_digest=digest(authority['bindings']),selector_sha256=old.SELECTOR_SHA,
            model_report_sha256=old.selector.REPORT_SHA,model_strict_receipt_sha256=old.selector.RECEIPT_SHA,
            actual_report_sha256=rank.REPORT_SHA,actual_strict_receipt_sha256=rank.RECEIPT_SHA,
            panel_sha256=old.PANEL_SHA,compiler_array_folder=str(compiler_folder),actual_array_folder=str(actual_folder)),
        source_files={str(f):sha(f) for f in own},
        protocol=dict(pair_order='i<j within each original five-word anchor; delta=cost_i-cost_j; lower is better',
            predicted_coordinate='original native projected CLS192, last row of shifted context only',
            metric_operator='identity192',MSE_divisor=DIM,effective_A='I/sqrt(192)',
            effective_cometric='I/192; original operator named Q is I, not a second weighting',
            exact_arithmetic='rhat=p-g; e=a-p; (Qa-Qp)=2mean(rhat*e)+mean(e^2); pair is i-j',
            factorization='s=rhat_i-rhat_j; c=e_j; delta=e_i-e_j; correction=2mean(s*c)+2mean((rhat_i+c)*delta)+mean(delta^2)',
            raw_MSE_normalization='sum squared native residual /192 exactly once, no sqrt rounding in calculation',
            float64_rounding='reported separately; no tolerance, clipping, fit, sign snapping or gate relaxation',
            actual_absolute_step=20,teacher_context_absolute_step=15,unknown_index12='no actual terminal; physical return20 retained',
            physical_orientations=METRICS,independent_sample_count=None,rank_or_policy_selection=False,
            retrospective_known_roots_only=True,capability_or_causality_claim=False))
    return value


def write(path,value):
    p=Path(path).absolute();raw=canonical(value)
    require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and p.suffix=='.json' and
            len(raw)<=CAP, 'exclusive bounded new own-stem output, never old producer paths')
    with p.open('xb') as stream:
        require(stream.write(raw)==len(raw),'whole output write');stream.flush();os.fsync(stream.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(output=str(p),analysis_sha256=sha(p),bytes=len(raw))


def verify(path,expected,rebuilt):
    same(read(path,expected),rebuilt,'full deterministic own report rebuild, not hash-only')
    return dict(verified=True,analysis_sha256=expected,fixed_pairs=80,P1_admission=False)


def consume(path,expected):
    value=build();verify(path,expected,value)
    return dict(schema=NAME+'.consume',analysis_sha256=expected,fixed_slots=40,fixed_pairs=80,
                comparable_pairs=76,unknown_pairs=4,full_saved_native_rebuilt=True,
                P1_admission=False,fresh_confirmation=False,C2_claim=False,physical_release=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('analyze','verify','consume'))
    parser.add_argument('--output');parser.add_argument('--analysis');parser.add_argument('--analysis-sha')
    args=parser.parse_args()
    if args.command=='consume':
        require(args.analysis and args.analysis_sha,'consume requires analysis and SHA')
        out=consume(args.analysis,args.analysis_sha)
    else:
        value=build()
        if args.command=='analyze':
            require(args.output,'analyze requires exclusive output');out=write(args.output,value)
        else:
            require(args.analysis and args.analysis_sha,'verify requires analysis and SHA')
            out=verify(args.analysis,args.analysis_sha,value)
    print(canonical(out).decode())


if __name__=='__main__':
    main()
