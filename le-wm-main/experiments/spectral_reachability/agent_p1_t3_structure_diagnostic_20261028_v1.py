"""Pinned saved80 T3 attribution; exact term-removal is not a deployable correction."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

NAME = 'agent_p1_t3_structure_diagnostic_20261028_v1'
UPSTREAM_NAME = 'agent_p1_pair_margin_decomposition_20261028_v1'
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'results'/f'{UPSTREAM_NAME}_MAIN01.json'
SOURCE_SHA = 'f723c5eb73ecebd0615a2ded0aa4b741c6f132ca5f6063c673f12d5e98539143'
FROZEN_SOURCES = {
    str(ROOT/'experiments/spectral_reachability'/f'{UPSTREAM_NAME}.py'):
        'bbbb39ef5000f9ca343dc512ba5de1f0a9795c680cc2d8f7492a3f000a542426',
    str(ROOT/'tests'/f'test_{UPSTREAM_NAME}.py'):
        'c7bf1f9449a1705a5e1883f71ff07bf489bf513b0e53c570ec8e7c78bad3583b',
    str(ROOT/'innerdocs'/f'{UPSTREAM_NAME}.md'):
        '6f1f806327862c8acb78dea2d021910d23bc2270e5addf3e24fb72690cba2766',
}
TERMS = ('common_bias_projection', 'relative_response_projection', 'relative_response_quadratic')
CLASSES = ('concordant','discordant','both_tie','left_tie_only','right_tie_only','unknown')
CAP = 4*1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def read(path, expected):
    p=Path(path).absolute()
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=CAP,
            'bounded regular canonical input')
    raw=p.read_bytes()
    require(len(raw)<=CAP and hashlib.sha256(raw).hexdigest()==expected,'input SHA')
    def unique(items):
        out={}
        for k,v in items:
            require(k not in out,'duplicate JSON key')
            out[k]=v
        return out
    value=json.loads(raw,object_pairs_hook=unique)
    canonical(value)  # Reject nonfinite JSON scalars, not just the fields used below.
    return value


def pack(value):
    return dict(numerator=str(value.numerator),denominator=str(value.denominator))


def rational(value):
    require(isinstance(value,dict) and set(value)=={'numerator','denominator'} and
            all(type(x) is str and len(x)<=1024 for x in value.values()),'canonical rational strings')
    try:
        n,d=int(value['numerator']),int(value['denominator'])
    except ValueError as exc:
        raise ValueError('canonical rational integers') from exc
    require(d>0,'positive rational denominator')
    x=Fraction(n,d)
    require(pack(x)==value,'canonical reduced rational encoding')
    return x


def sign(x):
    return int(x>0)-int(x<0)


def classification(a,b):
    sa,sb=sign(a),sign(b)
    return ('both_tie' if sa==sb==0 else 'left_tie_only' if sa==0 else
            'right_tie_only' if sb==0 else 'concordant' if sa==sb else 'discordant')


def analyze_pair(row):
    """No physical label is read. Fixed i<j reference e_j, no alternative gauge search."""
    out={k:row[k] for k in ('i','j','anchor_index','missing_indices')}
    if row['missing_indices']:
        require(row['exact'] is row['float64'] is None and row['model_to_observed']['classification']=='unknown',
                'unknown remains unfilled')
        return dict(**out,available=False,original_float64=dict(row['model_to_observed']),
                    full_exact=None,margin_threshold=None,contributions=None,cancellation=None,counterfactuals=None)
    ex=row['exact']; require(isinstance(ex,dict),'available exact terms')
    m,o,correction=(rational(ex[k]) for k in
                    ('predicted_margin','observed_margin','observed_minus_predicted_margin'))
    terms=tuple(rational(ex[k]) for k in TERMS)
    require(rational(ex['identity_residual'])==rational(ex['factorization_residual'])==0 and
            o-m==correction==sum(terms,Fraction()),'exact factorization identity')
    require(terms[2]>=0,'nonnegative relative response quadratic')
    f=row['float64']; fp,fo=f['predicted_margin'],f['observed_margin']
    require(all(type(x) in (float,int) and math.isfinite(x) for x in (fp,fo)),'finite original scalar margins')
    original=row['model_to_observed']
    require(original['classification']==classification(fp,fo) and
            canonical([original['delta_a'],original['delta_b']])==canonical([fp,fo]),
            'original float64 strict classification and signed scalar binding')
    direction=sign(m)
    full_strict=m*o<0
    subsets=[]
    for mask in range(8):
        retained=[k for bit,k in enumerate(TERMS) if mask & (1<<bit)]
        delta=sum((v for bit,v in enumerate(terms) if mask & (1<<bit)),Fraction())
        cf=m+delta
        subsets.append(dict(retained_mask=mask,retained_terms=retained,
                            removed_terms=[k for k in TERMS if k not in retained],
                            correction=pack(delta),margin=pack(cf),sign=sign(cf),
                            classification=classification(m,cf),strict_crossing=m*cf<0,
                            oriented_progress=None if not direction else pack(-direction*delta)))
    contributions={}
    for bit,(k,v) in enumerate(zip(TERMS,terms,strict=True)):
        progress=None if not direction else -direction*v
        alone=subsets[1<<bit];without=subsets[7 ^ (1<<bit)]
        contributions[k]=dict(value=pack(v),oriented_progress=None if progress is None else pack(progress),
            fraction_of_original_margin=None if progress is None else pack(progress/abs(m)),
            direction='undefined_zero_original_margin' if progress is None else
                'toward_crossing' if progress>0 else 'opposes_crossing' if progress<0 else 'zero',
            alone_strict_crossing=alone['strict_crossing'],
            necessary_for_full_strict_crossing=full_strict and not without['strict_crossing'],
            alone_mask=1<<bit,without_mask=7 ^ (1<<bit))
    positive=sum((v for v in terms if v>0),Fraction())
    negative=-sum((v for v in terms if v<0),Fraction())
    l1=positive+negative; net=sum(terms,Fraction());opposing=min(positive,negative)
    require(l1-abs(net)==2*opposing,'exact cancellation identity')
    return dict(**out,available=True,
        original_float64=dict(predicted_margin=fp,observed_margin=fo,classification=original['classification'],
            predicted_sign=sign(fp),observed_sign=sign(fo),signs_equal_exact=sign(fp)==sign(m) and sign(fo)==sign(o)),
        full_exact=dict(predicted_margin=pack(m),observed_margin=pack(o),correction=pack(net),
                        classification=classification(m,o),strict_crossing=full_strict),
        margin_threshold=None if not direction else pack(abs(m)),contributions=contributions,
        cancellation=dict(positive_mass=pack(positive),negative_mass=pack(negative),l1_mass=pack(l1),
            net_correction=pack(net),opposing_mass=pack(opposing),cancelled_l1_mass=pack(2*opposing),
            cancelled_l1_fraction=None if l1==0 else pack(2*opposing/l1)),counterfactuals=subsets)


def counts(values):
    out={k:values.count(k) for k in CLASSES}
    require(sum(out.values())==80,'all80 classification denominator')
    return out


def assemble(source):
    require(source['schema']==UPSTREAM_NAME+'.analysis' and
            source['status']=='saved_T3_T4_exact_pair_algebra_not_admitted' and
            [source[k] for k in ('fixed_slots','observed_latent_slots','missing_indices','fixed_pairs','known_roots','fresh_roots')]
                ==[40,39,[12],80,2,0] and
            source['P1_admission'] is source['fresh_confirmation'] is False,'fixed original retrospective snapshot')
    p=source['protocol']
    require([p[k] for k in ('MSE_divisor','metric_operator','actual_absolute_step','teacher_context_absolute_step')]
            ==[192,'identity192',20,15],'same original coordinates, metric and time')
    roster=[(a,i,j) for a in range(8) for i,j in itertools.combinations(range(5*a,5*a+5),2)]
    require(len(source['pairs'])==80,'fixed80 input pairs')
    rows=[]
    for row,(anchor,i,j) in zip(source['pairs'],roster,strict=True):
        require(all(type(row[k]) is int for k in ('anchor_index','i','j')) and
                (row['anchor_index'],row['i'],row['j'])==(anchor,i,j) and
                row['missing_indices']==([12] if 12 in (i,j) else []),'fixed roster and original unknown12')
        rows.append(analyze_pair(row))
    available=[x for x in rows if x['available']]
    require(len(available)==76,'fixed76 comparable/4unknown')
    summaries=[]
    for mask in range(8):
        labels=[x['counterfactuals'][mask]['classification'] if x['available'] else 'unknown' for x in rows]
        summaries.append(dict(retained_mask=mask,fixed_pairs=80,available=76,unknown=4,
            classifications=counts(labels),strict_crossings=labels.count('discordant')))
    return dict(fixed_pairs=80,comparable_pairs=76,unknown_pairs=4,pairs=rows,
        summary=dict(fixed_pairs=80,available=76,unknown=4,
            original_float64_classifications=counts([x['original_float64']['classification'] for x in rows]),
            full_exact_classifications=counts([x['full_exact']['classification'] if x['available'] else 'unknown' for x in rows]),
            float64_exact_sign_disagreements=sum(not x['original_float64']['signs_equal_exact'] for x in available),
            counterfactuals=summaries,contributions={k:dict(fixed_pairs=80,available=76,unknown=4,
                alone_strict_crossings=sum(x['contributions'][k]['alone_strict_crossing'] for x in available),
                necessary_for_full_strict_crossings=sum(x['contributions'][k]['necessary_for_full_strict_crossing'] for x in available))
                for k in TERMS},pairs_with_opposing_terms=sum(rational(x['cancellation']['opposing_mass'])>0 for x in available),
            independent_sample_count=None,full_library_reversal_rate=None),
        known_roots=2,fresh_roots=0,P1_admission=False,fresh_confirmation=False,
        deployed_correction=False,new_selector=False,fitted_parameters=False,physical_label_selection=False,
        science_gate_created=False,causal_mechanism_identified=False,new_neural_calls=0,new_physics_calls=0)


def load_source():
    value=read(SOURCE,SOURCE_SHA)
    require(value['source_files']==FROZEN_SOURCES,'original frozen source manifest')
    for path,expected in FROZEN_SOURCES.items():
        p=Path(path)
        require(p.resolve()==p and p.is_file() and not p.is_symlink() and sha(p)==expected,'original frozen source SHA')
    return value


def build():
    source=load_source()
    value=assemble(source)
    own=[Path(__file__).resolve(),ROOT/'tests'/f'test_{NAME}.py',ROOT/'innerdocs'/f'{NAME}.md']
    value.update(schema=NAME+'.analysis',status='retrospective_exact_term_attribution_not_a_repair',
        authority=dict(snapshot=str(SOURCE),snapshot_sha256=SOURCE_SHA,frozen_source_files=FROZEN_SOURCES,
            input_full_saved_native_rebuilt=source['authority']['full_saved_native_rebuilt'],
            input_closure_digest=source['authority']['closure_digest'],
            upstream_closure_rebuilt_by_this_consumer=False),source_files={str(p):sha(p) for p in own},
        protocol=dict(terms=list(TERMS),mask_bits='bit0 common, bit1 relative projection, bit2 relative quadratic; retain bits=1',
            exact_margin='m=predicted cost_i-cost_j; observed=m+B+R+Q; no second normalization',
            crossing='strict iff m*counterfactual_margin<0; zero is tie, not crossing',
            oriented_progress='-sign(m)*term; crossing iff summed progress>abs(m), for m!=0',
            reference='fixed i<j, c=e_j; B=2mean((rhat_i-rhat_j)*e_j); not a fitted global shared bias',
            cancellation='sum(abs(terms))-abs(sum(terms))=2*min(positive_mass,negative_mass)',
            nondeployability='algebraic pair term deletion; not necessarily realizable as consistent endpoint corrections',
            interpretation='sufficiency/necessity only for this fixed additive decomposition, not causal intervention',
            exhaustive_masks=list(range(8)),choose_best_mask=False,physical_labels_used=False,
            original_float64='retained separately from exact counterfactuals; no tolerances or sign snapping',
            full_denominator=80,comparable=76,unknown=4,prospective_confirmation=False))
    return value


def write(path,value):
    p=Path(path).absolute();raw=canonical(value)
    require(p.resolve()==p and p.parent==ROOT/'results' and p.name.startswith(NAME+'_') and
            p.suffix=='.json' and len(raw)<=CAP,'exclusive bounded own-stem output')
    with p.open('xb') as stream:
        require(stream.write(raw)==len(raw),'complete report write');stream.flush();os.fsync(stream.fileno())
    fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(output=str(p),analysis_sha256=sha(p),bytes=len(raw))


def verify(path,expected,rebuilt):
    require(canonical(read(path,expected))==canonical(rebuilt),'complete deterministic diagnostic rebuild')
    return dict(verified=True,analysis_sha256=expected,fixed_pairs=80,comparable_pairs=76,unknown_pairs=4,
                upstream_closure_rebuilt_by_this_consumer=False,P1_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('analyze','verify','consume'))
    parser.add_argument('--output');parser.add_argument('--analysis');parser.add_argument('--analysis-sha')
    args=parser.parse_args();value=build()
    if args.command=='analyze':
        require(args.output,'analyze needs --output');out=write(args.output,value)
    else:
        require(args.analysis and args.analysis_sha,'verify/consume need --analysis and --analysis-sha')
        out=verify(args.analysis,args.analysis_sha,value)
        if args.command=='consume':out.update(schema=NAME+'.consume',deployed_correction=False,new_selector=False)
    print(canonical(out).decode())


if __name__=='__main__':
    main()
