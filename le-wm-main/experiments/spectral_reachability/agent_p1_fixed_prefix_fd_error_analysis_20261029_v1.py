"""Fixed-denominator saved finite-difference error analysis; no neural calls."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
from experiments.spectral_reachability import agent_p1_fixed_prefix_fd_host_cuda_20261029_v1 as host

NAME='agent_p1_fixed_prefix_fd_error_analysis_20261029_v1'
HOST_SHA='dd27c1fa7ab8873e745098b0d8ad380642f6859543fa8c3fc876392bf021bb26'
ROOT,F,core,signal=host.ROOT,host.F,host.core,host.signal
require,raw,sha=host.require,host.raw,host.sha
SPACES=('raw','chart_logit')
STEPS=host.fd.STEPS


def number(x,label):
    require(type(x) in (int,float) and math.isfinite(x),label)
    return float(x)


def sign(x):return 'positive' if x>0 else 'negative' if x<0 else 'zero'


def summary(rows):
    available=[v for v in rows if v['finite_difference'] is not None]
    errors=[v['absolute_error'] for v in available]
    relative=[v['relative_error'] for v in available if v['relative_error'] is not None]
    return dict(denominator=len(rows),available=len(available),missing=len(rows)-len(available),
        relative_available=len(relative),relative_zero_denominator=len(available)-len(relative),
        abs_error_mean=None if not errors else math.fsum(errors)/len(errors),abs_error_max=None if not errors else max(errors),
        relative_error_mean=None if not relative else math.fsum(relative)/len(relative),
        relative_error_max=None if not relative else max(relative),
        fd_sign={s:sum(v['finite_difference_sign']==s for v in available) for s in ('negative','zero','positive')},
        ad_sign={s:sum(v['saved_AD_sign']==s for v in available) for s in ('negative','zero','positive')},
        sign_relation={s:sum(v['sign_relation']==s for v in available) for s in ('same_nonzero','opposite','one_zero','both_zero')})


def analyze(receipt,report_sha):
    """Pure arithmetic: caller must first rebuild and authenticate full consume."""
    core.hash_value(report_sha)
    require(receipt['schema']==host.NAME+'.consume' and receipt['status']=='verified_model_only' and receipt['complete'] is True
        and receipt['report_sha256']==report_sha and receipt['group_denominator']==64 and receipt['pair_denominator']==2560
        and all(receipt[k] is False for k in ('P1_admission','freshness','physical_release')),'complete authenticated schema required')
    require(len(receipt['rows'])==len(receipt['audits'])==len(receipt['invalid_causes'])==64,'complete group roster')
    rows=[]
    for i,(group,audit,invalid) in enumerate(zip(receipt['rows'],receipt['audits'],receipt['invalid_causes'],strict=True)):
        require(group['index']==i and group['status']=='sealed' and audit['index']==i and audit['pairs']==40,'sealed audited group')
        result=group['result']; require(result['index']==i and result['pair_denominator']==40 and len(result['pairs'])==40,'40 fixed pair slots')
        invalid_by_index={v['pair']:v for v in invalid}
        require(len(invalid_by_index)==len(invalid),'unique invalid causes')
        for j,p in enumerate(result['pairs']):
            space=SPACES[j//20];h=STEPS[(j%20)//10];coordinate=j%10
            require(p['space']==space and p['requested_step']==h and type(p['coordinate']) is int and p['coordinate']==coordinate
                and type(p['valid']) is bool,'fixed pair order')
            ad=number(p['saved_AD_direction'],'finite AD'); valid=p['valid']
            row=dict(group=i,slot=i//16,space=space,scale=h,coordinate=coordinate,valid=valid,
                saved_AD=ad,saved_AD_sign=sign(ad),finite_difference=None,finite_difference_sign=None,
                signed_error=None,absolute_error=None,relative_denominator=None,relative_error=None,
                relative_undefined_reason=None,sign_relation=None,null_causes=None,
                plus_step=number(p['plus_step'],'plus step'),minus_step=number(p['minus_step'],'minus step'))
            if not valid:
                require(j in invalid_by_index and type(invalid_by_index[j]['reasons']) is list and invalid_by_index[j]['reasons']
                    and invalid_by_index[j]['space']==space and invalid_by_index[j]['coordinate']==coordinate
                    and invalid_by_index[j]['requested_step']==h and all(p[k] is None for k in ('plus_Q','minus_Q','finite_difference','error_vs_saved_AD')),
                    'explicit null with cause and no fabricated score')
                row['null_causes']=deepcopy(invalid_by_index[j]['reasons'])
            else:
                require(j not in invalid_by_index,'valid pair cannot have null reason')
                plus=number(p['plus_Q'],'finite plus Q');minus=number(p['minus_Q'],'finite minus Q')
                denom=number(p['denominator'],'finite denominator')
                require(denom>0 and row['plus_step']>0 and row['minus_step']>0
                        and denom==row['plus_step']+row['minus_step'],'actual float32 spacing')
                fd=number(p['finite_difference'],'finite FD'); error=number(p['error_vs_saved_AD'],'finite error')
                require(fd==(plus-minus)/denom and error==fd-ad,'saved FD and error arithmetic')
                absolute=abs(error);rel_denom=abs(ad)
                require(math.isfinite(absolute),'finite absolute error')
                relation=('both_zero' if fd==0 and ad==0 else 'one_zero' if fd==0 or ad==0 else
                          'same_nonzero' if sign(fd)==sign(ad) else 'opposite')
                relative=None if rel_denom==0 else absolute/rel_denom
                require(relative is None or math.isfinite(relative),'relative error overflow')
                row.update(finite_difference=fd,finite_difference_sign=sign(fd),signed_error=error,absolute_error=absolute,
                           relative_denominator=rel_denom,relative_error=relative,
                           relative_undefined_reason='saved_AD_is_zero' if rel_denom==0 else None,sign_relation=relation)
            rows.append(row)
        require(result['valid_pairs']==sum(p['valid'] for p in result['pairs']) and
            result['predict_calls']==2*result['valid_pairs'] and audit['valid_pairs']==result['valid_pairs']
            and audit['native_forwards']==result['predict_calls'] and len(invalid)==40-result['valid_pairs'],'native/valid/null counts')
    require(len(rows)==2560,'all fixed pair positions')
    bins=[dict(space=s,scale=h,**summary([v for v in rows if v['space']==s and v['scale']==h])) for s in SPACES for h in STEPS]
    crosses=[]
    by={(v['group'],v['space'],v['coordinate'],v['scale']):v for v in rows}
    for i in range(64):
        for space in SPACES:
            for j in range(10):
                a,b=(by[(i,space,j,h)] for h in STEPS)
                require(a['saved_AD']==b['saved_AD'],'same saved gradient across scales')
                if a['valid'] and b['valid']:
                    delta=b['finite_difference']-a['finite_difference']
                    require(math.isfinite(delta),'finite scale difference')
                    reason=None
                else:delta=None;reason='missing_one_or_both_scales'
                crosses.append(dict(group=i,slot=i//16,space=space,coordinate=j,large_scale=STEPS[0],small_scale=STEPS[1],
                    large_fd=a['finite_difference'],small_fd=b['finite_difference'],small_minus_large=delta,
                    absolute_difference=None if delta is None else abs(delta),null_reason=reason))
    require(len(crosses)==1280,'all cross-scale positions')
    cross_summary=[]
    for space in SPACES:
        subset=[x for x in crosses if x['space']==space]; vals=[x['absolute_difference'] for x in subset if x['absolute_difference'] is not None]
        cross_summary.append(dict(space=space,denominator=640,available=len(vals),missing=640-len(vals),
            abs_difference_mean=None if not vals else math.fsum(vals)/len(vals),abs_difference_max=None if not vals else max(vals)))
    return dict(rows=rows,by_space_scale=bins,cross_scale=crosses,cross_scale_summary=cross_summary,
        group_denominator=64,pair_denominator=2560,cross_scale_denominator=1280,
        relative_definition='abs(FD-AD)/abs(AD); null if AD exactly zero',thresholds=None,
        gradient_correctness_established=False,P1_admission=False,freshness=False,physical_release=False)


def rebuild(output,report_sha):return host.consume(output,report_sha)


def build(output,report_sha,receipt,receipt_sha):
    require(sha(host.__file__)==HOST_SHA,'frozen FD host source')
    out=Path(output);path=Path(receipt)
    for p in (out,path):require(p.is_absolute() and p.resolve()==p,'absolute actual source')
    core.hash_value(report_sha);core.hash_value(receipt_sha)
    require(sha(out/'report.json')==report_sha and sha(path)==receipt_sha,'external SHA')
    rebuilt=rebuild(str(out),report_sha)
    require(path.read_bytes()==raw(rebuilt) and sha(path)==receipt_sha,'full consume equals persisted receipt bytes')
    analysis=analyze(rebuilt,report_sha)
    report=signal.read(out/'report.json',256*1024**2);p=signal.read(report['prepared'],256*1024**2)
    require(sha(report['prepared'])==report['prepare_sha256'],'actual FD prepared SHA')
    bindings=deepcopy(p['bindings']); bindings[str(out/'report.json')]=report_sha;bindings[str(path)]=receipt_sha
    bindings[report['prepared']]=report['prepare_sha256'];bindings[str(Path(host.__file__).resolve())]=HOST_SHA
    for name,h in report['artifact_sha256'].items():bindings[str(out/name)]=h
    for f in (Path(__file__).resolve(),ROOT/'tests'/('test_'+NAME+'.py'),ROOT/'innerdocs'/(NAME+'.md')):bindings[str(f)]=sha(f)
    F.check_bindings(bindings)
    return dict(schema=NAME+'.analysis',arguments=dict(output=output,report_sha=report_sha,receipt=receipt,receipt_sha=receipt_sha),
                bindings=bindings,analysis=analysis)


def verify(path,expected):
    core.hash_value(expected);require(sha(path)==expected,'analysis SHA');a=signal.read(path,256*1024**2)
    require(raw(a)==raw(build(**a['arguments'])) and sha(path)==expected,'full receipt and analysis rebuild');return True


def write(path,value):
    path=Path(path).absolute();data=raw(value)
    require(path.parent==ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path and not path.exists()
        and not path.is_symlink() and len(data)<=256*1024**2,'exclusive bounded output')
    require(shutil.disk_usage(path.parent).free>=200*1024**2+len(data),'disk floor')
    with path.open('xb') as f:require(f.write(data)==len(data),'full write');f.flush();os.fsync(f.fileno())
    signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('analyze','verify'))
    for k in ('output','report-sha','receipt','receipt-sha','analysis','analysis-sha','save'):ap.add_argument('--'+k)
    a=ap.parse_args()
    if a.command=='analyze':v=write(a.save,build(a.output,a.report_sha,a.receipt,a.receipt_sha))
    else:verify(a.analysis,a.analysis_sha);v=dict(verified=True,P1_admission=False)
    print(raw(v).decode(),flush=True)


if __name__=='__main__':main()
