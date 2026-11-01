"""Saved fixed-prefix gradient wiring and fixed shadow candidate diagnostic."""
import argparse
from pathlib import Path
from copy import deepcopy
import numpy as np
from experiments.spectral_reachability import agent_p1_fixed_prefix_saved_handoff_20261029_v1 as handoff

NAME = 'agent_p1_fixed_prefix_gradient_diagnostic_20261029_v1'
host, F, core = handoff.host, handoff.F, handoff.core
require, raw, sha, digest = handoff.require, handoff.raw, handoff.sha, handoff.digest
HANDOFF_SHA = '0b9442da65a96e9a8aa1eb467d30ba660747122cbc977144e34f1e4fb887a106'


def analyze(prepared, receipt, report_sha):
    """Pure saved arithmetic; caller must independently rebuild host strict."""
    handoff.select(prepared, receipt, report_sha)
    groups = []
    for q, saved in zip(prepared['groups'], receipt['rows'], strict=True):
        r = saved['result']; iterations = r['iterates']; steps = []
        for i, gradient in enumerate(r['gradients'], 1):
            prev, cur = iterations[i-1], iterations[i]
            require(raw(gradient['context']) == raw(prev['context']), 'gradient evaluated at preceding suffix context')
            x = core.finite(prev['logits'], (10,)).astype(float)
            a = F.chart(x); g_raw = core.finite(gradient['raw_gradient'], (5,2), True).astype(float)
            g_logit = (g_raw*(1-a.astype(float)**2)).reshape(10)
            proposed = F.proposal(x, g_logit)
            require(raw(proposed.tolist()) == raw(cur['logits']) and F.bits(F.chart(proposed)) == cur['raw_bits'],
                    'exact original chart chain and projected eta1 update')
            shadow = F.proposal(x, 0.5*g_logit)
            direction = proposed-x; predicted = float(np.dot(g_logit, direction))
            actual = float(cur['terminal_Q']-prev['terminal_Q'])
            steps.append(dict(iteration=i, gradient_raw_l2=float(np.linalg.norm(g_raw)),
                chart_gradient_l2=float(np.linalg.norm(g_logit)), input_logit_l2=float(np.linalg.norm(x)),
                unprojected_eta1_l2=float(np.linalg.norm(x-g_logit)), step_l2=float(np.linalg.norm(direction)),
                projected=bool(np.linalg.norm(x-g_logit)>F.CAP),
                local_linear_delta_Q=predicted, actual_delta_Q=actual,
                linear_residual_Q=actual-predicted, previous_terminal_Q=prev['terminal_Q'],
                next_terminal_Q=cur['terminal_Q'], shadow_eta=0.5, shadow_logits=shadow.tolist(),
                shadow_raw_float32_bits=F.bits(F.chart(shadow)), shadow_terminal_Q=None,
                shadow_score_status='not_evaluated', native_gradient_correctness='not_established'))
        best = min(iterations, key=lambda x:(x['terminal_Q'],x['iteration']))
        groups.append(dict(index=q['index'],slot=q['slot'],source_arm=q['source_arm'],source_iteration=q['source_iteration'],
            stage_Q=r['stage_Q'], initial_Q=r['initial_Q'], stage_uphill=r['stage_Q']>r['initial_Q'],
            anchor_terminal_Q=r['baseline_Q'], initial_terminal_Q=iterations[0]['terminal_Q'],
            best_terminal_Q=best['terminal_Q'], best_terminal_iteration=best['iteration'],
            best_minus_anchor=best['terminal_Q']-r['baseline_Q'],
            best_minus_initial=best['terminal_Q']-iterations[0]['terminal_Q'],
            selected_iteration=r['selected_iteration'], status=r['status'], steps=steps))
    require(len(groups)==64 and sum(len(g['steps']) for g in groups)==512, 'fixed diagnostic roster')
    return groups


def build(host_output, host_report_sha, host_prepared, host_prepare_sha, host_receipt, host_receipt_sha):
    require(sha(handoff.__file__) == HANDOFF_SHA, 'sealed model-only handoff source')
    # This performs the actual full producer strict, including inherited source closure.
    authority = handoff.build(host_output,host_report_sha,host_prepared,host_prepare_sha,host_receipt,host_receipt_sha)
    p = host.signal.read(host_prepared,256*1024**2); r = host.signal.read(host_receipt,256*1024**2)
    groups = analyze(p,r,host_report_sha); bindings = deepcopy(authority['bindings'])
    for f in (Path(__file__).resolve(), handoff.ROOT/'tests'/('test_'+NAME+'.py'), handoff.ROOT/'innerdocs'/(NAME+'.md')):
        bindings[str(f)] = sha(f)
    F.check_bindings(bindings)
    args = dict(host_output=host_output,host_report_sha=host_report_sha,host_prepared=host_prepared,
                host_prepare_sha=host_prepare_sha,host_receipt=host_receipt,host_receipt_sha=host_receipt_sha)
    return dict(schema=NAME+'.analysis',arguments=args,bindings=bindings,groups=groups,group_denominator=64,
        source_candidate_denominator=576,shadow_candidate_denominator=512, shadow_scores_available=0,
        gradient_correctness_established=False, physical_outcomes_used_for_selection=False,
        training=False, physical_release=False,P1_admission=False,freshness=False)


def verify(path, expected):
    core.hash_value(expected); require(sha(path)==expected, 'actual analysis SHA')
    saved=host.signal.read(path,256*1024**2)
    require(raw(saved)==raw(build(**saved['arguments'])) and sha(path)==expected, 'full saved analysis rebuild')
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('command',choices=('analyze','verify'))
    for k in ('host-output','host-report-sha','host-prepared','host-prepare-sha','host-receipt','host-receipt-sha','output','analysis','analysis-sha'):
        ap.add_argument('--'+k)
    a=ap.parse_args()
    if a.command=='analyze':
        # handoff.write enforces its own filename prefix, so this analysis has its own exclusive writer.
        v=write(a.output,build(a.host_output,a.host_report_sha,a.host_prepared,a.host_prepare_sha,a.host_receipt,a.host_receipt_sha))
    else: verify(a.analysis,a.analysis_sha); v=dict(verified=True,P1_admission=False)
    print(raw(v).decode(),flush=True)


def write(path,value):
    import os, shutil
    path=Path(path).absolute(); data=raw(value)
    require(path.parent==handoff.ROOT/'results' and path.name.startswith(NAME+'_') and path.resolve()==path
            and not path.exists() and not path.is_symlink() and len(data)<=256*1024**2, 'exclusive bounded output')
    require(shutil.disk_usage(path.parent).free>=200*1024**2+len(data), 'disk floor')
    with path.open('xb') as f:
        require(f.write(data)==len(data),'full write'); f.flush(); os.fsync(f.fileno())
    host.signal.fsync_dir(path.parent);return dict(path=str(path),sha256=sha(path))


if __name__=='__main__':main()
