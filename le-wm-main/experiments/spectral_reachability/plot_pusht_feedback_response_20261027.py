"""Saved known-root response plots; no new neural work or policy selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability.pusht_geometry_feedback_response_20261027_v1 import (
    verify_frozen, sha, ROLES, ARMS, analyze, require_close,
)
from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import write


def summarize(report, protocol):
    if report['status'] != 'diagnostic_recorded' or report['failure'] is not None:
        raise ValueError('complete diagnostic required')
    if (len(report['rows']) != 6 or len(report['contexts']) != 3
            or sum(p['predict_calls'] for p in report['predict']) != 249
            or report['image_work'] != dict(calls=3,frames=9)):
        raise ValueError('full actual work census required')
    idx = {(r['role'],r['arm']):r for r in report['rows']}
    if set(idx) != {(r,a) for r in ROLES for a in ARMS}:
        raise ValueError('complete unique role-arm census required')
    cases = {c['role']:c for c in protocol['cases']}
    rows = []
    for role in ROLES:
        for arm in ARMS:
            saved = idx[role,arm]['analysis']
            predicted = {p['probe_id']:p['predicted'] for p in saved['probes']}
            recomputed = analyze(predicted,cases[role]['real'],protocol['bank'],protocol['q'],protocol['goal'])
            if saved != recomputed:
                raise ValueError('per-probe arithmetic binding differs')
            zero = saved['zero_prediction_residual']['projected_mse']
            all_error = float(np.mean([p['projected_mse'] for p in saved['probes']]))
            for grid in saved['grids']:
                require_close(np.asarray(grid['real_K']),np.asarray(recomputed['grids'][
                    0 if grid['epsilon']==.05 else 1]['real_K']),'physical matrix',atol=0)
            rows.append(dict(role=role,arm=arm,zero_prediction_mse=zero,
                all_41_probe_mean_prediction_mse=all_error,
                response_cosines=[g['cosine'] for g in saved['grids']],
                response_relative_errors=[g['relative_error'] for g in saved['grids']]))
    return dict(status='saved_arithmetic_recomputed',rows=rows,new_neural_calls=0,
        independent_roots=0,P1_admission=False,
        scope='one reused model-qualified root, fixed suffix probes; diagnostic, not efficacy')


def figures(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
    idx={(r['role'],r['arm']):r for r in result['rows']}
    x=np.arange(3); colors=('#0072B2','#D55E00')
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.4),layout='constrained')
    for ax,field,label in zip(axs,('zero_prediction_mse','all_41_probe_mean_prediction_mse'),
            ('Zero-suffix prediction Q-MSE','All 41 probes: mean prediction Q-MSE')):
        for i,arm in enumerate(ARMS):
            ax.bar(x+(i-.5)*.34,[idx[r,arm][field] for r in ROLES],width=.32,
                color=colors[i],label='Imagined prefix' if i==0 else 'RGB-reencoded prefix')
        ax.set(xticks=x,xticklabels=ROLES,ylabel=label,yscale='log',xlabel='Saved role; one known root')
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
    for ext in ('png','pdf'):fig.savefig(output/f'Figure_1.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.4),layout='constrained',sharey=True)
    for j,ax in enumerate(axs):
        for i,arm in enumerate(ARMS):
            ax.bar(x+(i-.5)*.34,[idx[r,arm]['response_cosines'][j] for r in ROLES],width=.32,
                color=colors[i],label='Imagined prefix' if i==0 else 'RGB-reencoded prefix')
        ax.axhline(0,color='black',linewidth=.7)
        ax.set(xticks=x,xticklabels=ROLES,ylim=(-1,1),xlabel=f'Fixed logit epsilon = {(.05,.1)[j]}')
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
    axs[0].set_ylabel('Predicted vs real Q-response cosine')
    for ext in ('png','pdf'):fig.savefig(output/f'Figure_2.{ext}')
    plt.close(fig)


def run(input_dir,output):
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal
    input_dir,output=Path(input_dir).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    protocol=verify_frozen(input_dir)
    path=input_dir/'report.json'; expected=sha(path)
    result=summarize(json.loads(path.read_text()),protocol)
    output.mkdir(exist_ok=False)
    result['inputs']={str(path):expected,str(input_dir/'protocol.json'):sha(input_dir/'protocol.json')}
    result['source_seal']=create_seal(files=[Path(__file__),Path(__file__).resolve().parents[2]/
        'tests/test_plot_pusht_feedback_response_20261027.py'],args_json=json.dumps(result['inputs']),
        output_dir=output/'source_seal')
    write(output/'report.json',result);figures(result,output)
    with (output/'notes.txt').open('x') as f:
        f.write('Nominal prediction repair is not controllability repair. One reused root, no policy improvement or P1 pass.\n')
    with (output/'captions.tex').open('x') as f:
        f.write('% One reused calibration root; paired fixed-probe diagnostics only.\n'
            '\\includegraphics[width=\\linewidth]{Figure_1.pdf}\n'
            '\\includegraphics[width=\\linewidth]{Figure_2.pdf}\n')
    if sha(path)!=expected:raise ValueError('input changed')
    print(json.dumps(result['rows']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(**vars(p.parse_args()))
