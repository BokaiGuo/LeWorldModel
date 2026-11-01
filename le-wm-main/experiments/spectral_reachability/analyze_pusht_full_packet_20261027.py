"""Separate model capacity from retained physical endpoints; no neural work."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import digest,write


def require(value,message):
    if not value:
        raise ValueError(message)


def summarize(planning,physical):
    require(planning['status']=='plans_sealed' and all(planning['gates'].values()),'complete sealed planning required')
    require(physical['failures']==[],'runtime failures are not guarded censoring')
    require(all(v is True for k,v in physical['gates'].items() if k!='encodings_complete'),'other physical integrity failed')
    require(len(physical['rows'])==80 and len(physical['encodings'])==160,'complete fixed census required')
    rows={(r['root_sha256'],r['method']):r for r in physical['rows']}
    enc={(r['root_sha256'],r['method'],r['kind']):r for r in physical['encodings']}
    require(len(rows)==80 and len(enc)==160,'duplicate observations')
    expected={(p['root_sha256'],a['arm']+'_'+c['candidate_id']) for p in planning['plans']
              for a in p['arms'] for c in a['candidates']}
    require(set(rows)==expected and set(enc)=={(*k,t) for k in expected for t in ('prefix','terminal')},'identity census differs')
    blocked=[]
    for (root,method,kind),e in enc.items():
        branch=rows[root,method]
        target=5 if kind=='prefix' else 10
        if e['status']=='completed':
            require(e['comparable'] is True and e['attempted_encode_samples']==1 and
                    branch['completed_commands']>=target and np.isfinite(e['vector']).all(),'invalid completed encoding')
        else:
            require(e['status']=='blocked' and e['attempted_encode_samples']==0 and not e['comparable'] and
                branch['status']=='observation_unavailable' and branch['completed_commands']<target,
                'missing endpoint not explained by retained guard abort')
            blocked.append(dict(root_sha256=root,method=method,kind=kind,completed_commands=branch['completed_commands']))
    groups=[]
    for p in planning['plans']:
        root=p['root_sha256']
        for arm in p['arms']:
            roles={}
            require(len(arm['candidates'])==10,'ten candidates required')
            for role,cid in [('selected',arm['selected_candidate_id']),('greedy',arm['greedy_candidate_id']),('zero','prefix_0')]:
                candidate=next(c for c in arm['candidates'] if c['candidate_id']==cid)
                method=arm['arm']+'_'+cid;b=rows[root,method]
                prefix,terminal=enc[root,method,'prefix'],enc[root,method,'terminal']
                roles[role]=dict(candidate_id=cid,status=b['status'],completed_commands=b['completed_commands'],
                    full_horizon=b['full_horizon_completed'],pose_cost=b['pose']['pose_cost'],
                    official_state_distance=b['official']['official_state_distance'],official_success=b['official']['official_success'],
                    predicted_stage_Q=candidate['stage_cost'],predicted_terminal_Q=candidate['nonlinear_terminal_cost'],
                    observed_stage_Q=prefix.get('observed_Q_goal_cost'),observed_terminal_Q=terminal.get('observed_Q_goal_cost'),
                    prefix_logit_norm=float(np.linalg.norm(candidate['prefix_params'])),
                    suffix_logit_norm=float(np.linalg.norm(candidate['suffix_params'])),
                    prefix_raw_norm=float(np.linalg.norm(candidate['prefix_commands'])),
                    suffix_raw_norm=float(np.linalg.norm(candidate['suffix_commands'])))
            groups.append(dict(root_index=p['root_index'],root_sha256=root,arm=arm['arm'],roles=roles,
                selected_is_greedy=arm['selected_candidate_id']==arm['greedy_candidate_id']))
    return dict(status='retained_guarded_diagnostic',original_physical_status=physical['status'],
        original_gates=physical['gates'],missing_endpoints_explained_by_guard=blocked,
        physical_branches=80,actual_steps=sum(r['completed_commands'] for r in rows.values()),
        encoder_samples=sum(e['attempted_encode_samples'] for e in enc.values()),
        complete_prefixes=sum(e['status']=='completed' for k,e in enc.items() if k[-1]=='prefix'),
        complete_terminals=sum(e['status']=='completed' for k,e in enc.items() if k[-1]=='terminal'),
        all_selected_equal_greedy=all(g['selected_is_greedy'] for g in groups),groups=groups,
        P1_admission=False,stage_gate_changed=False,theory_negative=False,unique_cause_proven=False,
        scope='known calibration roots only; all aborts retained, no physical reselection or population inference')


def figures(result,planning,output):
    # Styling follows figure-generation's bar/line template; real data only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':300,
        'axes.spines.top':False,'axes.spines.right':False})
    colors={'constant':'#0072B2','full':'#D55E00'}
    fig,ax=plt.subplots(figsize=(5.8,3.5),layout='constrained')
    for arm,offset in [('constant',-.17),('full',.17)]:
        ax.bar(np.arange(4)+offset,[p['q_response_rank_'+arm] for p in planning['plans']],.3,
               label='Repeated (2D)' if arm=='constant' else 'Per-step (10D)',color=colors[arm])
    ax.set(xticks=np.arange(4),xticklabels=np.arange(1,5),xlabel='Known calibration root',
           ylabel='Model Q-response rank',ylim=(0,11),yticks=[0,2,4,6,8,10])
    ax.legend(frameon=False)
    for ext in ('png','pdf'):fig.savefig(output/f'Figure_1.{ext}')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.5),layout='constrained')
    for ax,field,label in zip(axes,('pose_cost','official_state_distance'),
                             ('Object pose cost (lower is better)','Official state distance (lower is better)')):
        for arm,marker in [('constant','o'),('full','s')]:
            groups=sorted((g for g in result['groups'] if g['arm']==arm),key=lambda g:g['root_index'])
            require(all(g['roles']['selected']['full_horizon'] and g['roles']['zero']['full_horizon'] for g in groups),
                    'do not plot aborted roles as equal-horizon effects')
            ax.plot(np.arange(1,5),[g['roles']['selected'][field] for g in groups],marker+'-',
                color=colors[arm],label=('Repeated' if arm=='constant' else 'Per-step')+' selected')
            ax.plot(np.arange(1,5),[g['roles']['zero'][field] for g in groups],marker+'--',
                color=colors[arm],alpha=.65,label=('Repeated' if arm=='constant' else 'Per-step')+' zero-prefix')
        ax.set(xlabel='Known calibration root',ylabel=label,xticks=np.arange(1,5));ax.grid(alpha=.2)
    axes[0].legend(frameon=False,fontsize=8)
    for ext in ('png','pdf'):fig.savefig(output/f'Figure_2.{ext}')
    plt.close(fig)


def run(planning_path,physical_path,output):
    from experiments.spectral_reachability.pusht_full_packet_physical_20261027 import prepare
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    planning_path,physical_path,output=map(lambda p:Path(p).resolve(),(planning_path,physical_path,output))
    require(not output.exists(),'exclusive new output required')
    prepare(planning_path)  # revalidate 80 model plans and all original bindings
    inputs={str(p):digest(p) for p in (planning_path,physical_path)}
    planning,physical=[json.loads(p.read_text()) for p in (planning_path,physical_path)]
    for seal in physical['source_seals']:
        verify_seal(output_dir=seal['output_dir'],expected_manifest_sha256=seal['manifest_sha256'])
    for item in physical['binding_checks']:
        require(item['matched'] is True and digest(item['path'])==next(
            v for k,v in json.loads((physical_path.parent/'protocol.json').read_text())['bindings'].items() if k==item['path']),
            'physical input binding changed')
    result=summarize(planning,physical)
    output.mkdir(exist_ok=False)
    seal=create_seal(files=[Path(__file__),Path(__file__).resolve().parents[2]/'tests/test_analyze_pusht_full_packet_20261027.py'],
        args_json=json.dumps(inputs,sort_keys=True),output_dir=output/'source_seal')
    result.update(inputs=inputs,source_seal=seal)
    write(output/'report.json',result)
    figures(result,planning,output)
    with (output/'notes.txt').open('x') as stream:
        stream.write('Model rank rises from 2 to 10; selected remains greedy. Original physical integrity_fail is retained: three missing terminal images follow guard aborts, not encoder failures. No P1 admission or theoretical rejection.\n')
    with (output/'captions.tex').open('x') as stream:
        stream.write('% Diagnostic, four previously used calibration roots; not submission efficacy evidence.\n')
        stream.write('\\includegraphics[width=.48\\linewidth]{Figure_1.pdf}\n\\includegraphics[width=.98\\linewidth]{Figure_2.pdf}\n')
    require(inputs=={str(p):digest(p) for p in (planning_path,physical_path)},'inputs changed')
    print(json.dumps({k:result[k] for k in ('status','actual_steps','encoder_samples','complete_prefixes','complete_terminals','all_selected_equal_greedy')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planning-path',type=Path,required=True)
    parser.add_argument('--physical-path',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(**vars(parser.parse_args()))
