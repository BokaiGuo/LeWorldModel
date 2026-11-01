"""Saved model/physics geometry-generation ablation; no neural work or admission."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability.p3_matched_physical_20261027_v1 import digest, write


def require(value,message):
    if not value:raise ValueError(message)


def generation(planning):
    require(planning['status']=='plans_sealed' and bool(planning['gates'])
        and all(v is True for v in planning['gates'].values()),'sealed planning required')
    roots=[]
    for p in planning['plans']:
        full,stop=[np.asarray(p['gradient_grids'][0]['gradients'][a]) for a in ('geometry','stopK')]
        fine=np.asarray(p['gradient_grids'][1]['gradients']['geometry'])
        norm=np.linalg.norm(full); other=np.linalg.norm(stop)
        require(norm>0 and other>0,'nonzero diagnostic derivative required')
        arms=[]
        for a in p['arms']:
            common=min(a['candidates'][:10],key=lambda c:c['objective'])
            selected=min(a['candidates'],key=lambda c:c['objective'])
            own=min(a['candidates'],key=lambda c:c['own_reach_value'])
            arms.append(dict(arm=a['arm'],selected=a['selected_candidate_id'],greedy=a['greedy_candidate_id'],
                selected_is_common_winner=selected['candidate_id']==common['candidate_id'],
                objective_improvement_vs_common=common['objective']-selected['objective'],
                equal_duration_stage_argmin=own['candidate_id']))
        roots.append(dict(root_index=p['root_index'],root_sha256=p['root_sha256'],
            geometry_term_relative_norm=float(np.linalg.norm(full-stop)/norm),
            gradient_cosine=float(full@stop/(norm*other)),
            epsilon_halfstep_relative_difference=float(np.linalg.norm(full-fine)/norm),arms=arms))
    return roots


def summarize(planning,physical):
    from experiments.spectral_reachability.run_pusht_d2p_p1_physical_calibration import _pose_label
    roots=generation(planning)
    require(not physical['failures'],'runtime failures cannot be silently treated as guard censoring')
    require(set(physical['gates'])=={'all96branchcensus','all_ninety_six_retained','encodings_complete',
        'frozen_eval','inputs_unchanged','model_unchanged','no_execution_errors','sources_unchanged',
        'trace_matches_saved_commands'},'physical integrity gate census differs')
    require(all(v is True for k,v in physical['gates'].items() if k!='encodings_complete'),'other physical integrity failed')
    rows={(r['root_sha256'],r['method']):r for r in physical['rows']}
    enc={(e['root_sha256'],e['method'],e['kind']):e for e in physical['encodings']}
    expected={(p['root_sha256'],a['arm']+'_'+c['candidate_id']) for p in planning['plans'] for a in p['arms'] for c in a['candidates']}
    require(len(rows)==len(physical['rows'])==len(expected)==96 and set(rows)==expected,'physical census differs')
    require(len(enc)==len(physical['encodings'])==192 and set(enc)=={(*k,t) for k in expected for t in ('prefix','terminal')},'encoding census differs')
    blocked=[]
    for (root,method,kind),e in enc.items():
        branch=rows[root,method]; target=5 if kind=='prefix' else 10
        if e['status']=='completed':
            require(e['comparable'] and e['attempted_encode_samples']==1 and branch['completed_commands']>=target,
                    'incomplete endpoint encoded as fixed horizon')
        else:
            require(e['status']=='blocked' and not e['comparable'] and e['attempted_encode_samples']==0
                and branch['status'] in ('observation_unavailable','environment_terminated') and branch['completed_commands']<target,
                'missing encoding not explained by retained endpoint guard')
            blocked.append(dict(root_sha256=root,method=method,kind=kind,completed_commands=branch['completed_commands']))
    require(physical['gates']['encodings_complete'] is (len(blocked)==0),'encoding gate contradicts retained slots')
    require(physical['status']==('integrity_fail' if blocked else 'calibration_recorded'),
        'physical status must retain its original integrity verdict')
    groups=[]
    for p in planning['plans']:
        for a in p['arms']:
            roles={}
            for role,cid in (('selected',a['selected_candidate_id']),('greedy',a['greedy_candidate_id']),('zero','prefix_0')):
                c=next(c for c in a['candidates'] if c['candidate_id']==cid)
                key=(p['root_sha256'],a['arm']+'_'+cid); b=rows[key]
                stage_pose=None
                if b['completed_commands']>=5:
                    stage_pose=_pose_label(np.asarray(b['trace'][4]['state']),np.asarray(b['context_identity']['goal']))['pose_cost']
                roles[role]=dict(candidate_id=cid,status=b['status'],completed_commands=b['completed_commands'],
                    predicted_stage_Q=c['stage_cost'],predicted_terminal_Q=c['nonlinear_terminal_cost'],
                    observed_stage_Q=enc[*key,'prefix'].get('observed_Q_goal_cost'),
                    observed_terminal_Q=enc[*key,'terminal'].get('observed_Q_goal_cost'),
                    prefix_pose=stage_pose,terminal_pose=b['pose']['pose_cost'] if b['full_horizon_completed'] else None,
                    terminal_official_distance=b['official']['official_state_distance'] if b['full_horizon_completed'] else None,
                    official_success=b['official']['official_success'] if b['full_horizon_completed'] else None,
                    fixed_terminal_available=b['full_horizon_completed'])
            groups.append(dict(root_index=p['root_index'],root_sha256=p['root_sha256'],arm=a['arm'],
                model_non_greedy=a['selected_candidate_id']!=a['greedy_candidate_id'],roles=roles))
    return dict(status='retained_guarded_diagnostic',original_physical_status=physical['status'],original_gates=physical['gates'],
        generation=roots,groups=groups,blocked_endpoints=blocked,branches=96,
        actual_steps=sum(r['completed_commands'] for r in rows.values()),
        actual_encoder_samples=sum(e['attempted_encode_samples'] for e in enc.values()),
        all_winners_unchanged_common=all(a['selected_is_common_winner'] for r in roots for a in r['arms']),
        P1_admission=False,theory_negative=False,new_neural_calls=0,
        scope='four reused calibration roots; changed greedy reference is not improved method; no physical reselection')


def figures(result,output):
    # figure-generation template's line/bar styling, real saved data only.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':300,
        'axes.spines.top':False,'axes.spines.right':False})
    rows=sorted(result['generation'],key=lambda r:r['root_index']); x=[r['root_index'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.2),layout='constrained')
    for ax,field,label,color in zip(axes,('geometry_term_relative_norm','epsilon_halfstep_relative_difference'),
        ('Geometry-term norm / full gradient (%)','Half-step derivative discrepancy (%)'),('#0072B2','#D55E00')):
        ax.bar(x,[100*r[field] for r in rows],color=color,width=.6)
        ax.set(xlabel='Known calibration root index',ylabel=label,xticks=x); ax.grid(axis='y',alpha=.2)
    for ext in ('png','pdf'):fig.savefig(output/f'Figure_1.{ext}')
    plt.close(fig)
    qualified=[g for g in result['groups'] if g['arm']=='geometry' and g['model_non_greedy']]
    require(len(qualified)==1,'this figure protocol expects the single preregistered model-qualified root')
    g=qualified[0]; roles=['selected','greedy','zero']; positions=np.arange(3)
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.2),layout='constrained')
    for ax,fields,labels,ylabel in ((axes[0],('prefix_pose','terminal_pose'),('Physical step 5','Physical step 10'),'Object pose cost (lower is better)'),
        (axes[1],('predicted_terminal_Q','observed_terminal_Q'),('Model terminal','Real RGB terminal'),'Goal-Q cost (lower is better)')):
        for offset,field,label,color in zip((-.17,.17),fields,labels,('#0072B2','#D55E00')):
            values=[g['roles'][r][field] for r in roles]
            ax.bar(positions+offset,[np.nan if v is None else v for v in values],width=.3,label=label,color=color)
            for i,v in enumerate(values):
                if v is None:ax.text(i+offset,0,'missing',rotation=90,va='bottom',ha='center',fontsize=8)
        ax.set(xticks=positions,xticklabels=roles,ylabel=ylabel,xlabel=f'Known root {g["root_index"]}: saved model roles')
        ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.2)
    for ext in ('png','pdf'):fig.savefig(output/f'Figure_2.{ext}')
    plt.close(fig)


def run(planning_path,physical_path,output):
    from experiments.spectral_reachability.pusht_geometry_physical_20261027_v2 import prepare
    from experiments.spectral_reachability.d2p_prerun_source_seal import create_seal,verify_seal
    planning_path,physical_path,output=map(lambda x:Path(x).resolve(),(planning_path,physical_path,output))
    require(not output.exists(),'exclusive output required')
    prepare(planning_path)
    inputs={str(p):digest(p) for p in (planning_path,physical_path)}
    planning,physical=[json.loads(p.read_text()) for p in (planning_path,physical_path)]
    for s in physical['source_seals']:verify_seal(output_dir=s['output_dir'],expected_manifest_sha256=s['manifest_sha256'])
    for b in physical['binding_checks']:require(b['matched'] and digest(b['path'])==json.loads((physical_path.parent/'protocol.json').read_text())['bindings'][b['path']],'binding drift')
    result=summarize(planning,physical)
    output.mkdir(exist_ok=False)
    seal=create_seal(files=[Path(__file__),Path(__file__).resolve().parents[2]/'tests/test_analyze_pusht_geometry_20261027.py'],
        args_json=json.dumps(inputs,sort_keys=True),output_dir=output/'source_seal')
    result.update(inputs=inputs,source_seal=seal)
    write(output/'report.json',result); figures(result,output)
    with (output/'notes.txt').open('x') as f:f.write('Known-root diagnosis only. Original failed encoding gate retained. No improved winner, new independent root, P1 admission or theoretical rejection.\n')
    with (output/'captions.tex').open('x') as f:
        f.write('% Four reused calibration roots; not submission efficacy evidence.\n\\includegraphics[width=\\linewidth]{Figure_1.pdf}\n\\includegraphics[width=\\linewidth]{Figure_2.pdf}\n')
    require(inputs=={str(p):digest(p) for p in (planning_path,physical_path)},'inputs changed')
    print(json.dumps({k:result[k] for k in ('status','original_physical_status','actual_steps','actual_encoder_samples','all_winners_unchanged_common')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('planning-path','physical-path','output'):p.add_argument('--'+name,type=Path,required=True)
    run(**vars(p.parse_args()))
