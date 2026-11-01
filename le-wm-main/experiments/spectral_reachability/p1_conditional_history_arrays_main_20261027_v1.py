"""Independent NumPy reconstruction of actual H2 action histories.

No neural/physics runtime imports. Constants are AST literals from the exact
source file bound by the compiler protocol, not checkpoint-training statistics.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_goal_patch_consistent_compiler_20261027_v1 as compiler

NAME='p1_conditional_history_arrays_main_20261027_v1'
META=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10)


def literal_stats(path, expected_sha):
    raw=Path(path).read_bytes()
    compiler.require(compiler.io.sha(raw)==expected_sha,'sealed normalizer source bytes')
    found={}
    for node in ast.parse(raw).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            if key in ('PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'):
                compiler.require(key not in found and isinstance(node.value,ast.Call) and
                    isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='asarray' and
                    isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='np' and
                    len(node.value.args)==1,'explicit unique NumPy literal stats')
                found[key]=np.asarray(ast.literal_eval(node.value.args[0]),dtype=np.float32)
    compiler.require(set(found)=={'PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'},'both source constants')
    mean,std=(found[k] for k in ('PUBLIC_ACTION_MEAN','PUBLIC_ACTION_STD'))
    compiler.require(mean.shape==std.shape==(2,) and np.isfinite(mean).all() and
        np.isfinite(std).all() and np.all(std>0),'finite positive per-command stats')
    return mean,std


def check_context(context,commands,mean,std):
    compiler.require(all(context.get(k)==v for k,v in META.items()),'context layout/dimension metadata')
    raw=np.asarray(commands,dtype=np.float32)
    compiler.require(raw.shape==(10,2) and np.isfinite(raw).all() and np.all(np.abs(raw)<=1),
        'exact two bounded five-command packets')
    # Separate float32 subtraction and division, exactly the sealed packet rule.
    expected=((raw-np.asarray(mean,dtype=np.float32))/np.asarray(std,dtype=np.float32)).reshape(2,10)
    actual=np.asarray(context['action_history'],dtype=np.float64)
    latent=np.asarray(context['latent_history'],dtype=np.float64)
    compiler.require(actual.shape==(2,10) and latent.shape==(3,192) and np.isfinite(latent).all() and
        np.array_equal(actual,expected.astype(np.float64)), 'normalized action history chronology/content')
    return compiler.digest(expected.astype(float).tolist())


def audit(cuda_output,expected_sha,output):
    cuda_output=Path(cuda_output).absolute();output=Path(output).absolute()
    compiler.require(not output.exists(),'exclusive new audit directory')
    raw=(cuda_output/'report.json').read_bytes()
    compiler.require(compiler.io.sha(raw)==expected_sha,'explicit actual report SHA')
    plans=compiler.selected_actions(cuda_output)
    binding=compiler.io.decode((cuda_output/'execution_protocol.json').read_bytes())
    body,identity=compiler.validate_prepared(binding['prepared'],binding['prepare_sha256'])
    path=Path(body['project_root'])/'experiments/run_pusht_counterfactual.py'
    mean,std=literal_stats(path,body['sources_sha256'][str(path)])
    report=compiler.io.decode(raw);rows=[];checks=0
    for row in report['rows']:
        compilation=row['compilation'];histories=[]
        for candidate in compilation['candidates']:
            h=check_context(candidate['context'],candidate['commands'],mean,std);checks+=1
            histories.append(dict(index=candidate['index'],history_sha256=h))
        winner=compilation['winner']
        for key in ('context',):check_context(winner[key],winner['commands'],mean,std);checks+=1
        for key in ('whole_replay_context','segmented_replay_context'):
            check_context(compilation[key],winner['commands'],mean,std);checks+=1
        rows.append(dict(root_sha256=row['root_sha256'],arm=row['arm'],candidates=histories,all_history_checks=True))
    for plan in plans:check_context(plan['selected_predicted_context'],plan['actions'],mean,std);checks+=1
    compiler.require(len(rows)==8 and checks==112,'all 80 candidates, 24 repeated winner contexts, eight exports')
    result=dict(schema=NAME+'.report',status='actual_saved_action_histories_consistent',
        source_report_sha256=expected_sha,source_protocol_id=identity,
        source_normalizer=str(path),source_normalizer_sha256=body['sources_sha256'][str(path)],
        public_mean=mean.astype(float).tolist(),public_std=std.astype(float).tolist(),
        normalizer_training_provenance_verified=False,history_checks=checks,all_checks=True,rows=rows,
        auditor_source_sha256=compiler.io.sha(Path(__file__).read_bytes()),saved_array_svds=96,
        new_neural_calls=0,new_physical_steps=0,P1_admission=False,
        scope='actual successful v1 report content; not a repair of v1 failure-handling windows')
    output.mkdir(exist_ok=False);compiler.write(output/'report.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-output',type=Path,required=True)
    parser.add_argument('--report-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();r=audit(a.cuda_output,a.report_sha256,a.output)
    print(compiler.io.canonical({k:v for k,v in r.items() if k!='rows'}))


if __name__=='__main__':main()
