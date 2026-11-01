"""Fixed-area metric on all four historical words: calibration, never confirmation."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from experiments.spectral_reachability import main_goal_area_weights_20261028_v1 as weights
from experiments.spectral_reachability import agent_p5_taskmetric_patch_bridge_20261028_v1 as patch

NAME = 'main_goal_area_diagnostic_20261028_v1'
ROOT = weights.ROOT
WEIGHTS = ROOT/'results/main_goal_area_weights_20261028_v1_prepare_MAIN01.json'
WEIGHTS_SHA = '6f23e75db8152a1e6b831afca02eff5305d92e7540e87e20a75462c571030f09'
PATCH = ROOT/'results/agent_p5_taskmetric_patch_bridge_20261028_v1_MAIN01.json'
PATCH_SHA = 'f960d9f3b537a62cb98c3bd70c45380bdc4e9c8717e0c70edbb852120073b8ad'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def agreement(a, b):
    return None if a == 0 or b == 0 else (a > 0) == (b > 0)


def paired(rows):
    require([r['word'] for r in rows] == list(patch.WORDS), 'all four fixed words')
    pairs=[]
    for a,b in itertools.combinations(rows,2):
        deltas={k:a[k]-b[k] for k in ('uniform_actual_Q','area_actual_Q','area_predicted_Q','coverage')}
        pairs.append(dict(left=a['word'],right=b['word'],left_minus_right=deltas,
            uniform_to_coverage=agreement(deltas['uniform_actual_Q'],-deltas['coverage']),
            area_to_coverage=agreement(deltas['area_actual_Q'],-deltas['coverage']),
            area_prediction_to_actual=agreement(deltas['area_predicted_Q'],deltas['area_actual_Q'])))
    return pairs


def build():
    # Verify goal-only weight producer completely BEFORE loading candidate data.
    require(sha(WEIGHTS)==WEIGHTS_SHA, 'sealed weights before candidate inputs')
    w=json.loads(WEIGHTS.read_bytes())
    require(weights.canonical(w)==weights.canonical(weights.build()), 'goal-only weight reconstruction')
    require(sha(PATCH)==PATCH_SHA, 'pinned all4 same-time data')
    patch.verify(PATCH,PATCH_SHA)
    source=json.loads(PATCH.read_bytes());authority=source['authority']
    prep=patch.v4.read_json(authority['prepare']['path'],authority['prepare']['sha256'])
    report=patch.v4.read_json(authority['report']['path'],authority['report']['sha256'])
    goal_rgb=patch.v4.owner.array(prep['inputs'][-1]['rgb'],(224,224,3),np.uint8)
    original=weights.goal.read_json(weights.INPUT,weights.INPUT_SHA)
    area_goal=np.load(original['goal_refs']['rgb.npy']['path'],allow_pickle=False)
    require(area_goal.tobytes()==goal_rgb.tobytes(), 'weight RGB is the identical original encoded goal')
    g=patch.v4.owner.array(report['token_artifacts']['goal__native'],(256,384))
    rows=[]
    for item in source['words']:
        word=item['word'];actual=patch.v4.owner.array(report['token_artifacts'][word+'__frame20'],(256,384))
        predicted=patch.v4.owner.array(prep['predictions'][word],(3,256,384))[-1]
        actual_q=weights.weighted_cost(actual,g,w['weights_float64'])
        predicted_q=weights.weighted_cost(predicted,g,w['weights_float64'])
        # Independent scalar reconstruction from the all256 attribution output.
        require(actual_q==float(np.dot(w['weights_float64'],item['spatial']['observed_goal']['patch_mse'])) and
                predicted_q==float(np.dot(w['weights_float64'],item['spatial']['predicted_goal']['patch_mse'])),
                'saved patch versus raw native NPY reduction')
        native=item['native']
        rows.append(dict(word=word,uniform_actual_Q=item['spatial']['observed_goal']['reference_float64'],
            area_actual_Q=actual_q,area_predicted_Q=predicted_q,
            area_prediction_MSE=weights.weighted_cost(predicted,actual,w['weights_float64']),
            coverage=native['coverage'],object_error_axes=native['object_error_x_y_theta_raw'],task_success=None))
    pairs=paired(rows)
    return dict(schema=NAME,status='historical_all4_fixed_rule_calibration_NOT_confirmation',rows=rows,pairs=pairs,
        comparisons={key:{label:sum(p[key] is value for p in pairs)
            for label,value in (('agree',True),('reverse',False),('tie',None))}
            for key in ('uniform_to_coverage','area_to_coverage','area_prediction_to_actual')},
        authority=dict(weights_path=str(WEIGHTS),weights_sha256=WEIGHTS_SHA,all_patch_path=str(PATCH),all_patch_sha256=PATCH_SHA),
        root_denominator=1,word_denominator=4,new_neural_calls=0,new_physics_calls=0,
        no_parameter_search=True,no_winner_selected=True,scientific_admission=False,P1_admission=False,
        original_gate_unchanged=True,independent_confirmation=False,
        interpretation='Historical same-time descriptive test of a separately sealed goal-only rule. Not a planner/controller intervention or evidence of full object geometry.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).absolute()
    require(out.resolve()==out and out.parent==ROOT/'results' and out.name.startswith(NAME+'_'), 'own exclusive output')
    result=build()
    with out.open('xb') as stream:stream.write(weights.canonical(result)+b'\n')
    print(json.dumps(dict(output=str(out),sha256=sha(out),comparisons=result['comparisons'])))


if __name__=='__main__':main()
