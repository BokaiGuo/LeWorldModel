"""Descriptive saved-only H25 model/physical pairing; no inference or simulator."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
JOIN=ROOT/'results/codex_p2_program_h25_physical_20261030_v1_JOIN_01.json'
CUDA=ROOT/'results/codex_p2_program_host_runner_20261030_v1_RUN_02'
JOIN_SHA='51fe16a15e2e83b255e1e5a4b7b644b82a263c17db94ca38c29d9ea1ce5d1283'
CUDA_SHA='f2daa8b1702eefb0d3d32384394900cf24a5d3cbd85f9c838e487cc76cb42296'
NAME='codex_p2_program_h25_paired_analysis_20261030_v1'


def checked(path, expected):
    path=Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('missing or nonregular source')
    body=path.read_bytes()
    if hashlib.sha256(body).hexdigest()!=expected:
        raise ValueError('source hash mismatch')
    return json.loads(body)


def analyze(join, report, branches):
    slots=join['policy_slots']; rows=report['rows']
    if len(slots)!=8 or len(rows)!=8 or len(branches)!=8:
        raise ValueError('fixed eight slots')
    output=[]
    for base in range(4):
        pair=[]
        for i,arm in enumerate(('joint','library_chain')):
            j=base*2+i; row=slots[j]['row']; br=branches[j]; meta=rows[j]
            if (slots[j]['base_index'],slots[j]['policy'],br['base'],br['arm'],
                meta['base'],meta['arm'],meta['status'])!=(base,arm,base,arm,base,arm,'complete'):
                raise ValueError('slot identity')
            plan=row['plan']; winner=br['result']['winner']; bits=bytes.fromhex(plan['raw_float32_hex'])
            suffix=bytes.fromhex(winner['candidate']['raw_float32_hex'])
            if (len(bits)!=200 or len(suffix)!=80 or bits[120:]!=suffix or
                hashlib.sha256(suffix).hexdigest()!=plan['new_model_plan']['suffix_sha256'] or
                br['result']['winner_index']!=plan['new_model_plan']['winner_index'] or
                winner['candidate']['program_ids']!=plan['new_model_plan']['program_ids']):
                raise ValueError('winner or physical action mismatch')
            if (row['status']!='horizon_completed' or row['own_returned']!=25 or
                row['own_attempted']!=25 or row['terminated'] or row['truncated'] or
                not row['feedback_guard'] or len(row['responses'])!=25):
                raise ValueError('missing complete actual physical trajectory')
            before=row['responses'][14]; after=row['responses'][24]
            if before['step']!=15 or after['step']!=25 or after['native_metrics']!=row['final_metrics']:
                raise ValueError('time-aligned actual endpoint')
            cost=winner['cost']
            if not isinstance(cost,(int,float)) or not math.isfinite(cost):
                raise ValueError('nonfinite model objective')
            pre=before['native_metrics']; post=after['native_metrics']
            pair.append(dict(arm=arm,model_Q=cost,winner_index=br['result']['winner_index'],
                program_ids=winner['candidate']['program_ids'],suffix_sha256=hashlib.sha256(suffix).hexdigest(),
                prefix_hex=bits[:120].hex(),cp15_state=before['state'],
                cp15_object_position_error=pre['object_position_error'],
                cp25_object_position_error=post['object_position_error'],
                cp15_angle_error=pre['wrapped_object_angle_error'],
                cp25_angle_error=post['wrapped_object_angle_error'],
                cp15_object_pose=pre['object_pose'],cp25_object_pose=post['object_pose'],
                native_success=row['native_success'],object_success=post['object_success'],
                actual_model_resource_counts=meta['resource_counts']))
        a,b=pair
        if a['prefix_hex']!=b['prefix_hex'] or a['cp15_state']!=b['cp15_state']:
            raise ValueError('same literal prefix and cp15 state required')
        def preference(x,y):
            return 'joint' if x<y else 'library_chain' if y<x else 'tie'
        output.append(dict(base=base,arms=pair,
            model_preference=preference(a['model_Q'],b['model_Q']),
            position_preference=preference(a['cp25_object_position_error'],b['cp25_object_position_error']),
            angle_preference=preference(a['cp25_angle_error'],b['cp25_angle_error']),
            position_difference_library_minus_joint=b['cp25_object_position_error']-a['cp25_object_position_error'],
            angle_difference_library_minus_joint=b['cp25_angle_error']-a['cp25_angle_error']))
    return dict(schema=NAME+'.analysis',pairs=output,known_roots=2,independent_samples=0,
        limitations=['descriptive known-root observations; base 2 and 3 share context',
            'model Q and physical errors have different scales and meanings',
            'no statistical inference, equal total compute or P1/P2 admission',
            'cost axes nest; model load and inherited acquisition are separate or unknown'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(); output=Path(args.output)
    if not output.is_absolute() or output.parent!=ROOT/'results' or not output.name.startswith(NAME+'_') or os.path.lexists(output):
        raise ValueError('exclusive analysis output')
    join=checked(JOIN,JOIN_SHA); report=checked(CUDA/'report.json',CUDA_SHA)
    branches=[checked(CUDA/row['file'],row['sha256']) for row in report['rows']]
    result=analyze(join,report,branches)
    result['sources']=dict(join_sha256=JOIN_SHA,cuda_report_sha256=CUDA_SHA,
        branch_sha256=[x['sha256'] for x in report['rows']])
    body=(json.dumps(result,sort_keys=True,allow_nan=False,indent=2)+'\n').encode()
    with output.open('xb') as f:
        f.write(body);f.flush();os.fsync(f.fileno())
    print(json.dumps(dict(path=str(output),sha256=hashlib.sha256(body).hexdigest())))


if __name__=='__main__':
    main()
