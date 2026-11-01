"""Authenticate exact historical allmode cp10 encodings; no new encoder calls."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import main_terminal_witness_rgb_contrast_20261028_v1 as math
from experiments.spectral_reachability import main_nonlinear_suffix_physical_contrast_20261028_v1 as physical
from experiments.spectral_reachability import p1_moving_overlay_recompile_rgb_cuda_20261027_v1 as legacy
from experiments.spectral_reachability import agent_nonlinear_suffix_terminal_rgb_20261028_v1 as current

ROOT=physical.ROOT
NAME='main_nonlinear_suffix_rgb_reuse_20261028_v1'
OLD=ROOT/'results/p1_moving_overlay_recompile_rgb_cuda_20261027_v1_MAIN01'
OLD_SHA='18d2f192a75e397c01e6c7eb7f6422e9bb073af865107ef469debe5b8ae32d1e'
NEW=ROOT/'results/agent_nonlinear_suffix_terminal_rgb_20261028_v1_cuda_MAIN01'
NEW_SHA='78bb8cfcb051623e64017da0061dea4ad00ea90daaf89bfbf774b20a50727ba5'
RECEIPT_SHA='18d281427d34c1afce2054308208e7a9561df24a5d117cc851a5d143f17939e4'


def exact(a,b):
    math.require(current.producer.mixed.raw10(a).tobytes()==current.producer.mixed.raw10(b).tobytes(),'exact full raw10')


def build():
    old=math.read(OLD/'report.json',OLD_SHA)
    accepted=legacy.consume(OLD,OLD_SHA)  # Rebuild actual historical source/model/encoding authority.
    new=math.read(NEW/'report.json',NEW_SHA)
    new_strict=current.consume(NEW,NEW_SHA)  # Rebuild new all168/native/RGB closure.
    persisted=math.read(ROOT/'results/agent_nonlinear_suffix_terminal_rgb_20261028_v1_consume_MAIN01.json',RECEIPT_SHA)
    math.require(current.raw(new_strict)==current.raw(persisted),'entire actual new persisted strict receipt')
    binding=old['input_binding']
    prior=math.read(binding['physical_report'],binding['physical_sha'])
    prep=math.read(prior['execution_binding']['prepared'],prior['execution_binding']['prepare_sha256'])
    allmode=math.read(ROOT/'results/agent_allmode_physical_20261028_v1_MAIN01/report.json',physical.OLD_SHA)
    model=math.read(ROOT/'results/agent_allmode_prefix_20261028_v2_cuda_MAIN01/report.json',
        'c3e38899a12cd163d75918da79ef029adb14b2479f74f15bdfac0b6ffc57242d')
    rows=[]
    for i,image_index in enumerate((9,15,21,27)):
        j=2*i+1;im=old['images'][image_index];a=allmode['rows'][i];p=prior['rows'][j];plan=prep['plans'][j]
        b=new['rows'][i];accepted_row=accepted[j]
        math.require(all(a[k]==p[k] for k in ('root_slot','condition','task_id','root_state','goal','goal_pose','final_state'))
            and (im['row'],im['root_slot'],im['condition'],im['step'],im['kind'])==
            (j,a['root_slot'],a['condition'],10,'checkpoint') and p['status']==a['status']=='horizon_completed'
            and a['own_returned']==p['own_returned']==10 and plan['task_id']==a['task_id']
            and plan['blind_sha256']==a['plan']['blind_sha256']
            and plan['source_report_sha256']==a['plan']['source_report_sha256']
            and plan['selection_digest']==p['selection_digest']==old['rows'][j]['selection_digest'],
            'whole same original task/native state/time/plan identity, not same root only')
        exact(plan['commands'],a['plan']['commands'])
        cp=[c for c in a['checkpoints'] if c['step']==10]
        math.require(len(cp)==1 and all(im['physical'][k]==cp[0][k] for k in
            ('step','file_sha256','pixel_sha256','state','snapshot')),'exact source cp10 PNG/state/full snapshot')
        z0=math.latent(OLD,im['latent']);z1=math.latent(NEW,b['latent'])
        pixels=legacy.load_array(im['rgb'],lambda n:(OLD/n).read_bytes())
        math.require(np.array_equal(pixels,current.bridge.load_pixels(dict(im['physical'],path=im['path']))),
            'actual source historical PNG pixels')
        pre=legacy.load_array(im['modules'][0]['preprocessed'],lambda n:(OLD/n).read_bytes())
        math.require(np.array_equal(pre,current.bridge.pixels.preprocess_pixels(pixels)), 'independent original preprocessing')
        g0=math.latent(OLD,old['images'][i]['latent']);g=math.vector(b['goal192'])
        math.require(np.array_equal(g0,g) and all(a[k]==b[k] for k in ('root_slot','condition','task_id'))
            and a['plan']['task_sha256']==b['task_sha256'] and b['native_goal']==a['goal'],
            'same current coordinates/goal/task, not another renderer')
        candidate=model['rows'][i]['result']['winner'];p0=math.vector(candidate['predicted_terminal']);p1=math.vector(b['predicted192'])
        exact(candidate['commands'],a['plan']['commands'])
        math.require(candidate['index']==a['plan']['selected_index'] and math.mse(p0,g)==a['plan']['model_Q']
            and math.mse(z0,g)==accepted_row['checkpoints'][-1]['Q']
            and math.mse(z1,g)==persisted['rows'][i]['actual_Q'],'independent actual/predicted original NPY cost')
        ma,mb=physical.base.metrics(a['final_state'],a['goal']),physical.base.metrics(b['evaluation']['final_state'],a['goal'])
        rows.append(dict(index=i,root_slot=a['root_slot'],condition=a['condition'],historical_image=image_index,
            old_index=a['plan']['selected_index'],new_index=b['selected_index'],old_actual_Q=math.mse(z0,g),
            new_actual_Q=math.mse(z1,g),physical_delta={k:mb[k]-ma[k] for k in ma if k!='native_success'},
            **math.decompose(p0,p1,z0,z1,g)))
    return dict(schema=NAME,rows=rows,reversals=sum(r['ranking_reversed'] for r in rows),known_roots=2,
        fixed_denominator=4,fresh_roots=0,full_old_RGB_consume_rebuilt_here=True,full_new_RGB_consume_rebuilt_here=True,
        new_neural_calls=0,new_physics_calls=0,P1_admission=False,scientific_admission=False,
        inputs=dict(old_RGB_report_sha256=OLD_SHA,new_RGB_report_sha256=NEW_SHA,new_RGB_receipt_sha256=RECEIPT_SHA),
        caveat='Exact saved authority reuse, not new execution or independent confirmation; signed terms are descriptive, not causal deletion.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output).absolute();math.require(out.parent==ROOT/'results' and out.name.startswith(NAME+'_') and not out.exists(),'new own artifact')
    result=build()
    with out.open('xb') as f:f.write(json.dumps(result,sort_keys=True,allow_nan=False).encode()+b'\n')
    print(json.dumps(dict(output=str(out),sha256=physical.base.sha(out),rows=result['rows'])))


if __name__=='__main__':main()
