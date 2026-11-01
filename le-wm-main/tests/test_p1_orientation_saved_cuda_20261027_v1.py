"""Independent NumPy audit of actual saved CUDA outputs, no neural execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/p1_orientation_cuda_main_20261027_v1_cuda_main01'
CAT=ROOT/'results/p1_goal_orientation_contrast_20261027_v1_prepared01'


def test_actual_all_frames_and_analysis_recomputed():
    raw=(RUN/'report.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='ad5649a5168123611b3bbec743b4886b7ef98854f19df82a7db5d8bcc5456234'
    r=json.loads(raw)
    frames=json.loads((CAT/'manifest.json').read_text())['frames']
    events=[json.loads(x) for x in (RUN/'image.events.jsonl').read_bytes().splitlines()]
    completes=[x for x in events if x['kind']=='native_image_completed']
    assert len(completes)==len(frames)==104
    assert [x['order'] for x in events if x['kind']=='frame_completed']==list(range(104))
    assert [x['rgb_sha256'] for x in completes]==[f['pixel_sha256'] for f in frames]
    assert r['resources']=={k:dict(attempted=104,completed=104) for k in ('encoder','projector')}
    for k in r['resources']:
        assert sum(x['samples'] for x in events if x['kind']=='native_image_attempted' and x['resource']==k)==104
    z=np.asarray([x['latent'] for x in completes],dtype=np.float64)
    assert z.shape==(104,192) and np.isfinite(z).all()
    a=json.loads((RUN/'analysis.json').read_text())
    for root,row in enumerate(a['rows']):
        v=z[root*26:(root+1)*26]
        assert row['original_to_clean_mse']==float(np.mean((v[0]-v[1])**2))
        for i,c in enumerate(row['contrasts']):
            n,b=v[2+2*i:4+2*i]
            assert c['nearest_to_clean_mse']==float(np.mean((n-v[1])**2))
            assert c['bilinear_to_clean_mse']==float(np.mean((b-v[1])**2))
            assert c['sampler_pair_mse']==float(np.mean((n-b)**2))
    first={};diffs=[]
    for f,v in zip(frames,z):
        if f['pixel_sha256'] in first:
            diffs.append(float(np.max(np.abs(v-first[f['pixel_sha256']]))))
        else:first[f['pixel_sha256']]=v
    assert len(first)==27 and len(diffs)==77
    assert max(diffs)==a['maximum_exact_input_repeat_abs_difference']==0
    assert r['errors']==[] and r['model_hash_before']==r['model_hash_after']
    assert r['P1_admission'] is False and a['P1_admission'] is False
