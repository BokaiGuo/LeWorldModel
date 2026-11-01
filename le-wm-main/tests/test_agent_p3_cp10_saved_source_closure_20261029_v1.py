from copy import deepcopy
import gzip
import hashlib
import io

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import agent_p3_cp10_saved_source_closure_20261029_v1 as m


def sha(b): return hashlib.sha256(b).hexdigest()


def stored(path, arr):
    buf=io.BytesIO(); np.save(buf,arr,allow_pickle=False); body=gzip.compress(buf.getvalue(),mtime=0)
    path.write_bytes(body)
    return dict(file=path.name,sha256=sha(body),array_sha256=sha(arr.tobytes()),shape=list(arr.shape),dtype=str(arr.dtype))


@pytest.fixture
def row(tmp_path):
    frames=[];hist=[]
    for i,step in enumerate((0,5,10)):
        pix=np.full((224,224,3),i,dtype=np.uint8); b=io.BytesIO();Image.fromarray(pix).save(b,format='PNG')
        path=tmp_path/f'{step}.png';path.write_bytes(b.getvalue());state=[float(i)]*7;snapshot={'state':state}
        f=dict(phase='plan',overlay_index=0,step=step,condition='off',png=path.name,
               file_sha256=sha(b.getvalue()),pixel_sha256=sha(pix.tobytes()),state=state,snapshot=snapshot)
        frames.append(f);hist.append(dict(absolute_step=step+10,png=str(path),**{k:f[k] for k in ('file_sha256','pixel_sha256','state','snapshot')}))
    source=dict(path=hist[2]['png'],**{k:hist[2][k] for k in ('file_sha256','pixel_sha256','state','snapshot')})
    raw=np.arange(20,dtype=np.float32).reshape(10,2)
    plan=dict(task_id='id',selection_digest='selection',model_task={'fixture':True},commands=raw.tolist())
    physical=dict(index=0,root_slot=0,arm='H1',status='horizon_completed',own_returned=10,initial_exact=True,
        prefix_reference_exact=True,prefix51={'complete':True,'frames':[{}]*51},missing_checkpoints=[],plan=plan,trajectory=frames)
    spec=dict(source_slot=0,root_slot=0,role='C',base_task_id='id',selection_digest='selection',cp5_task=plan['model_task'],
              causal_raw10=raw.tolist(),causal_raw10_bits=raw.tobytes().hex(),history_sources=hist,source=source)
    latent=np.arange(192,dtype=np.float32);ld=stored(tmp_path/'latent.npy.gz',latent)
    pd=stored(tmp_path/'pixels.npy.gz',np.full((224,224,3),2,dtype=np.uint8))
    encoded=dict(source=source,status='encoded',latent=ld,pixels=pd)
    task=dict(task={'root_context':{'latent_history':[[],[],latent.tolist()]}},identity={'encoded_latent':ld})
    return spec,encoded,task,physical,tmp_path


def test_complete_fixture(row):
    assert m.verify_row(*row).shape == (192,)


@pytest.mark.parametrize('damage',['slot','guard','raw','frame','snapshot','pixel','latent','task','descriptor','missing'])
def test_tampering_rejected(row,damage):
    s,e,t,p,d=deepcopy(row)
    if damage=='slot': p['index']=1
    elif damage=='guard': p['prefix51']['complete']=False
    elif damage=='raw': p['plan']['commands'][0][0]=100
    elif damage=='frame': p['trajectory'][1]['pixel_sha256']='0'*64
    elif damage=='snapshot': s['history_sources'][1]['snapshot']['state']=[3]*7
    elif damage=='pixel': e['pixels']['array_sha256']='0'*64
    elif damage=='latent': (d/e['latent']['file']).write_bytes(b'bad')
    elif damage=='task': t['task']['root_context']['latent_history'][-1][0]=5
    elif damage=='descriptor': e['latent']['file']='../latent.npy.gz'
    elif damage=='missing': p['missing_checkpoints']=[10]
    with pytest.raises((ValueError,KeyError)): m.verify_row(s,e,t,p,d)


def test_no_fake_authority_from_descriptor(tmp_path):
    p=tmp_path/'a';p.write_bytes(b'abc')
    with pytest.raises(ValueError): m.data(p,'0'*64)
