import gzip
import hashlib
import io
import json
from pathlib import Path
import runpy

import numpy as np
from PIL import Image
import pytest

from experiments.spectral_reachability import agent_p3_future_cp10_receipt_join_20261029_v1 as m

BASE=runpy.run_path(str(Path(__file__).parent/'test_agent_p3_same_time_capability_probe_cuda_20261029_v1.py'))


def sha(b): return hashlib.sha256(b).hexdigest()


def put(path,body):
    path.write_bytes(body); return {'path':str(path),'sha256':sha(body)}


def js(path,obj): return put(path,json.dumps(obj,sort_keys=True,allow_nan=False).encode())


def arr(path,x):
    b=io.BytesIO();np.save(b,x,allow_pickle=False);raw=gzip.compress(b.getvalue(),mtime=0);path.write_bytes(raw)
    return dict(file=path.name,sha256=sha(raw),array_sha256=sha(x.tobytes()),shape=list(x.shape),dtype=str(x.dtype))


@pytest.fixture
def bundle(tmp_path):
    bp,_,_,_,_,tasks=BASE['bridge_fixture'](); specs=bp['rows']; encoded=[]; latents=[]; bindings={}
    for i,s in enumerate(specs):
        frames=[];history=[]
        for j,step in enumerate((0,5,10)):
            pixels=np.full((224,224,3),i*3+j,dtype=np.uint8);b=io.BytesIO();Image.fromarray(pixels).save(b,format='PNG')
            image=put(tmp_path/f'{i}_{step}.png',b.getvalue());state=[float(i+j)]*7
            f=dict(phase='plan',overlay_index=0,step=step,condition='off',png=Path(image['path']).name,
                   file_sha256=image['sha256'],pixel_sha256=sha(pixels.tobytes()),state=state,snapshot={'state':state})
            frames.append(f);history.append(dict(absolute_step=step+10,png=image['path'],**{k:f[k] for k in ('file_sha256','pixel_sha256','state','snapshot')}))
        source=dict(path=history[-1]['png'],**{k:history[-1][k] for k in ('file_sha256','pixel_sha256','state','snapshot')})
        s.update(source=source,history_sources=history,source_slot=i,selection_digest='sel'+str(i))
        plan=dict(task_id=s['base_task_id'],selection_digest=s['selection_digest'],model_task=s['cp5_task'],commands=s['causal_raw10'])
        row=dict(index=i,root_slot=s['root_slot'],arm='H1' if s['role']=='C' else s['role'],status='horizon_completed',
            own_returned=10,initial_exact=True,prefix_reference_exact=True,prefix51=dict(complete=True,frames=[{}]*51),
            missing_checkpoints=[],plan=plan,trajectory=frames)
        report=js(tmp_path/f'physical{i}.json',dict(rows=[row],errors=[]))
        # Actual row indices are source slots, so the fixture preserves all prior slots.
        rows=[{}]*i+[row];report=js(tmp_path/f'physical{i}.json',dict(rows=rows,errors=[]))
        receipt=js(tmp_path/f'physical_receipt{i}.json',dict(physical_report_sha256=report['sha256'],bindings={report['path']:report['sha256']},rows=rows,errors=[]))
        s.update(source_report=report,source_receipt=receipt);bindings.update({report['path']:report['sha256'],receipt['path']:receipt['sha256']})
        z=np.asarray(tasks[i]['task']['root_context']['latent_history'][-1],dtype=np.float32);latents.append(z)
        latent=arr(tmp_path/f'latent{i}.npy.gz',z);pix=arr(tmp_path/f'pixels{i}.npy.gz',np.full((224,224,3),i*3+2,dtype=np.uint8))
        encoded.append(dict(source=source,status='encoded',latent=latent,pixels=pix))
        for d in (latent,pix):bindings[str(tmp_path/d['file'])]=d['sha256']
    projected=m.projection.project(specs,latents,[x['latent'] for x in encoded])
    src=put(tmp_path/'future_source.py',b'fixture only')
    prepare=js(tmp_path/'prepare.json',dict(schema=m.PRODUCER+'.prepare',producer_source=src,expected_model_hash=bp['expected_model_hash'],rows=specs))
    flags=dict(execution_authorized=False,scientific_admission=False,freshness=False,independent_confirmation=False)
    counts=dict(encoder=5,projector=5,predict=0,physics=0,training=0)
    report=js(tmp_path/'report.json',dict(schema=m.PRODUCER+'.report',producer_source=src,prepared=prepare['path'],prepare_sha256=prepare['sha256'],status='all5_sealed_not_admitted',errors=[],blocked=[],model_hash_before=bp['expected_model_hash'],model_hash_after=bp['expected_model_hash'],rows=encoded,fixed_denominator=5,counts=counts,**flags))
    completion=js(tmp_path/'completion.json',dict(schema=m.PRODUCER+'.completion',prepared=prepare['path'],prepare_sha256=prepare['sha256'],report_sha256=report['sha256'],status='published_within_cap',artifact_sha256={d['file']:d['sha256'] for x in encoded for d in (x['pixels'],x['latent'])},**flags))
    bindings.update({x['path']:x['sha256'] for x in (src,prepare,report,completion)})
    c=json.loads(Path(completion['path']).read_bytes())
    receipt=js(tmp_path/'receipt.json',dict(schema=m.PRODUCER+'.consume',producer_source=src,report_sha256=report['sha256'],completion_sha256=completion['sha256'],completion=c,status='all5_sealed_not_admitted',errors=[],blocked=[],tasks=projected['endpoints'],fixed_denominator=5,counts=counts,bindings=bindings,**flags))
    return dict(prepare=prepare,report=report,completion=completion,receipt=receipt,source=src)


def test_complete_tempdisk_join_is_not_authorization(bundle):
    r=m.join(bundle)
    assert len(r['task_projection']['signals'])==7 and r['slots']==5
    assert not r['execution_authorized'] and not r['source_authority_verified'] and not r['fresh_roots_certified']


@pytest.mark.parametrize('which',['prepare','report','completion','receipt','source'])
def test_each_external_byte_sha_checked(bundle,which):
    Path(bundle[which]['path']).write_bytes(b'tampered')
    with pytest.raises(ValueError):m.join(bundle)


def test_physical_and_latent_tamper_fail(bundle):
    r=json.loads(Path(bundle['report']['path']).read_bytes());d=r['rows'][0]['latent']
    (Path(bundle['report']['path']).parent/d['file']).write_bytes(b'tampered')
    with pytest.raises(ValueError):m.join(bundle)


def test_historical_source_hash_cannot_be_reclassified(bundle,monkeypatch):
    monkeypatch.setattr(m,'OLD_SOURCE',bundle['source']['sha256'])
    with pytest.raises(ValueError):m.join(bundle)


def test_report_rehash_does_not_bypass_completion(bundle):
    p=Path(bundle['report']['path']); x=json.loads(p.read_bytes());x['counts']['encoder']=4
    bundle['report']=js(p,x)
    with pytest.raises(ValueError):m.join(bundle)


def test_original_physical_receipt_bytes_checked(bundle):
    p=json.loads(Path(bundle['prepare']['path']).read_bytes())
    Path(p['rows'][0]['source_receipt']['path']).write_bytes(b'bad')
    with pytest.raises(ValueError):m.join(bundle)


def test_missing_source_descriptor_fails(bundle):
    bundle.pop('source')
    with pytest.raises(ValueError):m.join(bundle)
