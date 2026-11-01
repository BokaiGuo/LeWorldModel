"""RAM and temporary-disk only; not a production prepare or experiment."""
import copy
import gzip
import io
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.spectral_reachability import agent_e1_six_slot_common_prepare_20261029_v1 as m


def test_actual_inputs_rebuild_and_seal_all_six():
    x = m.build()
    assert [(r['root_slot'],r['arm']) for r in x['rows']] == [(r,a) for r in (0,1) for a in m.ARMS]
    assert x['target_seal_sha256'] == m.hash_bytes(m.canonical(x['rows']))
    assert all(len(__import__('base64').b64decode(r['target_f32_le_base64'])) == 768 for r in x['rows'])
    assert x['protocol']['controller']['total_predict_API_cap'] == 864
    assert x['protocol']['replay']['extra_model_predict_total_cap'] == 888
    assert x['protocol']['physical']['physics_step_cap'] == 1224
    assert x['protocol']['resources']['output_byte_cap'] is None
    assert x['dispatch_allowed'] is False


def make_npy():
    b=io.BytesIO(); a=np.arange(192,dtype='<f4').reshape(1,192)
    np.save(b,a,allow_pickle=False); data=b.getvalue()
    return data, dict(file_sha256=m.hash_bytes(data),file_bytes=len(data),data_sha256=m.hash_bytes(a.tobytes()),dtype='<f4',shape=[1,192])


def test_npy_exact_roundtrip():
    data,spec=make_npy()
    assert m.hash_bytes(m.npy(data,spec)) == spec['data_sha256']


@pytest.mark.parametrize('fault', ['payload','trailing','data_digest','shape','dtype'])
def test_npy_rejects_tamper_even_with_resealed_file(fault):
    data,spec=make_npy()
    if fault == 'payload': data=data[:-1]+bytes([data[-1]^1])
    if fault == 'trailing': data+=b'junk'
    if fault == 'data_digest': spec['data_sha256']='0'*64
    if fault == 'shape': spec['shape']=[192]
    if fault == 'dtype': spec['dtype']='<f8'
    spec['file_bytes']=len(data);spec['file_sha256']=m.hash_bytes(data)
    with pytest.raises(ValueError): m.npy(data,spec)


def test_no_nonfinite_or_float64_rounding_target():
    with pytest.raises(ValueError): m.vector_bytes([float('nan')]*192)
    with pytest.raises(ValueError): m.vector_bytes([0.1]*192)


def test_json_rejects_duplicates():
    with pytest.raises(ValueError): m.parse(b'{"a":1,"a":2}')


def test_tempdisk_actual_prepare_verify_and_mutation(tmp_path):
    base=[sys.executable,'-B','-m','experiments.spectral_reachability.'+m.NAME]
    p=tmp_path/'manifest.json'
    cmd=subprocess.run(base+['prepare','--output',str(p)],cwd=m.ROOT,capture_output=True,text=True)
    assert cmd.returncode==0,cmd.stderr
    h=json.loads(cmd.stdout)['sha256']
    verified=subprocess.run(base+['verify','--manifest',str(p),'--manifest-sha256',h],cwd=m.ROOT,capture_output=True,text=True)
    assert verified.returncode==0,verified.stderr
    again=subprocess.run(base+['prepare','--output',str(p)],cwd=m.ROOT,capture_output=True)
    assert again.returncode!=0
    v=json.loads(p.read_bytes()); v['rows'][1]['target_data_sha256']='0'*64
    p.write_bytes(m.canonical(v)); h=m.hash_bytes(p.read_bytes())
    changed=subprocess.run(base+['verify','--manifest',str(p),'--manifest-sha256',h],cwd=m.ROOT,capture_output=True)
    assert changed.returncode!=0


def test_persisted_input_corruption_and_symlink_rejected(tmp_path):
    p=tmp_path/'x';p.write_bytes(b'1')
    with pytest.raises(ValueError):m.bytes_at(p,'0'*64)
    s=tmp_path/'link';s.symlink_to(p)
    with pytest.raises(ValueError):m.bytes_at(s,m.hash_bytes(b'1'))


def test_seal_order_and_bit_changes_affect_digest():
    rows=m.build()['rows']
    changed=copy.deepcopy(rows);changed.reverse()
    assert m.hash_bytes(m.canonical(changed))!=m.hash_bytes(m.canonical(rows))


def actual_join_inputs():
    keys=('controller_prepare','controller_report','controller_receipt','hi_manifest','hi_report','hi_receipt','parity_receipt','pairing_receipt')
    docs=[m.parse(m.bytes_at(m.ROOT/m.FILES[k][0],m.FILES[k][1])) for k in keys]
    cp,hm,hr=docs[0],docs[3],docs[4]
    tasks=[]; payloads={}
    for i in (0,1):
        s=cp['slots'][i]
        tasks.append(m.parse(gzip.decompress(m.bytes_at(s['task_export'],s['task_sha256']))))
        r=hr['rows'][i*2]
        for name in (r['high_target'],r['low']['z_subgoal']):
            path=m.ROOT/'results/agent_hilewm_rgb_high_low_20261028_v1_cuda_MAIN01'/name
            payloads[name]=m.bytes_at(path,hr['arrays'][name]['file_sha256'])
    return docs,tasks,payloads


@pytest.mark.parametrize('fault', ['blind','scaler','receipt','target','compiled','parity','low'])
def test_actual_saved_join_mutations_fail(fault):
    docs,tasks,payloads=actual_join_inputs()
    cp,cr,cu,hm,hr,hu,parity,pairing=docs
    if fault=='blind': hm['slots'][0]['blind_sha256']='0'*64
    if fault=='scaler': tasks[0]['task']['action_std'][0]=1.0
    if fault=='receipt': hu['rows'][0]['target_delivered_exact']=False
    if fault=='target': cp['targets'][0]['targets'][m.ARMS[0]][0]+=1.0
    if fault=='compiled': cp['compiled'][0]['target'][0]+=1.0
    if fault=='parity': parity['diagnostics']['normalizer_pairing']=True
    if fault=='low':
        name=hr['rows'][0]['low']['z_subgoal'];payloads[name]=payloads[name][:-1]+b'X'
    with pytest.raises(ValueError):m.join(*docs,tasks,payloads)
