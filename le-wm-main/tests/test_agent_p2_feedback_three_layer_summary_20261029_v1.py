"""Saved ABI snapshots rebound exclusively into pytest temporary directories.

No producer imports, model execution, physics, or upstream strict calls.
"""
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import struct

import pytest

ROOT=Path(__file__).resolve().parents[1]
STEM='agent_p2_feedback_three_layer_summary_20261029_v1'
spec=importlib.util.spec_from_file_location(STEM,ROOT/'experiments/spectral_reachability'/(STEM+'.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def load(path):return json.loads(Path(path).read_bytes())


@pytest.fixture
def bundle(tmp_path):
    """Transport actual saved ABI to temp; never produce a production summary."""
    results=ROOT/'results'
    names={
        'if_rgb':s.RGB+'_consume_MAIN01.json',
        'O_rgb':s.ORG+'_consume_MAIN01.json',
        'feedback_prepare':s.MODEL+'_prepare_MAIN01.json',
        'feedback_receipt':s.MODEL+'_consume_MAIN01.json',
        'model_summary':s.MS+'_MAIN01.json',
        'native_summary':s.NS+'_MAIN01.json'}
    if not all((results/n).is_file() for n in names.values()):pytest.skip('local saved ABI assets unavailable')
    values={k:load(results/n) for k,n in names.items()}
    def save(name,x=None,original=None):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(Path(original).read_bytes() if original else s.raw(x))
        return dict(path=str(p),sha256=s.sha(p))
    def copy_report(ref,directory):
        x=load(ref['path']);dest=tmp_path/directory;dest.mkdir()
        for row in x['rows']:
            d=row['latent']
            if d is not None:shutil.copyfile(Path(ref['path']).parent/d['file'],dest/d['file'])
        return x,save(directory+'/report.json',x)
    f,o,p,m,ms,ns=[values[k] for k in s.KEYS]
    original=load(ns['arguments']['original'])
    original_ref=save('native_O.json',original=ns['arguments']['original'])
    ns['arguments']['original']=original_ref['path'];ns['arguments']['original_sha']=original_ref['sha256']
    jobs=[]
    for i in range(4):
        jobs.append(load(ns['arguments']['jobs'][i]))
        ref=save(f'job{i}.json',original=ns['arguments']['jobs'][i])
        ns['arguments']['jobs'][i]=ref['path'];ns['arguments']['job_shas'][i]=ref['sha256']
        f['source_authorities'][f'job{i}']=ref
    oreport,oref=copy_report(f['source_authorities']['O_report'],'O')
    o['report_sha256']=oref['sha256'];o['physical_receipt_sha256']=original_ref['sha256']
    oref_receipt=save('O_receipt.json',o)
    f['source_authorities']['O_report']=oref;f['source_authorities']['O_receipt']=oref_receipt
    oldfr=next(path for path,h in f['bindings'].items() if h==f['report_sha256'])
    fr,unused=copy_report(dict(path=oldfr),'IF')
    fp=load(f['completion']['prepared']);fp['authorities']=deepcopy(f['source_authorities'])
    fpref=save('IF_prepare.json',fp)
    fr['prepared']=fpref['path'];fr['prepare_sha256']=fpref['sha256']
    frref=save('IF/report.json',fr)
    f['prepare_sha256']=fpref['sha256'];f['report_sha256']=frref['sha256']
    f['completion']['prepared']=fpref['path'];f['completion']['report_sha256']=frref['sha256']
    f['bindings']={frref['path']:frref['sha256']}
    pref=save('feedback_prepare.json',original=results/names['feedback_prepare'])
    mref=save('feedback_receipt.json',original=results/names['feedback_receipt'])
    ms['sources']['prepare']=pref;ms['sources']['receipt']=mref
    refs=dict(if_rgb=save('IF_receipt.json',f),O_rgb=oref_receipt,feedback_prepare=pref,
        feedback_receipt=mref,model_summary=save('model_summary.json',ms),native_summary=save('native_summary.json',ns))
    manifest=dict(schema=STEM+'.inputs',trust='MAIN_complete_external_receipt_SHAs_not_rebuilt',inputs=refs)
    mr=save('inputs.json',manifest)
    return dict(v=values,original=original,jobs=jobs,manifest=mr,save=save,tmp=tmp_path)


def test_complete_temporary_disk_chain(bundle):
    b=bundle;ref=b['manifest'];x=s.analyze(ref['path'],ref['sha256'])
    assert len(x['summary']['rows'])==len(x['summary']['pairs'])==84
    assert len(x['checked_saved_bytes'])==67  # 15 JSON + 52 native192 gzip arrays
    out=s.write(str(b['tmp']/(STEM+'_fixture.json')),x)
    assert s.verify(out['path'],out['sha256'])['verified']
    assert x['summary']['new_fullstrict_calls']==0
    assert all(x['summary'][k] is False for k in s.FLAGS)
    assert x['summary']['resources']['FLOPs'] is None


def test_ram_complete_and_cross_context_null(bundle):
    x=s.summarize(bundle['v'],bundle['original'],bundle['jobs'])
    assert len(x['rows'])==84
    for r in x['rows']:
        if r['policy']!='O' and r['control_step']<=15:assert r['model_Q'] is None
        if r['control_step']>15 and r['policy']!='O':assert r['actual_encoding_origin']=='new_IF_actual_image'
    for p in x['pairs']:
        if not p['same_model_input']:assert p['model_Q_delta'] is None


def test_native_axes_and_original_angle_semantics():
    x=s.native([1,2,3,4,8,5,6],[0]*7)
    assert x['joint_xy_squared']==30 and x['velocity_squared']==61
    assert x['native7_squared']==155
    assert x['native_angle_term']<0 and x['wrapped_angle_error']>0


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),'0',None])
def test_vector_rejects_nonliteral(value):
    with pytest.raises(ValueError):s.vector([value]*192,192,True)


def test_fsum_no_epsilon_and_ties():
    a=[0.0]*192;b=a.copy();b[0]=struct.unpack('<f',b'\x01\0\0\0')[0]
    assert s.mse(a,b)>0
    assert s.sign(s.mse(a,a))=='tie'
    assert s.sign(-1e-300)=='negative' and s.sign(None)=='null'
    assert s.mse([1.0]*192,[0.0]*192)==1


@pytest.mark.parametrize('mutation',['schema','raw','task','cp','goal','native','witness','modelQ','gate'])
def test_join_mutations_failclosed(bundle,mutation):
    v=deepcopy(bundle['v']);old=deepcopy(bundle['original']);jobs=deepcopy(bundle['jobs'])
    if mutation=='schema':v['if_rgb']['schema']='fake'
    if mutation=='raw':v['if_rgb']['rows'][0]['raw30_float32_bits']='00'*240
    if mutation=='task':v['O_rgb']['rows'][0]['task_id']='wrong'
    if mutation=='cp':v['if_rgb']['rows'][0]['checkpoint']=25
    if mutation=='goal':v['if_rgb']['rows'][0]['goal192'][0]=True
    if mutation=='native':v['native_summary']['summary']['panel'][0]['state7'][0]+=1
    if mutation=='witness':v['feedback_receipt']['plans'][0]['new_witness_sha256']='0'*64
    if mutation=='modelQ':v['model_summary']['jobs'][0]['candidates'][v['model_summary']['jobs'][0]['original_selected']]['Q_fsum']+=1
    if mutation=='gate':v['if_rgb']['P2_admission']=True
    with pytest.raises(ValueError):s.summarize(v,old,jobs)


def test_saved_sha_mutation_rejected(bundle):
    ref=bundle['manifest'];p=Path(ref['path']);p.write_bytes(p.read_bytes()+b' ')
    with pytest.raises(ValueError,match='SHA'):s.analyze(ref['path'],ref['sha256'])


def test_missing_authority_not_status_only(bundle):
    mr=bundle['manifest'];m=load(mr['path']);del m['inputs']['native_summary']
    ref=bundle['save']('missing.json',m)
    with pytest.raises(ValueError,match='complete explicit'):s.analyze(ref['path'],ref['sha256'])


def test_encoding_census_no_free_dedup(bundle):
    v=deepcopy(bundle['v']);v['if_rgb']['encoded']=True
    with pytest.raises(ValueError,match='census'):s.summarize(v,bundle['original'],bundle['jobs'])


def test_snapshot_bit_mutation_rejected(bundle):
    jobs=deepcopy(bundle['jobs']);v=deepcopy(bundle['v'])
    jobs[0]['rows'][0]['checkpoints'][4]['snapshot']['state_float64_bits']='0'*112
    # The RGB source must follow this coherent mutation; the native literal bits still reject it.
    v['if_rgb']['rows'][0]['source']['snapshot']['state_float64_bits']='0'*112
    with pytest.raises(ValueError,match='snapshot float64'):s.summarize(v,bundle['original'],jobs)


def test_whole_summary_mutation_rejected(bundle):
    ref=bundle['manifest'];x=s.analyze(ref['path'],ref['sha256'])
    x['summary']['rows'][0]['actual_encoded_Q']+=1
    out=s.write(str(bundle['tmp']/(STEM+'_bad.json')),x)
    with pytest.raises(ValueError,match='whole own'):s.verify(out['path'],out['sha256'])


def test_source_seal_mutation_rejected(bundle):
    ref=bundle['manifest'];x=s.analyze(ref['path'],ref['sha256'])
    x['own_sha256'][str(s.own_files()[0])]='0'*64
    out=s.write(str(bundle['tmp']/(STEM+'_seal.json')),x)
    with pytest.raises(ValueError,match='whole own'):s.verify(out['path'],out['sha256'])


def test_missing_checkpoint_stays_null(bundle):
    v=deepcopy(bundle['v']);jobs=deepcopy(bundle['jobs']);row=jobs[0]['rows'][0]
    row['own_returned']=29;row['checkpoints']=[c for c in row['checkpoints'] if c['step']<30]
    v['if_rgb']['rows'][2]['actual192']=None
    v['if_rgb']['encoded']=23;v['if_rgb']['missing']=1;v['if_rgb']['counts']=s.counts(23)
    nr=v['native_summary']['summary']['panel'][13];nr['state7']=None;nr['metrics']=None
    out=s.summarize(v,bundle['original'],jobs)
    r=out['rows'][13]
    assert r['control_step']==30 and r['actual_encoded_Q'] is None and r['native'] is None
    assert r['missing_reason']=='native_checkpoint_not_reached'
    assert len(out['rows'])==84


@pytest.mark.parametrize('kind',['tail','shape','hash','truncate'])
def test_native192_codec_rejects(tmp_path,kind):
    header=b"{'descr': '<f4', 'fortran_order': False, 'shape': (192,), }\n"
    data=b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header+struct.pack('<192f',*([0.0]*192))
    if kind=='shape':data=data.replace(b'(192,)',b'(191,)')
    body=gzip.compress(data)
    if kind=='tail':body+=b'x'
    if kind=='truncate':body=body[:-2]
    p=tmp_path/'z.npy.gz';p.write_bytes(body)
    ref=dict(file=p.name,sha256=s.sha(p),array_sha256=s.hashlib.sha256(data[-768:]).hexdigest(),shape=[192],dtype='float32')
    if kind=='hash':ref['array_sha256']='0'*64
    with pytest.raises(ValueError):s.latent(ref,tmp_path,{})


def test_duplicate_json_and_relative_path_rejected():
    with pytest.raises(ValueError):s.decode(b'{"a":1,"a":2}')
    with pytest.raises(ValueError):s.absolute('results/x.json')
    with pytest.raises(ValueError):s.raw({'x':float('nan')})
