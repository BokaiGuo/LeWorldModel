from copy import deepcopy
import hashlib
import pytest
from experiments.spectral_reachability import agent_e3_control5_authority_join_20261029_v1 as j
from test_agent_e3_gd_host_cuda_20261029_v1 import body


def fixture():
    b=body(); h=j.host; flags={k:False for k in h.FLAGS}
    cls=dict(name='',class_module='fixture',class_qualname='Model',path='/fixture.py',sha256='a'*64)
    prep=dict(schema=h.NAME+'.prepare',fixed_cells=16,bodies=[deepcopy(b) for _ in range(16)],
              expected_model_hash=h.numerical.resolver.MODEL_SHA,bindings_sha256={'/fixture.py':'a'*64})
    plan=dict(first5=[[0.0,0.0]]*5,commands=[[0.0,0.0]]*5)
    report=dict(schema=h.NAME+'.report',cell=0,boundary=5,body_sha256=h.digest(b),errors=[],sources_verified=True,
                model_before=prep['expected_model_hash'],model_after=prep['expected_model_hash'],
                precision_before=h.primitives.precision_expected(),precision_after=h.primitives.precision_expected(),
                classes_before=[cls],classes_after=[deepcopy(cls)],axes={},**flags)
    completion=dict(schema=h.NAME+'.completion',cell=0,complete=True,boundary='after_report_fsync_hash',
                    elapsed_seconds=1.0,report_sha256='r')
    receipt=dict(schema=h.NAME+'.consume',cell=0,status='control5_cell_native_GD_verified_NOT_admitted',
                 report_sha256='r',axes={},plan=plan,**flags)
    return prep,report,completion,receipt,dict(result=deepcopy(plan)),0


def test_join_fixture():
    assert j.join_objects(*fixture())['first5'].shape==(5,2)


@pytest.mark.parametrize('field', ['model','precision','class','body','receipt','commands','elapsed','gate'])
def test_reject_mutation(field):
    p,r,c,t,row,cell=fixture()
    if field=='model':r['model_after']='bad'
    if field=='precision':r['precision_after']['MATH']=False
    if field=='class':r['classes_after'][0]['path']='/wrong'
    if field=='body':r['body_sha256']='bad'
    if field=='receipt':t['report_sha256']='bad'
    if field=='commands':
        t['plan']['commands'][0][0]=0.5
        row['result']=deepcopy(t['plan'])
    if field=='elapsed':c['elapsed_seconds']=True
    if field=='gate':t['P1_admission']=True
    with pytest.raises(ValueError):j.join_objects(p,r,c,t,row,cell)


def test_full_tempdisk_pinned_join_and_tamper(tmp_path):
    p,r,c,t,row,cell=fixture()
    def write(path, value):
        data=j.host.raw(value);path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()
    source=tmp_path/'fixture.py';source.write_bytes(b'fixture source\n')
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    p['bindings_sha256']={str(source):source_sha}
    r['classes_before'][0].update(path=str(source),sha256=source_sha)
    r['classes_after']=deepcopy(r['classes_before'])
    prep_path=tmp_path/'prepared.json';prep_sha=write(prep_path,p)
    output=tmp_path/'output';output.mkdir()
    row_sha=write(output/'row.json',row)
    r.update(prepared=str(prep_path),prepare_sha256=prep_sha,artifacts={'row.json':row_sha})
    report_path=output/'report.json';report_sha=write(report_path,r)
    c['report_sha256']=report_sha
    completion_path=output/'completion.json';completion_sha=write(completion_path,c)
    t.update(report_sha256=report_sha,completion_sha256=completion_sha)
    receipt_path=tmp_path/'receipt.json';receipt_sha=write(receipt_path,t)
    args=(str(prep_path),prep_sha,str(report_path),report_sha,str(completion_path),completion_sha,
          str(receipt_path),receipt_sha,cell)
    assert j.load(*args)['cell']==0
    source.write_bytes(b'changed')
    with pytest.raises(ValueError):j.load(*args)
