"""RAM real-schema MAIN-reader receipt fixtures; no fullstrict/production."""
from copy import deepcopy
import json
import runpy
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_p3_equal_cap_comparison_20261029_v2 as m

f=runpy.run_path(str(m.ROOT/'tests/test_agent_p3_equal_cap_comparison_20261029_v1.py'))


def bundle():
    p,r,s=f['bundle'](True)
    den=dict(arms=4,requests_per_arm=144,total_requests=576)
    p['fixed_denominator']=r['fixed_denominator']=deepcopy(den)
    p['profile'].update(final_reserve_bytes=256*1024**2,output_bytes=1024**3)
    for k in ('equal_actual_resources','equal_total_compute','equal_wall'):r[k]=s[k]=False
    s.update(schema=m.READER+'.consume',common_budget=deepcopy(p['profile']),equal_cap_contract=True,
        matching_axes=[],total_FLOPs=None,inherited_producer_sha256=m.PRODUCER_SHA,
        compatibility=dict(original_reader_record_cap=64*1024**2,record_cap=256*1024**2,total_cap=1024**3,
            original_producer_unchanged=True,original_full_math_resource_timing_validation=True,new_neural_calls=0,new_physics_calls=0))
    files=[m.ROOT/'experiments/spectral_reachability'/(m.READER+'.py'),m.ROOT/'tests'/('test_'+m.READER+'.py'),m.ROOT/'innerdocs'/(m.READER+'.md')]
    s['consumer_sources']={str(path):m.sha(path) for path in files}
    s.pop('slot_statuses')
    for row,a in zip(r['rows'],s['arms']):
        row['result']['backend_kind']='HOST_CUDA'
        for k in ('realized_progress_depth','realized_widths','model_plan_available'):a.pop(k)
        a.update(plan_available=True,timing=None,backend_kind='HOST_CUDA')
    return p,r,s


def inputs(tmp,b=None):
    return dict(old=f['save_bundle'](tmp,'old',f['bundle'](False)),new=f['save_bundle'](tmp,'new',b or bundle()))


def test_actual_schema_without_legacy_fields(tmp_path):
    refs=inputs(tmp_path);out=m.analyze(refs)
    assert out['reader_authority']['schema']==m.READER+'.consume'
    assert len(out['arms'])==4 and sum(len(a['jobs']) for a in out['arms'])==576
    assert out['arms'][0]['delta_new_minus_old']['attempted']['predict_API']==1
    assert out['arms'][0]['same_released_action_bits'] is True
    assert out['reader_authority']['compatibility']['new_neural_calls']==0
    with pytest.raises(ValueError,match='schemas'):
        p,r,s=(m.v1.load(refs['new'][k]) for k in ('prepare','report','receipt'))
        m.v1.validate_bundle(p,r,s,True,refs['new'])


@pytest.mark.parametrize('key',('schema','cap','new_calls','math','producer','source','sourceset','budget','release','timing','backend','denominator','unknown','flags','resource'))
def test_new_authority_mutations_rejected(tmp_path,key):
    p,r,s=bundle()
    if key=='schema':s['schema']=m.v1.NEW+'.consume'
    elif key=='cap':s['compatibility']['record_cap']+=1
    elif key=='new_calls':s['compatibility']['new_neural_calls']=1
    elif key=='math':s['compatibility']['original_full_math_resource_timing_validation']=False
    elif key=='producer':s['inherited_producer_sha256']='0'*64
    elif key=='source':s['consumer_sources'][next(iter(s['consumer_sources']))]='0'*64
    elif key=='sourceset':s['consumer_sources'].pop(next(iter(s['consumer_sources'])))
    elif key=='budget':s['common_budget']['output_bytes']+=1
    elif key=='release':s['arms'][0]['plan_available']=False
    elif key=='timing':s['arms'][0]['timing']={'invented':True}
    elif key=='backend':s['arms'][0]['backend_kind']='RAM_FIXTURE'
    elif key=='denominator':s['fixed_denominator']['total_requests']=575
    elif key=='unknown':s['arms'][0]['Jacobian_elements']=0
    elif key=='flags':s['equal_actual_resources']=True
    else:s['arms'][0]['resources']['completed']['predict_API']+=1
    with pytest.raises(ValueError):m.analyze(inputs(tmp_path,(p,r,s)))


def test_partial_fixed_denominator_missing_arm(tmp_path):
    p,r,s=bundle();r['rows'][3].update(result=None,released_plan=None,plan_available=False,status='blocked_after_runtime_failure')
    s['arms'].pop();s['status']='saved_equal_cap_partial_verified_not_admitted'
    a=m.analyze(inputs(tmp_path,(p,r,s)))['arms'][3]
    assert a['delta_new_minus_old'] is None and len(a['jobs'])==144
    assert all(j['new'] is None for j in a['jobs'])


def test_real_cli_writer_verify_and_no_overwrite(tmp_path,monkeypatch,capsys):
    # Only RAM temporary reader-source ABI files, not real consumer execution.
    monkeypatch.setattr(m,'ROOT',tmp_path)
    for folder,name in [('experiments/spectral_reachability',m.READER+'.py'),('tests','test_'+m.READER+'.py'),('innerdocs',m.READER+'.md')]:
        d=tmp_path/folder;d.mkdir(parents=True,exist_ok=True);(d/name).write_text('fixture')
    (tmp_path/'results').mkdir();refs=inputs(tmp_path);path=tmp_path/'results'/(m.NAME+'_fixture.json')
    args=['analyze','--output',str(path)]
    for side in ('old','new'):
        for kind in ('prepare','report','receipt'):
            args+=['--'+side+'-'+kind,refs[side][kind]['path'],'--'+side+'-'+kind+'-sha',refs[side][kind]['sha256']]
    m.main(args);assert json.loads(capsys.readouterr().out)['sha256']==m.sha(path)
    m.main(['verify','--summary',str(path),'--summary-sha',m.sha(path)])
    assert json.loads(capsys.readouterr().out)['verified']
    x=json.loads(path.read_bytes());x['arms'][0]['delta_new_minus_old']['completed']['predict_API']=999
    path.write_bytes(m.canonical(x))
    with pytest.raises(ValueError,match='recomputation'):m.main(['verify','--summary',str(path),'--summary-sha',m.sha(path)])
    with pytest.raises(ValueError,match='exclusive'):m.write(path,x)


def test_import_never_loads_producer_or_torch():
    code=f'from experiments.spectral_reachability import {m.NAME} as m; import sys; assert "torch" not in sys.modules; assert not any("parent_native" in s or "saved_consumer" in s for s in sys.modules); m.main(["--help"])'
    p=subprocess.run([sys.executable,'-c',code],cwd=m.ROOT,capture_output=True,text=True)
    assert p.returncode==0
