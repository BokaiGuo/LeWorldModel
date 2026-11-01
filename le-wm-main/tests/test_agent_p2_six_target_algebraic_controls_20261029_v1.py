"""Pure RAM/synthetic-files tests; no actual results are analyzed."""
import ast
import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from experiments.spectral_reachability import agent_p2_six_target_algebraic_controls_20261029_v1 as m

FIXTURE_FILE=Path(__file__).with_name('test_agent_p2_six_target_error_decomposition_20261029_v1.py')
spec=importlib.util.spec_from_file_location('frozen_algebraic_fixtures',FIXTURE_FILE)
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


def vec(x):return np.full(192,x,np.float32)


@pytest.mark.parametrize('p,z,t,g',[(3.,1.,2.,0.),(-2.,1.,-1.,0.),(.25,.5,-.25,.125),(0.,0.,0.,0.)])
def test_exact_dyadic_controls(p,z,t,g):
    out=m.controls(vec(p),vec(z),vec(t),vec(g))
    for v in out['geometry_controls']:
        aux=np.full(192,t,np.float64);aux[0]+=v['sign']*2**-8
        assert v['auxiliary_target']['dtype']=='float64'
        assert v['auxiliary_target']['values']==aux.tolist()
        for route,x in [('model',p),('actual',z)]:
            assert v[route]['original_task_MSE']==(x-g)**2
            d=v[route]
            expected=math.fsum((d['tracking_MSE'],d['target_goal_geometry'],d['signed_cross_term']))
            assert d['identity_roundoff_residual']==(x-g)**2-expected
            if v['sign']==0:assert d['identity_roundoff_residual']==0
        assert v['unchanged_task_residual']==out['original_residual']['task_goal']
    for v in out['residual_controls']:
        a=v['amplitude'];x=z+a*(p-z)
        assert v['derived_prediction']['values']==[x]*192
        for name,r in [('task_goal',g),('tracking_target',t)]:
            d=v['by_reference'][name]
            assert d['energy']==a*a*(p-z)**2
            assert d['signed_cross_term']==a*2*(p-z)*(z-r)
            assert d['derived_MSE']==(x-r)**2
            assert d['identity_roundoff_residual']==0
            assert d['difference_scaling_roundoff']==0


def test_f64_aux_not_silently_rounded_to_f32():
    target=vec(2**20);out=m.controls(vec(1),vec(2),target,vec(0))
    aux=out['geometry_controls'][2]['auxiliary_target']
    assert aux['values'][0]==2**20+2**-8
    assert float(np.float32(aux['values'][0]))!=aux['values'][0]
    assert target.tobytes()==vec(2**20).tobytes()


def test_f64_residual_prediction_not_f32():
    out=m.controls(vec(2**24),vec(1),vec(0),vec(0))
    derived=out['residual_controls'][2]['derived_prediction']
    assert derived['values'][0]==8388608.5
    assert float(np.float32(derived['values'][0]))!=derived['values'][0]


def test_negative_cross_term_preserved():
    out=m.controls(vec(-2),vec(1),vec(0),vec(0))
    low,zero,high,original=out['residual_controls']
    a,b=low['by_reference']['task_goal'],high['by_reference']['task_goal']
    assert a['energy']==b['energy']
    assert a['signed_cross_term']==-b['signed_cross_term']
    assert b['signed_cross_term']<0
    assert zero['by_reference']['task_goal']['energy']==0
    assert zero['by_reference']['task_goal']['signed_cross_term']==0
    assert original['derived_prediction']['values']==[-2.]*192


def test_missing_never_filled():
    out=m.controls(vec(1),None,vec(2),vec(0))
    assert out['actual_missing']
    for d in out['geometry_controls']:
        assert d['model'] is not None and d['actual'] is None
        assert d['unchanged_task_residual'] is None
    for d in out['residual_controls']:
        assert d['derived_prediction'] is None and d['by_reference'] is None
        assert d['reason']=='actual_encoding_missing'


def test_roundoff_reported_not_thresholded():
    p=np.linspace(-7,3,192,dtype=np.float32);z=np.linspace(11,-5,192,dtype=np.float32)
    out=m.controls(p,z,vec(.3),vec(-.7))
    for v in out['residual_controls']:
        for a in v['by_reference'].values():
            assert a['identity_roundoff_residual']==a['derived_MSE']-a['reconstructed_MSE']
            assert a['energy_scaling_roundoff']==a['energy']-a['expected_scaled_energy']
            assert a['cross_scaling_roundoff']==a['signed_cross_term']-a['expected_scaled_cross_term']
    assert 'passed' not in str(out)


@pytest.mark.parametrize('bad',[np.zeros(192,np.float32),np.zeros(191,np.float64),np.full(192,np.nan),[0.]*192])
def test_bad_f64_rejected(bad):
    with pytest.raises(ValueError):m.vec64(bad)


def test_original_inputs_not_changed_signed_zero():
    arrays=[vec(-0.),vec(1.),vec(2.),vec(0.)];before=[a.tobytes() for a in arrays]
    m.controls(*arrays)
    assert [a.tobytes() for a in arrays]==before
    assert m.record64(np.full(192,-0.,np.float64))['sha256']!=m.record64(np.zeros(192,np.float64))['sha256']


def test_fixed_panel_and_original_scalars():
    p,r,c,z=fixtures.fixture_panel();out=m.panel(p,r,c,z)
    assert out['denominator']==dict(slots=6,positions=24,geometry_controls=72,residual_controls=96)
    assert out['missing_actual']==0 and out['winner'] is None and out['threshold'] is None
    assert out['controller_inputs'] is False and out['counterfactual_only'] is True
    assert [(v['slot'],v['cp_public_step']) for v in out['rows']]==[(s,cp) for s in range(6) for cp in (0,5,10,15)]
    assert [v['original_scalars'] for v in out['rows']]==c['rows']


def test_missing_panel_keeps_design_denominator():
    p,r,c,z=fixtures.fixture_panel();i=1
    for row in (p['rows'][i],r['rows'][i]):
        row.update(frame=None,pixel_source=None,response=None,latent=None,reason='native_missing')
    r['rows'][i]['status']='missing';z[i]=None
    original,_=m.base.original_rows(p,r,z);c['rows']=original
    out=m.panel(p,r,c,z)
    assert out['denominator']['geometry_controls']==72 and out['denominator']['residual_controls']==96
    assert out['missing_actual']==1 and len(out['rows'])==24
    assert out['rows'][i]['residual_controls'][0]['derived_prediction'] is None


def test_source_closure_only_small_frozen_sources():
    c=m.closure()
    assert c[str(Path(m.base.__file__))]==m.BASE_SHA
    assert c[str(m.ROOT/m.PROTOCOL)]==m.PROTOCOL_SHA
    assert str(Path(m.__file__)) in c and str(Path(__file__).resolve()) in c
    assert all('/results/' not in path for path in c)


@pytest.mark.parametrize('what',['scalar','cp','history','prediction','cp0'])
def test_scalar_source_mutations(what):
    p,r,c,z=fixtures.fixture_panel()
    if what=='scalar':c['rows'][1]['model_task_Q']+=1
    elif what=='cp':p['rows'][1]['cp_public_step']=4
    elif what=='history':p['rows'][1]['original_a_sha256']='other'
    elif what=='prediction':p['rows'][1]['prediction_origin']='normalized'
    else:z[0]=vec(9)
    with pytest.raises(ValueError):m.panel(p,r,c,z)


def test_fixture_four_files_build_write_verify(tmp_path,monkeypatch):
    _,_,_,_,refs=fixtures.fixture_documents(tmp_path/'input')
    monkeypatch.setattr(m.base,'actual_refs',lambda:refs)
    monkeypatch.setattr(m,'closure',lambda:{str(Path(m.__file__)):m.sha(Path(m.__file__))})
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    value=m.build(refs)
    assert value['upstream_authentication_imported'] and not value['independent_fullstrict']
    assert value['implemented_section']==2 and not value['action_interventions']
    assert value['new_calls']==dict(encoder=0,predict=0,AD=0,physics=0)
    out=tmp_path/'results'/(m.NAME+'_fixture.json');h=m.write(out,value)['sha256']
    assert m.verify(out,h)['verified']
    with pytest.raises(FileExistsError):m.write(out,value)
    data=m.read(out,h);data['rows'][1]['residual_controls'][1]['amplitude']=2.
    tampered=tmp_path/'results'/(m.NAME+'_tamper.json');th=m.write(tampered,data)['sha256']
    with pytest.raises(ValueError,match='reconstructed'):m.verify(tampered,th)
    (tmp_path/'input'/'r0_latent.npy.gz').write_bytes(b'changed')
    with pytest.raises(ValueError):m.build(refs)


@pytest.mark.parametrize('what',['authority','wall','partial','model','seal','raw'])
def test_build_failclosed(tmp_path,monkeypatch,what):
    p,r,d,c,refs=fixtures.fixture_documents(tmp_path/'input')
    monkeypatch.setattr(m.base,'actual_refs',lambda:refs)
    monkeypatch.setattr(m,'closure',lambda:{})
    if what=='authority':
        other=copy.deepcopy(refs);other['report']['sha256']='0'*64
        with pytest.raises(ValueError):m.build(other)
        return
    if what=='wall':
        tick=iter((0.,121.))
        with pytest.raises(ValueError):m.build(refs,clock=lambda:next(tick))
        return
    if what=='seal':
        (tmp_path/'input'/'native_seal.json').write_bytes(b'{}')
    else:
        # Re-hash fixture metadata, then require semantic joins to reject it.
        if what=='partial':r['errors']=['failed']
        elif what=='model':r['model_after']='changed'
        else:p['rows'][1]['raw_prefix_bits']='wrong'
        for key,doc in [('prepare',p),('report',r)]:
            Path(refs[key]['path']).write_bytes(m.encode(doc));refs[key]['sha256']=m.sha(refs[key]['path'])
    with pytest.raises(ValueError):m.build(refs)


def test_only_frozen_local_helpers_no_fullstrict_or_inference():
    tree=ast.parse(Path(m.__file__).read_text())
    calls={ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not {'base.build','base.verify','base.analyze','base.pairs'} & calls
    assert not any('consume' in c or 'encode_one' in c for c in calls)
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert imports==['__future__','pathlib','experiments.spectral_reachability']
    assert not any(isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names) for n in ast.walk(tree))


def test_cli_fixture_build_then_verify(tmp_path,monkeypatch,capsys):
    (tmp_path/'results').mkdir();monkeypatch.setattr(m,'ROOT',tmp_path)
    # Exercise only new CLI plumbing on a RAM value, never substitute a production callback.
    monkeypatch.setattr(m,'build',lambda refs=None:{'authorities':{},'fixture':True})
    out=tmp_path/'results'/(m.NAME+'_cli.json')
    m.main(['build','--output',str(out)])
    m.main(['verify','--summary',str(out),'--summary-sha256',m.sha(out)])
    assert '"verified":true' in capsys.readouterr().out
