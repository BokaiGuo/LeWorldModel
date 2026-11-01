"""Array/AST/contract tests only. Never instantiate or step a simulator."""
import ast
from pathlib import Path
import numpy as np
import pytest
from experiments.spectral_reachability import p5_jepa_physical_bridge_20261027_v1 as b


def test_normalized_zero_is_native_mean_not_raw_zero():
    result=b.command_packet(np.zeros((5,2)))
    np.testing.assert_allclose(result['raw_target_displacements_pixels'],np.tile([-.87,.68],(5,1)),atol=1e-7)
    assert np.any(result['native_relative_env_arguments'] != 0)


def test_raw_zero_has_nonzero_normalized_coordinates():
    z=np.tile(-b.MEAN.astype(float)/b.STD.astype(float),(5,1))
    np.testing.assert_allclose(b.command_packet(z)['raw_target_displacements_pixels'],0,atol=1e-12)


def test_no_tanh_clipping_velocity_or_absolute_target_reuse():
    z=np.full((5,2),20.)
    u=b.command_packet(z)['native_relative_env_arguments'][0]
    assert np.all(u>1)
    np.testing.assert_allclose(b.absolute_target([100.,200.],u)-b.absolute_target([90.,180.],u),[10,20])


@pytest.mark.parametrize('bad',[np.zeros(10),np.zeros((4,2)),np.ones((5,2),bool),[['x','y']]*5,np.full((5,2),np.nan),np.full((5,2),np.inf)])
def test_reject_invalid_packets(bad):
    with pytest.raises(ValueError):b.command_packet(bad)


def test_negation_is_about_native_mean_not_raw_origin():
    z=np.arange(10,dtype=float).reshape(5,2)
    a=b.command_packet(z)['raw_target_displacements_pixels']
    c=b.command_packet(-z)['raw_target_displacements_pixels']
    np.testing.assert_allclose(a+c,np.tile(200*b.MEAN.astype(float),(5,1)),atol=1e-12)


def test_reverse_preserves_step_axis_xy_and_history_scope():
    z=np.arange(10,dtype=float).reshape(5,2)
    np.testing.assert_array_equal(b.command_packet(z[::-1])['raw_target_displacements_pixels'],b.command_packet(z)['raw_target_displacements_pixels'][::-1])
    assert b.protocol()['history_action_slices']==[[0,5],[5,10]]
    assert b.protocol()['new_packet_slice']==[10,15]


def test_missing_prerequisites_remain_blocked_even_compatible_runtime():
    ids={x['id'] for x in b.blockers({'pymunk':'6.8.0','gym':'0.23.1'},True,True)}
    assert ids=={'incomplete_physics_snapshot','reset_advances_physics','generator_render_identity_unbound','video_timebase_not_physics_clock'}


def test_runtime_breakages_exact():
    ids={x['id'] for x in b.blockers({'pymunk':'7.2.0','gym':'0.26.2'},False,False)}
    assert {'missing_scikit_image','pymunk_api_version_mismatch','gym_version_mismatch'}<=ids


def test_protocol_has_no_execution_or_planning_admission():
    p=b.protocol()
    assert not p['physics_executed'] and not p['neural_executed'] and not p['P5_admission']
    assert p['execution_command'] is None
    assert p['timing']['reset_extra_substeps']==1
    assert p['timing']['packet_simulation_seconds']==.5


def test_source_no_simulator_import_or_execution_or_unsafe_load():
    tree=ast.parse(Path(b.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            module=n.module if isinstance(n,ast.ImportFrom) else ' '.join(a.name for a in n.names)
            assert not any(x in module for x in ['pusht_env','stable_worldmodel','pickle'])
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):
            assert n.func.attr not in ['step','reset','Space','Body','forward','add_safe_globals','safe_globals']
            if ast.unparse(n.func)=='torch.load':
                kw={k.arg:ast.literal_eval(k.value) for k in n.keywords}
                assert kw=={'weights_only':True,'map_location':'cpu'}


def test_native_stats_ast_only():
    root=Path(b.__file__).resolve().parents[2]
    stats=b.source_stats(root/b.DSET)
    assert len(stats)==4


def test_restricted_array_rejects_sha_before_loading(tmp_path):
    p=tmp_path/'bad.pth';p.write_bytes(b'not a pickle')
    with pytest.raises(ValueError,match='SHA/size'):b.restricted_array(p,'0'*64,5,'torch.float32')


def test_symlink_seal_rejected(tmp_path):
    p=tmp_path/'source';p.write_text('x');link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError):b.sha(link)


def test_actual_prepare_no_physics_construction_and_v8_identity(monkeypatch):
    import pymunk
    def forbidden(*args,**kwargs):
        raise AssertionError('physics construction prohibited')
    monkeypatch.setattr(pymunk.Space,'__init__',forbidden)
    monkeypatch.setattr(pymunk.Body,'__init__',forbidden)
    root=Path(b.__file__).resolve().parents[2]
    r=b.prepare(root)
    assert r['status']=='blocked_exact_native_physical_bridge'
    assert r['source_data_hashes_unchanged']
    assert r['row']['v8_input_identity']
    assert r['row']['abs_minus_rel_minus_agent_max_abs']==0
    assert r['row']['video']['decoded_frames']==111
    assert r['cost']['env_construct']==r['cost']['physics_step']==r['cost']['neural_forward']==0
    assert {'incomplete_physics_snapshot','reset_advances_physics'}<={v['id'] for v in r['blockers']}
