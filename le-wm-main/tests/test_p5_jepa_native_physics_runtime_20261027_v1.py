"""No installations, physics construction or neural calls in these tests."""
import ast
import json
from pathlib import Path
import sys
import pytest
from experiments.spectral_reachability import p5_jepa_native_physics_runtime_20261027_v1 as r


def test_six_exact_requirements_no_torch():
    r.verify_lock()
    assert r.LOCK=={'wheel':'0.45.1','gym':'0.23.1','pymunk':'6.8.0','scikit-image':'0.25.2','lazy-loader':'0.4','tifffile':'2025.6.11'}
    assert all('torch' not in k for k in r.LOCK)


def test_target_is_not_venv_or_old_source():
    target,stage=r.paths(r.ROOT)
    assert target==r.ROOT/'.jepa-physics-site'
    assert stage==r.ROOT/f'results/{r.NAME}/bootstrap_artifacts'


def test_proposal_fresh_seed_and_all_roots():
    p=r.fresh_proposal()
    assert p['seeds']==[2026102701,2026102702,2026102703,2026102704]
    assert p['not_archived_episode1'] and not p['P5_admission']
    assert p['env_constructor']['reset_to_state'] is None
    assert p['env_constructor']['legacy'] is False
    assert p['env_constructor']['relative'] is True
    assert p['env_constructor']['with_velocity'] is True
    assert p['env_constructor']['action_scale']==100


def test_reset_and_zero_semantics_not_pause_or_none():
    p=r.fresh_proposal()['factual_stage']
    assert p['space_step_calls']==1+15*10==151
    assert p['reset_substeps']==1 and p['simulation_seconds']==1.51
    assert 'NOT None' in p['action'] and 'NOT normalized zero' in p['action']
    assert 'NOT physics pause' in p['zero_limit']
    assert p['frames_to_save']==list(range(16))


def test_no_false_full_contact_snapshot_restore_claim():
    p=r.fresh_proposal()['capture']
    assert {'velocity','angular_velocity','force','torque'}<=set(p['bodies'])
    assert 'NOT a complete contact-cache serialization' in p['hidden_solver']
    assert 'retain live native state' in p['hidden_solver']
    assert 'never pickle' in p['before_reset']


def test_zero_trace_not_action_fidelity():
    p=r.fresh_proposal()['next_action_fidelity']
    assert 'zero-only traces cannot' in p['no_efficacy_from_zero']
    assert p['contexts']==[0,5,10] and p['target']==15
    assert 'BEFORE' in p['requires']


def test_source_no_install_or_physics_or_neural_execution():
    tree=ast.parse(Path(r.__file__).read_text())
    for n in ast.walk(tree):
        if isinstance(n,ast.Call):
            name=ast.unparse(n.func)
            assert name not in ['subprocess.run','subprocess.Popen','os.system','exec','eval']
            if isinstance(n.func,ast.Attribute):
                assert n.func.attr not in ['step','reset','Space','Body','forward','load','add_safe_globals','safe_globals']
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            name=n.module if isinstance(n,ast.ImportFrom) else ' '.join(x.name for x in n.names)
            assert not any(s in name for s in ['torch','pickle','transformers'])


def test_installed_records_safe(tmp_path):
    a=tmp_path/'a';a.write_bytes(b'a')
    assert r.installed_records(tmp_path)['a']['sha256']==r.sha(a)
    (tmp_path/'bad').symlink_to(a)
    with pytest.raises(ValueError):r.installed_records(tmp_path)


def test_budget_rejects_without_deleting(tmp_path,monkeypatch):
    f=tmp_path/'x';f.write_bytes(b'0123456789')
    monkeypatch.setattr(r,'MAX_NEW_BYTES',9)
    with pytest.raises(RuntimeError):r.budget([tmp_path])
    assert f.exists()


def test_dependency_metadata_missing_closed():
    result=r.dependency_constraints(['p5-nonexistent-package-20261027'])
    assert result['problems']==['missing:p5-nonexistent-package-20261027']


def test_actual_inherited_metadata_satisfied():
    closure=r.dependency_constraints(r.INHERITED)
    assert not closure['problems']
    assert closure['records']['numpy']['version']=='2.4.6'


def test_prepare_and_print_only_bootstrap(tmp_path):
    m=r.prepare(r.ROOT)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(m))
    plan=r.bootstrap(r.ROOT,m,manifest)
    assert plan['status']=='installer_plan_only_not_executed'
    assert not plan['installation_executed']
    for k in ['stage1_argv','stage2_argv']:
        assert '--target' in plan[k] and '--no-deps' in plan[k] and '--no-cache' in plan[k]
        assert '--python' in plan[k]
    assert '--no-build-isolation' in plan['stage2_argv']
    assert 'gym==0.23.1' in plan['stage2_argv']
    assert plan['environment']['PYTHONPATH']==str(r.ROOT/'.jepa-physics-site')
    assert 'inherited_dependency_closure' in m


def test_tampered_protocol_fails_before_import(tmp_path):
    m=r.prepare(r.ROOT);m['fresh_calibration_proposal']['seeds']=[0]
    with pytest.raises(ValueError,match='protocol'):r.verify(m,r.ROOT)


def test_current_base_probe_fails_without_native_import(tmp_path):
    m=r.prepare(r.ROOT);p=tmp_path/'manifest.json';p.write_text(json.dumps(m))
    # This test is for the unmodified main/base process, not an installed overlay.
    assert str(r.paths(r.ROOT)[0]) not in sys.path
    before=set(sys.modules)
    result=r.probe(r.ROOT,m,p)
    assert result['status']=='runtime_probe_failed'
    assert 'target_not_before_base_site' in result['problems']
    assert result['physics_step_calls']==result['env_construct_calls']==0
    assert not any(n.endswith('_official_source_import') for n in set(sys.modules)-before)
