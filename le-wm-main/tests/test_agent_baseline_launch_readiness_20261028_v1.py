"""RAM metadata fixtures, not actual model/consumer certification."""
import ast
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.spectral_reachability import agent_baseline_launch_readiness_20261028_v1 as m


class RAM:
    root = Path('/RAM/le-wm-main')
    def __init__(self):
        self.hashes = {}
        self.data = {}
        self.expected = {}
        self.bytes_hashed = 0
        for path, sha in [(m.MATRIX,m.MATRIX_SHA),(m.REPORTED,m.REPORTED_SHA),
                          (m.COMPARISON,m.COMPARISON_SHA),(m.BUDGET,m.BUDGET_SHA),
                          (m.HILEWM_HELPER,m.HILEWM_HELPER_SHA),(m.HILEWM_FAILURE,m.HILEWM_FAILURE_SHA)]:
            self.expected[self.path(path)] = sha
        self.data[self.path(m.HILEWM_FAILURE)] = {'status':'failed_preserved'}
        for w in m.WORKFLOWS:
            self.expected[self.path('experiments/spectral_reachability/'+w.module+'.py')] = w.source_sha
            self.expected[self.path(w.prepared)] = w.prepare_sha
            self.expected[self.path(w.output)/'report.json'] = w.report_sha
            bindings = {str(self.root/f'closure/{w.key}_{i}'): 'a'*64 for i in range(w.binding_count)}
            self.expected.update({Path(k):v for k,v in bindings.items()})
            p = {'schema':w.module+'.prepare','expected_model_hash':m.MODEL_SHA,
                 'slots':[{'source_slot':i,'task_sha256':s} for i,s in enumerate(m.TASK_SHAS)],
                 'bindings_sha256':bindings,
                 'contract':{'P1_admission':False,'training':False,'with_target':False,'feedback_MPC':False}}
            artifacts = {f'fixture{i}.bin':'b'*64 for i in range(w.artifact_count)}
            self.expected.update({self.path(w.output)/k:v for k,v in artifacts.items()})
            r = {'schema':w.module+'.report','status':w.status,'errors':[],
                 'prepared':str(self.path(w.prepared)),'prepare_sha256':w.prepare_sha,'artifact_sha256':artifacts}
            self.data[self.path(w.prepared)],self.data[self.path(w.output)/'report.json'] = p,r
    def path(self, path):
        return Path(m.os.path.abspath(self.root/path))
    def digest(self, path):
        if path not in self.expected:
            raise FileNotFoundError(str(path))
        self.hashes[path] = self.expected[path]
        return self.expected[path]
    def json(self, path):
        return copy.deepcopy(self.data[path])
    def is_file(self, path):
        return True


def test_four_methods_ready_local_not_official():
    r = m.build(RAM())
    assert [x['method'] for x in r['methods']] == ['GRASP','Hi-LeWM','LeWM','PlanWAM']
    assert [x['local_status'] for x in r['methods']] == ['ready_for_MAIN_dry_preflight','blocked','ready_for_MAIN_dry_preflight','blocked']
    assert all(x['official_status']=='blocked' and x['baseline_runnable'] is None for x in r['methods'])
    assert not r['scientific_admission'] and not r['P1_admission']
    for w in r['local_workflows']:
        assert len(w['main_dry_preflight']) == 2 and not w['cuda_launch_authorized']
        assert w['actual_run_command'] is None
        assert [c['argv'][4] for c in w['main_dry_preflight']] == ['verify','consume']


@pytest.mark.parametrize('mutate,code', [('missing','FileNotFoundError'),('hash','sha_mismatch'),
                                       ('root','fixed_two_root_identity'),('MPC','gradient_is_open_loop'),
                                       ('artifact','unsafe_artifact_name')])
def test_hard_failures_no_dispatch(mutate,code):
    f = RAM(); w=m.WORKFLOWS[2]
    p=f.data[f.path(w.prepared)]
    if mutate=='missing': del f.expected[Path(next(iter(p['bindings_sha256'])))]
    if mutate=='hash': f.expected[Path(next(iter(p['bindings_sha256'])))]='0'*64
    if mutate=='root': p['slots'][0]['task_sha256']='0'*64
    if mutate=='MPC': p['contract']['feedback_MPC']=True
    if mutate=='artifact':
        a=f.data[f.path(w.output)/'report.json']['artifact_sha256']
        a['../bad']=a.pop(next(iter(a)))
    r=m.local_workflow(f,w,True)
    assert r['status']=='blocked' and not r['main_dry_preflight']
    assert code in [e['code'] for e in r['errors']]


def test_reported_values_never_unlock_local():
    f=RAM()
    del f.expected[f.path(m.REPORTED)]
    r=m.build(f)
    assert r['local_workflows'][0]['status']=='ready_for_MAIN_dry_preflight'
    assert not r['evidence_columns'][0]['file_verified']
    assert not r['evidence_columns'][0]['affects_launch_readiness']
    assert r['methods'][1]['local_status']=='blocked'
    assert r['methods'][3]['official_run_command'] is None


def test_matrix_drift_blocks_all_local_workflows():
    f=RAM();f.expected[f.path(m.MATRIX)]='0'*64
    r=m.build(f)
    assert r['shared_errors'] and all(w['status']=='blocked' for w in r['local_workflows'])


def test_HiLeWM_reuses_receipt_validator_never_opens_part():
    f=RAM();r=m.build(f)
    c=r['methods'][1]['diagnostic_commands'][0]
    assert c['expected_exit_code']==1 and 'receipt' in c['argv']
    assert all(not str(x).endswith('.part') for x in f.hashes)


def test_propose_main_only_no_execution_no_purpose():
    f=RAM();r=m.build(f)
    with pytest.raises(ValueError,match='purpose'):
        m.propose_run(f,r,'moving_grasp',Path('/RAM/le-wm-main/results/NEW'),'')
    with patch.object(Path,'exists',return_value=False),patch.object(Path,'is_dir',return_value=True),\
         patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),{'free':1024**3})()):
        m.propose_run(f,r,'moving_grasp',Path('/RAM/le-wm-main/results/NEW'),'MAIN new diagnostic')
    row=r['local_workflows'][0]
    assert row['actual_run_command']['argv'][4]=='run'
    assert not row['actual_run_command']['executed'] and not row['cuda_launch_authorized']


def test_existing_output_refused():
    f=RAM();r=m.build(f)
    with patch.object(Path,'exists',return_value=True):
        with pytest.raises(ValueError,match='NEW'):
            m.propose_run(f,r,'moving_grasp',Path('/RAM/le-wm-main/results/old'),'reason')


def test_venv_symlink_not_resolved_in_commands():
    f=m.Files(Path(__file__).resolve().parents[1])
    c=m.command(f,'module','verify')
    assert c['argv'][0].endswith('/.venv/bin/python')


def test_implementation_has_no_model_or_process_import():
    tree=ast.parse(Path(m.__file__).read_text())
    imports={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
    assert not imports & {'torch','pickle','numpy','subprocess','urllib','requests','socket'}
