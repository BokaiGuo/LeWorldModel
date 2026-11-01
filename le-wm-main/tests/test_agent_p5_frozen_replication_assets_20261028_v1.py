"""RAM metadata only; fake SHA maps never constitute actual model evidence."""
import ast
import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.spectral_reachability import agent_p5_frozen_replication_assets_20261028_v1 as m


class RAM:
    root=Path('/RAM/le-wm-main')
    def __init__(self):
        self.hashes={};self.bytes_hashed=0;self.expected={};self.data={}
        self.expected[Path(m.saved.__file__)]=m.HELPER_SHA
        for k,(path,sha) in m.SPECS.items():
            self.expected[self.path(path)]=sha
            self.data[self.path(path)]={}
        for module,sha in m.SOURCES.values():
            self.expected[self.path('experiments/spectral_reachability/'+module+'.py')]=sha
        entries=[{'path':str(self.root/f'asset_{i}'),'exists':True,'sha256':'a'*64} for i in range(41)]
        self.expected.update({Path(r['path']):r['sha256'] for r in entries})
        self.data[self.path(m.SPECS['assets'][0])]={
            'schema':'p5_local_asset_binding_20261027_v1.v1','P5_admission':False,'P1_admission':False,
            'calibration_roots':list(range(202610360,202610368)),'files':entries,
            'dataset':{'path':str(m.CACHE/'datasets/tworoom.h5'),'bytes':12775849984}}
        self.data[self.path(m.SPECS['tworoom'][0])]={'status':'replay_completed','P5_admission':False,'gates':{'frozen':True}}
        binds={str(self.root/f'native_{i}'):'b'*64 for i in range(988)}
        del binds[str(self.root/'native_987')]
        binds[str(m.OFFICIAL_SOURCE_TAR)]=m.OFFICIAL_SOURCE_SHA
        self.expected.update({Path(k):v for k,v in binds.items()})
        manifest={'P5_admission':False,'bindings':binds}
        self.data[self.path(m.SPECS['native_manifest'][0])]=manifest
        self.data[self.path(m.SPECS['native'][0])]={
            'status':'fresh_native_panel_fidelity_recorded','P5_admission':False,'model_unchanged':True,
            'sources_unchanged':True,'new_physical_steps':0,
            'strict_loads':{k:{'exact_match':True} for k in ('predictor','proprio_encoder')},
            'protocol':{'training':False,'cpu_neural_fallback':False,'root_selection_changed':False,
                        'seeds':[2026102702,2026102703,2026102704]},
            'manifest_sha256':m.canonical_digest(manifest)}
    def path(self,name):return self.root/name
    def digest(self,path):
        self.hashes[path]=self.expected[path]
        return self.expected[path]
    def json(self,path):return copy.deepcopy(self.data[path])
    def is_file(self,path):return path in self.expected
    def size(self,path):return 12775849984


def test_two_frozen_runtimes_not_P5_and_cube_same_family():
    r=m.build(RAM())
    assert [x['status'] for x in r['candidates']]==['ready_for_MAIN_frozen_runtime_preflight']*2+['blocked']
    assert r['candidates'][0]['architecture_family']==r['candidates'][2]['architecture_family']
    assert r['candidates'][1]['architecture_family']!=r['candidates'][0]['architecture_family']
    assert r['independent_architecture_replications_established']==0 and r['P5_admission'] is False
    assert all(x['actual_run_command'] is None for x in r['candidates'])
    assert r['saved_admission_commands'][0]['expected_exit_code']==3


@pytest.mark.parametrize('kind', ['missing_weight','strict','root','manifest','admission','source'])
def test_native_mutations_failclosed(kind):
    f=RAM();n=f.data[f.path(m.SPECS['native'][0])]
    if kind=='missing_weight':del f.expected[f.root/'native_0']
    elif kind=='strict':n['strict_loads']['predictor']['exact_match']=False
    elif kind=='root':n['protocol']['seeds'][0]=1
    elif kind=='manifest':n['manifest_sha256']='0'*64
    elif kind=='admission':n['P5_admission']=True
    else:f.expected[f.path('experiments/spectral_reachability/'+m.SOURCES['native_jepa'][0]+'.py')]='0'*64
    r=m.build(f)['candidates'][1]
    assert r['status']=='blocked' and r['errors'] and not r['actual_run_command']


def test_environment_multiplicity_not_architecture_and_Cube_adapter_excluded():
    r=m.build(RAM())
    assert 'SAME architecture' in r['family_boundary']
    c=r['candidates'][2]
    assert 'no_training_free_cube_runner' in [e['code'] for e in c['errors']]
    width=next(e for e in c['errors'] if e['code']=='cube_packet_width')
    assert width['actual']==25 and width['expected']==10


def test_twroom_calibration_stays_fixed_no_outcome_selection():
    f=RAM();f.data[f.path(m.SPECS['assets'][0])]['calibration_roots'][0]=123
    r=m.build(f)['candidates'][0]
    assert r['status']=='blocked'


def test_no_neural_imports_and_canonical_hash_not_file_hash():
    tree=ast.parse(Path(m.__file__).read_text())
    names={a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
    assert not names & {'torch','numpy','pickle','subprocess','socket','urllib','requests'}
    assert m.canonical_digest({'a':1,'b':2})==m.canonical_digest({'b':2,'a':1})


def test_generate_native_command_only_no_physics():
    f=RAM();r=m.build(f)
    with patch.object(Path,'exists',return_value=False),\
         patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),{'free':1024**3})()):
        m.propose(f,r,'native_jepa',f.root/'results/NEW','MAIN new purpose')
    c=r['candidates'][1]['actual_run_command']
    assert 'cuda' in c['argv'] and '0 new physics' in c['physics_scope']
    assert not c['executed'] and not r['candidates'][1]['launch_authorized']


def test_twroom_command_discloses_warmup_physics_and_original_prefix():
    f=RAM();r=m.build(f)
    with patch.object(Path,'exists',return_value=False),\
         patch.object(m.shutil,'disk_usage',return_value=type('Disk',(),{'free':1024**3})()):
        with pytest.raises(ValueError,match='prefix'):
            m.propose(f,r,'tworoom',f.root/'results/NEW','purpose')
        m.propose(f,r,'tworoom',f.root/'results/p5_local_asset_binding_20261027_v1_NEW','purpose')
    assert '15 original warmup physics' in r['candidates'][0]['actual_run_command']['physics_scope']


def test_existing_output_or_missing_purpose_refused():
    f=RAM();r=m.build(f)
    with pytest.raises(ValueError,match='purpose'):
        m.propose(f,r,'native_jepa',f.root/'results/NEW','')
    with patch.object(Path,'exists',return_value=True):
        with pytest.raises(ValueError,match='NEW'):
            m.propose(f,r,'native_jepa',f.root/'results/OLD','purpose')


def test_missing_actual_report_not_claimed_completed():
    f=RAM();del f.expected[f.path(m.SPECS['native'][0])]
    r=m.build(f)['candidates'][1]
    assert r['status']=='blocked' and not r['existing_runtime_completed']


def test_tmp_source_allowlist_exact_not_directory_scan():
    f=m.Files(Path(__file__).resolve().parents[1])
    assert f.allowed(m.OFFICIAL_SOURCE_TAR)
    assert not f.allowed(m.OFFICIAL_SOURCE_TAR.parent/'unbound.tar.gz')
    ram=RAM()
    ram.data[ram.path(m.SPECS['native_manifest'][0])]['bindings'][str(m.OFFICIAL_SOURCE_TAR)]='0'*64
    assert 'official_source_binding' in [e['code'] for e in m.build(ram)['candidates'][1]['errors']]
