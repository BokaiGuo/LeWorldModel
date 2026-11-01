"""Read-only saved ABI fixtures + exclusive temporary output; no fullstrict/NN/physics."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from experiments.spectral_reachability import agent_e1_executable_gap_contract_20261029_v1 as m


@pytest.fixture(scope='module')
def saved():
    # Fixed existing public known-root saved files, not reserved outcomes. No old
    # consumers run and no metric is used to choose a root/arm/threshold.
    refs={k:dict(path=str(m.ROOT/'results'/p),sha256=h) for k,(p,h) in m.CATALOG.items()}
    data={k:m.read(r['path'],r['sha256']) for k,r in refs.items()}
    return data,refs


def put(p,x):p.write_bytes(m.raw(x));return m.sha(p)


def test_all6_existing_chain_no_missing_baseline_substitution(saved):
    data,refs=saved;r=m.joined(deepcopy(data),refs)
    assert len(r)==6 and len(m.field_matrix(r))==2*len(m.SHARED)
    for i in (1,4):
        assert r[i]['target'] is r[i]['shared'] is r[i]['native_success'] is None
    for i in (0,2,3,5):
        assert r[i]['native_executed'] is True and r[i]['arrival'] is None
        assert [x['step'] for x in r[i]['observed_tracking']]==[5,10]
        assert r[i]['controller_action_witness']['pointer'].startswith('/plans/')
    assert r[0]['proposal_action_witness'] is None
    assert r[2]['proposal_action_witness']['pointer']=='/compiled/0'
    assert all(x['status']=='unknown' and x['two_active_match'] is True for x in m.field_matrix(r))


@pytest.mark.parametrize('case',['root','arm','commands','target','compiled','task','native_selection','controller_report',
    'rgb_time','model','missing_prior','prior_zero','denominator','native_goal','scientific_row'])
def test_resealed_metadata_tamper_rejected(saved,case):
    d,refs=saved;d=deepcopy(d)
    if case=='root':d['controller_receipt']['plans'][0]['source_slot']=1
    elif case=='arm':d['physical_receipt']['rows'][0]['arm']=m.ARMS[2]
    elif case=='commands':d['rgb_receipt']['rows'][0]['commands'][0][0]+=.01
    elif case=='target':d['controller_receipt']['plans'][0]['target'][0]+=.01
    elif case=='compiled':d['controller_prepare']['compiled'][0]['prefix_context']['latent_history'][-1][0]+=.01
    elif case=='task':d['controller_receipt']['plans'][0]['task_id']='f'*64
    elif case=='native_selection':d['physical_receipt']['rows'][0]['selection_digest']='f'*64
    elif case=='controller_report':d['controller_report']['prepare_sha256']='f'*64
    elif case=='rgb_time':d['rgb_receipt']['rows'][0]['checkpoints'][0]['step']=10
    elif case=='model':d['rgb_report']['meta']['model_hash_after']='f'*64
    elif case=='missing_prior':d['controller_receipt']['rows'].pop(1)
    elif case=='prior_zero':d['controller_receipt']['rows'][1]['resources']=0
    elif case=='denominator':d['rgb_receipt']['scientific_slot_denominator']=4
    elif case=='native_goal':d['physical_receipt']['rows'][0]['goal'][0]+=.01
    else:d['rgb_receipt']['scientific_rows'][1]['native_success']=False
    with pytest.raises(ValueError):m.joined(d,refs)


def temp_catalog(tmp,mp,saved):
    d,refs=saved;d=deepcopy(d)
    paths={k:tmp/(k+'.json') for k in d}
    # Rebind the complete declared test chain to real temporary files.
    cp_sha=put(paths['controller_prepare'],d['controller_prepare'])
    d['controller_report'].update(prepared=str(paths['controller_prepare']),prepare_sha256=cp_sha)
    ch=put(paths['controller_report'],d['controller_report'])
    for p in d['controller_receipt']['plans']:p['whole_run_report_sha256']=ch
    ph=put(paths['physical_prepare'],d['physical_prepare'])
    d['physical_report'].update(prepared=str(paths['physical_prepare']),prepare_sha256=ph)
    nh=put(paths['physical_report'],d['physical_report'])
    d['physical_receipt']['physical_report_sha256']=nh
    d['rgb_report']['input_binding']=dict(physical_report=str(paths['physical_report']),physical_sha=nh)
    cat={k:(str(paths[k]),put(paths[k],v)) for k,v in d.items()}
    mp.setattr(m,'CATALOG',cat)
    return paths


def test_tempdisk_complete_prepare_cli_and_wholeverify(saved,tmp_path,monkeypatch,capsys):
    temp_catalog(tmp_path,monkeypatch,saved);out=tmp_path/(m.NAME+'_RAM.json')
    m.main(['prepare','--output',str(out)]);o=json.loads(capsys.readouterr().out)
    x=m.read(out,o['sha256']);assert x['three_arm_comparison_complete'] is x['dispatch_allowed'] is False
    assert x['existing_prior_evidence']['hi_delivery']['frozen_learned_integration_exists'] is True
    assert x['existing_prior_evidence']['hi_tracker']['normalized_history3_controller_exists'] is True
    assert x['existing_prior_evidence']['hi_parity']['bounded_computation_pairing'] is True
    assert x['new_neural_calls']==x['new_physics_calls']==x['new_fullstrict_calls']==0
    m.main(['verify','--prepared',str(out),'--prepare-sha',o['sha256']]);assert json.loads(capsys.readouterr().out)['verified'] is True
    with pytest.raises(ValueError):m.main(['prepare','--output',str(out)])


def test_tempdisk_missing_is_missing_not_zero_or_three_arm_ready(saved,tmp_path,monkeypatch):
    paths=temp_catalog(tmp_path,monkeypatch,saved);paths['controller_receipt'].unlink()
    x=m.build();assert len(x['rows'])==6 and x['missing'][0]['role']=='controller_receipt'
    assert all(r['native_success'] is None for r in x['rows'])
    assert x['production_ready'] is False


def test_tempdisk_sha_and_resealed_own_summary_tamper(saved,tmp_path,monkeypatch):
    paths=temp_catalog(tmp_path,monkeypatch,saved);x=m.build();out=tmp_path/(m.NAME+'_tamper.json')
    x['rows'][1]['native_success']=False;h=put(out,x)
    with pytest.raises(ValueError):m.verify(out,h)
    paths['controller_prepare'].write_bytes(b'{}')
    with pytest.raises(ValueError):m.build()


def test_exact_pointer_and_no_threshold_defaults(saved):
    d,_=saved;v=m.pointer(d['controller_receipt'],'/plans/0/target');m.vector(v,192)
    assert m.pointer({'a/b':{'~x':[1]}},'/a~1b/~0x/0')==1
    with pytest.raises(ValueError):m.pointer([1],'/00')
    c=m.contract();assert c['old_controller']['predict_per_dispatch']==144 and c['raw_steps']==10
    assert c['training'] is c['dispatch_allowed'] is False
    assert 'threshold' in c['arrival']


def test_known_mismatch_cannot_be_hidden_by_missing_third_arm(saved):
    d,refs=saved;r=m.joined(deepcopy(d),refs);r[2]['shared']['normalizer']='other'
    v=[x for x in m.field_matrix(r) if x['root_slot']==0 and x['field']=='normalizer'][0]
    assert v['status']=='mismatch' and v['two_active_match'] is False
