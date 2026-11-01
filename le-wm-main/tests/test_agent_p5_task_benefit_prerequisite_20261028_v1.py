"""RAM task arithmetic and failclosed wiring, not scientific evidence."""
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.spectral_reachability import agent_p5_task_benefit_prerequisite_20261028_v1 as m


def q(c):
    return dict(coverage=c,native_reward=min(c/.95,1.),native_success_threshold=.95,
        native_done=False,task_success=None,
        native_goal_pose=dict(dtype='<f8',shape=[3],values=[256.,256.,math.pi/4]))


def fixture():
    return dict(schema=m.OWNER+'.report',status='all4_saved_native_word_responses_analyzed',
        scientific_admission=False,task_success=None,
        baselines=[dict(word=w,frame=10,quantities=q(.3)) for w in m.WORDS],
        suffix_frame_rows=[dict(word=w,frame=f,suffix_step=f-10,quantities=q(.3+(i-2)*.01*(f-10)))
            for i,w in enumerate(m.WORDS) for f in range(11,21)])


def test_fixed_metrics_allwords_no_winner():
    r=m.metrics(fixture())
    assert len(r['word_rows'])==4 and len(r['frame_rows'])==40 and len(r['paired_contrasts'])==2
    assert r['word_rows'][0]['terminal_coverage_gain']==pytest.approx(-.2)
    assert r['word_rows'][3]['terminal_coverage_gain']==pytest.approx(.1)
    assert r['word_rows'][0]['mean_suffix_coverage_gain']==pytest.approx(-.11)
    assert r['paired_contrasts'][0]['terminal_coverage_gain']==pytest.approx(-.1)
    assert r['selected_word'] is None and r['C2_established'] is False
    assert all(x['task_success'] is None for x in r['word_rows'])


@pytest.mark.parametrize('change',['missing','order','duplicate','frame','step','prefix','goal','threshold','reward','done','success','nan','range','admission','status'])
def test_mutations(change):
    a=fixture(); row=a['suffix_frame_rows'][0]; v=row['quantities']
    if change=='missing': a['suffix_frame_rows'].pop()
    elif change=='order': a['suffix_frame_rows'].reverse()
    elif change=='duplicate': a['suffix_frame_rows'][-1]=deepcopy(row)
    elif change=='frame': row['frame']=10
    elif change=='step': row['suffix_step']=2
    elif change=='prefix': a['baselines'][0]['quantities']=q(.4)
    elif change=='goal': v['native_goal_pose']['values'][0]+=1
    elif change=='threshold': v['native_success_threshold']=.9
    elif change=='reward': v['native_reward']=1.
    elif change=='done': v['native_done']=True
    elif change=='success': v['task_success']=True
    elif change=='nan': v['coverage']=math.nan
    elif change=='range': v['coverage']=1.1
    elif change=='admission': a['scientific_admission']=True
    else: a['status']='partial'
    with pytest.raises(ValueError): m.metrics(a)


def test_object_movement_not_metric_and_reward_saturation():
    a=fixture()
    for row in a['suffix_frame_rows']:
        row['quantities']=q(.99)
        row['quantities']['block_xy']=[10000.,-10000.]
    r=m.metrics(a)
    assert all(x['terminal_native_reward']==1. for x in r['word_rows'])
    assert all(x['terminal_coverage_gain']==pytest.approx(.69) for x in r['word_rows'])
    assert all(x['terminal_coverage_gain']==0 for x in r['paired_contrasts'])
    assert m.protocol()['new_success_threshold'] is None


def test_recompute_exact_strict_chain(tmp_path):
    physical='physical'
    paths=['physical_prepare_MAIN01.json','physical_physics_MAIN01/report.json',
        'consume_contact_words_MAIN_20261028_v1.py','physical_consume_MAIN01.json']
    saved=dict(inputs={str(tmp_path/p):str(i)*64 for i,p in enumerate(paths)})
    seen=[]
    def analyze(*args): seen.append(args); return saved
    a=SimpleNamespace(PHYSICAL=physical,analyze=analyze)
    m.recompute(a,saved)
    assert len(seen)==1 and len(seen[0])==8
    a.analyze=lambda *args:{}
    with pytest.raises(ValueError,match='full strict'): m.recompute(a,saved)
    saved['inputs'].pop(str(tmp_path/paths[0]))
    with pytest.raises(ValueError,match='unique saved authority'): m.recompute(a,saved)


def test_prepare_hardblocks_C2_even_with_valid_task_metrics(tmp_path,monkeypatch):
    saved=fixture(); saved.update(model_only_parallel={'RAM':True},physical_meter={'RAM':1},
        operation_meter={'RAM':2},wall_seconds_physical=3.)
    def fake_sha(p):
        if str(p).endswith(m.ENV): return m.ENV_SHA
        if Path(p)==tmp_path/'analysis': return m.ANALYSIS_SHA
        return 'a'*64
    fake=SimpleNamespace(read_json=lambda *args:saved,sha=fake_sha)
    monkeypatch.setattr(m,'owner',lambda:fake)
    monkeypatch.setattr(m,'recompute',lambda *args:None)
    # Explicit fake hashes only for RAM seam; never an actual asset certificate.
    monkeypatch.setattr(m,'OWNER_SHA','a'*64)
    result=m.prepare(tmp_path/'analysis',m.ANALYSIS_SHA)
    assert result['engineering_ready'] is True
    assert result['prospective_trial_ready'] is result['C2_established'] is result['scientific_admission'] is False
    assert len(result['hard_missing'])==4 and len(result['metrics']['word_rows'])==4
    with pytest.raises(ValueError,match='actual MAIN'): m.prepare(tmp_path/'analysis','0'*64)


def test_real_source_identity_no_saved_execution():
    a=m.owner()
    assert a.sha(m.ROOT/m.ENV)==m.ENV_SHA
    assert (m.ROOT/'innerdocs'/(m.TAG+'.md')).is_file()
    assert m.protocol()['timing'].startswith('specified after existing outcomes')


def test_cli_wiring_no_real_prepare(tmp_path,monkeypatch):
    a=m.owner(); value=dict(RAM=True,C2_established=False,status='fixture')
    monkeypatch.setattr(m,'prepare',lambda *args:value)
    args=['--analysis',str(tmp_path/'input'),'--analysis-sha256',m.ANALYSIS_SHA,'--output',str(tmp_path/'out')]
    assert m.main(['prepare',*args])==0
    assert m.main(['verify',*args,'--output-sha256',a.sha(tmp_path/'out')])==0
    with pytest.raises(ValueError,match='exclusive'): m.main(['prepare',*args])
    with pytest.raises(ValueError,match='actual output'): m.main(['verify',*args])
