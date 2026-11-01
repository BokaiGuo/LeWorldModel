"""Saved-source/mock precondition checks; no old runtime imports or execution."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from experiments.spectral_reachability import p4_feedback_second_batch_preconditions_20261027_v1 as m


ROOTS=[str(i)*64 for i in range(4)]


def plans():
    return [dict(root_index=i,root_sha256=root,arms=[dict(arm='geometry',selected_candidate_id='b' if i==1 else 'a',
        greedy_candidate_id='a',candidates=[dict(candidate_id='a'),dict(candidate_id='b')])]) for i,root in enumerate(ROOTS)]


def qualified(): return m.qualification(plans(),ROOTS)


def test_complete_root_census():
    q=qualified()
    assert len(q)==4 and [r['root_sha256'] for r in q if r['old_model_only_qualification']]==[ROOTS[1]]


@pytest.mark.parametrize('change',['missing','duplicate','wrong_index','no_geometry','duplicate_geometry','unknown_winner','duplicate_candidate'])
def test_bad_planner_census(change):
    p=plans()
    if change=='missing': p.pop()
    if change=='duplicate': p[0]['root_sha256']=ROOTS[1]
    if change=='wrong_index': p[0]['root_index']=1
    if change=='no_geometry': p[0]['arms'][0]['arm']='stopK'
    if change=='duplicate_geometry': p[0]['arms'].append(deepcopy(p[0]['arms'][0]))
    if change=='unknown_winner': p[0]['arms'][0]['selected_candidate_id']='unknown'
    if change=='duplicate_candidate': p[0]['arms'][0]['candidates'].append({'candidate_id':'a'})
    with pytest.raises(ValueError): m.qualification(p,ROOTS)


def test_same_root_aliases_never_second_batch():
    q=qualified(); rows=m.exclude_roots(q,ROOTS[1],[ROOTS[1]],ROOTS[1],ROOTS[1],set(ROOTS))
    assert not any(r['engineering_prepare_eligible'] for r in rows)
    assert rows[1]['exclusion_reasons']==['same_root_as_first_batch','previously_used_calibration_root_not_fresh_holdout']
    assert rows[1]['source_chain_complete']


@pytest.mark.parametrize('index',[0,2,3])
def test_root_disjoint_not_source_complete_or_fresh(index):
    row=m.exclude_roots(qualified(),ROOTS[1],[ROOTS[1]],ROOTS[1],ROOTS[1],set(ROOTS))[index]
    assert row['root_disjoint_from_first'] and not row['fresh_holdout_verified']
    assert not row['source_chain_complete'] and not row['engineering_prepare_eligible']
    assert 'old_geometry_selected_equals_greedy' in row['exclusion_reasons']
    assert 'no_root_specific_capability_pack_in_existing_chain' in row['exclusion_reasons']


def test_no_outcome_ranking_in_qualification():
    p=plans(); q=m.qualification(p,ROOTS)
    for v in p: v['official_success']=True; v['pose_cost']=-1000
    assert m.qualification(p,ROOTS)==q


def test_full_new_engineering_chain_does_not_certify_fresh_holdout():
    q=qualified()
    rows=m.exclude_roots(q,ROOTS[1],[ROOTS[0]],ROOTS[0],ROOTS[0],set(ROOTS))
    # Helper separates existing-chain engineering compatibility from independent
    # scientific readiness; actual audit rejects newly changed unknown producers.
    assert rows[0]['engineering_prepare_eligible'] and not rows[0]['fresh_holdout_verified']


def test_roots_not_confused_with_file_or_model_hash():
    obj=dict(sha256='f'*64,model_bindings={'/weights':'e'*64},rows=[{'root_sha256':ROOTS[0]}],
             roots=ROOTS,qualified_roots=[ROOTS[1]],root_id=ROOTS[0])
    assert m.roots_of(obj)==ROOTS


def source_texts():
    folder=m.REPO/'experiments/spectral_reachability'
    return ((folder/(m.RUNTIME+'.py')).read_text(),
            (folder/'p4_feedback_matched_20261027_v2.py').read_text())


def test_real_saved_ast_feedback_consumption():
    records=m.consumption_contract(*source_texts())
    assert len(records)==5 and records[1]['legal_at_primitive_steps']==[5,10,15]
    assert records[-1]['legal_at_primitive_steps']==[20] and 'evaluation only' in records[-1]['rule']


@pytest.mark.parametrize('before,after',[
    ("observed if arm=='actual' else predicted","predicted if arm=='actual' else observed"),
    ('backend.encode(frames, between,','backend.encode(other_frames, between,'),
    ('backend.compile(deepcopy(chosen_context),','backend.compile(deepcopy(observed),'),
    ('predicted = successor','predicted = observed'),
    ('terminal_rgb, commands,','other_rgb, commands,'),
])
def test_wrong_consumer_ast_rejected(before,after):
    runtime,feedback=source_texts()
    assert before in runtime
    with pytest.raises(ValueError): m.consumption_contract(runtime.replace(before,after),feedback)


@pytest.mark.parametrize('before,after',[
    ('execution["completed_commands"] != 5','execution["completed_commands"] != 4'),
    ('execution["remaining_commands_not_executed"] != 0','execution["remaining_commands_not_executed"] != 1'),
    ('frames[1],frames[2],terminal_rgb','frames[0],frames[2],terminal_rgb'),
    ('between[1],executed_packet','between[0],executed_packet'),
])
def test_wrong_shift_or_partial_gate_rejected(before,after):
    runtime,feedback=source_texts()
    assert before in feedback
    with pytest.raises(ValueError): m.consumption_contract(runtime,feedback.replace(before,after))


@pytest.mark.parametrize('raw',['{"x":1,"x":2}','{"x":NaN}','{"x":1e999}'])
def test_strict_json(raw):
    with pytest.raises(ValueError): m.decode(raw)


def test_input_budget_and_changes(tmp_path):
    path=tmp_path/'a.json'; path.write_text('{}'); inputs=m.Inputs()
    assert inputs.json(path)=={}
    path.write_text('{"changed":1}')
    with pytest.raises(ValueError): inputs.stable()
    with pytest.raises(ValueError): m.Inputs().json(path,limit=1)


def test_inventory_separates_protocols_and_derived_receipts(tmp_path):
    a=tmp_path/'p4_20steps'; a.mkdir()
    (a/'protocol.json').write_bytes(m.canonical(dict(schema='test',contract=dict(planned_steps_per_arm=20,boundaries=[5,10,15]),roots=[ROOTS[1]])))
    b=tmp_path/'p4_derived'; b.mkdir()
    (b/'receipt.json').write_bytes(m.canonical(dict(schema='test.derived_receipt',root_sha256=ROOTS[1])))
    inv=m.inventory(tmp_path,m.Inputs())
    assert len(inv)==2 and len({tuple(r['root_sha256s']) for r in inv})==1
    assert {r['category'] for r in inv}=={'20step3round_protocol_not_an_extra_execution','derived_encoding_not_new_batch'}


def test_exclusive_output(tmp_path):
    path=tmp_path/(m.TAG+'_existing'); path.mkdir()
    with pytest.raises(ValueError): m.run(path)


def test_no_runtime_imports_or_model_calls():
    tree=ast.parse(Path(m.__file__).read_text()); names=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): names.extend(v.name for v in node.names)
        elif isinstance(node,ast.ImportFrom): names.append(node.module or '')
    assert not any(any(word in name for word in ('torch','pymunk','gym','runtime','checkpoint_compat','attribution_audit')) for name in names)
