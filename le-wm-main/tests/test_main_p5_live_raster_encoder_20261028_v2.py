"""Pure replay authority/unchanged-kernel tests; no model or environment."""
import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from experiments.spectral_reachability import main_p5_live_raster_encoder_20261028_v2 as m
from experiments.spectral_reachability import agent_p5_target_raster_encoder_20261028_v1 as old


def receipt():
    args=dict(render_manifest_sha256='a'*64,render_report_sha256='b'*64)
    c=dict(schema=m.consumer.TAG+'.receipt',status='strict_live_raster_complete_v2',
        consumer_bindings=m.consumer.source_seals(),renderer_schema=m.renderer.NAME,baseline_to_current_anchor=[dict(state_id=s) for s in m.STATES],
        prepare_sha256=args['render_manifest_sha256'],report_sha256=args['render_report_sha256'],
        states=5,cells=10,native_entries=deepcopy(m.renderer.EXPECTED),scientific_admission=False,
        physics_steps=605,old_renderer_release=False,historical_full_public_equivalence=None,
        source_comparisons=[{} for _ in range(65)],encoder_inputs=[dict(image_key=s+'/'+r) for s in m.STATES for r in m.RASTERS])
    return c,args


def test_actual_replay_contract_not_static():
    c,args=receipt();m.check_replay_receipt(c,c,args)
    assert m.protocol()['upstream_physics_steps']==605
    assert m.protocol()['physics']==0
    assert m.protocol()['cpu_neural_fallback'] is False


@pytest.mark.parametrize('key,value', [('schema','old.static.consume'),('status','partial'),
    ('prepare_sha256','c'*64),('report_sha256','d'*64),('states',4),('cells',9),('scientific_admission',True)])
def test_bad_replay_receipt(key,value):
    original,args=receipt();c=dict(original,**{key:value})
    with pytest.raises(ValueError):m.check_replay_receipt(c,c,args)


def test_saved_receipt_must_match_full_consumer():
    c,args=receipt();exact=deepcopy(c);exact['extra']='not persisted'
    with pytest.raises(ValueError):m.check_replay_receipt(c,exact,args)


def test_static_zero_physics_cannot_be_relabelled():
    c,args=receipt();c['native_entries']['space_step']=0
    with pytest.raises(ValueError):m.check_replay_receipt(c,c,args)


@pytest.mark.parametrize('name',['run_cuda','consume','Meter','Store','encode','scores','load_runtime','source_closure'])
def test_frozen_cuda_and_saved_kernel_copied_without_global_replacement(name):
    def node(module):
        tree=ast.parse(Path(module.__file__).read_text())
        return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    assert ast.dump(node(m),include_attributes=False)==ast.dump(node(old),include_attributes=False)
    assert m.sha(old.__file__)==m.KERNEL_SOURCE_SHA
    assert old.TAG!=m.TAG and old.renderer is not m.renderer


def test_full10_then16_scores_without_dedup_or_mask():
    features={key:np.full(m.SHAPE,i,np.float32) for i,key in enumerate(m.KEYS)}
    result=m.scores(features)
    assert len(result['cross_condition'])==16 and len(result['fixed_within_state_raster_difference'])==5
    assert result['cross_condition'][0]['full_patch_goal_mse']==64.
    assert all(x['visible_white_feature_mse']==1. for x in result['fixed_within_state_raster_difference'])
    features.pop(m.KEYS[-1])
    with pytest.raises(ValueError):m.scores(features)


def test_fixed_meter_all_ten_and_no_predictor():
    meter=m.Meter(lambda *args:None)
    for key in m.KEYS:
        meter.item=key;meter.before('official_encoder',1);meter.after('official_encoder',1)
    assert meter.completed==m.EXPECTED
    with pytest.raises(ValueError):meter.before('official_encoder',1)
    with pytest.raises(ValueError):m.Meter(lambda *args:None).before('predictor',1)


def test_no_dispatch_without_confirmation(monkeypatch):
    monkeypatch.setattr(m,'verify',lambda *args:pytest.fail('authority first'))
    with pytest.raises(ValueError):m.run_cuda({},'/output',False)


@pytest.mark.parametrize('value',[None,'future','0'*63,'A'*64])
def test_actual_sha_only(value):
    with pytest.raises(ValueError):m.actual_sha(value)


@pytest.mark.parametrize('field,value',[('old_renderer_release',True),('historical_full_public_equivalence',True),('physics_steps',805),('source_comparisons',[])])
def test_no_historical_gate_relabelling(field,value):
    c,args=receipt();c[field]=value
    with pytest.raises(ValueError):m.check_replay_receipt(c,c,args)


def test_live_input_order_cannot_be_selected_or_deduplicated():
    c,args=receipt();c['encoder_inputs'].reverse()
    with pytest.raises(ValueError):m.check_replay_receipt(c,c,args)



def test_old_clock_consumer_receipt_rejected():
    c,args=receipt();c['schema']=m.renderer.NAME+'.receipt';c['status']='strict_live_raster_complete'
    with pytest.raises(ValueError):m.check_replay_receipt(c,c,args)


def test_consumer_source_closure_cannot_drift():
    c,args=receipt();c['consumer_bindings']={}
    with pytest.raises(ValueError):m.check_replay_receipt(c,c,args)


def test_upstream_snapshot_digest_domain_not_encoder_manifest_domain():
    state={'public_state':{'value':1.0},'global_rng':{'numpy':[1,2]}}
    cell=dict(fresh_snapshot_digest=m.renderer.digest(state),
        fresh_public_state_digest=m.renderer.digest(state['public_state']))
    m.check_fresh_snapshot(state,deepcopy(state),cell)
    with pytest.raises(ValueError):m.check_fresh_snapshot(state,state,dict(cell,fresh_snapshot_digest='0'*64))
    changed=deepcopy(state);changed['global_rng']['numpy'][0]=3
    with pytest.raises(ValueError):m.check_fresh_snapshot(state,changed,cell)

