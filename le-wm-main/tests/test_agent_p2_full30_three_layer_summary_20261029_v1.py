"""RAM saved receipts only; no producer/model/environment/fullstrict calls."""
from copy import deepcopy
import math
import subprocess
import sys
import pytest
from experiments.spectral_reachability import agent_p2_full30_three_layer_summary_20261029_v1 as m


def fixture(missing=False):
    physical=dict(public_step=dict(attempted=160,completed=160))
    rgb=dict(schema=m.RGB+'.consume',status='all28_full30_RGB_verified_NOT_admitted',fixed_denominator=28,plan_denominator=4,
        rows=[],known_roots=2,fresh_roots=0,consume_new_neural_calls=0,consume_new_physics=0,full_upstream_rebuilt_here=False,
        own_native_encoding_saved_reconstructed_here=True,physical_report_sha256='report',physical_receipt_sha256='receipt',
        inherited_physical_counts=deepcopy(physical),counts={},inherited_model_resources=dict(Jacobian_elements=None),
        inherited_model_wall_seconds=2.,total_FLOPs=None,
        **{k:False for k in ('P1_admission','P2_admission','P2_mechanism','C3','equal_total_compute','physical_release')})
    s=dict(denominator=dict(branches=4,known_roots=2,fresh_roots=0,checkpoint_positions=28,pairs=14),
           panel=[],pairs=[],counts=deepcopy(physical),**{k:False for k in m.FLAGS})
    scalar=dict(schema=m.SCALAR+'.summary',new_neural_calls=0,new_physics_calls=0,new_fullstrict_calls=0,
                arguments=dict(report_sha='report',receipt_sha='receipt'),summary=s)
    for slot in range(4):
        for step in m.CP:
            i=7*slot+m.CP.index(step);root=slot//2;arm=m.ARMS[slot%2]
            absent=missing and slot==1 and step>=25
            goal=[0.]*192;pred=None if step==0 else [.25+step/128-(.125 if root==0 and slot%2 else 0)]*192
            actual=None if absent else [(.5 if root==0 and slot%2 and step else .25)]*192
            cp=None if absent else dict(step=step,png=f'b{slot}_{step}.png',file_sha256='same',pixel_sha256='same',state=[float(step)]*7,
                snapshot=dict(state=[float(step)]*7,bodies=dict(RAM=True)))
            native=None if absent else {k:float(step+slot%2) if root==0 else float(step) for k in m.NATIVE_KEYS}
            if native is not None:native['native_success']=False
            model=None if pred is None else m.original_mean(pred,goal)
            aq=None if actual is None else m.original_mean(actual,goal)
            error=None if model is None or aq is None else model-aq
            source=None if absent else dict(**deepcopy(cp),path='/RAM/'+cp['png'])
            response=None if absent or step==0 else dict(step=step,state=deepcopy(cp['state']),native_success=False)
            rgb['rows'].append(dict(index=i,plan_index=slot,root_slot=root,condition='off',arm=arm,checkpoint=step,
                goal192=goal,predicted192=pred,actual192=actual,model_Q=model,actual_encoded_Q=aq,model_minus_actual_Q=error,
                absolute_Q_error=None if error is None else abs(error),prediction_vs_encoded_MSE=None if pred is None or actual is None else m.original_mean(pred,actual),
                role='observed_anchor_reencode' if step==0 else 'same_action_H6_prediction',latent_missing=absent,source=source,
                native_metrics=None if absent or step==0 else dict(native_success=False),response=response,
                observed_anchor_reencode_MSE=0. if step==0 else None,task_id='task'+str(root),selection_digest='seal'+str(slot),
                causal_raw10_bits='bits'+str(i),physical_status='native_stopped' if missing and slot==1 else 'horizon_completed',
                P1_admission=False,P2_admission=False,C3=False,physical_benefit_claim=False))
            s['panel'].append(dict(slot=slot,root=root,arm=arm,control_step=step,absolute_step=10+step,checkpoint=deepcopy(cp),
                metrics=native,missing_reason='checkpoint_not_reached' if absent else None))
    n=sum(r['actual192'] is not None for r in rgb['rows'])
    rgb['counts']={k:dict(attempted=n if k in ('encoder','projector') else 0,completed=n if k in ('encoder','projector') else 0)
                   for k in ('encoder','projector','action_encoder','predict_API','predictor','pred_proj')}
    for root in (0,1):
        for j,step in enumerate(m.CP):
            a,b=[s['panel'][slot*7+j]['metrics'] for slot in (2*root,2*root+1)]
            known=a is not None and b is not None
            s['pairs'].append(dict(root=root,control_step=step,direction='joint_two_round minus two_program_transfer',
                comparable=known,delta=None if not known else {k:b[k]-a[k] for k in m.NATIVE_KEYS},native_success_delta=0 if known else None))
    return rgb,scalar


def test_full28_24forecasts_4anchors_all14_pairs_no_filled_model():
    out=m.summarize(*fixture())
    assert out['denominator']==dict(positions=28,anchors=4,forecasts=24,pairs=14,anchor_pairs=2,forecast_pairs=12)
    assert len(out['rows'])==28 and len(out['pairs'])==14
    assert all(r['model_Q'] is r['model_minus_actual_Q'] is None for r in out['rows'] if r['role']=='anchor')
    assert out['comparison_census']['all14']['model_to_actual']['null']==2
    assert out['comparison_census']['forecast12']['model_to_actual']['reversal']==6
    assert out['comparison_census']['forecast12']['model_to_actual']['both_tie']==6
    assert out['inherited_model_resources']['Jacobian_elements'] is None and out['total_FLOPs'] is None


def test_root1_equal_vectors_and_all_native_ties_remain_in_denominator():
    out=m.summarize(*fixture())
    for p in out['pairs'][7:]:
        assert set(p['native_delta'].values())=={0.}
        assert p['delta_actual_Q']==0 and p['vector_equalities']['actual']
        assert p['delta_model_Q']==(None if p['control_step']==0 else 0.)


def test_missing_kept_both_native_and_actual_but_model_forecast_survives():
    out=m.summarize(*fixture(True))
    assert len(out['rows'])==28 and len(out['pairs'])==14
    r=out['rows'][13];assert r['actual_Q'] is r['native'] is None and r['model_Q'] is not None
    p=out['pairs'][6];assert p['delta_actual_Q'] is None and p['delta_model_Q'] is not None
    assert p['comparisons']['model_to_actual']=='null' and p['missing_slots']==[1]
    assert all(sum(v.values())==14 for v in out['comparison_census']['all14'].values())


@pytest.mark.parametrize('a,b,label',[(1e-300,-1e-300,'reversal'),(-1e-300,1e-300,'reversal'),
    (0.,-0.,'both_tie'),(0.,1e-300,'left_tie'),(1e-300,0.,'right_tie'),(None,0.,'null'),(-1.,-2.,'same')])
def test_strict_signs_do_not_use_underflowing_products(a,b,label):assert m.compare(a,b)==label


def test_fsum_closed_form_and_original_reduction_difference_separate():
    assert m.mse([1.]*192,[0.]*192)==1.
    a=[float(2**26)]+[.125]*191;z=[0.]*192
    assert m.mse(a,z)==math.fsum(v*v for v in a)/192
    rgb,s=fixture();r=rgb['rows'][1];r['actual192']=a
    r['actual_encoded_Q']=m.original_mean(a,r['goal192']);r['model_minus_actual_Q']=r['model_Q']-r['actual_encoded_Q']
    r['absolute_Q_error']=abs(r['model_minus_actual_Q']);r['prediction_vs_encoded_MSE']=m.original_mean(r['predicted192'],a)
    out=m.summarize(rgb,s)['rows'][1]
    assert out['actual_Q']==m.mse(a,z)
    assert out['fsum_minus_producer_numpy']['actual_Q']==out['actual_Q']-r['actual_encoded_Q']
    assert out['fsum_minus_producer_numpy']['actual_Q']!=0.


def test_missing_anchor_retains_null_and_fixed_comparison_axes():
    rgb,s=fixture();r=rgb['rows'][0];n=s['summary']['panel'][0]
    r.update(actual192=None,actual_encoded_Q=None,latent_missing=True,source=None,observed_anchor_reencode_MSE=None)
    n.update(metrics=None,checkpoint=None,missing_reason='checkpoint_not_reached')
    s['summary']['pairs'][0].update(comparable=False,delta=None,native_success_delta=None)
    for key in ('encoder','projector'):rgb['counts'][key]=dict(attempted=27,completed=27)
    out=m.summarize(rgb,s)
    assert out['available']['anchors']==3 and out['denominator']['anchors']==4
    assert out['pairs'][0]['comparisons']['actual_to_native7_distance']=='null'
    assert out['pairs'][0]['delta_model_Q'] is None


def test_saved_reader_rejects_duplicate_keys_nonfinite_and_changed_bytes(tmp_path):
    for payload in (b'{"a":1,"a":2}',b'{"a":NaN}'):
        p=tmp_path/'bad.json';p.write_bytes(payload)
        with pytest.raises(ValueError):m.read(str(p),m.sha(p))
    p.write_bytes(b'{}');h=m.sha(p);p.write_bytes(b'{"changed":1}')
    with pytest.raises(ValueError):m.read(str(p),h)


@pytest.mark.parametrize('kind',['schema','count','known','upstream','physicalsha','physicalcounts','order','time','png','snapshot',
    'model','actualQ','error','mse','anchorzero','anchorpred','native','nativedelta','goal','task','rawvector','flag'])
def test_mutation_failclosed(kind):
    rgb,s=deepcopy(fixture());r=rgb['rows'][1]
    if kind=='schema':rgb['schema']='old.consume'
    elif kind=='count':rgb['counts']['encoder']['completed']=27
    elif kind=='known':rgb['known_roots']=4
    elif kind=='upstream':rgb['full_upstream_rebuilt_here']=True
    elif kind=='physicalsha':rgb['physical_report_sha256']='bad'
    elif kind=='physicalcounts':rgb['inherited_physical_counts']['public_step']['completed']=0
    elif kind=='order':r['plan_index']=3
    elif kind=='time':s['summary']['panel'][1]['absolute_step']=5
    elif kind=='png':r['source']['pixel_sha256']='bad'
    elif kind=='snapshot':r['source']['snapshot']['state'][0]=9.
    elif kind=='model':r['model_Q']=0.
    elif kind=='actualQ':r['actual_encoded_Q']=0.
    elif kind=='error':r['model_minus_actual_Q']=0.
    elif kind=='mse':r['prediction_vs_encoded_MSE']=0.
    elif kind=='anchorzero':rgb['rows'][0]['model_Q']=0.
    elif kind=='anchorpred':rgb['rows'][0]['predicted192']=[0.]*192
    elif kind=='native':r['response']['native_success']=True
    elif kind=='nativedelta':s['summary']['pairs'][1]['delta']['velocity_squared']+=1.
    elif kind=='goal':r['goal192'][0]=1.
    elif kind=='task':r['task_id']='other'
    elif kind=='rawvector':r['actual192'][0]=.1
    else:rgb['P2_admission']=True
    with pytest.raises((ValueError,KeyError)):m.summarize(rgb,s)


def test_persisted_analysis_verify_fixture_and_wrong_pins(tmp_path,monkeypatch):
    rgb,s=fixture();rp,sp=tmp_path/'rgb.json',tmp_path/'scalar.json'
    rp.write_bytes(m.encode(rgb));sp.write_bytes(m.encode(s))
    monkeypatch.setattr(m,'RGB_SHA',m.sha(rp));monkeypatch.setattr(m,'SCALAR_SHA',m.sha(sp))
    out=tmp_path/(m.NAME+'_RAM.json')
    args=['analyze','--rgb-receipt',str(rp),'--rgb-sha',m.sha(rp),'--scalar-summary',str(sp),'--scalar-sha',m.sha(sp),'--output',str(out)]
    m.main(args);m.main(['verify','--summary',str(out),'--summary-sha',m.sha(out)])
    with pytest.raises(ValueError):m.main(args)
    with pytest.raises(ValueError):m.analyze(str(rp),'0'*64,str(sp),m.sha(sp))
    with pytest.raises(ValueError):m.absolute('relative.json')


def test_cold_import_has_no_producer_neural_or_native_modules():
    code="import sys;from experiments.spectral_reachability import "+m.NAME+";assert not any(k.startswith(('torch','gym','pymunk','stable_worldmodel')) or k.endswith('agent_p2_two_program_full30_rgb_encode_20261029_v1') for k in sys.modules)"
    subprocess.run([sys.executable,'-B','-c',code],check=True,cwd=m.ROOT)


def test_actual_saved_abi_only_no_summary_build_or_vector_arithmetic():
    rgb=m.read(str(m.ROOT/'results/agent_p2_two_program_full30_rgb_encode_20261029_v1_consume_MAIN01.json'),m.RGB_SHA)
    scalar=m.read(str(m.ROOT/'results/main_p2_full30_same_time_summary_20261029_v1_MAIN01.json'),m.SCALAR_SHA)
    assert len(rgb['rows'])==len(scalar['summary']['panel'])==28
    assert len(scalar['summary']['pairs'])==14
    for r,n in zip(rgb['rows'],scalar['summary']['panel']):
        assert {'actual192','goal192','predicted192','model_Q','actual_encoded_Q','native_metrics','source','response'}<=r.keys()
        assert set(n['metrics'])==set(m.NATIVE_KEYS)|{'native_success'}
        assert r['checkpoint']==n['control_step'] and r['plan_index']==n['slot']
        assert len(r['actual192'])==len(r['goal192'])==192
        if r['checkpoint']==0:assert r['predicted192'] is r['model_Q'] is r['response'] is None
        else:assert len(r['predicted192'])==192
