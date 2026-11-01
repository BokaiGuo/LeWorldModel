"""Real saved metadata/NumPy validation and RAM mutations; no neural/physics."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.spectral_reachability import agent_p2_guard_saved_consume_20261028_v1 as m

OUTPUT=m.ROOT/'results/agent_p2_runtime_guard_20261028_v1_known_MAIN01'
REPORT_SHA='0cfbe37a7783c6df331305e95e3c058a9b006f72bb1ca5b59d24fae047ee23a0'
PREPARED=m.ROOT/'results/agent_p2_runtime_guard_20261028_v1_prepared01.json'
PREPARE_SHA='02269ae5d08d1f645e9bcf2473a7c7cbb2a16f3125c3fad62c23859380cc6c48'
REVIEW_SHA='52599f94ff9e60db7673310521a1536b9d24fe033bcca8be6206e814946918c8'


@pytest.fixture(scope='module')
def actual():
    assert m.sha(OUTPUT/'report.json')==REPORT_SHA
    p=m.native.read('results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01/protocol.json')
    ready=p['context']['cases'][0]
    row=m.read(OUTPUT/'root_0.json');records=m.producer.read_journal(OUTPUT/'root_0_guard.jsonl')
    return records,row,ready,'1d79f5b17002334434454dfb5a94e6d918de1d603afdf84918e5ded6eb980d1b'


def first(records,kind):return next(r['payload'] for r in records if r['kind']==kind)


def test_actual_journal_reconciles_not_merely_hashes(actual):
    audit=m.reconcile(*actual)
    assert audit['primitive_count']==867 and audit['module_entries']==664 and audit['operation_pairs']==83
    assert audit['candidate_count']==120 and audit['arrays']==82
    assert audit['unique']['predict_calls']==audit['unique']['action_encoder_calls']==332
    assert audit['unique']['math_vjp_192x10_directions']==7680
    assert audit['unique']['flops'] is None


def test_explicit_root_VJP_direction_request_not_silently_ignored(actual):
    e=deepcopy(actual[1]['operations'][0]);e['scalar_output_vjp_directions_requested']=1
    with pytest.raises(ValueError,match='mathematical VJP'):m.work_operation(e)


@pytest.mark.parametrize('damage',[
    'pending','return_missing','entry_missing','reserve_missing','fake_completion','group_wrong','group_upper',
    'actual_entry','op_stream','candidate_hash','candidate_missing','snapshot_axis','snapshot_unknown',
    'array_receipt','array_order','metric_return','header','extra_failure','return_before_entry','bool_root'])
def test_real_saved_semantic_mutations_rejected_even_without_relying_on_hash_chain(actual,damage):
    records,row,ready,native_sha=deepcopy(actual)
    if damage=='pending':row['guard']['pending_operations']={'x':{'entry':True}};records[-1]['payload']=deepcopy(row['guard'])
    elif damage in ('return_missing','entry_missing','reserve_missing','candidate_missing'):
        kind={'return_missing':'return_observed','entry_missing':'entry_boundary','reserve_missing':'operation_reserve_intent','candidate_missing':'candidate_terminal'}[damage]
        records.pop(next(i for i,r in enumerate(records) if r['kind']==kind))
    elif damage=='fake_completion':first(records,'return_observed')['success']=False
    elif damage=='group_wrong':first(records,'group_reserve_intent')['group']='other'
    elif damage=='group_upper':first(records,'group_reserve_intent')['upper']['predict_calls']=999
    elif damage=='actual_entry':row['actual_model_entries'][0]['samples']=2
    elif damage=='op_stream':row['operations'][0]['shape']=[192,10]
    elif damage=='candidate_hash':first(records,'candidate_terminal')['native_event_sha256']='a'*64
    elif damage=='snapshot_axis':row['guard']['actual_complete_totals']['predict_calls']=0;records[-1]['payload']=deepcopy(row['guard'])
    elif damage=='snapshot_unknown':row['guard']['actual_complete_totals']['flops']=0;records[-1]['payload']=deepcopy(row['guard'])
    elif damage=='array_receipt':first(records,'native_evidence')['event']['receipt']['sha256']='a'*64
    elif damage=='array_order':
        k=next(i for i,r in enumerate(records) if r['kind']=='native_evidence');saved=records.pop(k);records.insert(2,saved)
    elif damage=='metric_return':next(r['payload'] for r in records if r['kind']=='return_observed' and r['payload']['token'].startswith('metric.'))['native_evidence']['kind']='not_a_return'
    elif damage=='header':records[0]['payload']['binding']['wrapper_source_sha256']='a'*64
    elif damage=='extra_failure':first(records,'operation_reserve_intent')['hidden_failure']=True
    elif damage=='return_before_entry':
        k=next(i for i,r in enumerate(records) if r['kind']=='entry_boundary');records[k],records[k+1]=records[k+1],records[k]
    else:first(records,'entry_boundary')['root_slot']=False
    # Hash-chain verification is intentionally bypassed in this unit test: semantic verifier MUST still reject.
    with pytest.raises((ValueError,KeyError,IndexError)):m.reconcile(records,row,ready,native_sha)


@pytest.mark.parametrize('damage',['status','failure','source','model','model_hash','denominator','review','prepare','unknown0','extra'])
def test_terminal_report_strict_failclosed_before_partial_rows(damage):
    r=m.read(OUTPUT/'report.json')
    if damage=='status':r['status']='in_progress'
    elif damage=='failure':r['error']={'message':'native failed'}
    elif damage=='source':r['source_unchanged']=False
    elif damage=='model':r['model_unchanged']=None
    elif damage=='model_hash':r['model_after']['buffers']='a'*64
    elif damage=='denominator':r['candidate_denominator']=479
    elif damage=='review':r['main_review_sha256']='a'*64
    elif damage=='prepare':r['wrapper_prepare_sha256']='a'*64
    elif damage=='unknown0':r['physics_endpoint_cost']=0
    else:r['partial_but_ok']=True
    with pytest.raises(ValueError):m.terminal_report(r,PREPARE_SHA,REVIEW_SHA)


def test_nonterminal_refused_before_any_row_or_prepare_reads(monkeypatch):
    r=m.read(OUTPUT/'report.json');r['status']='in_progress';called=[]
    def read(path,*args):
        called.append(Path(path).name)
        assert Path(path).name=='report.json'
        return m.producer.encode(r)
    monkeypatch.setattr(m,'raw_bytes',read)
    with pytest.raises(ValueError,match='terminal successful'):m.consume(OUTPUT,m.C.digest(r),PREPARED,PREPARE_SHA,REVIEW_SHA)
    assert called==['report.json']


def test_confirmatory_denies_before_reads(monkeypatch):
    monkeypatch.setattr(m,'read',lambda *a:pytest.fail('no read permitted'))
    with pytest.raises(ValueError,match='confirmatory/physics'):
        m.consume(None,None,None,None,None,confirmatory=True)


@pytest.fixture(scope='module')
def native_inputs(actual):
    records,row,ready,native_sha=actual
    p=m.native.read('results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01/protocol.json')
    source=m.native.read(p['context']['compilation'])
    compiled=next(r['compiled'] for r in source['rows'] if r['root_sha256']==ready['root_sha256'])
    saved=m.native.read(OUTPUT/'root_0_bank.json')
    return p,compiled,saved


@pytest.mark.parametrize('damage',['raw_action','history','Q','suffix_J','provenance'])
def test_reused_native_result_validator_rejects_saved_witness_mutations(actual,native_inputs,damage):
    records,row,ready,native_sha=actual;p,compiled,saved=native_inputs
    arm=m.native.ARMS[1];result=deepcopy(row['results'][1]);e=result['attempts'][0]
    prov=m.native.provenance(p,m.native.read('results/p2_composition_predict_frontier_cuda_20261027_v2_prepared01/prepared.json'),ready,saved['expanded'],arm)
    if damage=='raw_action':e['actions'][0][0]+=.125
    elif damage=='history':e['witness']['receipt']['terminal_context']['action_history'][0][0]+=.125
    elif damage=='Q':e['cost']+=.125
    elif damage=='suffix_J':e['suffix_compilation']['jacobian'][0][0]+=.125
    else:e['witness']['edges'][0]['provenance']['checkpoint_sha256']='a'*64
    with pytest.raises(ValueError):m.native.validate_result(result,arm=arm,bank=saved['expanded'],ready=ready,prov=prov,compiled=compiled)


@pytest.mark.parametrize('damage',['value','shape','context','receipt_sha','stage'])
def test_reused_native_FULL_array_validator_rejects_saved_mutations(actual,native_inputs,monkeypatch,damage):
    records,row,ready,native_sha=actual;p,compiled,saved=native_inputs;r=deepcopy(row)
    if damage=='receipt_sha':r['array_receipts'][0]['sha256']='a'*64
    elif damage=='stage':r['derivative_stages'][0]['persisted']=False
    else:
        original=m.native.read
        def changed(path,*args):
            body=original(path,*args)
            if Path(path).name=='root_0_shared_nominal.json':
                if damage=='value':body['value'][0]+=.125
                elif damage=='shape':body['shape']=[1,192]
                else:body['binding']['root_sha256']='a'*64
            return body
        monkeypatch.setattr(m.native,'read',changed)
    with pytest.raises(ValueError):m.native.validate_array_artifacts(OUTPUT,p,native_sha,ready,compiled,saved,r)


def test_actual_complete_saved_native_validators_and_4root8arm480():
    out=m.consume(OUTPUT,REPORT_SHA,PREPARED,PREPARE_SHA,REVIEW_SHA)
    assert [out[k] for k in ('root_denominator','arm_denominator','candidate_denominator')]==[4,8,480]
    assert len(out['selected_plans'])==8 and len(out['root_audits'])==4
    assert out['unique_resources']['predict_calls']==1328 and out['unique_resources']['selection_evaluations']==480
    assert out['unique_resources']['jacobian_192x10_calls']==160 and out['unique_resources']['flops'] is None
    assert all(len(p['commands'])==10 for p in out['selected_plans'])
    assert out['new_neural_calls']==out['new_physics_steps']==0
    assert out['P2_admission'] is out['physical_execution_authorized'] is out['independent_neural_replay'] is False
    assert out['full_resource_equal'] is out['hardware_cost_equivalence_certified'] is False
    for audit in out['root_audits']:
        comparison=audit['arm_resource_comparison']
        assert comparison['predict_frontier_equal'] is True and comparison['full_resource_equal'] is False
        assert 'flops' in comparison['unknown_axes'] and comparison['differences']
        assert [x['packets'] for x in audit['packet_ledgers']]==[164,84]


@pytest.mark.parametrize('damage',['missing_packet','duplicate_packet','wrong_return_scope','fake_flops',
    'fake_equal_resources','bool_count','bad_wall','missing_candidate','failed_candidate'])
def test_packet_actual_return_join_and_no_hidden_cost_claims(actual,damage):
    row=deepcopy(actual[1]);result=row['results'][0];packet=result['ledger'][0]
    if damage=='missing_packet':result['ledger'].pop()
    elif damage=='duplicate_packet':result['ledger'][1]=deepcopy(packet)
    elif damage=='wrong_return_scope':row['actual_model_entries'][8]['scope']['path']=[1,0]
    elif damage=='fake_flops':packet['flops']=0
    elif damage=='fake_equal_resources':result['full_resource_equal']=True
    elif damage=='bool_count':packet['predict_calls']=True
    elif damage=='bad_wall':packet['wall_time_s']=-1
    elif damage=='missing_candidate':result['attempts'].pop()
    else:result['attempts'][0]['completed']=False
    with pytest.raises(ValueError):
        m.reconcile_packet_ledgers(row)


def test_cli_failed_report_keeps_declared_denominator_census_without_plans(tmp_path,monkeypatch,capsys):
    r=m.read(OUTPUT/'report.json');r['status']='failed_closed';r['census'][0]['candidates'][0]['status']='failed_or_pending'
    monkeypatch.setattr(m,'consume',lambda *a:(_ for _ in ()).throw(ValueError('pending operation')))
    monkeypatch.setattr(m,'read',lambda *a:r);monkeypatch.setattr(m,'sha',lambda *a:REPORT_SHA)
    written=[];monkeypatch.setattr(m.native,'write',lambda p,v:written.append(v))
    monkeypatch.setattr(m,'ROOT',tmp_path)
    path=tmp_path/'results'/(m.NAME+'_failure.json')
    code=m.main(['--input',str(OUTPUT),'--report-sha',REPORT_SHA,'--prepared',str(PREPARED),
        '--prepare-sha',PREPARE_SHA,'--review-sha',REVIEW_SHA,'--output',str(path)])
    assert code==1 and written[0]['candidate_denominator']==480 and written[0]['selected_plans']==[]
    assert written[0]['declared_census'][0]['candidates'][0]['status']=='failed_or_pending'
