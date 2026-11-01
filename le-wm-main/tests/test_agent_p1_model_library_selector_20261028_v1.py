"""RAM-only ABI tests, no production consume, neural model or environment calls."""
from copy import deepcopy
import json
import numpy as np
import pytest
from experiments.spectral_reachability import agent_p1_model_library_selector_20261028_v1 as m


def candidates():
    zero = np.zeros((5,2),'<f4')
    uz = np.arange(10,dtype='<f4').reshape(5,2)*np.float32(.01)+np.float32(.1)
    us = np.full((5,2),.2,'<f4'); bank = [zero,uz,-uz,us,-us]
    out = []
    for i in range(40):
        task,rem = divmod(i,10); role,word = divmod(rem,5)
        prefix = np.full((5,2),0. if role == 0 else .125,'<f4')
        p = dict(schema=m.teacher.NAME+'.model_plan',index=i,task=task,root_slot=task//2,
            condition=('off','on')[task%2],role=('Z','S')[role],library_index=word,library_label=m.teacher.LIBRARY[word],
            anchor_id=str(i//5),blind_sha256='b'*64,task_id=str(task),prefix_commands=prefix.astype(float).tolist(),
            suffix_commands=bank[word].astype(float).tolist(),commands=np.concatenate((prefix,bank[word])).astype(float).tolist(),
            raw_authority='literal_float32',teacher_context_sha256=str(i//5),model_Q=float((3,2,1,2,4)[word]),
            selection_uses_probe_response=False,selection_uses_physics=False,P1_admission=False)
        p['plan_digest'] = m.digest(p)
        w = dict(source_report_sha256=m.REPORT_SHA,source_receipt_sha256=m.RECEIPT_SHA,anchor_index=i//5,
            library_index=word,native_prediction=dict(file='native.npy.gz'),shifted_context=dict(file='shift.npy.gz'),
            endpoint_sha256='e'*64,goal_sha256='g'*64,Q=p['model_Q'],semantics='fixture')
        out.append(dict(original_plan=p,prediction_witness=w,prediction_digest=m.digest(w),
            full10_raw_bits=np.concatenate((prefix,bank[word])).tobytes().hex(),suffix_raw_bits=bank[word].tobytes().hex()))
    return out


def recost(c, value):
    p,w = c['original_plan'],c['prediction_witness']; p['model_Q']=value; w['Q']=value
    p['plan_digest']=m.digest({k:v for k,v in p.items() if k != 'plan_digest'})
    c['prediction_digest']=m.digest(w)


def test_all40_preserved_eight_ordered_selections_and_diagonal_identity():
    original = candidates(); before = deepcopy(original); a = m.select40(original)
    assert original == before and a['candidates'] == original
    assert a['fixed_candidates'] == 40 and a['fixed_selections'] == len(a['selections']) == 8
    assert [s['selected_original_index'] for s in a['selections']] == [5*i+2 for i in range(8)]
    for i,s in enumerate(a['selections']):
        assert len(s['fixed_candidate_costs']) == 5 and s['library_index'] == 2
        assert s['original_generated_diagonal']['library_index'] == (1 if i%2 == 0 else 3)
        assert s['original_generated_diagonal']['was_five_word_argmin_policy'] is False
        assert not s['physical_release'] and not s['P1_admission']
        assert s['selection_digest'] == m.digest({k:v for k,v in s.items() if k != 'selection_digest'})


def test_stable_argmin_equal_cost_keeps_lowest_original_index():
    c = candidates()
    for row in c: recost(row, 1.)
    a = m.select40(c)
    assert all(s['library_index'] == 0 and s['exact_minimum_tie_indices'] == [0,1,2,3,4] for s in a['selections'])
    assert len(a['candidates']) == 40


def test_exact_tie_not_epsilon_and_negative_zero_stable():
    c = candidates()
    recost(c[0],1e-300); recost(c[1],0.); recost(c[2],-0.)
    s = m.select40(c)['selections'][0]
    assert s['library_index'] == 1 and s['exact_minimum_tie_indices'] == [1,2]


def test_selected_raw_suffix_five_not_full10_prefix_replay():
    c = candidates(); s = m.select40(c)['selections'][1]; p=c[7]['original_plan']
    assert s['commands'] == p['suffix_commands'] and s['original_full10_commands'] == p['commands']
    assert m.raw(s['commands'],5).tobytes().hex() == s['commands_raw_bits']
    assert s['commands_dtype'] == '<f4' and s['commands_shape'] == [5,2]
    assert s['selected_original_plan_digest'] == p['plan_digest']
    assert s['selected_prediction_digest'] == c[7]['prediction_digest']


@pytest.mark.parametrize('cost',[None,float('nan'),float('inf'),-1.,True])
def test_missing_invalid_cost_no_subset_fallback(cost):
    c = candidates(); c[3]['original_plan']['model_Q']=cost
    with pytest.raises(ValueError): m.select40(c)


@pytest.mark.parametrize('mutation',['drop','reorder','label','role','root','plan_digest','prediction_digest','prediction_Q',
    'source_report','context','prefix','rawbits','zero','inverse','cross_role','physical_label','plan_label','witness_label'])
def test_full_model_abi_mutation_rejection(mutation):
    c = candidates()
    if mutation == 'drop': c.pop()
    elif mutation == 'reorder': c[0],c[1]=c[1],c[0]
    elif mutation == 'label': c[1]['original_plan']['library_label']='other'
    elif mutation == 'role': c[1]['original_plan']['role']='S'
    elif mutation == 'root': c[1]['original_plan']['root_slot']=1
    elif mutation == 'plan_digest': c[1]['original_plan']['plan_digest']='0'*64
    elif mutation == 'prediction_digest': c[1]['prediction_digest']='0'*64
    elif mutation == 'prediction_Q': c[1]['prediction_witness']['Q']=99.
    elif mutation == 'source_report': c[1]['prediction_witness']['source_report_sha256']='0'*64
    elif mutation == 'context': c[1]['original_plan']['teacher_context_sha256']='other'; recost(c[1],2.)
    elif mutation == 'prefix': c[1]['original_plan']['prefix_commands'][0][0]=.125; recost(c[1],2.)
    elif mutation == 'rawbits': c[1]['suffix_raw_bits']='00'
    elif mutation in ('zero','inverse','cross_role'):
        i = {'zero':0,'inverse':2,'cross_role':6}[mutation]
        p = c[i]['original_plan']; a=m.raw(p['suffix_commands'],5).copy(); a[0,0]=np.float32(.25)
        p['suffix_commands']=a.astype(float).tolist(); p['commands']=p['prefix_commands']+p['suffix_commands']
        c[i]['suffix_raw_bits']=a.tobytes().hex();c[i]['full10_raw_bits']=m.raw(p['commands'],10).tobytes().hex();recost(c[i],p['model_Q'])
    elif mutation == 'physical_label': c[1]['native_success']=True
    elif mutation == 'plan_label': c[1]['original_plan']['native_success']=True
    else: c[1]['prediction_witness']['actual_Q']=0.
    with pytest.raises(ValueError): m.select40(c)


def test_physical_missing_index12_is_not_a_model_cost_filter():
    c = candidates(); recost(c[12],0.)
    assert m.select40(c)['selections'][2]['selected_original_index'] == 12


def test_duplicate_raw_words_kept_in_five_cost_denominator():
    c = candidates()
    for base in range(0,40,5):
        for to,fr in ((3,1),(4,2)):
            c[base+to]['original_plan']['suffix_commands']=deepcopy(c[base+fr]['original_plan']['suffix_commands'])
            p=c[base+to]['original_plan'];p['commands']=p['prefix_commands']+p['suffix_commands']
            c[base+to]['suffix_raw_bits']=c[base+fr]['suffix_raw_bits']
            c[base+to]['full10_raw_bits']=m.raw(p['commands'],10).tobytes().hex();recost(c[base+to],p['model_Q'])
    assert len(m.select40(c)['candidates']) == 40


@pytest.mark.parametrize('value',[1.00001, float('nan'), .100000000001])
def test_literal_raw_no_silent_clip_round_or_nan(value):
    with pytest.raises(ValueError): m.raw([[value,0.]]*5,5)


def test_make_candidate_endpoint_only_Q_no_shifted_history_comparison(monkeypatch):
    p=deepcopy(candidates()[0]['original_plan']); p['model_Q']=.25
    native=np.full((3,192),.5,np.float32);shifted=native.copy();shifted[:2]=99.
    row=dict(index=0,blind_input=dict(goal=[0.]*192,q=np.eye(192).tolist(),context=dict(identity=1)),
        evaluations=[dict(stage='evaluate_0',status='completed',cost=.25,raw_commands=p['suffix_commands'],latents=dict(file='shift'))],
        forwards=[{}, {}, dict(stage='evaluate_0',modules=[dict(output=dict(file='native'))])])
    p['teacher_context_sha256']=m.digest(row['blind_input']['context'])
    monkeypatch.setattr(m.teacher,'load_array',lambda directory,r: native if r['file']=='native' else shifted)
    c=m.make_candidate(p,row,None)
    assert c['prediction_witness']['Q'] == .25
    assert c['prediction_digest'] == m.digest(c['prediction_witness'])
    shifted[-1]=.6
    with pytest.raises(ValueError): m.make_candidate(p,row,None)


def test_whole_json_roundtrip_verify_and_rehashed_selector_tamper(tmp_path):
    a=dict(schema=m.NAME+'.selector',**m.select40(candidates()))
    p=tmp_path/'selector.json';p.write_bytes(m.canonical(a))
    assert json.loads(p.read_text()) == a and m.verify(p,m.sha(p),a)['verified']
    a2=deepcopy(a);a2['selections'][0]['library_index']=4
    p.write_bytes(m.canonical(a2))
    with pytest.raises(ValueError): m.verify(p,m.sha(p),a)
    with pytest.raises(ValueError): m.verify(p,'0'*64,a)


def test_consume_rebuilds_not_only_hash_and_retains_no_release(tmp_path,monkeypatch):
    a=dict(schema=m.NAME+'.selector',**m.select40(candidates()));p=tmp_path/'selector.json';p.write_bytes(m.canonical(a))
    called=[]
    def build(): called.append(True);return a
    monkeypatch.setattr(m,'build',build)
    out=m.consume(p,m.sha(p))
    assert called == [True] and len(out['selected_plans']) == 8 and not out['physical_release']


def test_write_exclusive_and_contract_no_retroactive_policy_claim(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'ROOT',tmp_path);(tmp_path/'results').mkdir()
    p=tmp_path/'results'/(m.NAME+'_test.json');m.write(p,{})
    with pytest.raises(FileExistsError):m.write(p,{})
    with pytest.raises(ValueError):m.write(tmp_path/'results/MAIN.json',{})
    c=m.contract()
    assert c['original_policy_used_this_selector'] is False and c['fresh_confirmation'] is False
    assert c['total_FLOPs'] is None and not c['physical_release']


def test_build_full_model_native_schema_wiring_RAM_only(monkeypatch):
    c=candidates(); model_rows=[]
    for anchor in range(8):
        context=dict(anchor=anchor)
        r=dict(index=anchor,blind_input=dict(context=context,goal=[0.]*192,q=np.eye(192).tolist()),evaluations=[],forwards=[{},{}])
        for word in range(5):
            p=c[anchor*5+word]['original_plan'];p['teacher_context_sha256']=m.digest(context);recost(c[anchor*5+word],1.)
            r['evaluations'].append(dict(stage=f'evaluate_{word}',status='completed',cost=1.,
                raw_commands=p['suffix_commands'],latents=dict(file='shift')))
            r['forwards'].append(dict(stage=f'evaluate_{word}',modules=[dict(output=dict(file='native'))]))
        model_rows.append(r)
    report=dict(rows=model_rows,prepared='/ram/prepare.json',prepare_sha256='p'*64,artifact_sha256={})
    receipt=dict(plans=[x['original_plan'] for x in c],resources=dict(predict=96))
    preparation=dict(bindings={})
    def read(path,*args):
        return report if path==m.REPORT else receipt if path==m.RECEIPT else preparation
    invoked=[]
    def native_consume(directory,h): invoked.append((directory,h));return receipt
    monkeypatch.setattr(m.teacher,'read',read);monkeypatch.setattr(m.teacher,'consume',native_consume)
    monkeypatch.setattr(m,'sha',lambda *args:'a'*64)
    # Preserve heterogeneous external SHA checks without touching files.
    hashes={str(m.REPORT):m.REPORT_SHA,str(m.RECEIPT):m.RECEIPT_SHA,'/ram/prepare.json':'p'*64}
    monkeypatch.setattr(m,'sha',lambda path:hashes.get(str(path),'a'*64))
    monkeypatch.setattr(m.teacher,'load_array',lambda *args:np.ones((3,192),np.float32))
    result=m.build()
    assert invoked==[(m.REPORT.parent,m.REPORT_SHA)] and len(result['candidates'])==40
    assert [s['library_index'] for s in result['selections']]==[0]*8
    assert result['original_resources']==dict(predict=96)
    assert result['physical_labels_consumed_by_selector'] is False
