"""Pure saved-schema arithmetic fixtures; no physics/model or science evidence."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p5_contact_word_response_analysis_20261028_v1 as m

A=m.helper(m.ARITHMETIC)


def frame(f,w,block=False):
    k={'xy':1.,'yx':2.,'x_minus_x':-1.,'zero_zero':0.}[w] if f>10 else 0.
    p=dict(bodies=dict(agent=dict(position=[180.+k*(f-10),271.],velocity=[k,0.]),
        block=dict(position=[256.+(k if block else 0),256.],velocity=[k if block else 0.,0.],angle=.01*k if block else 0.,angular_velocity=0.)),
        env=dict(coverage_arr=[.2]*f,n_contact_points=0,success_threshold=.95,
            goal_pose=dict(dtype='<f8',shape=[3],values=[256.,256.,float(np.pi/4)]),latest_action=[0.,0.]))
    t=dict(kind='step',done=False,reward=float(.2/.95),info=dict(n_contacts=0,final_coverage=.2,max_coverage=.2))
    rgb=np.zeros((224,224,3),np.uint8); rgb[0,0,0]=int(abs(k)*25)
    return p,t,rgb


def fixture():
    x=np.tile(np.array([.1,0.],np.float64),(5,1)); y=x[:,::-1].copy(); z=np.zeros_like(x)
    arrays=(np.concatenate((x,y)),np.concatenate((y,x)),np.concatenate((x,-x)),np.concatenate((z,z)))
    plans=[dict(arm=w,reference_suffix_raw_u=a.tolist()) for w,a in zip(m.WORDS,arrays)]
    rows=[dict(arm=w,status='completed',error=None) for w in m.WORDS]
    model=dict(status='public_native_action_response_recorded',errors=[],finalization_errors=[],
        words=[dict(word=w,status='completed',error=None) for w in m.WORDS],
        contrasts=dict(xy_vs_yx_patch_mse=1.,x_minus_x_vs_zero_patch_mse=2.),meter=dict(RAM_token=True))
    def command(row,f):
        raw=plans[m.WORDS.index(row['arm'])]['reference_suffix_raw_u'][f-11]
        return dict(frame=f,raw_env_argument=raw,pixel_displacement=(100*np.asarray(raw)).tolist(),absolute_target=[0.,0.])
    return plans,rows,model,lambda row,f:frame(f,row['arm']),command,lambda row,f:[]


def assemble(data=None):
    return m.assemble(*(data or fixture()),A)


def test_full40_20_and_agent_not_object():
    result=assemble()
    assert len(result['baselines'])==4 and len(result['suffix_frame_rows'])==40 and len(result['pair_frame_rows'])==20
    assert [(x['word'],x['frame']) for x in result['suffix_frame_rows']]==[(w,f) for w in m.WORDS for f in range(11,21)]
    assert any(x['delta']['agent_xy']['l2']>0 for x in result['pair_frame_rows'])
    assert any(x['rgb']['changed_pixels']>0 for x in result['pair_frame_rows'])
    assert all(x['delta']['block_xy']['l2']==0 for x in result['pair_frame_rows'])
    assert not any(x['any_exact_block_pose_difference'] or x['object_noncommutativity_claim'] for x in result['pair_descriptive_summaries'])
    assert result['model_only_parallel']['meter']==dict(RAM_token=True)
    for x in result['suffix_frame_rows']:
        assert x['rgb_vs_own_prefix10']['object_attribution'].startswith('unknown')


def test_intermediate_object_difference_not_endpoint_only():
    data=list(fixture())
    data[3]=lambda row,f:frame(f,row['arm'],block=f==13)
    result=assemble(data)
    pairs=result['pair_frame_rows'][:10]
    assert pairs[2]['delta']['block_xy']['l2']>0 and pairs[-1]['delta']['block_xy']['l2']==0
    assert result['pair_descriptive_summaries'][0]['any_exact_block_pose_difference']
    assert not result['pair_descriptive_summaries'][0]['object_noncommutativity_claim']


@pytest.mark.parametrize('change',['missing','reordered','failed','missing_model_word','model_error','negative_patch','bad_raw','command','goal','baseline','done','contact_formula'])
def test_mutations(change):
    data=list(fixture()); plans,rows,model=data[:3]
    if change=='missing': rows.pop()
    elif change=='reordered': rows.reverse()
    elif change=='failed': rows[0]['status']='partial_failed'
    elif change=='missing_model_word': model['words'].pop()
    elif change=='model_error': model['errors']=['failure']
    elif change=='negative_patch': model['contrasts']['xy_vs_yx_patch_mse']=-1.
    elif change=='bad_raw': plans[0]['reference_suffix_raw_u'].pop()
    elif change=='command':
        orig=data[4]
        def command(row,f):
            c=deepcopy(orig(row,f)); c['raw_env_argument']=[1.,1.]; return c
        data[4]=command
    else:
        def load(row,f):
            p,t,img=frame(f,row['arm'])
            if row['arm']=='xy':
                if change=='goal' and f==11: p['env']['goal_pose']['values'][0]+=1
                if change=='baseline' and f==10: p['bodies']['agent']['position'][0]+=1
                if change=='done': t['done']=True
                if change=='contact_formula': t['info']['n_contacts']=2
            return p,t,img
        data[3]=load
    with pytest.raises(ValueError): assemble(data)


def callback(pair=('agent','block'),before=0):
    return dict(returned=True,counter_before=before,counter_after=before+1,body_roles=list(pair),
                points=[dict(point_a=[1.,2.],point_b=[1.,2.],distance=-.1)],total_impulse=[2.,3.])


def test_callback_pair_identity_not_unique_contact_or_force():
    events=[callback(),callback(('agent','static'),1)]
    result=m.contacts(events,2)
    assert result['callback_entries']==result['points']==2
    assert result['by_actual_body_pair']['agent:block']==dict(callback_entries=1,points=1)
    assert result['by_actual_body_pair']['agent:static']==dict(callback_entries=1,points=1)
    assert result['events']==events and 'total_force' not in result


@pytest.mark.parametrize('change',['return','before','after','role','points'])
def test_callback_mutations(change):
    e=callback(); points=1
    if change=='return': e['returned']=False
    elif change=='before': e['counter_before']=1
    elif change=='after': e['counter_after']=2
    elif change=='role': e['body_roles']=['invented','agent']
    else: points=2
    with pytest.raises(ValueError): m.contacts([e],points)


def test_actual_raw_order_and_net_displacement_are_separate():
    result=assemble()
    rows=result['suffix_frame_rows']
    assert rows[9]['cumulative_raw_u']==rows[19]['cumulative_raw_u']
    assert rows[0]['command']['raw_env_argument']!=rows[10]['command']['raw_env_argument']
    assert rows[29]['cumulative_raw_u'][0]==pytest.approx(0.)
    assert rows[39]['cumulative_raw_u']==[0.,0.]
    assert m.policy()['denominator']==4 and m.policy()['task_success'] is None


def test_angles_and_rgb_units_not_combined():
    left=A.quantities(*frame(11,'xy')[:2],11); right=deepcopy(left)
    left['block_angle_native_rad']=2*np.pi+.01
    d=A.difference(left,right)
    assert d['block_angle']['left_minus_right_native_rad']>6
    assert d['block_angle']['principal_left_minus_right_rad']==pytest.approx(.01)
    a=np.zeros((224,224,3),np.uint8); b=a.copy(); b[0,0,0]=255
    rgb=A.rgb_difference(a,b)
    assert rgb['max_abs_intensity']==255 and rgb['changed_channels']==1


def test_record_loader_original_layout_mutation(tmp_path):
    p=tmp_path/'xy/frame_11'; p.mkdir(parents=True)
    public,t,rgb=frame(11,'xy')
    for name,v in [('full_public_state.json',public),('transition.json',t)]: (p/name).write_bytes(m.canonical(v))
    with (p/'rgb.npy').open('xb') as f: np.save(f,rgb,allow_pickle=False)
    records=[dict(frame=f,files={}) for f in range(21)]
    records[11]['files']={x.name:dict(sha256=m.sha(x),bytes=x.stat().st_size) for x in p.iterdir()}
    load=m.record_loader(tmp_path,dict(arm='xy',records=records))
    assert np.array_equal(load(11)[2],rgb)
    (p/'transition.json').write_text('{}')
    with pytest.raises(ValueError,match='SHA/size'): load(11)


def test_no_actual_receipt_no_wrapper_import(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'load_bound',lambda *args:pytest.fail('must validate saved inputs before wrapper import'))
    with pytest.raises(ValueError,match='regular nonsymlink'):
        m.analyze(tmp_path/'m',m.MANIFEST_SHA,tmp_path/'r',m.REPORT_SHA,
            tmp_path/'c',m.CONSUMER_SHA,tmp_path/'receipt',m.RECEIPT_SHA)


@pytest.mark.parametrize('index',[1,3,5,7])
def test_external_actual_sha_required(tmp_path,index):
    args=[tmp_path/'m',m.MANIFEST_SHA,tmp_path/'r',m.REPORT_SHA,
          tmp_path/'c',m.CONSUMER_SHA,tmp_path/'receipt',m.RECEIPT_SHA]
    args[index]='0'*64
    with pytest.raises(ValueError,match='actual new four-word'): m.analyze(*args)


def test_authority_strict_return_must_match(tmp_path,monkeypatch):
    manifest=tmp_path/'m'; report=tmp_path/'r'; receipt=tmp_path/'receipt'
    for p,v in ((manifest,{}),(report,{}),(receipt,dict(status='all4_saved_native_words_verified',denominator=4,
        scientific_admission=False,task_success=None))): p.write_bytes(m.canonical(v))
    monkeypatch.setattr(m,'MANIFEST_SHA',m.sha(manifest)); monkeypatch.setattr(m,'REPORT_SHA',m.sha(report))
    monkeypatch.setattr(m,'RECEIPT_SHA',m.sha(receipt))
    monkeypatch.setattr(m,'load_bound',lambda *args:SimpleNamespace(consume=lambda *args:{}))
    with pytest.raises(ValueError,match='exact MAIN strict'):
        m.analyze(manifest,m.MANIFEST_SHA,report,m.REPORT_SHA,tmp_path/'consumer',m.CONSUMER_SHA,receipt,m.RECEIPT_SHA)


def test_strict_json_exclusive_write(tmp_path):
    p=tmp_path/'out.json'; m.write_new(p,dict(x=1))
    with pytest.raises(FileExistsError): m.write_new(p,dict(x=2))
    with pytest.raises(ValueError,match='FILE SHA'): m.read_json(p,'0'*64)
    p.write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError,match='duplicate'): m.read_json(p)
    p.write_text('{"x":NaN}')
    with pytest.raises(ValueError,match='nonfinite'): m.read_json(p)
    link=tmp_path/'link'; link.symlink_to(p)
    with pytest.raises(ValueError,match='nonsymlink'): m.sha(link)


def test_frozen_saved_authorities_only():
    for name,h in m.PINS.items(): assert m.sha(m.ROOT/'experiments/spectral_reachability'/(name+'.py'))==h
    report=m.ROOT/f'results/{m.PHYSICAL}_physics_MAIN01/report.json'
    assert m.sha(report)==m.REPORT_SHA
    receipt=m.read_json(m.ROOT/f'results/{m.PHYSICAL}_consume_MAIN01.json',m.RECEIPT_SHA)
    assert receipt['denominator']==4 and len(receipt['private_suffix_frame_rows'])==40
    assert m.sha(m.ROOT/'experiments/spectral_reachability/consume_contact_words_MAIN_20261028_v1.py')==m.CONSUMER_SHA
    before=set(sys.modules)
    m.helper(m.ARITHMETIC)
    assert not {x for x in set(sys.modules)-before if x.split('.')[0] in ('torch','pymunk','pygame')}


def test_cli_saved_write_verify_and_tamper(tmp_path,monkeypatch,capsys):
    result=dict(schema=m.TAG+'.report',RAM_fixture_only=True,scientific_admission=False)
    calls=[]
    def analyze(*args):
        calls.append(args)
        return result
    monkeypatch.setattr(m,'analyze',analyze)
    output=tmp_path/'analysis.json'
    args=[]
    for key in ('manifest','report','consumer','receipt'):
        args.extend(('--'+key,str(tmp_path/key),'--'+key+'-sha256','1'*64))
    args.extend(('--output',str(output)))
    assert m.main(['analyze',*args])==0
    assert m.read_json(output)==result and len(calls)==1
    h=m.sha(output)
    assert m.main(['verify',*args,'--output-sha256',h])==0 and len(calls)==2
    with pytest.raises(ValueError,match='exclusive new output'): m.main(['analyze',*args])
    with pytest.raises(ValueError,match='actual saved analysis SHA'): m.main(['verify',*args])
    with pytest.raises(ValueError,match='FILE SHA'): m.main(['verify',*args,'--output-sha256','0'*64])
    result['mutated']=True
    with pytest.raises(ValueError,match='recomputation mismatch'): m.main(['verify',*args,'--output-sha256',h])
    assert 'saved_analysis_verified' in capsys.readouterr().out


def test_documented_source_bundle_exists():
    assert (m.ROOT/'innerdocs'/(m.TAG+'.md')).is_file()
    assert m.policy()['scientific_admission'] is False
