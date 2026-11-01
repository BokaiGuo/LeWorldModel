"""Assembler RAM fixtures only; no production synthesis/consumers/models/files."""
from copy import deepcopy
from types import SimpleNamespace
import ast
import json
import numpy as np
import pytest
from experiments.spectral_reachability import p1_moving_baseline_comparison_20261027_v1 as m


def fixture():
    raw=np.asarray([[.1,0]]*5+[[0,.1]]*5,np.float32)
    actions=((raw-np.asarray([-.0087,.0068],np.float32))/np.asarray([.2019,.2002],np.float32)).reshape(2,10).tolist()
    tasks=[dict(source_slot=s,task_id=m.TASK_IDS[s],task_sha256=m.TASK_SHAS[s],task_export=f'/approved/task_{s:02d}.json.gz',
        blind_sha256=str(s+1)*64,task_contract_sha256=str(s+3)*64,with_target=False,raw_history=raw.tolist(),goal=[1.]*192,
        root_context=dict(layout='states_with_between_state_actions_v1',context_length=3,latent_dim=192,action_dim=10,
            latent_history=[[0.]*192,[.125]*192,[.25]*192],action_history=deepcopy(actions))) for s in (0,1)]
    encoder=dict(schema=m.LOADER+'.report',status='moving_blind_tasks_encoded_and_sealed',errors=[],with_target=False,model_unchanged=True,
        resources={k:dict(attempted=8,completed=8) for k in ('encoder','projector')},
        rows=[dict(source_slot=s,task_id=m.TASK_IDS[s],task_file_sha256=m.TASK_SHAS[s]) for s in (0,1)])
    bundles=[]
    for p,spec in m.PROVIDERS.items():
        arms=list(spec['arms']);rows=[];plans=[]
        c=dict(with_target=False,per_root_budget={},information_contract='fixture predeclared protocol, no outcome selection')
        if p=='moving':c.update(ADforward_in_predict_per_root={m.ARMS[0]:8,m.ARMS[1]:11},mathematical_vjp_per_root={m.ARMS[0]:768,m.ARMS[1]:2112})
        elif p=='gradient':c.update(gradient_forward_predict_per_root=64,scalar_reverse_sweeps_per_root=32,jacobian=0,svd=0)
        for s in (0,1):
            for arm in arms:
                n=dict(zip(m.ARMS,(78,66,134,72,136,84)))[arm]
                resources={k:dict(attempted=n,completed=n) for k in ('predict','action_encoder')};api={}
                if p=='moving':
                    j,svd,gr=(4,4,768) if arm==m.ARMS[0] else (11,12,11)
                    resources.update(jacobian=dict(attempted=j,completed=j),svd=dict(attempted=svd,completed=svd))
                    api=dict(grad=dict(attempted=gr,completed=gr),jacobian=dict(attempted=0 if arm==m.ARMS[0] else 11,completed=0 if arm==m.ARMS[0] else 11))
                elif p=='gradient':resources['backward']=dict(attempted=32,completed=32);api=dict(grad=dict(attempted=32,completed=32))
                elif p=='kino':resources['pred_projector']=dict(attempted=84,completed=84)
                rows.append(dict(source_slot=s,arm=arm,task_id=tasks[s]['task_id'],status='model_plan_sealed',resources=resources,AD_API_resources=api))
                t=tasks[s];prediction=dict(t['root_context'],latent_history=[t['root_context']['latent_history'][1],t['root_context']['latent_history'][2],[.5]*192])
                plan={k:t[k] for k in ('source_slot','task_id','task_sha256','task_export','blind_sha256','task_contract_sha256')}
                plan.update(arm=arm,commands=[[.125,0.]]*10,with_target=False,selection_uses_physics=False,physical_execution=False,
                    prediction=prediction,model_terminal_cost=.25)
                if p=='kino':plan.update(terminal_cost=.25,path=[1,2])
                plans.append(plan)
        e=dict(provider=p,state='complete',arms=arms,report=dict(path=f'/actual/{p}/report.json',sha256='a'*64),consume=dict(path=f'/actual/{p}/consume.json',sha256='b'*64))
        r=dict(schema=spec['module']+'.report',status=spec['status'],errors=[],commands_released=True,source_verified=True,model_unchanged=True,
            model_hash_before='c'*64,model_hash_after='c'*64,contract=c,rows=rows)
        if p=='gradient':r['limits']=dict(elapsed_seconds=17.5)
        bundles.append(dict(entry=e,report=r,consume=dict(schema=spec['module']+'.consume',P1_admission=False,**{spec['consume_rows']:plans})))
    return tasks,encoder,bundles,deepcopy(m.template()['downstream'])


def assembled():return m.assemble(*fixture())


def test_complete_model_layer_all12_and_strict_JSON():
    r=assembled();assert r['status']=='model_layer_complete_physical_rgb_missing'
    assert len(r['rows'])==12 and len(r['summary'])==6
    assert r['denominator']==dict(algorithms=6,roots_per_algorithm=2,expected_model_rows=12,actual_model_rows=12,expected_physical_rows=12,actual_physical_rows=0)
    assert all(x['mean_model_Q']['value']==.25 and x['actual_model_roots']==2 for x in r['summary'])
    assert all(x['native_successes']['value'] is None and x['mean_realQ_terminal10']['value'] is None for x in r['summary'])
    assert json.loads(m.canonical(r))==r
    assert all(r[k] is False for k in ('P1_admission','P2_admission','independent_confirmation','official_reproduction','equal_total_compute','equal_search_domain'))


def test_forward_AD_not_added_twice_and_Jacobian_distinctions():
    r=assembled();rows={x['arm']:x for x in r['rows']}
    gn=rows[m.ARMS[0]]['resources']['value'];d2p=rows[m.ARMS[1]]['resources']['value'];gd=rows[m.ARMS[4]]['resources']['value']
    assert gn['forward_samples']['value']['completed']==78 and gn['AD_forward_in_predict']['value']==8
    assert gn['jacobian']['value']['completed']==4 and gn['jacobian_API']['value']['completed']==0
    assert gn['grad_API']['value']['completed']==768 and gn['mathematical_reverse_directions']['value']==768
    assert gn['backward']['value'] is None
    assert d2p['predict']['value']['completed']==66 and d2p['AD_forward_in_predict']['value']==11
    assert d2p['grad_API']['value']['completed']==11 and d2p['jacobian_API']['value']['completed']==11
    assert d2p['mathematical_reverse_directions']['value']==2112
    assert gd['predict']['value']['completed']==136 and gd['AD_forward_in_predict']['value']==64
    assert gd['backward']['value']==gd['grad_API']['value']==dict(attempted=32,completed=32)


def test_shared_encoding_once_standalone_each_not_summed():
    r=assembled();once=r['inherited_encoding_shared_once']['value']
    assert once=={k:dict(attempted=8,completed=8) for k in ('encoder','projector')}
    assert all(x['standalone_inherited_encoding']['value']==once for x in r['summary'])
    assert all('8+8 per standalone TWO-ROOT' in x['standalone_inherited_encoding']['basis'] for x in r['summary'])


def test_latency_run_scope_not_divided_or_duplicated():
    r=assembled();runs={x['provider']:x for x in r['runs']}
    assert runs['gradient']['latency_seconds']['value']==17.5
    assert runs['moving']['latency_seconds']['value'] is None
    assert all(x['latency_seconds']['value'] is None for x in r['rows'])
    assert len([x for x in r['runs'] if x['latency_seconds']['value'] is not None])==1


def test_null_latency_stays_unmeasured_not_zero():
    f=list(fixture());f[2][2]['report']['limits']['elapsed_seconds']=None
    r=m.assemble(*f);assert all(x['latency_seconds']['value'] is None for x in r['runs'])


def test_all_missing_model_still_full_denominator():
    tasks,enc,b,down=fixture();r=m.assemble(tasks,enc,[],down)
    assert r['denominator']['actual_model_rows']==0 and len(r['rows'])==12
    assert all(x['model_Q']['status']=='missing' and x['model_Q']['value'] is None for x in r['rows'])
    assert all(x['mean_model_Q']['value'] is None for x in r['summary'])
    assert 'missing' in m.markdown(r) and '0/2' in m.markdown(r)


def test_explicit_missing_reason_preserved():
    tasks,enc,b,down=fixture();b[3]=dict(entry=dict(provider='kino',state='missing',arms=[m.ARMS[-1]],reason='CUDA尚未实际运行'))
    r=m.assemble(tasks,enc,b,down);rows=[x for x in r['rows'] if x['arm']==m.ARMS[-1]]
    assert all(x['model_Q']['reason']=='CUDA尚未实际运行' for x in rows)
    assert r['denominator']['actual_model_rows']==10


def test_partial_is_not_complete_and_no_success_conversion():
    tasks,enc,b,down=fixture();part=b[2];part['entry'].pop('consume');part['entry'].update(state='partial',reason='timeout')
    part.pop('consume');part['report'].update(status='partial_or_failed',commands_released=False,errors=[dict(message='timeout')])
    part['report']['rows'][0]['resources']['predict']=dict(attempted=6,completed=5)
    part['report']['rows'][1]['resources']={};part['report']['rows'][1]['status']='not_started'
    r=m.assemble(tasks,enc,b,down);a=next(x for x in r['summary'] if x['arm']==m.ARMS[4])
    assert a['actual_model_roots']==0 and a['partial_model_roots']==2 and a['mean_model_Q']['value'] is None
    row=next(x for x in r['rows'] if x['arm']==m.ARMS[4]);assert row['resources']['reported_counters']['predict']==dict(attempted=6,completed=5)
    assert row['resources']['value'] is None and row['commands_sha256']['value'] is None
    assert row['native_success']['value'] is None


@pytest.mark.parametrize('edit',['old_task','task_sha','target_on','zero_history','zero_tokens','swap','model_cost','short_commands','wrong_blind','duplicate','missing_root','negative_counter','bool_counter','nonfinite_time'])
def test_strict_invalid_input_rejected(edit):
    tasks,enc,b,down=fixture()
    if edit=='old_task':tasks[0]['task_id']='f'*64
    elif edit=='task_sha':tasks[0]['task_sha256']='f'*64
    elif edit=='target_on':tasks[0]['with_target']=True
    elif edit=='zero_history':tasks[0]['raw_history']=[[0.,0.]]*10
    elif edit=='zero_tokens':tasks[0]['root_context']['action_history']=[[0.]*10]*2
    elif edit=='swap':tasks.reverse()
    elif edit=='model_cost':b[0]['consume']['plans'][0]['model_terminal_cost']=0.
    elif edit=='short_commands':b[0]['consume']['plans'][0]['commands'].pop()
    elif edit=='wrong_blind':b[0]['consume']['plans'][0]['blind_sha256']='f'*64
    elif edit=='duplicate':b.append(deepcopy(b[0]))
    elif edit=='missing_root':b[0]['report']['rows'].pop()
    elif edit=='negative_counter':b[0]['report']['rows'][0]['resources']['predict']['completed']=-1
    elif edit=='bool_counter':b[0]['report']['rows'][0]['resources']['predict']['attempted']=True
    else:b[2]['report']['limits']['elapsed_seconds']=float('nan')
    with pytest.raises(ValueError):m.assemble(tasks,enc,b,down)


def test_encoder_must_be_actual8_same_tasks_targetfree():
    for key in ('with_target','resources','rows'):
        f=list(fixture())
        if key=='with_target':f[1][key]=True
        elif key=='resources':f[1][key]['encoder']['completed']=7
        else:f[1][key].reverse()
        with pytest.raises(ValueError):m.assemble(*f)


def test_template_valid_and_pending_schema_cannot_silently_connect():
    v=m.template();m.manifest_check(v)
    v['downstream']['physical']=dict(state='actual',reason='invented')
    with pytest.raises(ValueError,match='schema'):m.manifest_check(v)


@pytest.mark.parametrize('edit',['unknown_arm','duplicate_arm','duplicate_provider','extra','wrong_encoder','wrong_order'])
def test_manifest_failclosed(edit):
    v=m.template()
    if edit=='unknown_arm':v['sources'][0]['arms']=['gradient_MPC']
    elif edit=='duplicate_arm':v['sources'][0]['arms']*=2
    elif edit=='duplicate_provider':v['sources'].append(deepcopy(v['sources'][0]))
    elif edit=='extra':v['sources'][0]['terminal_witness']=0
    elif edit=='wrong_encoder':v['encoder']['sha256']='f'*64
    else:v['sources'][0]['arms'].reverse()
    with pytest.raises(ValueError):m.manifest_check(v)


def test_sampling_single_arm_invocations_supported_no_overwrite():
    v=m.template();source=v['sources'].pop(1)
    for a in m.ARMS[2:4]:v['sources'].append(dict(source,arms=[a]))
    m.manifest_check(v)


def test_chinese_markdown_and_missing_are_not_zero_failure():
    r=assembled();md=m.markdown(r)
    assert '两根平均modelQ' in md and 'missing不代表0' in md and '不可均摊' in md
    assert '16/16' not in md and '6/6' not in md
    assert all(x['native_success']['value'] is None for x in r['rows'])


def test_IO_load_sources_calls_ONLY_own_saved_consumers(monkeypatch):
    tasks,enc,b,down=fixture();e=b[0]['entry'];lookup={e['report']['path']:b[0]['report'],e['consume']['path']:b[0]['consume']};called=[]
    monkeypatch.setattr(m,'load_ref',lambda ref:deepcopy(lookup[ref['path']]))
    monkeypatch.setattr(m,'sha',lambda path:m.PROVIDERS['moving']['sha'])
    def consume(path,h):called.append((str(path),h));return deepcopy(b[0]['consume'])
    monkeypatch.setattr(m.importlib,'import_module',lambda name:SimpleNamespace(consume=consume))
    loaded=m.load_sources(dict(sources=[e]));assert loaded[0]['consume']==b[0]['consume']
    assert called==[('/actual/moving','a'*64)]


def test_IO_saved_ownconsume_disagreement_fails(monkeypatch):
    tasks,enc,b,down=fixture();e=b[0]['entry']
    monkeypatch.setattr(m,'load_ref',lambda ref:deepcopy(b[0]['report'] if ref==e['report'] else b[0]['consume']))
    monkeypatch.setattr(m,'sha',lambda path:m.PROVIDERS['moving']['sha'])
    monkeypatch.setattr(m.importlib,'import_module',lambda name:SimpleNamespace(consume=lambda *a:dict(schema='wrong')))
    with pytest.raises(ValueError,match='ownconsume'):m.load_sources(dict(sources=[e]))


def test_partial_loader_never_calls_a_consumer(monkeypatch):
    tasks,enc,b,down=fixture();e=deepcopy(b[2]['entry']);e.pop('consume');e.update(state='partial',reason='failed')
    monkeypatch.setattr(m,'load_ref',lambda ref:deepcopy(b[2]['report']))
    def forbidden(*a):raise AssertionError('partial must not consume')
    monkeypatch.setattr(m.importlib,'import_module',forbidden)
    assert 'consume' not in m.load_sources(dict(sources=[e]))[0]


def test_read_duplicate_nonfinite_and_reference_SHA_RAM(monkeypatch):
    monkeypatch.setattr(m.Path,'is_file',lambda p:True);monkeypatch.setattr(m.Path,'is_symlink',lambda p:False)
    monkeypatch.setattr(m.Path,'stat',lambda p:SimpleNamespace(st_size=30))
    monkeypatch.setattr(m.Path,'read_bytes',lambda p:b'{"a":1,"a":2}')
    with pytest.raises(ValueError,match='duplicate'):m.read('/ram.json')
    monkeypatch.setattr(m.Path,'read_bytes',lambda p:b'{"a":NaN}')
    with pytest.raises(ValueError):m.read('/ram.json')
    monkeypatch.setattr(m,'sha',lambda p:'a'*64)
    with pytest.raises(ValueError):m.reference(dict(path='/ram.json',sha256='b'*64))


def test_source_has_no_execution_or_arbitrary_schema_dispatch():
    source=m.Path(m.__file__).read_text();tree=ast.parse(source)
    imports=[x.name for n in ast.walk(tree) if isinstance(n,ast.Import) for x in n.names]
    assert 'torch' not in imports and 'subprocess' not in imports
    attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
    assert 'step' not in attrs and 'run_model' not in attrs and 'load_compatible_pretrained' not in attrs
    assert 'if __name__' in source and 'physical/RGB' not in m.PROVIDERS


def test_pure_assembler_also_rejects_unfrozen_downstream_actual():
    f=list(fixture());f[3]['rgb']=dict(state='actual',reason='not an accepted schema')
    with pytest.raises(ValueError,match='schema'):m.assemble(*f)


@pytest.mark.parametrize('case',['floor','cap'])
def test_output_limits_fail_before_any_write_RAM(monkeypatch,case):
    tasks,encoder,bundles,downstream=fixture();manifest=m.template();manifest['sources']=[]
    monkeypatch.setattr(m,'load_ref',lambda ref:deepcopy(manifest if ref['path']=='/ram/manifest.json' else encoder))
    monkeypatch.setattr(m,'task_inputs',lambda:deepcopy(tasks))
    monkeypatch.setattr(m,'load_sources',lambda value:[])
    monkeypatch.setattr(m,'reference',lambda ref:m.Path(ref['path']))
    bindings={t['task_export']:t['task_sha256'] for t in tasks}
    bindings[str(m.ROOT/'experiments/spectral_reachability'/(m.LOADER+'.py'))]=m.LOADER_SHA
    monkeypatch.setattr(m,'sha',lambda p:bindings.get(str(p),'a'*64))
    monkeypatch.setattr(m.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    def forbidden(*a,**k):raise AssertionError('no write before limits')
    monkeypatch.setattr(m.Path,'mkdir',forbidden)
    if case=='cap':monkeypatch.setattr(m,'markdown',lambda result:'x'*(m.CAP+1))
    with pytest.raises(ValueError,match='reserve' if case=='floor' else 'cap'):
        m.run('/ram/manifest.json','a'*64,'/__ram_output_must_not_exist__')


def test_assembled_payload_small_and_deterministic():
    a=assembled();b=assembled()
    assert m.canonical(a)==m.canonical(b)
    assert len(m.canonical(a))+len(m.markdown(a).encode())<m.CAP
