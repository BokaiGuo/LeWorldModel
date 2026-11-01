"""Saved-only phase-goal regression tests, using the real first branch schema."""
from copy import deepcopy
from functools import lru_cache
import pytest
from experiments.spectral_reachability import agent_p1_prospective_action_effect_20261028_v4 as m


@lru_cache(maxsize=1)
def actual():
    output=m.ROOT/'results/agent_p1_prospective_action_effect_20261028_v2_MAIN01'
    report=m.v2.teacher.read(output/'report.json',m.ACTUAL_SHA,m.v2.phys.REPORT_CAP)
    events=m.v2.read_events(output/'journal.jsonl')
    return output,report,events


def first():
    output,r,events=actual();row=deepcopy(r['rows'][0])
    es=deepcopy([e for e in events if e.get('job_digest')==row['job_digest']])
    return output,row,es


def rehash(events):
    # Test mutations are internally re-signed; chronology/payload must still reject.
    for e in events:e['event_sha256']=m.v2.digest({k:v for k,v in e.items() if k!='event_sha256'})


def test_actual_first_row_original_failure_and_all25_phase_labels():
    output,row,es=first();raw=[e for e in es if e['kind']=='raw_return']
    assert raw[0]['native_distance']==56.512040741808306
    assert m.v2.native_labels(row['goal'],raw[0]['state'])['native_distance']==242.84466686189273
    records=m.validate_branch(output,row,es)
    assert len(records)==25
    assert [x['event']['phase'] for x in records]==['warmup']*10+['prefix']*5+['suffix']*10
    assert records[0]['goal']!=row['goal'] and records[10]['goal']==row['goal']


@pytest.mark.parametrize('mutation',['attached_goal','row_goal','native_goal','missing_bind','extra_bind',
    'late_bind','early_bind','extra_setter','swap_raw','phase','snapshot','reward','distance',
    'success','terminated','truncated','goal_rgb','bind_snapshot'])
def test_actual_shape_rejects_mutation(mutation):
    output,row,es=first()
    attached=next(e for e in es if e['kind']=='goal_attached')
    recipe=next(e for e in es if e['kind']=='placement_recipe')
    bind=next(e for e in es if e['kind']=='operation_attempted' and e.get('resource')=='goal_bind')
    raw=[e for e in es if e['kind']=='raw_return']
    if mutation=='attached_goal':attached['goal'][5]+=1
    elif mutation=='row_goal':row['goal'][5]+=1
    elif mutation=='native_goal':recipe['native_goal'][5]+=1
    elif mutation=='missing_bind':es.remove(bind)
    elif mutation=='extra_bind':es.insert(es.index(bind),deepcopy(bind))
    elif mutation=='late_bind':bind['index']=raw[10]['index']+1
    elif mutation=='early_bind':bind['index']=raw[0]['index']-1
    elif mutation=='extra_setter':
        e=deepcopy(bind);e['resource']='goal_setter';e['index']=raw[-1]['index']+1;es.append(e)
    elif mutation=='swap_raw':
        i,j=es.index(raw[0]),es.index(raw[1]);es[i],es[j]=es[j],es[i]
    elif mutation=='phase':raw[0]['phase']='prefix'
    elif mutation=='snapshot':raw[0]['snapshot']['state'][0]+=1
    elif mutation=='reward':raw[0]['reward']+=1
    elif mutation=='distance':raw[0]['native_distance']+=1
    elif mutation=='success':raw[0]['native_success']=True
    elif mutation=='terminated':raw[0]['terminated']=True
    elif mutation=='truncated':raw[0]['truncated']=True
    elif mutation=='goal_rgb':attached['goal_rgb_sha256']='0'*64
    elif mutation=='bind_snapshot':attached['snapshot']['state'][0]+=1
    rehash(es)
    with pytest.raises(ValueError):m.validate_branch(output,row,es)


def test_complete1000_checked_including_saved_native_info_goal_state():
    output,r,es=actual();records,summary=m.validate_phases(output,r,es)
    assert len(records)==1000 and summary==dict(rows=40,raw_returns=1000,warmup=400,prefix=200,suffix=400)


def test_event_adapter_uses_actual_event_and_order_not_goal_state_key():
    output,row,es=first();records=m.validate_branch(output,row,es)
    call,finish=m.event_adapter(records)
    for record in records:
        assert call(row,deepcopy(record['event']))==m.v2.native_labels(record['goal'],record['event']['state'])
    assert finish()==25
    with pytest.raises(ValueError):call(row,records[-1]['event'])


def test_same_state_across_phases_remains_distinct_authenticated_calls():
    output,row,es=first();records=m.validate_branch(output,row,es)
    a,b=deepcopy(records[0]),deepcopy(records[10])
    b['event']['state']=deepcopy(a['event']['state'])
    b['event']['event_sha256']=m.v2.digest({k:v for k,v in b['event'].items() if k!='event_sha256'})
    # Adapter unit test: already-validated records with a repeated state; no state lookup.
    call,finish=m.event_adapter([a,b])
    assert call(row,a['event'])!=call(row,b['event'])
    assert finish()==2


@pytest.mark.parametrize('mutation',['order','job','event_digest','row_goal','skip'])
def test_adapter_rejects_wrong_event_or_unused_calls(mutation):
    output,row,es=first();records=m.validate_branch(output,row,es);call,finish=m.event_adapter(records)
    e=deepcopy(records[0]['event'])
    if mutation=='order':e=deepcopy(records[1]['event'])
    elif mutation=='job':e['job_digest']='0'*64
    elif mutation=='event_digest':e['event_sha256']='0'*64
    elif mutation=='row_goal':row['goal'][0]+=1
    with pytest.raises(ValueError):
        if mutation=='skip':finish()
        else:call(row,e)


def test_ast_only_rewrites_the_single_native_labels_caller():
    import ast,inspect
    tree,count=m.adapted_tree()
    assert count==1
    original=ast.parse(inspect.getsource(m.v2.consume))
    # The inverse substitution must recover the identical complete original AST.
    class Undo(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='phase_labels':
                return ast.parse("native_labels(row['goal'],e['state'])",mode='eval').body
            return self.generic_visit(node)
    assert ast.dump(Undo().visit(tree),include_attributes=False)==ast.dump(original,include_attributes=False)


@pytest.mark.parametrize('field',['goal_state','goal_proprio','pos_agent','vel_agent','block_pose'])
def test_independent_info_payload_mismatch_rejected(field,monkeypatch):
    output,row,es=first();original=m.native_info
    def changed(path,event):
        info=original(path,event)
        if event['phase']=='warmup' and event['step']==1:info[field][0]+=1
        return info
    monkeypatch.setattr(m,'native_info',changed)
    with pytest.raises(ValueError,match='native saved info'):
        m.validate_branch(output,row,es)


def test_native_info_hash_not_merely_trusted():
    output,row,es=first();event=next(e for e in es if e['kind']=='raw_return')
    event['info']['sha256']='0'*64
    with pytest.raises(ValueError,match='native info SHA'):m.native_info(output,event)


def test_task_goal_cannot_be_substituted_for_warmup_even_with_updated_distance_reward():
    output,row,es=first();raw=next(e for e in es if e['kind']=='raw_return')
    labels=m.v2.native_labels(row['goal'],raw['state'])
    raw.update(labels,reward=-labels['native_distance'],terminated=labels['native_success'])
    rehash(es)
    with pytest.raises(ValueError,match='independent full7'):m.validate_branch(output,row,es)


def test_phase_spoof_does_not_authorize_bind_before_warmup_finishes():
    output,row,es=first();attached=next(e for e in es if e['kind']=='goal_attached')
    raw=next(e for e in es if e['kind']=='raw_return')
    # Keep native labels unchanged; simply authenticating a phase name cannot suffice.
    attached['index']=raw['index']-1;rehash(es)
    with pytest.raises(ValueError):m.validate_branch(output,row,es)
