"""RAM delimiter mutations and validator inheritance; no neural/physics."""
import ast
from copy import deepcopy
import hashlib
import inspect

import pytest

from experiments.spectral_reachability import agent_p2_contextual_composition_saved_consume_20261028_v2 as m


def record(value):return m.producer.canonical(value)+b'\n'


def test_actual_canonical_ABI_and_strict_record_blank_pairs():
    assert m.producer.canonical({'x':1})==b'{"x":1}\n'
    raw=record({'x':1})+record({'string':'escaped\nnewline','zero':-0.0})
    events,e=m.framed_events(raw)
    assert events==[{'x':1},{'string':'escaped\nnewline','zero':-0.0}]
    assert e['decoded_sha256']==hashlib.sha256(raw).hexdigest()
    assert e['decoded_bytes']==len(raw) and e['record_count']==e['empty_separator_count']==2
    assert e['splitline_count']==4 and e['strictly_alternating_empty']
    assert e['original_bytes_preserved'] and not e['arbitrary_blank_filter']


@pytest.mark.parametrize('raw',[
    b'',b'\n',b'\n\n',b'{"x":1}\n',b'{"x":1}\n\n\n',
    b'\n{"x":1}\n\n',b'{"x":1}\n{"x":2}\n\n',
    b'{"x":1}\n \n',b'{"x":1}\r\n\r\n',
    b'{"x":1}\n\nBROKEN\n\n',b'{"x":1}\n\n{"x":2}\n',
    b'{"x":1}\n\n\n\n{"x":2}\n\n',b'{"x":NaN}\n\n',
    b'{"x":1,"x":2}\n\n',b'{ "x":1}\n\n',b'[]\n\n',
    b'null\n\n',b'{"x":1}\n\n\xff\n\n'])
def test_reject_extra_missing_blank_or_nonempty_bad_record(raw):
    with pytest.raises((ValueError,UnicodeDecodeError)):
        m.framed_events(raw)


def test_all_original_consume_checks_AST_preserved():
    """Only explicit parser/candidate ABI seams and return packaging differ."""
    old=ast.parse(inspect.getsource(m.producer.consume)).body[0]
    new=ast.parse(inspect.getsource(m.consume_native)).body[0]
    old.name=new.name
    replaced=0
    for i,node in enumerate(old.body):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='events':
            old.body[i]=ast.parse('events,framing=framed_events(raw)').body[0];replaced+=1
    assert replaced==1
    # Narrow, literal replacements: preserve every assertion and native helper.
    text=ast.unparse(old)
    assert text.count('check_module_event_order(own)')==1
    text=text.replace('check_module_event_order(own)',"check_module_event_order(tagged_events(own, row['results']))")
    assert text.count("for e in emitted")==1
    text=text.replace("for e in emitted", "for e in tagged_events(emitted, row['results'])")
    old=ast.parse(text).body[0]
    terminal=old.body.pop();assert isinstance(terminal,ast.Return)
    old.body.append(ast.Assign(targets=[ast.Name(id='native',ctx=ast.Store())],value=terminal.value))
    old.body.append(ast.parse('return native,framing').body[0])
    assert ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False)


def candidates():
    results=[]
    for arm,n in enumerate((80,40)):
        attempts=[]
        for i in range(n):
            c=dict(index=i,candidate_id=f'{arm}_{i}',completed=True,actions=[[0.,-0.]]*10,
                logits=[0.]*20,cost=float(i),metric_attempted=True,selection_wall_time_s=.001,
                witness=dict(edges=[],receipt=dict(terminal_context='RAM ONLY')))
            if arm==1:c['suffix_compilation']=dict(nominal=[0.]*192,jacobian='RAM ONLY')
            attempts.append(c)
        results.append(dict(attempts=attempts))
    return results


def test_exact_untagged_candidate_terminal_view():
    r=candidates(); terminals=[dict(kind='native_event',event=c) for arm in r for c in arm['attempts']]
    typed=dict(kind='native_event',event=dict(kind='array_saved',receipt={'x':1}))
    module=dict(kind='module_return',receipt={})
    assert m.tagged_events([typed,*terminals,module],r)==[typed,module]


def test_real_order_helper_with_representative_untagged_terminal_ABI():
    r=candidates();typed=[]
    for j,scope in enumerate(m.frontier.expected_scopes()[::2]):
        for kind,resource,index in [
            ('module_attempt','action_encoder',3*j),('entry_attempt','action_encoder',2*j),
            ('entry_terminal','action_encoder',2*j),('module_return','action_encoder',3*j),
            ('entry_attempt','predict',2*j+1),('module_attempt','predictor',3*j+1),
            ('module_return','predictor',3*j+1),('module_attempt','pred_proj',3*j+2),
            ('module_return','pred_proj',3*j+2),('entry_terminal','predict',2*j+1)]:
            if kind.startswith('module'):
                typed.append(dict(kind=kind,receipt=dict(resource=resource,index=index,scope=scope)))
            else:
                typed.append(dict(kind='native_event',event=dict(kind=kind,entry=dict(kind=resource,index=index,scope=scope))))
    terminals=[dict(kind='native_event',event=deepcopy(c)) for arm in r for c in arm['attempts']]
    own=typed[:10]+terminals+typed[10:]
    with pytest.raises(KeyError,match='kind'):m.producer.check_module_event_order(own)
    m.producer.check_module_event_order(m.tagged_events(own,r))
    own[-1]['event']['entry']['index']=-1
    with pytest.raises(ValueError,match='module hooks'):m.producer.check_module_event_order(m.tagged_events(own,r))


@pytest.mark.parametrize('mutation',['missing','extra','reorder','bad_record','kind','denominator','cost','signed_zero'])
def test_candidate_terminal_ABI_rejects_unknown_or_mismatched(mutation):
    r=candidates(); events=[dict(kind='native_event',event=deepcopy(c)) for arm in r for c in arm['attempts']]
    if mutation=='missing':events.pop()
    elif mutation=='extra':events.append(deepcopy(events[-1]))
    elif mutation=='reorder':events[0],events[1]=events[1],events[0]
    elif mutation=='bad_record':events[0]['event']={'unexpected':1}
    elif mutation=='kind':events.insert(0,dict(kind='native_event',event=dict(kind='UNKNOWN')))
    elif mutation=='denominator':r[1]['attempts'].pop()
    elif mutation=='cost':events[0]['event']['cost']=123.
    elif mutation=='signed_zero':
        r[0]['attempts'][0]['value']=0.;events[0]['event']['value']=-0.
    with pytest.raises(ValueError):m.tagged_events(events,r)


def test_source_runtime_and_actual_report_hashes_are_unchanged():
    assert m.sha(m.producer.__file__)==m.PRODUCER_SHA
    assert m.sha(m.DIRECTORY/'report.json')==m.REPORT_SHA


def test_receipt_writer_exclusive_small_stdout_no_overwrite(tmp_path):
    path=tmp_path/(m.STEM+'_RAM.json')
    value=dict(status='RAM_NOT_EVIDENCE',native_strict_receipt=dict(plans=[{}]*8),
        framing_evidence=dict(record_count=2))
    summary=m.write_receipt(path,value)
    assert path.read_bytes()==m.producer.canonical(value)
    assert summary['plans']==8 and summary['receipt_sha256']==m.sha(path)
    assert 'native_strict_receipt' not in summary and not summary['physical_release']
    with pytest.raises(ValueError):m.write_receipt(path,value)
    with pytest.raises(ValueError):m.write_receipt(tmp_path/'not_owned.json',value)


def test_bad_external_receipt_sha_rejected_before_analysis(tmp_path,monkeypatch):
    p=tmp_path/'RAM.json';p.write_bytes(b'{}')
    def forbidden(*a):raise AssertionError('must reject before reading source')
    monkeypatch.setattr(m,'analyze',forbidden)
    with pytest.raises(ValueError,match='receipt SHA'):m.verify_receipt(p,'0'*64)


def test_full_receipt_comparison_not_status_only(tmp_path,monkeypatch):
    saved=dict(planner_output='/RAM_ONLY',planner_report_sha256='1'*64,
        native_strict_receipt=dict(plans=[dict(raw_sha256='2'*64)]))
    p=tmp_path/'RAM.json';p.write_bytes(m.producer.canonical(saved))
    monkeypatch.setattr(m,'analyze',lambda *a:deepcopy(saved))
    assert m.verify_receipt(p,m.sha(p))==saved
    corrupt=deepcopy(saved);corrupt['native_strict_receipt']['plans'][0]['raw_sha256']='3'*64
    monkeypatch.setattr(m,'analyze',lambda *a:corrupt)
    with pytest.raises(ValueError,match='entire strict'):m.verify_receipt(p,m.sha(p))


def test_decompression_trailing_stream_rejected():
    import gzip
    raw=gzip.compress(record({'x':1}))+gzip.compress(record({'x':2}))
    with pytest.raises(ValueError):m.overlay.blind.inflate(raw,m.EXPANDED)
