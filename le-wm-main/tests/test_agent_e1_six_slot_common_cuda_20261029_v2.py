"""RAM and temporary-disk fixtures only; no model or CUDA execution."""
import hashlib
import json
import base64
from contextlib import nullcontext

import pytest
import numpy as np
import test_agent_experiment1_common_controller_20261028_v1 as fixture

from experiments.spectral_reachability import agent_e1_six_slot_common_cuda_20261029_v2 as m


class FakeModel:
    def load_state_dict(self, data, strict):
        self.state, self.strict = data, strict
        return [], []
    def to(self, device):
        self.device = device
        return self
    def eval(self):
        self.training = False
        return self
    def requires_grad_(self, value):
        self.grad = value
        return self


def test_same_buffers_load_and_hash():
    config, weights = b'{"architecture":"fixture"}', b'fixture weights\x00'
    seen = {}
    def instantiate(value):
        seen['config'] = value
        return FakeModel()
    def deserialize(buffer):
        seen['weights'] = buffer.read()
        return {'original': 1}
    model, receipt = m.load_bound_buffers(config, weights, instantiate, deserialize,
                                          lambda x: ({'remapped': x['original']}, 1), 'cuda:0')
    assert seen == {'config': {'architecture': 'fixture'}, 'weights': weights}
    assert model.state == {'remapped': 1} and not model.strict and not model.training and not model.grad
    assert receipt['weights_sha256'] == hashlib.sha256(seen['weights']).hexdigest()
    assert receipt['config_sha256'] == hashlib.sha256(config).hexdigest()


def test_key_mismatch_rejected():
    class Bad(FakeModel):
        def load_state_dict(self, data, strict):
            return ['missing'], []
    with pytest.raises(ValueError, match='checkpoint key'):
        m.load_bound_buffers(b'{}', b'x', lambda _: Bad(), lambda _: {}, lambda x: (x, 0), 'cuda:0')


def test_real_journal_two_events_and_mutation(tmp_path):
    out = tmp_path / 'exclusive'
    writer = m.Journal(out)
    writer.event({'kind': 'attempt', 'phase': 'controller', 'ticket': 0})
    writer.event({'kind': 'return', 'phase': 'controller', 'ticket': 0})
    writer.close()
    events, final = m.read_journal(out / 'events.jsonl')
    assert len(events) == 2 and events[1]['kind'] == 'return' and len(final) == 64
    with pytest.raises(ValueError):
        m.Journal(out)
    data = (out / 'events.jsonl').read_bytes().replace(b'controller', b'controlles', 1)
    (out / 'events.jsonl').write_bytes(data)
    with pytest.raises(ValueError, match='chain'):
        m.read_journal(out / 'events.jsonl')


@pytest.mark.parametrize('raw', [b'{}\n\n', b'{"a":1,"a":2}\n', b'{"x":NaN}\n', b'{}'])
def test_invalid_journal_rejected(tmp_path, raw):
    path = tmp_path / 'bad.jsonl'
    path.write_bytes(raw)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        m.read_journal(path)


def test_bound_file(tmp_path):
    path = tmp_path / 'source'
    path.write_bytes(b'abc')
    assert m.read_bound(path, m.sha(b'abc'), 3) == b'abc'
    with pytest.raises(ValueError):
        m.read_bound(path, m.sha(b'abc'), 2)
    with pytest.raises(ValueError):
        m.read_bound(path, m.sha(b'bad'), 3)


def test_full_controller_and_paid_replay_from_real_journal(tmp_path):
    task = fixture.task(0)
    writer = m.Journal(tmp_path / 'full')
    target = [0.0] * 192
    result, counts = m.controller(task, 0, m.core.ARMS[1], target, fixture.Backend(), writer.event, guard=lambda _: nullcontext())
    assert counts == m.phase.beam.budget(144)
    raw = result['candidates'][result['winner_index']]['commands']
    writer.close()
    events, _ = m.read_journal(tmp_path / 'full' / 'events.jsonl')
    assert m.verify_controller(task, 0, m.core.ARMS[1], target, result, events) == result
    events[-1]['index'] = 71 if events[-1]['index'] != 71 else 70
    with pytest.raises(ValueError):
        m.verify_controller(task, 0, m.core.ARMS[1], target, result, events)

    writer2 = m.Journal(tmp_path / 'replay')
    replay, counts2 = m.replay_fixture(task, raw, fixture.Backend(), writer2.event, guard=lambda _: nullcontext())
    writer2.close()
    replay_events, _ = m.read_journal(tmp_path / 'replay' / 'events.jsonl')
    assert counts2 == m.phase.beam.budget(4)
    assert m.phase.verify_saved_replay(task, raw, replay_events) == replay


def test_six_slot_tempdisk_barrier_and_saved_reconstruction(tmp_path):
    # Full fixed denominator and math; fake backend, never NN/physics.
    rows = []
    for index, (root, arm) in enumerate(m.core.SLOTS):
        task = fixture.task(root)
        out = tmp_path / str(index)
        writer = m.Journal(out)
        result, counts = m.controller(task, root, arm, [0.0] * 192, fixture.Backend(), writer.event, guard=lambda _: nullcontext())
        writer.close()
        events, _ = m.read_journal(out / 'events.jsonl')
        assert m.verify_controller(task, root, arm, [0.0] * 192, result, events) == result
        assert counts == m.phase.beam.budget(144)
        raw = result['candidates'][result['winner_index']]['commands']
        raw_sha = m.put_new(out, 'raw.json', m.canonical(raw))
        raw_reload = m.parse_canonical(m.read_bound(out / 'raw.json', raw_sha, 65536))
        replay_writer = m.Journal(tmp_path / ('r' + str(index)))
        replay, replay_counts = m.replay_fixture(task, raw_reload, fixture.Backend(), replay_writer.event, guard=lambda _: nullcontext())
        replay_writer.close()
        replay_events, _ = m.read_journal(replay_writer.path)
        assert m.phase.verify_saved_replay(task, raw_reload, replay_events) == replay
        assert replay_counts == m.phase.beam.budget(4)
        rows.append(dict(root=root, arm=arm, target=[0.0]*192,goal=task['goal'],
                         whole=replay['whole'], segmented=replay['segmented']))
    seal = m.put_new(tmp_path, 'all_replays.json', m.canonical(rows))
    loaded = m.parse_canonical(m.read_bound(tmp_path / 'all_replays.json', seal, 1024 * 1024))
    assert len(m.compare_replays(loaded, seal)) == 6
    with pytest.raises(ValueError, match='sealed'):
        m.compare_replays(loaded, '0' * 64)


def test_float32_exact_difference():
    a = np.array([0., 1.], dtype=np.float32)
    b = np.array([-0., np.nextafter(np.float32(1), np.float32(2))], dtype=np.float32)
    diff = m.f32_diff(a, b)
    assert diff == dict(shape=[2], bitwise_equal=False, max_abs=2**-23, max_ulp=1)


def test_compressed_store_full_six_consumer_and_tamper(tmp_path, monkeypatch):
    tasks = [fixture.task(0), fixture.task(1)]
    slots = [dict(source_slot=i, task_id=m.core.old.core.task_id(t), blind_sha256=str(i+2)*64,
                  task_export='fixture'+str(i), task_sha256='a'*64,task_contract_sha256='b'*64)
             for i,t in enumerate(tasks)]
    policy = tmp_path / 'policy'
    loaded = dict(config_sha256='c'*64,weights_sha256='d'*64,config_bytes=2,weights_bytes=4,
                  remapped_keys=0,strict_key_coverage=True)
    previous = dict(slots=slots,policy=str(policy),expected_model_hash='e'*64,
        bindings_sha256={str(policy/'config.json'):'c'*64,str(policy/'weights.pt'):'d'*64})
    raw = np.zeros(192,dtype='<f4').tobytes()
    targets = [dict(root_slot=r,arm=a,task_id=slots[r]['task_id'],blind_sha256=slots[r]['blind_sha256'],
                    task_sha256='a'*64,target_f32_le_base64=base64.b64encode(raw).decode(),target_data_sha256=m.sha(raw))
               for r,a in m.core.SLOTS]
    six = dict(rows=targets,target_seal_sha256=m.sha(m.canonical(targets)))
    binding = dict(expected_model_hash='e'*64)
    monkeypatch.setattr(m,'authorities',lambda *args: (six,previous))
    monkeypatch.setattr(m,'verify_manifest',lambda *args: binding)
    monkeypatch.setattr(m.core.old.sampling.old,'load_export',
                        lambda path, _: {'gn_task':tasks[int(path[-1])]})
    output = tmp_path / 'full_store'
    store = m.Store(output)
    store.put('targets.z',targets)
    store.event(dict(kind='all_targets_sealed_before_model',sha256=six['target_seal_sha256']))
    store.event(dict(kind='model_loaded',loaded=loaded,model_hash='e'*64,cuda=True,tf32=False,autocast=False,math_requested=True))
    rows = [dict(root=r,arm=a,status='not_started',controller_resources=None,controller_samples=None,
                 replay_resources=None,replay_samples=None) for r,a in m.core.SLOTS]
    m.execute_groups(six,previous,store,lambda task:fixture.Backend(),lambda backend,meter:nullcontext(),
                     lambda:None,lambda:None,rows)
    assert store.index == 9382 and store.puts == 27
    store.close()
    report = dict(schema=m.NAME+'.report',status='all_six_sealed',prepared='fixture',prepare_sha256='f'*64,
        rows=rows,errors=[],artifacts={p.name:m.sha(p.read_bytes()) for p in output.iterdir()},
        journal_events=store.index,journal_sha256=store.previous,array_bytes=store.archive.size,
        metadata_bytes=store.meta,model_hash_before='e'*64,model_hash_after='e'*64,loaded=loaded,
        host=None,gpu=dict(peak_allocated=1,peak_reserved=2,cap=4*1024**3,requested_math=True,
            observed_math=None,tf32_matmul=False,tf32_cudnn=False,autocast=False),
        storage=m.storage_bound(),elapsed_before_report=1.,scientific_admission=False,
        training=False,new_physics=0,new_encoder=0,new_ad=0)
    report_sha=m.put_new(output,'report.json',m.canonical(report))
    completion=dict(schema=m.NAME+'.completion',report_sha256=report_sha,
                    timing_boundary='completion file and directory fsync measured by separate timing witness')
    completion_sha=m.put_new(output,'completion.json',m.canonical(completion))
    timing=dict(schema=m.NAME+'.timing',report_sha256=report_sha,completion_sha256=completion_sha,
                elapsed_after_completion_fsync=2.,wall_limit=m.WALL,wall_within_limit=True,
                timing_boundary='run entry through completion file and directory fsync; excludes timing witness fsync')
    timing_sha=m.put_new(output,'timing.json',m.canonical(timing))
    receipt=m.consume(output,report_sha,completion_sha,timing_sha)
    assert receipt['status']=='all_six_saved_controller_and_replays_verified'
    assert len(receipt['rows']) == len(receipt['scores']) == 6
    assert store.archive.size < store.meta  # actual fixture dedup, not a native bound
    late=dict(timing,elapsed_after_completion_fsync=901.,wall_within_limit=False)
    (output/'timing.json').write_bytes(m.canonical(late))
    with pytest.raises(ValueError,match='timing'):
        m.consume(output,report_sha,completion_sha,m.sha(m.canonical(late)))
    (output/'timing.json').write_bytes(m.canonical(timing))
    file=output/'slot_0.raw.z'
    file.write_bytes(file.read_bytes()+b'x')
    with pytest.raises(ValueError,match='artifact SHA'):
        m.consume(output,report_sha,completion_sha,timing_sha)


def test_store_lossless_array_dedup_and_hard_bound(tmp_path):
    s = m.Store(tmp_path / 'store')
    values = [float(i) for i in range(192)]
    s.event(dict(kind='one', values=values))
    n = s.archive.size
    s.event(dict(kind='two', values=values))
    assert s.archive.size == n == 192*8
    s.put('payload.z', {'values':values})
    s.close()
    report=dict(artifacts={p.name:m.sha(p.read_bytes()) for p in s.path.iterdir()},
                journal_events=s.index,journal_sha256=s.previous,array_bytes=s.archive.size,metadata_bytes=s.meta)
    m.put_new(s.path,'report.json',b'{}\n');m.put_new(s.path,'completion.json',b'{}\n');m.put_new(s.path,'timing.json',b'{}\n')
    payload, events = m.read_store(s.path,report)
    assert payload['payload.z']['values'] == events[0]['values'] == events[1]['values'] == values
    assert m.storage_bound()['worst_output_bytes'] <= m.CAP


def test_timing_sample_is_after_completion_real_write_and_late_fails(tmp_path):
    s = m.Store(tmp_path/'timed')
    observed = []
    original = s.write
    def wrapped(name, data, **kwargs):
        result = original(name, data, **kwargs)
        observed.append(name)
        return result
    s.write = wrapped
    def clock():
        assert observed == ['completion.json']
        assert (s.path/'completion.json').is_file()
        return s.started + 901.0
    completion,timing = m.publish_completion(s,'a'*64,clock)
    assert observed == ['completion.json','timing.json']
    assert timing['elapsed_after_completion_fsync'] == 901.0
    assert timing['wall_within_limit'] is False
    assert timing['completion_sha256'] == m.sha(m.canonical(completion))
    s.close()


def test_no_timing_witness_if_completion_write_fails(tmp_path):
    s = m.Store(tmp_path/'fault')
    def fail(name,data,**kwargs):
        raise OSError('fixture completion fsync failure')
    s.write = fail
    with pytest.raises(OSError):
        m.publish_completion(s,'a'*64)
    assert not (s.path/'timing.json').exists()


def test_saved_event_shape_audit_keeps_repeated_values():
    a=[float(x) for x in range(192)]
    audit=m.audit_packed_events([{'event':{'kind':'attempt','values':a}},
                                 {'event':{'kind':'return','values':a}}])
    assert audit['event_count']==2 and audit['archived_array_bytes']==192*8
    assert audit['by_kind']['attempt']['max_array_growth']==192*8
    assert audit['by_kind']['return']['max_array_growth']==0
    assert 'legacy events only' in audit['qualification']
