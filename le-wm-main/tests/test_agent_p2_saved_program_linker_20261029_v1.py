"""Synthetic native saved IO only; never execute producer/fullstrict/model/physics."""
from copy import deepcopy
import gzip
import hashlib
import io

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p2_saved_program_linker_20261029_v1 as m


def blob(x):
    a = np.ascontiguousarray(x, dtype='<f4')
    return dict(shape=list(a.shape), dtype='float32', raw_hex=a.tobytes().hex(),
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def make_data():
    """Complete 2x42x2 data and native compiler/forward schema, arithmetic RAM callback."""
    p = dict(schema=m.PRODUCER+'.prepare', expected_model_hash='a'*64, slots=[])
    r = dict(schema=m.PRODUCER+'.report', prepared='/fixture/prepare.json', prepare_sha256='b'*64,
             status='all4_full15_plans_sealed_not_admitted', errors=[], first_error=None,
             model_before='a'*64, model_after='a'*64, model_unchanged=True, source_verified=True, rows=[])
    c = dict(schema=m.PRODUCER+'.consume', status='full15_engineering_verified_science_blocked',
             errors=[], planner_report_sha256='c'*64, plans=[])
    authorities = {k: dict(path='/fixture/'+k+'.json', sha256=h*64) for k, h in zip(m.CAPS, 'bcd')}
    authorities['prepared']['path'] = r['prepared']
    rows, arrays = [], {}
    for root in range(2):
        ctx = dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
                   action_dim=10, latent_history=np.full((3, 192), root, np.float32).tolist(),
                   action_history=np.zeros((2, 10), np.float32).tolist())
        slot = dict(root_slot=root, condition='off', task_id=m.digest(['task', root]), task_sha256=m.digest(['export', root]),
                    blind_sha256=m.digest(['blind', root]), packet=dict(root_context=ctx, goal=np.zeros(192).tolist()),
                    q=np.eye(192).tolist(), warm_raw10=np.zeros((10, 2)).tolist())
        p['slots'].append(slot)
        normal = dict(mean=[0., 0.], std=[1., 1.])
        for arm in m.ARMS:
            c['plans'].append(dict(root_slot=root, arm=arm, root_context=deepcopy(ctx), goal=slot['packet']['goal'],
                task_id=slot['task_id'], task_sha256=slot['task_sha256'], model_hash=p['expected_model_hash'],
                program_count=1 if arm == m.ARMS[0] else 3, normalization=normal,
                root_warm_raw10=slot['warm_raw10'], planner_report_sha256='c'*64, prepare_sha256='b'*64))
        row = dict(index=root, status='sealed', forwards=[], AD=[], result=dict(bank=dict(params=[]),
                   candidates={arm: [] for arm in m.ARMS}))
        def path(start, logits, tag):
            current, contexts = deepcopy(start), []
            for k, raw in enumerate(m.chart(logits, len(logits)//10)):
                idx = len(row['forwards'])
                tokens = np.concatenate((np.array(current['action_history'], np.float32), raw.reshape(1, 10)))
                z = np.asarray(current['latent_history'][-1], np.float32)+np.float32(.0001+float(raw.sum())*.01)
                out2 = np.zeros((3, 192), np.float32)
                out2[-1] = z
                nxt = deepcopy(current)
                nxt.update(latent_history=np.concatenate((np.array(current['latent_history'], np.float32)[1:], z[None])).tolist(),
                           action_history=tokens[1:].tolist())
                data = blob(raw)
                zero = np.zeros((1, 3, 192), np.float32)
                ins = [[tokens[None]], [np.array(current['latent_history'], np.float32)[None], zero], [zero.reshape(3, 192)]]
                outs = [zero, zero, out2]
                modules = [dict(index=3*idx+j, resource=res, completed=True, inputs=[blob(a) for a in xs], output=blob(y))
                           for j, (res, xs, y) in enumerate(zip(('action_encoder', 'predictor', 'pred_proj'), ins, outs))]
                tagk = tag+'/'+str(k)
                row['forwards'].append(dict(index=idx, tag=tagk, input=current, output=nxt, raw=data, modules=modules,
                    completed=True, request=dict(index=idx, tag=tagk, input_context_sha256=m.digest(current),
                                                 raw_sha256=m.digest(data), completed=False)))
                current = nxt
                contexts.append(current)
            return contexts
        def ad(start, logits, tag):
            index = len(row['AD'])
            arr = np.zeros((192, len(logits)), np.float32)
            stream = io.BytesIO()
            np.save(stream, arr, allow_pickle=False)
            name = f'root{root}_J{index:03}.npy'
            arrays[name] = stream.getvalue()
            desc = dict(file=name, sha256=hashlib.sha256(arrays[name]).hexdigest(), shape=list(arr.shape), dtype='float32')
            contexts = path(start, logits, tag+'/forward')
            req = dict(index=index, tag=tag, input_context_sha256=m.digest(start), at_logits=logits,
                       raw=m.chart(logits, len(logits)//10).tolist(), coordinates='logit_zero' if tag == 'root_J' else 'raw_incumbent',
                       shape=[192, len(logits)], mathematical_VJP=192, vectorize=True, completed=False)
            row['AD'].append(dict(request=req, contexts=contexts, array=desc, completed=True))
        ad(ctx, [0.]*30, 'root_J')
        for i in range(42):
            u = np.full(30, .001*i, np.float64).tolist()
            joint_path = path(ctx, u, f'joint/{i}')
            row['result']['bank']['params'].append(u)
            row['result']['candidates'][m.ARMS[0]].append(dict(index=i, logits=u, contexts=joint_path))
            first = path(ctx, u[:10], f'compose/{i}/first')[0]
            ad(first, u[10:], f'compose/{i}/edge2/J')
            midpath = path(first, u[10:], f'compose/{i}/edge2/selected_replay')
            ad(midpath[0], u[20:], f'compose/{i}/edge3/J')
            last = path(midpath[0], u[20:], f'compose/{i}/edge3/selected_replay')
            edge2 = dict(input_context=first, fixed_prefix_logits=u[:10], incumbent_logits=u[10:],
                         selected_logits=u[10:], selected_contexts=midpath)
            edge3 = dict(input_context=midpath[0], fixed_prefix_logits=u[:20], incumbent_logits=u[20:],
                         selected_logits=u[20:], selected_contexts=last)
            row['result']['candidates'][m.ARMS[1]].append(dict(index=i, logits=u, contexts=[first, midpath[0], last[0]],
                root_seed_sha256=m.digest(u), edge2=edge2, edge3=edge3))
        # Preserve 1068 actual-format entries; unreferenced modules remain inherited, not re-certified.
        while len(row['forwards']) < 1068:
            f = dict(row['forwards'][0], tag='unreferenced/'+str(len(row['forwards'])))
            row['forwards'].append(f)
        rows.append(row)
        r['rows'].append(dict(root_slot=root, file=f'root{root}.json.gz', candidate_denominator=[42, 42]))
    def load_array(d):
        return np.load(io.BytesIO(arrays[d['file']]), allow_pickle=False)
    return p, r, c, rows, authorities, arrays, load_array


@pytest.fixture(scope='module')
def data():
    return make_data()


@pytest.fixture(scope='module')
def catalog(data):
    p, r, c, rows, a, _, load = data
    return m.extract_programs(p, r, c, rows, a, load)


def test_full168_denominator_rawbits_and_program_not_packet_count(catalog, data):
    assert len(catalog['catalog']) == 336
    assert len(catalog['candidates']) == 168
    assert all(x['result']['status'] == 'linked_saved_actions_NOT_admitted' for x in catalog['candidates'])
    for entry in catalog['candidates']:
        linked = entry['result']
        original = data[3][entry['root_slot']]['result']['candidates'][entry['arm']][entry['candidate_index']]
        assert linked['raw_float32_bits'] == m.chart(original['logits'], 3).tobytes().hex()
        assert linked['packet_count'] == 3
        assert linked['program_count'] == (1 if entry['arm'] == m.ARMS[0] else 3)
        assert [x['step'] for x in linked['step_provenance']] == list(range(1, 16))
        assert not any(linked[k] for k in m.FLAGS)
    h2 = next(p for p in catalog['catalog'] if p['program_index'] == 1)
    assert (h2['planned_steps'], h2['committed_steps']) == (10, 5)
    root = next(p for p in catalog['catalog'] if p['arm'] == m.ARMS[1] and p['program_index'] == 0)
    assert (root['planned_steps'], root['committed_steps']) == (15, 5)
    assert not catalog['evidence']['upstream_fullstrict_rebuilt_here']


def test_two_program_prefix_can_be_linked_without_synthesizing_extra_steps(catalog):
    refs = catalog['candidates'][1]['refs'][:2]
    out = m.link_programs(catalog['catalog'], refs, expected_steps=10)
    assert out['program_count'] == 2 and len(out['commands']) == 10
    assert m.link_programs(catalog['catalog'], refs)['reason'] == 'incomplete_requested_horizon'


@pytest.mark.parametrize('kind,reason', [
    ('unknown', 'missing_program_reference'), ('crossroot', 'different_task_goal_model_normalizer_domain_or_root'),
    ('overlap', 'overlap_gap_or_wrong_absolute_program_interval'),
    ('crosscandidate', 'disconnected_full_context_not_terminal_only')])
def test_fixed_link_failures(catalog, kind, reason):
    refs = deepcopy(catalog['candidates'][1]['refs'])
    if kind == 'unknown': refs[1] = 'absent'
    elif kind == 'crossroot': refs[1] = refs[1].replace('r0/', 'r1/')
    elif kind == 'overlap': refs[1] = refs[0]
    elif kind == 'crosscandidate': refs[1] = refs[1].replace('c00', 'c01')
    out = m.link_programs(catalog['catalog'], refs)
    assert out['status'] == 'blocked' and out['reason'] == reason
    assert out['raw_float32_bits'] is out['commands'] is None


def reseal(p):
    p['program_sha256'] = m.digest({k: v for k, v in p.items() if k != 'program_sha256'})


def test_terminal_same_but_history_different_is_not_a_seam(catalog):
    programs = deepcopy(catalog['catalog'][:4])
    # Same terminal192, different causal predecessor/action token.
    programs[2]['input_context']['action_history'][0][0] = .25
    reseal(programs[2])
    out = m.link_programs(programs, [p['id'] for p in programs[1:]])
    assert out['reason'] == 'disconnected_full_context_not_terminal_only'


def test_compiler_from_another_context_cannot_be_substituted(catalog):
    programs = deepcopy(catalog['catalog'][:4])
    programs[2]['compiler']['input_context_sha256'] = '0'*64
    reseal(programs[2])
    with pytest.raises(ValueError, match='compiler'):
        m.link_programs(programs, [p['id'] for p in programs[1:]])


def test_program_capsule_tamper_rejected(catalog):
    programs = deepcopy(catalog['catalog'][:4])
    programs[2]['commands'][0][0] = .5
    with pytest.raises(ValueError, match='immutable'):
        m.link_programs(programs, [p['id'] for p in programs[1:]])


def test_actual_missing_transition_keeps_complete_candidate_denominator(data):
    p, r, c, rows, a, _, load = data
    rows = list(rows)
    row = dict(rows[0])
    row['forwards'] = [dict(f, tag='not_the_required_tag') if f['tag'] == 'compose/0/edge3/selected_replay/0' else f
                       for f in row['forwards']]
    rows[0] = row
    out = m.extract_programs(p, r, c, rows, a, load)
    assert len(out['candidates']) == 168
    blocked = [x for x in out['candidates'] if x['result']['status'] == 'blocked']
    assert len(blocked) == 1 and blocked[0]['candidate_index'] == 0 and blocked[0]['arm'] == m.ARMS[1]
    assert blocked[0]['result']['commands'] is None


@pytest.mark.parametrize('kind', ['schema', 'receipt_sha', 'model', 'missing_root', 'winner_alias'])
def test_authority_and_fixed_denominator_mutation(data, kind):
    p, r, c, rows, a, _, load = data
    p, r, c = dict(p), dict(r), dict(c)
    if kind == 'schema': c['schema'] = 'other.consume'
    elif kind == 'receipt_sha': c['planner_report_sha256'] = '0'*64
    elif kind == 'model': r['model_after'] = '0'*64
    elif kind == 'missing_root': rows = rows[:1]
    elif kind == 'winner_alias':
        c['plans'] = [dict(x) for x in c['plans']]
        c['plans'][0]['program_count'] = 3
    with pytest.raises(ValueError):
        m.extract_programs(p, r, c, rows, a, load)


@pytest.mark.parametrize('kind', ['raw', 'input', 'output', 'complete', 'module', 'request'])
def test_referenced_real_native_record_mutation(data, kind):
    row = data[3][0]
    f = deepcopy(next(x for x in row['forwards'] if x['tag'] == 'joint/1/0'))
    if kind == 'raw': f['raw']['raw_hex'] = '00'*40
    elif kind == 'input': f['input']['action_history'][0][0] = .25
    elif kind == 'output': f['output']['latent_history'][0][0] = .25
    elif kind == 'complete': f['completed'] = 1
    elif kind == 'module': f['modules'][1]['inputs'][0] = blob(np.zeros((1, 3, 192), np.float32)+1)
    elif kind == 'request': f['request']['input_context_sha256'] = '0'*64
    candidate = row['result']['candidates'][m.ARMS[0]][1]
    with pytest.raises(ValueError):
        m.saved_path({f['tag']: f}, data[0]['slots'][0]['packet']['root_context'],
                     m.chart(candidate['logits'], 3)[:1], candidate['contexts'][:1], 'joint/1', dict(mean=[0., 0.], std=[1., 1.]))


@pytest.mark.parametrize('kind', ['completed', 'context', 'raw', 'shape', 'array'])
def test_compiler_native_schema_mutation(data, kind):
    row = dict(data[3][0])
    entry = deepcopy(row['AD'][0])
    row['AD'] = [entry]+row['AD'][1:]
    if kind == 'completed': entry['completed'] = False
    elif kind == 'context': entry['request']['input_context_sha256'] = '0'*64
    elif kind == 'raw': entry['request']['raw'][0][0][0] = .25
    elif kind == 'shape': entry['request']['shape'] = [192, 20]
    elif kind == 'array': entry['array']['file'] = 'other.npy'
    with pytest.raises(ValueError):
        m.compiler_evidence(row, {f['tag']: f for f in row['forwards']}, 'root_J',
            data[0]['slots'][0]['packet']['root_context'], [0.]*30, data[-1], dict(mean=[0., 0.], std=[1., 1.]))


@pytest.mark.parametrize('value', [[True]*10, [float('nan')]*10, [1e-100]*10])
def test_nonliteral_or_nonfinite_f32(value):
    with pytest.raises(ValueError):
        m.numeric(value, (10,))


def test_saved_decode_rejects_duplicate_and_nonfinite():
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
        with pytest.raises(ValueError):
            m.decode(raw)


def disk_data(data, tmp_path):
    p, r, c, rows, _, arrays, _ = data
    p, r, c = deepcopy(p), deepcopy(r), deepcopy(c)
    r['artifact_sha256'] = {}
    for root, row in enumerate(rows):
        path = tmp_path/f'root{root}.json.gz'
        path.write_bytes(gzip.compress(m.canonical(row), mtime=0))
        r['artifact_sha256'][path.name] = m.sha(path)
    for name, raw in arrays.items():
        path = tmp_path/name
        path.write_bytes(raw)
        r['artifact_sha256'][name] = m.sha(path)
    pp, rp, cp = [tmp_path/(k+'.json') for k in m.CAPS]
    pp.write_bytes(m.canonical(p))
    r.update(prepared=str(pp), prepare_sha256=m.sha(pp))
    rp.write_bytes(m.canonical(r))
    c['planner_report_sha256'] = m.sha(rp)
    for plan in c['plans']:
        plan.update(planner_report_sha256=m.sha(rp), prepare_sha256=m.sha(pp))
    cp.write_bytes(m.canonical(c))
    return {k: dict(path=str(path), sha256=m.sha(path)) for k, path in zip(m.CAPS, (pp, rp, cp))}


def test_real_saved_cli_and_requests_and_SHA(data, tmp_path, capsys):
    a = disk_data(data, tmp_path)
    requests = tmp_path/'jobs.json'
    jobs = [dict(id='unavailable', refs=['unseen/program'], expected_steps=15)]
    requests.write_bytes(m.canonical(jobs))
    out = tmp_path/'analysis.json'
    argv = [v for k, d in a.items() for v in ('--'+k, d['path'], '--'+k+'-sha', d['sha256'])]
    m.main(['analyze', *argv, '--output', str(out), '--requests', str(requests), '--requests-sha', m.sha(requests)])
    result = m.read_pin(out, m.sha(out), m.OUTPUT_CAP)
    assert result['summary'] == dict(linked=168, blocked=0)
    assert result['requested_links'][0]['result']['status'] == 'blocked'
    assert len(result['catalog']) == 336
    assert len(m.canonical(result))+1 < m.OUTPUT_CAP
    m.main(['verify', '--analysis', str(out), '--analysis-sha', m.sha(out)])
    assert 'verified' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        m.write_new(out, result)
    (tmp_path/'root0_J000.npy').write_bytes(b'bad')
    with pytest.raises(ValueError):
        m.analyze(a)


def test_pins_and_plain_LF_codec():
    assert m.ACTUAL['report']['sha256'] == '56e8b4f2df522f7a0c4cb9bc37ad74bcbc5d4f89e2c5cd601d42cd0eb071d8ef'
    assert m.digest({'x': 1}) == hashlib.sha256(b'{"x":1}\n').hexdigest()
    assert not m.canonical({'x': 1}).endswith(b'\n')


def test_file_fsync_while_open_then_dir(tmp_path, monkeypatch):
    import os
    import stat
    seen, real = [], os.fsync
    def sync(fd):
        seen.append('dir' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        real(fd)
    monkeypatch.setattr(m.os, 'fsync', sync)
    m.write_new(tmp_path/'test.json', {'fixture': True})
    assert seen == ['file', 'dir']
