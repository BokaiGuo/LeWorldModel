"""Synthetic saved ABI only: no real consumer, model or physics imports."""
from copy import deepcopy
import gzip
import hashlib

import numpy as np
import pytest

from experiments.spectral_reachability import agent_p3_actual_cp5_task_builder_20261029_v1 as m


def fixture():
    p = dict(schema=m.PLANNER+'.prepare', tasks=[], layers=[])
    r = dict(schema=m.PLANNER+'.report', prepared='/fixture/prepare.json', prepare_sha256='a'*64, rows=[], nodes=[])
    c = dict(schema=m.PLANNER+'.consume', complete=True, status='saved_shared_panel_verified_NOT_admitted',
             prepared=r['prepared'], prepare_sha256='a'*64, report_sha256='b'*64, audited=[])
    authorities = {k: dict(path='/fixture/'+k+'.json', sha256=h*64)
                   for k, h in zip(m.CAPS, 'abc')}
    authorities['prepared']['path'] = r['prepared']
    exports = []
    for root in range(2):
        warm = np.zeros((10, 2), np.float32)
        warm[:5, 0] = np.float32(.1)
        warm[5:, 1] = np.float32(.1)
        prefix = np.full((5, 2), np.float32(.125+root*.125), np.float32)
        base = dict(action_mean=[0., 0.], action_std=[1., 1.], goal=np.zeros(192).tolist(),
                    projection=np.eye(192).tolist(), root_context=dict(layout='states_with_between_state_actions_v1',
                    context_length=3, latent_dim=192, action_dim=10,
                    latent_history=np.full((3, 192), root, np.float32).tolist(), action_history=warm.reshape(2, 10).tolist()))
        domain = dict(raw_bounds=[-1, 1], packet_steps=5, action_dim=2,
                      normalization='original saved float32 subtract/divide; no /100 or clipping')
        b = {k: m.digest([k, root]) for k in m.BINDINGS}
        b.update(context_sha256=m.digest(base['root_context']), goal_sha256=m.digest(base['goal']),
                 projection_sha256=m.digest(base['projection']), action_domain_sha256=m.digest(domain),
                 normalizer_sha256=m.digest([base['action_mean'], base['action_std']]))
        base['binding'] = b
        exported_task = {k: deepcopy(v) for k, v in base.items() if k != 'binding'}
        exported_task['root_raw_action_history'] = warm.tolist()
        exported_task['task_contract_sha256'] = b['task_contract_sha256']
        exports.append(dict(schema='p1_constructive_moving_history_encode_cuda_20261027_v1.task_export',
                            source_slot=root, task_id=b['task_id'], task=exported_task,
                            task_contract=dict(observed_steps=[0, 5, 10]),
                            history_semantics='actual_moving_raw_between_observed0_5_10'))
        layer = dict(source_slot=root, task_id=b['task_id'], task_sha256=b['task_sha256'],
                     task_export=f'/fixture/export{root}.json.gz', commands=prefix.tolist(), selection_sha256='d'*64)
        p['tasks'].append(base)
        p['layers'].append(layer)
    populate(p, r, c, exports)
    return p, r, c, exports, authorities


def populate(p, r, c, exports):
    r['nodes'], r['rows'], c['audited'] = [], [], []
    for root, base in enumerate(p['tasks']):
        layer = p['layers'][root]
        prefix = np.array(layer['commands'][:5], np.float32)
        warm = np.array(exports[root]['task']['root_raw_action_history'], np.float32)
        raw = np.concatenate((warm[-5:], prefix))
        ctx = deepcopy(base['root_context'])
        ctx['latent_history'] = ctx['latent_history'][1:]+[np.full(192, root+1, np.float32).tolist()]
        ctx['action_history'] = m.normalize(base, raw).tolist()
        for arm in m.ARMS:
            eid = f's{root}_{arm}'
            ob = dict(boundary=5, png_sha256='1'*64, pixel_sha256='2'*64, encoding_sha256='3'*64,
                      observed_index=4, encoded_index=5,
                      observed_latent_sha256=hashlib.sha256(m.canonical(ctx['latent_history'][-1])+b'\n').hexdigest())
            identity = dict(episode_id=eid, source_slot=root, mode=arm, boundary=5, raw10=raw.tolist(), observation=ob)
            req = dict(schema=m.PLANNER+'.node', base_task_sha256=m.digest(base),
                       task=m.local_task(base, ctx, identity, m.LIVE), **identity)
            r['nodes'].append(dict(episode_id=eid, source_slot=root, mode=arm, boundary=5, status='sealed', request=req))
            r['rows'].append(dict(episode_id=eid, source_slot=root, mode=arm, prefix_exact=True,
                prefix=dict(commands=prefix.tolist(), raw_float32_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
                            source_selection_sha256=layer['selection_sha256'])))
            c['audited'].append(dict(episode_id=eid, boundary=5, node_verified=True))


def test_complete_join_timing_identity_and_no_mutation():
    args = fixture()
    before = deepcopy(args)
    out = m.join(*args)
    assert args == before
    assert len(out['tasks']) == len(out['provenance']) == 2
    for root, (task, prov) in enumerate(zip(out['tasks'], out['provenance'])):
        assert prov['actual_cp5_absolute_steps'] == [5, 10, 15]
        assert prov['actual_cp5_control_steps'] == [-5, 0, 5]
        assert prov['original_control_steps'] == [-10, -5, 0]
        assert len(prov['branch_witnesses']) == 4
        assert task['binding']['root_sha256'] == args[0]['tasks'][root]['binding']['root_sha256']
        assert task['binding']['task_id'] != args[0]['tasks'][root]['binding']['task_id']
        m.task_check(task)
    assert not out['evidence']['upstream_fullstrict_rebuilt_here']
    assert not any(out[k] for k in m.FLAGS)
    assert sum(out['new_calls'].values()) == 0


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'rootbool', 'boundarybool', 'goal', 'context',
    'raw10', 'normalizer', 'binding', 'original', 'prefix', 'observation', 'audit', 'receipt',
    'reportsha', 'preparedpath', 'times', 'booltime', 'warm', 'nan', 'boollatent', 'originalgoal',
    'exportschema', 'originalcontract'])
def test_mutations_rejected(kind):
    p, r, c, e, a = fixture()
    req = r['nodes'][0]['request']
    if kind == 'missing': r['nodes'].pop(0)
    elif kind == 'duplicate': r['nodes'][1] = deepcopy(r['nodes'][0])
    elif kind == 'rootbool': req['source_slot'] = False
    elif kind == 'boundarybool': r['nodes'][0]['boundary'] = True
    elif kind == 'goal': req['task']['goal'][0] = 1.
    elif kind == 'context': req['task']['root_context']['latent_history'][0][0] = 4.
    elif kind == 'raw10': req['raw10'][0][0] = .25
    elif kind == 'normalizer': req['task']['action_std'][0] = 2.
    elif kind == 'binding': req['task']['binding']['root_sha256'] = 'f'*64
    elif kind == 'original': req['base_task_sha256'] = 'f'*64
    elif kind == 'prefix': r['rows'][0]['prefix']['commands'][0][0] = .5
    elif kind == 'observation': req['observation']['observed_latent_sha256'] = 'f'*64
    elif kind == 'audit': c['audited'][0]['node_verified'] = False
    elif kind == 'receipt': c['complete'] = 1
    elif kind == 'reportsha': c['report_sha256'] = 'f'*64
    elif kind == 'preparedpath': c['prepared'] = '/other'
    elif kind == 'times': e[0]['task_contract']['observed_steps'] = [5, 10, 15]
    elif kind == 'booltime': e[0]['task_contract']['observed_steps'][0] = False
    elif kind == 'warm': e[0]['task']['root_raw_action_history'][5][0] = .25
    elif kind == 'nan': req['task']['goal'][0] = float('nan')
    elif kind == 'boollatent': req['task']['root_context']['latent_history'][0][0] = False
    elif kind == 'originalgoal': e[0]['task']['goal'][0] = 1.
    elif kind == 'exportschema': e[0]['schema'] = 'other.task_export'
    elif kind == 'originalcontract': e[0]['task']['task_contract_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        m.join(p, r, c, e, a)


def test_consistently_resealed_single_branch_cp5_still_rejected():
    p, r, c, e, a = fixture()
    req = r['nodes'][1]['request']
    ctx = deepcopy(req['task']['root_context'])
    ctx['latent_history'][-1][0] = 7.
    req['observation']['observed_latent_sha256'] = hashlib.sha256(m.canonical(ctx['latent_history'][-1])+b'\n').hexdigest()
    identity = {k: deepcopy(req[k]) for k in ('episode_id', 'source_slot', 'mode', 'boundary', 'raw10', 'observation')}
    req['task'] = m.local_task(p['tasks'][0], ctx, identity, m.LIVE)
    with pytest.raises(ValueError, match='all four branches'):
        m.join(p, r, c, e, a)


def disk_fixture(tmp_path):
    p, r, c, e, a = fixture()
    p['sources'] = {}
    for root in range(2):
        path = tmp_path/f'export{root}.json.gz'
        path.write_bytes(gzip.compress(m.canonical(e[root]), mtime=0))
        h = m.sha(path)
        p['layers'][root].update(task_export=str(path), task_sha256=h)
        p['tasks'][root]['binding']['task_sha256'] = h
        p['sources'][str(path)] = h
    populate(p, r, c, e)
    pp, rp, cp = [tmp_path/(k+'.json') for k in m.CAPS]
    pp.write_bytes(m.canonical(p))
    r.update(prepared=str(pp), prepare_sha256=m.sha(pp))
    rp.write_bytes(m.canonical(r))
    c.update(prepared=str(pp), prepare_sha256=m.sha(pp), report_sha256=m.sha(rp))
    cp.write_bytes(m.canonical(c))
    return {k: dict(path=str(path), sha256=m.sha(path)) for k, path in zip(m.CAPS, (pp, rp, cp))}


def test_real_saved_resolver_writer_and_verify_cli(tmp_path, capsys):
    a = disk_fixture(tmp_path)
    result = m.resolve_actual_cp5_tasks(a)
    assert len(result['bindings']) == 8  # three authorities, two exports, own triple
    output = tmp_path/'tasks.json'
    argv = [x for k, v in a.items() for x in ('--'+k, v['path'], '--'+k+'-sha', v['sha256'])]
    m.main(['prepare', *argv, '--output', str(output)])
    m.main(['verify', *argv, '--output', str(output), '--output-sha', m.sha(output)])
    assert 'verified' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        m.write_new(output, result)
    path = next(tmp_path.glob('export0*'))
    path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError, match='SHA'):
        m.resolve_actual_cp5_tasks(a)


@pytest.mark.parametrize('raw', [b'{"x":0,"x":1}', b'{"x":NaN}'])
def test_json_rejects_duplicate_and_nonfinite(raw):
    with pytest.raises(ValueError):
        m.decode(raw)


def test_bounded_reader_sha_symlink_and_cap(tmp_path):
    p = tmp_path/'a.json'
    p.write_bytes(b'{}')
    with pytest.raises(ValueError, match='SHA'):
        m.read_pinned(p, '0'*64, 2)
    with pytest.raises(ValueError, match='bounded'):
        m.read_pinned(p, m.sha(p), 1)
    link = tmp_path/'link'
    link.symlink_to(p)
    with pytest.raises(ValueError, match='bounded'):
        m.read_pinned(link, m.sha(p), 2)


def test_durable_file_fsync_before_directory(tmp_path, monkeypatch):
    import os
    import stat
    real = os.fsync
    seen = []
    def sync(fd):
        mode = os.fstat(fd).st_mode  # fails if the file has already been closed
        seen.append('dir' if stat.S_ISDIR(mode) else 'file')
        real(fd)
    monkeypatch.setattr(m.os, 'fsync', sync)
    m.write_new(tmp_path/'saved.json', {'fixture': True})
    assert seen == ['file', 'dir']
