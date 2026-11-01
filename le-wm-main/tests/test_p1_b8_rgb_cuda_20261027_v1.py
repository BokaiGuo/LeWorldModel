"""In-memory contract tests plus read-only real B8 evidence; no CUDA/tmp fixtures."""
from copy import deepcopy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import numpy as np
from experiments.spectral_reachability import p1_b8_rgb_cuda_20261027_v1 as m


def fixture():
    q = np.zeros((192, 192)); q[0, 0] = q[1, 1] = 1
    task = dict(environment='mock/contract-only', task_contract_sha256='a'*64,
        root_context=dict(layout='states_with_between_state_actions_v1', context_length=3, latent_dim=192,
            action_dim=10, latent_history=np.zeros((3, 192)).tolist(), action_history=np.zeros((2, 10)).tolist()),
        root_raw_action_history=np.zeros((10, 2)).tolist(), goal=np.zeros(192).tolist(), projection=q.tolist(),
        model_sha256='b'*64, model_config_sha256='c'*64, metric_source_sha256='d'*64,
        goal_source_sha256='e'*64, representation='native_projected_cls', latent_dim=192,
        horizon=2, packet_steps=5, control_dim=2, action_mean=[0., 0.], action_std=[1., 1.],
        packet_logit_cap=1., feedback='open_loop_model_only')
    tasks = {h*64:deepcopy(task) for h in ('1', '2')}; frames = []
    pred = np.zeros(192); pred[0], pred[3] = 2, 4
    for root in tasks:
        for arm in m.ARMS:
            for step in m.STEPS:
                frames.append(dict(order=len(frames), root_sha256=root, arm=arm, step=step,
                    task_id=m.core.task_id(tasks[root]), predicted_native192=pred.tolist(),
                    pixel_sha256='f'*64, file_sha256='e'*64, path='mock-not-read',
                    physical_diagnostic={'mock_only':True}))
    return frames, tasks


class MemoryStore:
    def __init__(self): self.files = {}
    def write(self, name, raw):
        if name in self.files: raise ValueError('exclusive mock file')
        self.files[name] = raw; return hashlib.sha256(raw).hexdigest()
    def put(self, name, value): return self.write(name, m.canonical(value))
    def native(self, name, value):
        b = io.BytesIO(); np.save(b, value, allow_pickle=False); return self.write(name, b.getvalue())


def exercise(bad=None):
    frames, _ = fixture(); events = []; ledger = m.Ledger(events.append)
    store = MemoryStore(); rows = []; errors = []
    def encode(image):
        ledger.call('encoder', lambda:None, shape=(1, 3, 224, 224))
        def project():
            if bad == 'projector': raise RuntimeError('original mock projector failure')
        ledger.call('projector', project, shape=(1, 192))
        value = np.zeros(192, dtype=np.float32)
        if bad == 'dtype': value = value.astype(np.float64)
        if bad == 'nan': value[0] = np.nan
        if bad == 'shape': value = value[:-1]
        return value
    m.encode_frames(frames, encode, ledger, store, rows, errors, reader=lambda *a:(None, None))
    return ledger, store, rows, errors, events


class B8Tests(unittest.TestCase):
    def test_twelve_no_pixel_dedup_and_native_saved(self):
        ledger, store, rows, errors, events = exercise()
        self.assertEqual(errors, [])
        self.assertEqual(ledger.counts(), {k:dict(attempted=12, completed=12) for k in ('encoder', 'projector')})
        self.assertEqual(sum(n.endswith('.npy') for n in store.files), 12)
        self.assertEqual(sum(e['kind'] == 'native_saved' for e in events), 12)
        for row in rows:
            self.assertEqual(row['latent_sha256'], hashlib.sha256(store.files[row['latent_file']]).hexdigest())
            value = np.load(io.BytesIO(store.files[row['latent_file']]), allow_pickle=False)
            self.assertEqual(value.dtype, np.float32); self.assertEqual(value.shape, (192,))

    def test_invalid_native_stops_without_fake_arrays(self):
        for bad in ('dtype', 'nan', 'shape'):
            with self.subTest(bad=bad):
                ledger, store, rows, errors, _ = exercise(bad)
                self.assertTrue(errors); self.assertTrue(ledger.failed)
                self.assertEqual(rows[0]['status'], 'failed')
                self.assertTrue(all(r['status'] == 'not_attempted' for r in rows[1:]))
                self.assertFalse(any(n.endswith('.npy') for n in store.files))

    def test_projector_first_error_separate_completion(self):
        ledger, _, _, errors, _ = exercise('projector')
        self.assertIn('original mock projector failure', str(errors[0]))
        self.assertEqual(ledger.counts(), {'encoder':dict(attempted=1, completed=1), 'projector':dict(attempted=1, completed=0)})

    def test_synchronize_error_is_not_completion(self):
        frames, _ = fixture(); ledger = m.Ledger(lambda e:None); ledger.begin(frames[0])
        def fail(): raise RuntimeError('mock synchronize')
        with self.assertRaisesRegex(RuntimeError, 'synchronize'):
            ledger.call('encoder', lambda:None, shape=(1, 3, 224, 224), synchronize=fail)
        self.assertEqual(ledger.counts()['encoder'], dict(attempted=1, completed=0))

    def test_cpu_shape_and_order_refused(self):
        frames, _ = fixture()
        for kwargs in (dict(device='cpu', shape=(1, 3, 224, 224)), dict(shape=(2, 3, 224, 224))):
            ledger = m.Ledger(lambda e:None); ledger.begin(frames[0])
            with self.assertRaises(ValueError): ledger.call('encoder', lambda:None, **kwargs)
            self.assertEqual(ledger.entries, [])
        ledger = m.Ledger(lambda e:None)
        with self.assertRaises(ValueError): ledger.begin(frames[1])
        ledger.begin(frames[0])
        with self.assertRaises(ValueError): ledger.call('projector', lambda:None, shape=(1, 192))

    def test_duplicate_charge_refused(self):
        frames, _ = fixture(); ledger = m.Ledger(lambda e:None); ledger.begin(frames[0])
        ledger.call('encoder', lambda:None, shape=(1, 3, 224, 224))
        with self.assertRaises(ValueError): ledger.call('encoder', lambda:None, shape=(1, 3, 224, 224))
        self.assertEqual(ledger.counts()['encoder']['attempted'], 1)

    def test_original_Q_not_identity(self):
        frames, tasks = fixture(); values = np.zeros((12, 192)); values[:, 0], values[:, 3] = 1, 6
        result = m.analyze(frames, values, tasks)
        for row in result['rows']:
            self.assertEqual(row['actual_Q_mse'], 1/192)
            self.assertEqual(row['predicted_Q_mse'], 4/192)
            self.assertEqual(row['native_prediction_mse'], 5/192)
        self.assertFalse(result['P1_admission']); self.assertFalse(result['new_Q'])

    def test_frame_mutations_fail_closed(self):
        for change in ('missing', 'order', 'task', 'root', 'step', 'arm'):
            frames, tasks = fixture()
            if change == 'missing': frames.pop()
            else:
                field = {'task':'task_id', 'root':'root_sha256'}.get(change, change)
                frames[0][field] = 99 if field in ('order', 'step') else 'wrong'
            with self.subTest(change=change), self.assertRaises(ValueError): m.validate_frames(frames, tasks)

    def test_nonfinite_analysis_rejected(self):
        frames, tasks = fixture(); values = np.zeros((12, 192)); values[0, 0] = np.nan
        with self.assertRaises(ValueError): m.analyze(frames, values, tasks)

    def test_floor_cap_and_final_reserve(self):
        for args in ((0, 1, m.FLOOR), (m.CAP-m.FINAL_RESERVE, 1, m.FLOOR+1)):
            with self.assertRaises(ValueError): m.check_budget(*args)
        m.check_budget(0, 1, m.FLOOR+1)
        m.check_budget(m.CAP-1, 1, m.FLOOR+1, final=True)
        with self.assertRaises(ValueError): m.check_budget(m.CAP, 1, m.FLOOR+1, final=True)

    def test_store_refuses_before_mkdir(self):
        with patch.object(m.shutil, 'disk_usage', return_value=NS(free=m.FLOOR+m.CAP-1)):
            with self.assertRaises(ValueError): m.Store(m.ROOT/'results'/('never_created_mock_'+m.NAME))

    def test_prepare_floor_and_size_no_write(self):
        for protocol, free in (({'mock':True}, m.FLOOR), ({'oversize':'x'*m.PREPARE_CAP}, m.FLOOR+m.CAP)):
            with patch.object(m, 'collect', return_value=protocol), patch.object(m.shutil, 'disk_usage', return_value=NS(free=free)):
                with self.assertRaises(ValueError): m.prepare(m.ROOT/'results'/('never_created_mock_'+m.NAME+'.json'))

    def test_wrong_prepare_sha_no_read(self):
        with patch.object(m.io, 'regular'), patch.object(m.io, 'sha', return_value='a'*64):
            with self.assertRaisesRegex(ValueError, 'external'): m.verify('not-read', 'b'*64)

    def test_conflicting_binding(self):
        with self.assertRaises(ValueError): m.merge_bindings({'file':'a'*64}, {'file':'b'*64})

    def test_hooks_deny_and_restore(self):
        def module(): return NS(forward=lambda *a, **kw:None)
        model = NS(encoder=module(), projector=module(), predictor=module(), action_encoder=module(), predict=lambda:None)
        autograd = NS(grad=lambda:None, backward=lambda:None, graph=NS(), functional=NS(jacobian=lambda:None))
        torch = NS(autograd=autograd, cuda=NS(synchronize=lambda device:None))
        for target, key in ((model, 'predict'), (model.predictor, 'forward'), (model.action_encoder, 'forward'), (autograd, 'grad'), (autograd, 'backward'), (autograd.functional, 'jacobian')):
            original = getattr(target, key); ledger = m.Ledger(lambda e:None); ledger.begin(fixture()[0][0]); hooks = m.io.Hooks()
            try:
                m.install_hooks(torch, model, ledger, hooks, 'cuda')
                with self.assertRaises(ValueError): getattr(target, key)()
                self.assertTrue(ledger.forbidden)
            finally:
                errors = []; hooks.restore(errors); self.assertEqual(errors, [])
            self.assertIs(getattr(target, key), original)

    def test_native_exact_keyword_transform_mock(self):
        from contextlib import nullcontext
        from experiments import spectral_reachability as package
        class Tensor:
            shape = (1, 1, 192); dtype = 'float32'; device = 'cuda'
            def __getitem__(self, index): return self
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.zeros(192, dtype=np.float32)
        seen = []
        def image_tensor(image, *, transform, device):
            seen.append((image, transform, device)); return 'pixels'
        def encode(packet): self.assertEqual(packet, {'pixels':'pixels'}); return {'emb':Tensor()}
        p0 = NS(_image_tensor=image_tensor, image_transform=lambda:'original-transform')
        with patch.dict(sys.modules, {'experiments.spectral_reachability.preflight_pusht_d2p_p0':p0}), patch.object(package, 'preflight_pusht_d2p_p0', p0, create=True):
            value = m.Native(NS(no_grad=nullcontext, float32='float32'), NS(encode=encode), 'cuda')('rgb')
        self.assertEqual(seen, [('rgb', 'original-transform', 'cuda')]); self.assertEqual(value.dtype, np.float32)

    def test_actual_saved_full_collect(self):
        protocol = m.collect()
        self.assertEqual(len(protocol['frames']), 12); self.assertEqual(len(protocol['tasks']), 2)
        self.assertEqual(len({f['path'] for f in protocol['frames']}), 12)
        self.assertLess(len(m.canonical(protocol)), m.PREPARE_CAP)
        self.assertEqual(protocol['encoder_cap'], 12); self.assertEqual(protocol['projector_cap'], 12)
        for key in ('predict_cap', 'AD_cap', 'physical_steps_cap', 'action_encoder_cap'): self.assertEqual(protocol[key], 0)
        self.assertFalse(protocol['P1_admission']); self.assertFalse(protocol['new_Q'])

    def test_actual_physical_scope_mutations(self):
        report = m.physical.read(m.RUN/'report.json'); prepared = m.physical.read(m.PREPARED)
        events = m.source.read_journal(m.RUN/'journal.jsonl')
        for key, value in (('neural_calls', 1), ('P1_admission', True), ('status', 'partial')):
            bad = deepcopy(report); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): m.validate_physical(bad, prepared, events)

    def test_z_no_neural_imports(self):
        self.assertFalse(any(n in sys.modules for n in ('torch', 'gymnasium', 'pymunk')))


if __name__ == '__main__': unittest.main()
