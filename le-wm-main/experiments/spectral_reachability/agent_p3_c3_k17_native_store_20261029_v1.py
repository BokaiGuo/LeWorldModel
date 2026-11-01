"""Native capture API over the prospective K17 bounded store, not a runner."""
import hashlib
import io
import shutil
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_p3_c3_k17_bounded_store_20261029_v1 as bound
from experiments.spectral_reachability import agent_p3_c3_k17_physical_transport_20261029_v1 as journal


class NativeStore:
    def __init__(self, path):
        self.inner = bound.Store(path)
        self.path = self.inner.path

    def check(self, size=4096, final=False):
        if type(size) is not int or size < 0 or final:
            raise ValueError('invalid native predispatch check')
        if self.inner.fault:
            raise ValueError('store fault')
        entries = self.inner._entries()
        if (sum(p.stat().st_size for p in entries) + size > bound.CAP - bound.FINAL_RESERVE or
                shutil.disk_usage(self.path).free < bound.FLOOR + size):
            self.inner.fault = True
            raise ValueError('aggregate predispatch cap/floor')

    def emit(self, kind, **fields):
        return self.inner.emit(kind, fields)

    def frame(self, name, value):
        a = np.asarray(value)
        if a.shape != (224, 224, 3) or a.dtype != np.uint8:
            raise ValueError('native uint8 RGB224')
        output = io.BytesIO()
        Image.fromarray(a).save(output, format='PNG')
        data = output.getvalue()
        self.inner.png(name, data)
        return hashlib.sha256(data).hexdigest()

    def arrays(self, name, **arrays):
        """Preserve invalid render data in reserved failure record; then stop.

        Payload is explicitly size-limited; if too large the failure is raised
        without silently replacing it by a summary.
        """
        import base64
        data = io.BytesIO()
        np.savez_compressed(data, **arrays)
        raw = data.getvalue()
        value = {'kind': 'invalid_native_array', 'requested_name': name,
                 'npz_base64': base64.b64encode(raw).decode('ascii'),
                 'npz_sha256': hashlib.sha256(raw).hexdigest()}
        self.inner.fail(value)
        raise ValueError('invalid native array preserved in failure.json')


def inspect_capture_journal(path):
    """Check journal chain and fresh-observation file hashes; not full consume."""
    directory = path
    events = journal.read_journal(directory / 'journal.jsonl')
    names = set()
    for event in events:
        if event['kind'] != 'fresh_observation':
            continue
        receipt = event['payload']['receipt']
        name = receipt['png']
        if type(name) is not str or '/' in name or not name.endswith('.png') or name in names:
            raise ValueError('PNG name or duplicate')
        names.add(name)
        p = directory / name
        if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != receipt['file_sha256']:
            raise ValueError('PNG hash')
    return {'journal_events': len(events), 'observation_pngs': len(names),
            'physical_provenance_verified': False, 'full_consume': False}
