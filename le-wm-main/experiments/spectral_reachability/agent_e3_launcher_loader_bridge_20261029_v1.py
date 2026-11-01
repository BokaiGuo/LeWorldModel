"""Loader-to-unified-journal bridge; not a launcher or semantic consumer."""
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub


class LoaderBridge:
    """Preserve the old loader's exact events and receipt in the unified ledger.

    The loader itself remains unchanged and must be invoked only by an
    authorized host launcher. No checkpoint is read by this adapter.
    """
    def __init__(self, store):
        pub.base.need(isinstance(store, pub.Store), 'unified publication store')
        self.store = store
        self.finished = False

    def event(self, event):
        pub.base.need(not self.finished and type(event) is dict, 'loader event')
        self.store.event('physical', 'record', 'loader_event', self.store.index,
                         {'event': event})

    def put(self, name, value, final=False):
        pub.base.need(not self.finished and name == 'loader.json' and final is False
                      and type(value) is dict, 'only original loader receipt')
        self.store.event('physical', 'record', 'loader_receipt', self.store.index,
                         {'receipt': value})
        self.finished = True


def verify_events(events, prepared, prepared_sha, loader):
    """Validate original loader ABI, including event order and buffer identity.

    Transport hashing and prepared/source authority are the caller's separate
    responsibility. A receipt does not itself prove a model was loaded.
    """
    records = [e for e in events if e.get('origin') == 'physical' and
               e.get('resource') in ('loader_event', 'loader_receipt')]
    pub.base.need(len(records) == 5 and all(e.get('kind') == 'record' for e in records)
                  and [e['resource'] for e in records] == ['loader_event'] * 4 + ['loader_receipt']
                  and all(records[i]['index'] < records[i+1]['index'] for i in range(4)),
                  'exact loader record chronology')
    receipt = records[-1]['detail']['receipt']
    loader.validate_loader(receipt, prepared, prepared_sha)
    files = receipt['files']
    pub.base.need(len(files) == 2, 'two checkpoint buffers')
    for i, name in enumerate(('config.json', 'weights.pt')):
        f = files[i]
        pub.base.need(f['path'] == str(__import__('pathlib').Path(prepared['policy']) / name),
                      'original checkpoint order')
        expected = [{'kind': 'load_intent', 'path': f['path'], 'expected_sha256': f['sha256']},
                    {'kind': 'consumed_buffer_frozen', 'receipt': f}]
        pub.base.need([records[2*i+j]['detail']['event'] for j in range(2)] == expected,
                      'exact consumed buffer lifecycle')
    return receipt
