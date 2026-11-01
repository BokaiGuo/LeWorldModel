"""Independent four-slot transport inventory audit, not a closed-loop receipt."""
import io
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_matched_order_feedback_store_codex_20261029 as store

base = store.base
need = store.need


def inspect(directory, journal_sha):
    root = Path(directory)
    need(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(), 'canonical directory')
    files = list(root.iterdir())
    need(all(f.is_file() and not f.is_symlink() for f in files) and
         sum(f.stat().st_size for f in files) <= store.CAP, 'bounded flat file inventory')
    journal = store.inspect_journal(root/'events.jsonl',journal_sha)
    refs = {}
    def collect(value):
        if type(value) is dict:
            if set(value) in ({'file','sha256','shape','dtype','origin','role'},
                              {'file','sha256','pixel_sha256','shape','dtype'}):
                name = value['file']
                need(type(name) is str and Path(name).name == name and name not in ('','.','..'), 'flat artifact')
                need(name not in refs or refs[name] == value, 'immutable descriptor')
                refs[name] = value
            else:
                for v in value.values(): collect(v)
        elif type(value) is list:
            for v in value: collect(v)
    for event in journal['events']:
        collect(event['detail'])
    charged = {k:0 for k in store.LIMITS}
    for event in journal['events']:
        charged[event['origin']+'_events'] += len(base.raw(event))+1
    for name,d in refs.items():
        if 'pixel_sha256' in d:
            need(name.endswith('.png') and d['shape'] == [224,224,3] and d['dtype'] == 'uint8', 'PNG descriptor')
            blob = base.read_bytes(root/name,store.publication.PNG_CAP)
            need(base.fsha(blob) == d['sha256'] and store.publication.png_data(blob) == d['pixel_sha256'], 'PNG bytes/pixels')
            charged['physical_png'] += len(blob); continue
        origin,role,shape,dtype = d['origin'],d['role'],d['shape'],d['dtype']
        need(origin in ('encoding','numerical') and type(shape) is list and
             all(type(x) is int and x >= 0 for x in shape) and dtype in ('<f4','|u1') and name.endswith('.npy'), 'NPY descriptor')
        if origin == 'encoding':
            need((role,tuple(shape),dtype) in base.ENC_SHAPES, 'encoding array ABI')
        else:
            need(dtype == '<f4' and np.prod(shape,dtype=object) <= 768, 'numeric array ABI')
        size = int(np.prod(shape,dtype=object))*np.dtype(dtype).itemsize
        blob = base.read_bytes(root/name,size+4096); need(base.fsha(blob) == d['sha256'], 'NPY SHA')
        stream = io.BytesIO(blob); version = np.lib.format.read_magic(stream)
        need(version in ((1,0),(2,0)), 'NPY version')
        read = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
        actual,fortran,dt = read(stream)
        need(list(actual) == shape and not fortran and dt.str == dtype and len(blob)-stream.tell() == size, 'NPY header/payload')
        stream.seek(0); value = np.load(stream,allow_pickle=False)
        need(stream.tell() == len(blob) and np.isfinite(value).all(), 'NPY exact finite payload')
        charged[origin+'_arrays'] += len(blob)
    need({p.name for p in files} == set(refs)|{'events.jsonl'}, 'all and only journal-referenced files')
    need(all(charged[k] <= store.LIMITS[k] for k in charged) and sum(charged.values()) <= store.CAP-store.RESERVE,
         'independent transport capacity')
    return dict(schema='p3_matched_order_feedback_store_saved_codex_20261029',
                counts=journal['counts'], pending=journal['pending'], bytes=charged,
                transport_complete=journal['transport_complete'], physical_certified=False,
                full_saved_strict=False, scientific_pass=False)
