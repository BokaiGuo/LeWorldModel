"""Bridge original single-frame capture to the unified publication journal.

Transport only: no source/model authentication or episode semantic certification.
"""
import io
import base64
from pathlib import Path
import numpy as np
from PIL import Image
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as pub


class CaptureStore:
    def __init__(self, store):
        self.store = store
        self.path = store.output
        self.frames = {}

    @property
    def index(self):
        return self.store.index

    def check(self, size=0):
        self.store.check('physical_events', size)

    def emit(self, kind, **fields):
        pub.base.need(type(kind) is str and bool(kind), 'native event kind')
        return self.store.event('physical', 'record', 'native_'+kind, self.store.index, fields)

    def frame(self, name, image):
        pub.base.need(type(name) is str and Path(name).name == name and name.endswith('.png')
                      and name not in self.frames and self.store.frames < pub.PNG_COUNT, 'unique native PNG name/count')
        a = np.asarray(image)
        pub.base.need(a.shape == (224, 224, 3) and a.dtype == np.uint8, 'native RGB')
        stream = io.BytesIO()
        Image.fromarray(a).save(stream, format='PNG')
        data = stream.getvalue()
        pixels = pub.png_data(data)
        d = dict(file=name, sha256=pub.base.fsha(data), pixel_sha256=pixels,
                 shape=[224,224,3], dtype='uint8')
        self.store.write(name, data, 'physical_png')
        self.store.frames += 1
        self.frames[name] = d
        self.store.event('physical', 'record', 'native_frame', self.store.index, {'frame':d})
        return d['sha256']

    def descriptor(self, receipt):
        p = Path(receipt['path'])
        pub.base.need(p.parent == self.path and p.name in self.frames, 'owned native frame')
        d = self.frames[p.name]
        pub.base.need(receipt['file_sha256'] == d['sha256'] and receipt['pixel_sha256'] == d['pixel_sha256'],
                      'native receipt exact frame hashes')
        return dict(d)

    def arrays(self, name, **values):
        """Retain an invalid render as bounded journal chunks, never a fake PNG."""
        pub.base.need(type(name) is str and Path(name).name == name and name.endswith('_invalid.npz')
                      and set(values) == {'raw_render'}, 'original invalid-render ABI')
        a = np.asarray(values['raw_render'])
        pub.base.need(a.dtype.kind in 'buif' and a.nbytes < 1024**2 and a.ndim <= 8,
                      'bounded primitive invalid render')
        buffer = io.BytesIO()
        np.save(buffer, a, allow_pickle=False)
        blob = buffer.getvalue()
        pub.base.need(len(blob) <= 1024**2 + 4096, 'bounded invalid NPY')
        sha = pub.base.fsha(blob)
        chunks = [blob[i:i+6000] for i in range(0, len(blob), 6000)]
        for i, chunk in enumerate(chunks):
            self.store.event('physical', 'record', 'native_invalid_array', self.store.index,
                             dict(name=name, index=i, total=len(chunks), sha256=sha,
                                  data=base64.b64encode(chunk).decode('ascii')))
        return sha


def decode_invalid_array(records, name):
    """Recover one ordered, bounded diagnostic array from verified event details."""
    pub.base.need(type(records) is list and bool(records), 'invalid array chunks')
    total = records[0]['total']; sha = records[0]['sha256']
    pub.base.need(type(total) is int and total == len(records) and 0 < total <= 176,
                  'invalid array chunk count')
    parts = []
    for i, d in enumerate(records):
        pub.base.need(type(d) is dict and set(d) == {'name','index','total','sha256','data'}
                      and d['name'] == name and type(d['index']) is int and d['index'] == i
                      and d['total'] == total and d['sha256'] == sha and type(d['data']) is str,
                      'invalid array chunk identity/order')
        b = base64.b64decode(d['data'], validate=True)
        pub.base.need(len(b) == 6000 if i < total-1 else 0 < len(b) <= 6000, 'invalid array chunk size')
        parts.append(b)
    blob = b''.join(parts)
    pub.base.need(len(blob) <= 1024**2 + 4096 and pub.base.fsha(blob) == sha, 'invalid array hash/cap')
    stream = io.BytesIO(blob)
    version=np.lib.format.read_magic(stream)
    pub.base.need(version in ((1,0),(2,0)), 'invalid array header version')
    shape,fortran,dtype=(np.lib.format.read_array_header_1_0(stream) if version==(1,0)
                         else np.lib.format.read_array_header_2_0(stream))
    size=int(np.prod(shape,dtype=object))*dtype.itemsize
    pub.base.need(len(shape)<=8 and dtype.kind in 'buif' and size<1024**2
                  and len(blob)-stream.tell()==size, 'invalid array header/payload bound')
    stream.seek(0)
    array = np.load(stream, allow_pickle=False)
    pub.base.need(stream.tell() == len(blob) and array.dtype.kind in 'buif' and array.ndim <= 8
                  and array.nbytes < 1024**2, 'invalid array bounded exact EOF')
    return array
