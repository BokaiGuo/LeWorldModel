"""Independent saved encoder-array connectivity audit, not a physical consumer."""
import io
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import p3_matched_order_feedback_encoder_codex_20261029 as producer

u = producer.original.u
ROLES = {'pixels','preprocessed','encoder_input','encoder_hidden','projector_input','projector_output','latent'}


def inspect(output, frame, arrays, boundary):
    u.need(type(boundary) is int and boundary in producer.BOUNDARIES, 'matched boundary')
    root = Path(output)
    u.need(root.is_absolute() and root.resolve() == root and root.is_dir() and not root.is_symlink(), 'canonical directory')
    u.need(type(frame) is dict and set(frame) == {'file','sha256','pixel_sha256'}, 'frame descriptor')
    name = frame['file']
    u.need(type(name) is str and Path(name).name == name and name.endswith('.png'), 'flat PNG')
    png = u.read_bytes(root/name, 256*1024)
    u.need(u.fsha(png) == frame['sha256'], 'PNG hash')
    u.need(type(arrays) is dict and set(arrays) == ROLES, 'complete role set')
    values = {}; names = {name}
    for role, descriptor in arrays.items():
        u.need(type(descriptor) is dict and set(descriptor) == {'file','sha256','shape','dtype','origin','role'} and
               descriptor['origin'] == 'encoding' and descriptor['role'] == role, 'exact array descriptor')
        filename = descriptor['file']
        u.need(type(filename) is str and Path(filename).name == filename and filename.endswith('.npy') and
               filename not in names, 'unique flat array')
        names.add(filename)
        shape = tuple(descriptor['shape']); dtype = descriptor['dtype']
        u.need((role,shape,dtype) in u.ENC_SHAPES, 'role shape dtype')
        blob = u.read_bytes(root/filename, 4096 + int(np.prod(shape))*np.dtype(dtype).itemsize)
        u.need(u.fsha(blob) == descriptor['sha256'], 'array hash')
        stream = io.BytesIO(blob); version = np.lib.format.read_magic(stream)
        u.need(version in ((1,0),(2,0)), 'NPY version')
        header = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
        actual_shape, fortran, actual_dtype = header(stream)
        u.need(actual_shape == shape and not fortran and actual_dtype.str == dtype and
               len(blob)-stream.tell() == int(np.prod(shape))*actual_dtype.itemsize, 'NPY header and length')
        stream.seek(0); value = np.load(stream, allow_pickle=False)
        u.need(stream.tell() == len(blob) and np.isfinite(value).all(), 'array exact finite data')
        values[role] = value
    u.need(u.fsha(values['pixels'].tobytes()) == frame['pixel_sha256'], 'pixel SHA')
    result = producer.verify_arrays(png, values)
    return dict(boundary=boundary, saved_io_connected=result['saved_io_connected'],
                neural_parity_recomputed=False, physical_certified=False, scientific_pass=False)
