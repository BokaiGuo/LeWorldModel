"""Conservative candidate partition for E3 continuous episode transport.

This is a bound and admission primitive, not a claim that an unwritten store
implements the partition or that any native run fits its category limits.
"""
import io
import numpy as np
from experiments.spectral_reachability import agent_e3_gd_host_cuda_20261029_v1 as host

MiB=1024**2
CAP=512*MiB
RESERVE=64*MiB
PHYSICAL_PNG=32*256*1024
PHYSICAL_JOURNAL=32*MiB
PHYSICAL_META=16*MiB
ENCODING_ARRAYS=8*MiB
ENCODING_EVENTS_META=8*MiB


def compress_bound(n):
    if type(n) is not int or n<0:raise ValueError('nonnegative integer size')
    return n+(n>>12)+(n>>14)+(n>>25)+32


def encoding_array_sizes():
    specs=[((224,224,3),np.uint8),((1,1,3,224,224),np.float32),
           ((1,3,224,224),np.float32),((1,257,192),np.float32),
           ((1,192),np.float32),((1,192),np.float32),((192,),np.float32)]
    values=[]
    for shape,dtype in specs:
        stream=io.BytesIO();np.save(stream,np.zeros(shape,dtype=dtype),allow_pickle=False)
        values.append(len(stream.getvalue()))
    return values


def budget(K):
    b=host.budget(K)
    encoding_npy=2*sum(encoding_array_sizes())
    encoding_bound=2*sum(map(compress_bound,encoding_array_sizes()))
    parts=dict(numerical_arrays=b['array_bytes'],numerical_events=b['event_bytes'],
        numerical_metadata=b['metadata_bytes'],physical_PNG=PHYSICAL_PNG,
        physical_journal=PHYSICAL_JOURNAL,physical_metadata=PHYSICAL_META,
        encoding_arrays=ENCODING_ARRAYS,encoding_events_metadata=ENCODING_EVENTS_META,
        final_reserve=RESERVE)
    return dict(parts=parts,total=sum(parts.values()),cap=CAP,remaining=CAP-sum(parts.values()),
        encoding_npy=encoding_npy,encoding_gzip_bound=encoding_bound,
        encoding_fixed_arrays_fit=encoding_bound<=ENCODING_ARRAYS,
        physical_32_PNG_assumes_per_image_max=256*1024,
        physical_and_metadata_fit_actual_writer_unproven=True,
        independent_working_caps_required=True,full_production_transport_proven=False)
