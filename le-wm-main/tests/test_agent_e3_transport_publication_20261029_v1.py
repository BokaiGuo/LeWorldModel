import io
import numpy as np
import pytest
from PIL import Image
from experiments.spectral_reachability import agent_e3_transport_publication_20261029_v1 as m


def png():
    b=io.BytesIO();Image.fromarray(np.zeros((224,224,3),np.uint8)).save(b,format='PNG');return b.getvalue()


def build(tmp_path,fail=False):
    p=tmp_path/'job';s=m.Store(p);f=s.frame(png());z=s.array('encoding','encoder_hidden',np.zeros((1,257,192),np.float32))
    s.event('physical','intent','step',0,{});s.event('encoding','intent','encoder',1,{'frame':f})
    s.event('encoding','return','encoder',1,{'hidden':z})
    s.event('physical','failure' if fail else 'return','step',0,{})
    return p,s


def test_full_temporary_png_publish_and_consume(tmp_path):
    p,s=build(tmp_path);seal=s.publish('completed');r=m.consume(p,seal['report_sha256'],seal['completion_sha256'])
    assert seal['complete'] and r['complete'] and not r['full_native_strict']


def test_failure_cannot_be_completed(tmp_path):
    p,s=build(tmp_path,True);seal=s.publish('completed')
    assert not seal['complete']
    assert not m.consume(p,seal['report_sha256'],seal['completion_sha256'])['complete']


def test_png_tamper_and_extra(tmp_path):
    p,s=build(tmp_path);seal=s.publish('completed');(p/'frame00.png').write_bytes(png()+b'x')
    with pytest.raises(ValueError):m.consume(p,seal['report_sha256'],seal['completion_sha256'])


def test_png_wrong_shape_before_write(tmp_path):
    p=tmp_path/'job';s=m.Store(p);b=io.BytesIO();Image.fromarray(np.zeros((8,8,3),np.uint8)).save(b,format='PNG')
    with pytest.raises(ValueError):s.frame(b.getvalue())
    assert list(p.iterdir())==[]
