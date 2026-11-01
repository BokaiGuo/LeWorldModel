import numpy as np
import pytest

from experiments.spectral_reachability import agent_e3_native_capture_transport_20261029_v1 as a


def test_native_frame_writer_publication_reader(tmp_path):
    image = np.arange(224*224*3, dtype=np.uint32).astype(np.uint8).reshape(224,224,3)
    store = a.pub.Store(tmp_path/'job')
    bridge = a.CaptureStore(store)
    name = 'cell00_plan_05.png'
    file_sha = bridge.frame(name, image)
    receipt = dict(path=str(store.output/name), file_sha256=file_sha,
                   pixel_sha256=a.pub.base.fsha(image.tobytes()))
    bridge.emit('fresh_observation', episode_id='cell00', receipt=receipt)
    d = bridge.descriptor(receipt)
    assert d['file'] == 'cell00_plan_05.png'
    seal = store.publish('completed')
    assert a.pub.consume(store.output, seal['report_sha256'], seal['completion_sha256'])['complete']
    (store.output/d['file']).write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        a.pub.consume(store.output, seal['report_sha256'], seal['completion_sha256'])


def test_frame_limit_and_invalid_before_write(tmp_path):
    store = a.pub.Store(tmp_path/'job')
    bridge = a.CaptureStore(store)
    with pytest.raises(ValueError): bridge.frame('../bad.png', np.zeros((224,224,3),np.uint8))
    with pytest.raises(ValueError): bridge.frame('bad.png', np.zeros((3,3,3),np.uint8))
    assert not list(store.output.iterdir())


def test_invalid_render_bytes_preserved_in_publication(tmp_path):
    store=a.pub.Store(tmp_path/'job'); bridge=a.CaptureStore(store)
    original=np.arange(10000,dtype=np.float32).reshape(100,100)
    original[0,0]=np.nan
    bridge.arrays('cell_plan_01.png_invalid.npz',raw_render=original)
    seal=store.publish('partial')
    receipt=a.pub.consume(store.output,seal['report_sha256'],seal['completion_sha256'])
    assert not receipt['complete']
    lines=(store.output/'events.jsonl').read_bytes().splitlines()
    details=[a.pub.base.decode(line)['detail'] for line in lines]
    recovered=a.decode_invalid_array(details,'cell_plan_01.png_invalid.npz')
    assert recovered.dtype==original.dtype and recovered.shape==original.shape
    assert recovered.tobytes()==original.tobytes()
    details[0]['sha256']='0'*64
    with pytest.raises(ValueError):a.decode_invalid_array(details,'cell_plan_01.png_invalid.npz')
