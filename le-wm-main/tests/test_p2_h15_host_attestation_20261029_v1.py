import hashlib
import pytest
from experiments.spectral_reachability import p2_h15_host_attestation_20261029_v1 as m


class Model:
    def modules(self): return []
    def parameters(self): return []
    def buffers(self): return []
    def state_dict(self): return {}
    def named_buffers(self): return []


class Device:
    type = 'cuda'


class Sink:
    def event(self, value): pass


class Flags:
    class backends:
        class cuda:
            class matmul: allow_tf32 = False
            @staticmethod
            def math_sdp_enabled(): return True
            @staticmethod
            def flash_sdp_enabled(): return False
            @staticmethod
            def mem_efficient_sdp_enabled(): return False
            @staticmethod
            def cudnn_sdp_enabled(): return False
        class cudnn: allow_tf32 = False
    @staticmethod
    def is_autocast_enabled(device): return False


def test_real_callback_construction_and_two_root_identity(tmp_path):
    bridge=m.bridge.build_bridge()
    src=tmp_path/'source'; src.write_bytes(b'sealed')
    files={str(src):hashlib.sha256(b'sealed').hexdigest()}
    obj=m.Session(bridge, Model(), Device(), Flags, files, m.model_hash(Model()), fixture=True)
    paired=m.bind_roots(obj, lambda root: Sink())
    for root, pair in enumerate(paired):
        cb=pair['callbacks']
        assert isinstance(cb.backend, m.native.NativeTree)
        assert cb.backend.task == bridge['roots'][root]['task']
        assert pair['ledger'].caps == m.budget()
        with pytest.raises(ValueError, match='not authorized'): cb.signal(None,pair['ledger'])
    assert obj.close()['fixture_only'] is True


def test_model_precision_and_source_mismatch_fail_closed(tmp_path):
    bridge=m.bridge.build_bridge()
    src=tmp_path/'source'; src.write_bytes(b'sealed')
    files={str(src):hashlib.sha256(b'sealed').hexdigest()}
    with pytest.raises(ValueError): m.Session(bridge, Model(), Device(), Flags, files, {'original_state_dict':'0'*64,'all_buffers':'0'*64}, fixture=True)
    class Bad(Flags):
        @staticmethod
        def is_autocast_enabled(device): return True
    with pytest.raises(ValueError): m.Session(bridge, Model(), Device(), Bad, files, m.model_hash(Model()), fixture=True)
    obj=m.Session(bridge,Model(),Device(),Flags,files,m.model_hash(Model()),fixture=True)
    src.write_bytes(b'changed')
    with pytest.raises(ValueError): obj.close()


def test_fixed_budget_and_host_refusal():
    assert m.budget() == m.single.caps()
    with pytest.raises(ValueError): m.host_dispatch(None)


def test_offline_preflight_saved_and_consumed(tmp_path):
    path = tmp_path/'preflight.json'
    digest = m.save_preflight(path)
    checked = m.consume_preflight(path, digest)
    assert checked['inventory_file_count'] == len(m.historical_inventory())
    assert not checked['host_authorized'] and not checked['runtime_model_verified']
    with pytest.raises(ValueError): m.consume_preflight(path, '0'*64)
    with pytest.raises(FileExistsError): m.save_preflight(path)
