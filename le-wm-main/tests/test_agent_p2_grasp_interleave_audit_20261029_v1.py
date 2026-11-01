import hashlib
import pytest
from experiments.spectral_reachability import agent_p2_grasp_interleave_audit_20261029_v1 as audit


def test_real_source_binding_and_order():
    out = audit.inspect_sources()
    assert out['cuda_parity'] == 'not_tested'
    assert out['original'] != out['host']
    assert out['production_ready'] is False


def test_tamper_rejected(tmp_path):
    path = tmp_path / 'source.py'
    path.write_text('changed')
    with pytest.raises(ValueError, match='source SHA mismatch'):
        audit.inspect_sources(tmp_path, {'original': ('source.py', '0' * 64)})


def test_method_requires_unique_class_and_method():
    with pytest.raises(ValueError, match='class identity'):
        audit.method('class A: pass\nclass A: pass\n', 'A', 'f')
    with pytest.raises(ValueError, match='method identity'):
        audit.method('class A:\n def f(self): pass\n def f(self): pass\n', 'A', 'f')


def test_changed_source_hash_is_not_accepted(tmp_path):
    p = tmp_path / 'x'
    p.write_bytes(b'old')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_bytes(b'new')
    with pytest.raises(ValueError, match='source SHA mismatch'):
        audit.inspect_sources(tmp_path, {'original': ('x', digest)})
