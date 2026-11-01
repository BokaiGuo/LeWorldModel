import pytest
from experiments.spectral_reachability import codex_p2_program_library_accounting_20261030_v1 as a


def test_fixed_caps_and_source_material():
    p=a.material()
    assert p["caps"]==a.CAPS and len(p["cases"])==4
    assert p["P2_admission"] is False and p["loaded_dynamic_closure"]=="unknown"
    assert any(x["path"].endswith("/weights.pt") for x in p["sources"])


def test_nested_calls_and_real_failure_are_not_returns():
    events=[];m=a.Meter(events.append)
    assert m.call("jacobian",lambda:m.call("packet",lambda:4))==4
    assert [(e["kind"],e["resource"],e["depth"]) for e in events]==[
        ("resource_attempt","jacobian",0),("resource_attempt","packet",1),
        ("resource_return","packet",1),("resource_return","jacobian",0)]
    def fail():
        raise RuntimeError("device failure")
    with pytest.raises(RuntimeError):
        m.call("packet",fail)
    assert m.attempted["packet"]==2 and m.returned["packet"]==1
    with pytest.raises(ValueError):
        m.call("packet",lambda:1)


def test_exhaustion_prevents_callback():
    events=[];m=a.Meter(events.append);called=[]
    m.attempted["jacobian"]=a.CAPS["jacobian"]
    with pytest.raises(ValueError):
        m.call("jacobian",lambda:called.append(1))
    assert not called and not events


def test_no_emit_failure_no_dispatch():
    def fail(_):
        raise OSError("durability failed")
    m=a.Meter(fail);called=[]
    with pytest.raises(OSError):
        m.call("packet",lambda:called.append(1))
    assert not called and m.attempted["packet"]==0
