"""Numeric file/algebra fixtures only; no neural imports or execution."""
import ast
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"experiments/spectral_reachability/p5_jepa_native_response_audit_20261027_v1.py"
spec=importlib.util.spec_from_file_location("response_audit",SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_scope_and_fixed_numeric_only_tolerance():
    p=m.protocol()
    assert p["episode"]==1 and p["damping"]==.01
    assert p["fd_accuracy_threshold"] is None and p["fd_acceptance_verdict"]=="not_assessed"
    assert p["arithmetic_tolerance"]["rtol"]==1e-8 and p["arithmetic_tolerance"]["atol"]==1e-10
    assert not p["neural_execution"] and not p["P5_admission"]


def test_recompute_oracle_and_sign():
    j=np.tile(np.eye(10),(2,1));r=np.arange(20,dtype=float)/20
    c=m.recompute(j,r)
    np.testing.assert_allclose(c["gram"],np.eye(10)/10)
    np.testing.assert_allclose((c["gram"]+.01*np.eye(10))@c["delta"],j.T@r/20)
    np.testing.assert_allclose(c["linear_residual_mse"],np.mean((r-j@c["delta"])**2))
    np.testing.assert_allclose(c["raw_singular_direct"],np.full(10,np.sqrt(2)))


def test_rank_deficient_columns_not_removed():
    c=m.recompute(np.zeros((7,10)),np.ones(7))
    np.testing.assert_array_equal(c["delta"],np.zeros(10))
    assert c["linear_residual_mse"]==1 and c["gram"].shape==(10,10)


def test_numeric_compare_rejects_shape_nan_and_bad_value():
    assert not m.numeric_compare([1],[1,2])["consistent"]
    assert not m.numeric_compare([np.nan],[0])["consistent"]
    assert not m.numeric_compare([1],[2])["consistent"]
    assert m.numeric_compare([1.],[1.+1e-12])["consistent"]


def fixture_file(tmp_path,dtype=np.float32,shape=(2,10),name="response_columns.npy",allow_pickle=False):
    path=tmp_path/name;array=np.ones(shape,dtype=dtype)
    with path.open("xb") as f:np.save(f,array,allow_pickle=allow_pickle)
    record=dict(shape=list(shape),dtype=str(array.dtype),sha256=m.sha(path),file_bytes=path.stat().st_size)
    return path,record,array


def test_numeric_readonly_array(tmp_path):
    path,record,array=fixture_file(tmp_path)
    value=m.load_array(path,record,(2,10),np.dtype("float32"),tmp_path)
    assert not value.flags.writeable
    np.testing.assert_array_equal(value,array)


def test_sha_tamper_rejected(tmp_path):
    path,record,_=fixture_file(tmp_path);record["sha256"]="0"*64
    with pytest.raises(ValueError,match="SHA/size"):m.load_array(path,record,(2,10),np.dtype("float32"),tmp_path)


def test_header_shape_rejected_even_matching_record(tmp_path):
    path,record,_=fixture_file(tmp_path,shape=(4,5));record["shape"]=[2,10]
    with pytest.raises(ValueError,match="header"):m.load_array(path,record,(2,10),np.dtype("float32"),tmp_path)


def test_object_payload_never_unpickled(tmp_path):
    path,record,_=fixture_file(tmp_path,dtype=object,allow_pickle=True);record.update(dtype="float32",shape=[2,10])
    with pytest.raises(ValueError):m.load_array(path,record,(2,10),np.dtype("float32"),tmp_path)


def test_symlink_rejected(tmp_path):
    original,record,_=fixture_file(tmp_path,name="official_target.npy")
    link=tmp_path/"response_columns.npy";link.symlink_to(original)
    with pytest.raises(ValueError,match="symlink"):m.load_array(link,record,(2,10),np.dtype("float32"),tmp_path)


def test_wrong_directory_rejected(tmp_path):
    path,record,_=fixture_file(tmp_path)
    with pytest.raises(ValueError,match="directory"):m.load_array(path,record,(2,10),np.dtype("float32"),tmp_path/"elsewhere")


def fd_fixture(j):
    rows=[]
    for eps in (.01,.005):
        for i in range(10):
            norm=float(np.linalg.norm(j[:,i]));rows.append(dict(epsilon=eps,coordinate=i,rmse=0.,max_abs=0.,relative_l2=0.,fd_l2=norm,cosine=1.,diagnostic_only=True))
    return rows


def test_fd_summaries_not_invented_derivative_pass():
    j=np.tile(np.eye(10),(2,1));out=m.audit_fd(fd_fixture(j),j)
    assert len(out)==20 and all(x["summaries_consistent"] for x in out)
    assert all(x["derivative_agreement_pass"] is None and not x["FD_recomputed_from_vectors"] for x in out)


def test_large_selfconsistent_fd_error_not_screened():
    # Algebra fixture where FD=2*J: huge relative difference is preserved, not failed scientifically.
    j=np.tile(np.eye(10),(2,1));rows=fd_fixture(j)
    for row in rows:
        norm=float(np.linalg.norm(j[:,row["coordinate"]]));row.update(rmse=norm/np.sqrt(len(j)),max_abs=1.,relative_l2=1.,fd_l2=2*norm)
    out=m.audit_fd(rows,j)
    assert all(x["summaries_consistent"] and x["derivative_agreement_pass"] is None for x in out)


def test_fd_missing_row_and_corrupt_relative_error():
    j=np.tile(np.eye(10),(2,1));rows=fd_fixture(j)
    with pytest.raises(ValueError,match="fixed FD"):m.audit_fd(rows[:-1],j)
    rows[0]["relative_l2"]=1
    assert not m.audit_fd(rows,j)[0]["summaries_consistent"]


def test_fd_bad_bounds():
    j=np.tile(np.eye(10),(2,1));rows=fd_fixture(j);rows[0]["cosine"]=1.1
    assert not m.audit_fd(rows,j)[0]["summaries_consistent"]


def test_no_torch_pickle_or_neural_source_imports():
    tree=ast.parse(SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert all(n.name not in {"torch","pickle","importlib"} for n in node.names)
        if isinstance(node,ast.ImportFrom):assert node.module not in {"torch","pickle","importlib"}
    assert "allow_pickle=False" in SOURCE.read_text() and "mmap_mode=\"r\"" in SOURCE.read_text()


def test_actual_saved_arrays_if_present():
    prepared=m.prepare(ROOT);path=Path(prepared["expected_report"])
    if not path.exists():pytest.skip("waiting actual v8 result; not synthetic evidence")
    out=m.audit(ROOT,prepared,path)
    assert out["status"]=="saved_array_arithmetic_consistent"
    assert all(out["checks"].values())
    assert len(out["columns"])==10 and len(out["FD_summary_checks"])==20
    assert out["FD_accuracy_verdict"]=="not_assessed_no_threshold"
