"""Independent NumPy-only audit of saved v8 arrays; never imports neural code."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

NAME="p5_jepa_native_response_audit_20261027_v1"
PARENT="p5_jepa_native_action_response_20261027_v8"
PARENT_MANIFEST_SHA="0588f825c29b884350e97ed0dc47d8872bb539d0ca7a7faf045190eee5f3340f"
D=98304
RTOL=1e-8
ATOL=1e-10
LAYOUT={"response_columns":((D,10),np.dtype("float32")),
    "nominal_successor":((1,256,384),np.dtype("float32")),
    "official_target":((1,256,384),np.dtype("float32")),
    "target_minus_successor":((1,256,384),np.dtype("float64"))}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024**2),b""):h.update(b)
    return h.hexdigest()


def array_sha(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def protocol():
    return dict(schema=NAME,episode=1,parent=PARENT,parent_manifest_sha256=PARENT_MANIFEST_SHA,
        array_policy="bounded numeric .npy headers/payload/SHA, mmap read-only, allow_pickle=False, no symlinks/escapes",
        arithmetic_tolerance=dict(rtol=RTOL,atol=ATOL,scope="CPU/GPU float64 arithmetic receipt consistency only, NOT FD/AD scientific accuracy"),
        residual="target.astype(float64)-nominal.astype(float64), exact saved-array equality",
        gram="J.T@J/98304",rhs="J.T@residual/98304",damping=.01,
        spectra="independent tall-thin SVD and 10x10 eigvalsh; gate reported singular values squared/normalized against Gram eigenvalues to avoid sqrt roundoff instability at zero",
        finite_difference="20 fixed summaries (.01/.005 x ten coordinates), algebra consistency only; raw FD/plus/minus vectors NOT saved, cannot independently recompute derivatives",
        fd_accuracy_threshold=None,fd_acceptance_verdict="not_assessed",selection="all saved columns/epsilons, no replacement/filtering",
        neural_execution=False,training=False,P5_admission=False)


def check_seals(seals):
    bad=[p for p,h in seals.items() if not Path(p).is_file() or sha(p)!=h]
    if bad:raise ValueError("sealed files differ: "+json.dumps(bad))


def parent_manifest(root):
    path=root/f"results/{PARENT}/manifest_frozen.json"
    if sha(path)!=PARENT_MANIFEST_SHA:raise ValueError("frozen v8 manifest SHA rejected")
    value=json.loads(path.read_text());check_seals(value["seals"])
    if value["schema"]!=PARENT or value["status"]!="prepared_non_neural" or value["row"]["episode"]!=1:
        raise ValueError("wrong v8 prepared protocol")
    return path,value


def prepare(root):
    path,parent=parent_manifest(root)
    seals=dict(parent["seals"])
    files=[path,Path(__file__).resolve(),root/"tests"/f"test_{NAME}.py",root.parent/"docs"/f"{NAME}.md"]
    seals.update({str(p):sha(p) for p in files});check_seals(seals)
    return dict(schema=NAME,status="prepared_non_neural",protocol=protocol(),seals=seals,
        expected_report=str(root/f"results/{PARENT}/episode1_native_action_response_cuda_main_v1.json"),
        actual_result_read=False,neural_execution=False,P5_admission=False)


def validate_manifest(root,manifest):
    if prepare(root)!=manifest:raise ValueError("audit source/protocol/seals replay differs")


def numeric_compare(actual,expected):
    a=np.asarray(actual,np.float64);b=np.asarray(expected,np.float64)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(consistent=False,actual_shape=list(a.shape),expected_shape=list(b.shape))
    diff=np.abs(a-b)
    return dict(consistent=bool(np.allclose(a,b,rtol=RTOL,atol=ATOL)),
        max_abs=float(diff.max()) if diff.size else 0.,reference_scale=float(np.abs(b).max()) if b.size else 0.)


def load_array(path,record,shape,dtype,allowed_dir):
    """Validate small header BEFORE mmap; do not trust path, shape or byte count."""
    path=Path(path);allowed_dir=Path(allowed_dir)
    if path.name not in {k+".npy" for k in LAYOUT} or path.parent!=allowed_dir or not path.is_absolute():
        raise ValueError("array path outside exact artifact directory")
    if any(p.is_symlink() for p in (path,*path.parents)) or not path.resolve().is_relative_to(allowed_dir.resolve()):
        raise ValueError("symlink/escape rejected")
    expected_bytes=math.prod(shape)*dtype.itemsize
    size=path.stat().st_size
    if not expected_bytes<size<=expected_bytes+1024 or size!=record["file_bytes"] or sha(path)!=record["sha256"]:
        raise ValueError("array SHA/size rejected")
    if record["shape"]!=list(shape) or record["dtype"]!=str(dtype):raise ValueError("array receipt shape/dtype rejected")
    with path.open("rb") as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):header=np.lib.format.read_array_header_1_0(f,max_header_size=1024)
        elif version==(2,0):header=np.lib.format.read_array_header_2_0(f,max_header_size=1024)
        else:raise ValueError("unsupported npy version")
        stored_shape,fortran,stored_dtype=header
        if stored_shape!=shape or stored_dtype!=dtype or stored_dtype.hasobject or fortran:
            raise ValueError("unsafe/mismatched npy header")
        if f.tell()+expected_bytes!=size:raise ValueError("npy payload/trailing bytes rejected")
    value=np.load(path,allow_pickle=False,mmap_mode="r",max_header_size=1024)
    if not np.isfinite(value).all():raise ValueError("nonfinite saved array")
    return value


def recompute(j,residual):
    j=np.asarray(j,np.float64);r=np.asarray(residual,np.float64).reshape(-1)
    if j.ndim!=2 or j.shape[1]!=10 or len(j)==0 or len(r)!=len(j) or not np.isfinite(j).all() or not np.isfinite(r).all():
        raise ValueError("finite aligned ten-column response required")
    count=len(j);g=j.T@j/count;b=j.T@r/count;e=np.linalg.eigvalsh(g)
    delta=np.linalg.solve(g+.01*np.eye(10),b);sv=np.linalg.svd(j,full_matrices=False,compute_uv=False)
    return dict(gram=g,rhs=b,eigenvalues=e,delta=delta,raw_singular_direct=sv,
        normalized_singular_direct=sv/np.sqrt(count),linear_residual_mse=float(np.mean((r-j@delta)**2)),
        solve_residual_norm=float(np.linalg.norm((g+.01*np.eye(10))@delta-b)))


def audit_fd(rows,j):
    expected=[(eps,i) for eps in (.01,.005) for i in range(10)]
    if [(r["epsilon"],r["coordinate"]) for r in rows]!=expected:
        raise ValueError("all fixed FD epsilon/coordinate rows required in order")
    results=[]
    for row in rows:
        i=row["coordinate"];adnorm=float(np.linalg.norm(np.asarray(j[:,i],np.float64)))
        if not row["diagnostic_only"] or not all(math.isfinite(row[k]) for k in ("rmse","max_abs","relative_l2","fd_l2","cosine")):
            raise ValueError("nonfinite/nondiagnostic FD receipt")
        if any(row[k]<0 for k in ("rmse","max_abs","relative_l2","fd_l2")):
            raise ValueError("negative FD norm/error receipt")
        errornorm=row["rmse"]*np.sqrt(len(j));product=row["fd_l2"]*adnorm
        norm_identity=row["fd_l2"]**2+adnorm**2-2*row["cosine"]*max(product,1e-24)
        bounds=(row["rmse"]<=row["max_abs"]+ATOL and row["max_abs"]<=errornorm+ATOL and abs(row["cosine"])<=1+ATOL)
        comparisons=dict(relative_l2=numeric_compare(row["relative_l2"],errornorm/max(adnorm,1e-12)),
            norm_cosine_identity=numeric_compare(errornorm**2,norm_identity))
        results.append(dict(epsilon=row["epsilon"],coordinate=i,recorded=row,
            saved_JVP_column_l2=adnorm,summary_bounds_consistent=bool(bounds),arithmetic=comparisons,
            summaries_consistent=bool(bounds and all(c["consistent"] for c in comparisons.values())),
            derivative_agreement_pass=None,FD_recomputed_from_vectors=False))
    return results


def audit(root,manifest,report_path):
    validate_manifest(root,manifest)
    report_path=Path(report_path).resolve()
    if report_path!=Path(manifest["expected_report"]):raise ValueError("fixed v8 report path required")
    if not report_path.is_file():raise FileNotFoundError("awaiting actual v8 report; do not fabricate result")
    before_sha=sha(report_path);report=json.loads(report_path.read_text());_,parent=parent_manifest(root)
    if report["schema"]!=PARENT or report["status"]!="native_action_response_recorded" or report["episode"]!=1:
        raise ValueError("v8 did not record complete native response; inspect original failure")
    checks={}
    checks["protocol_exact"]=report["protocol"]==parent["protocol"]
    checks["canonical_parent_manifest"]=report["manifest_canonical_sha256"]==hashlib.sha256(json.dumps(parent,sort_keys=True).encode()).hexdigest()
    checks["source_freeze_flags"]=all(report[k] for k in ("source_data_seals_unchanged","frozen_state_unchanged","count_contract_matches","frame_contract_matches","phase_contract_matches","cuda_executed"))
    checks["model_receipts"]=report["model_before"]==report["model_after"]==parent["expected_model_receipts"]
    checks["strict_dynamics"]=all(c["exact_match"] and not any(c[k] for k in ("missing_keys","unexpected_keys","shape_mismatches","dtype_mismatches")) for c in report["strict_comparison"].values())
    checks["native_sources"]=all(p in parent["seals"] and sha(p)==parent["seals"][p] for p in report["imported_native_sources"].values())
    checks["no_execution_claim"]=not report["compiled_delta_executed"] and not report["physical_execution"] and not report["P5_admission"]
    counts=Counter(t["module"] for t in report["trace"])
    checks["forward_counts"]=all(counts[k]==report["counts"][k]==parent["protocol"]["counts"][k] for k in ("official_encoder","proprio_encoder","predictor"))
    checks["AD_API_counts"]=report["counts"]==parent["protocol"]["counts"]
    phases=Counter(t["phase"]["kind"] for t in report["trace"] if t["module"]=="predictor")
    checks["phases"]=dict(phases)==report["predictor_phase_counts"]==dict(nominal=1,jvp=10,finite_difference=40)
    frames={k:sum(t["frames"] for t in report["trace"] if t["module"]==k) for k in counts}
    checks["frames"]=frames==report["frame_cost_observed"]==dict(official_encoder=4,proprio_encoder=3,predictor=153)
    predictor_trace=[t for t in report["trace"] if t["module"]=="predictor"]
    expected_phases=[dict(kind="nominal")]+[dict(kind="jvp",coordinate=i) for i in range(10)]+[dict(kind="finite_difference",epsilon=eps,coordinate=i,sign=sign) for eps in (.01,.005) for i in range(10) for sign in ("plus","minus")]
    checks["exact_predictor_phase_order"]=len(predictor_trace)==51 and all(t["phase"]==p and t["input_shapes"]==[[1,3,1,16,16,384],[1,3,10],[1,3,256,16]] for t,p in zip(predictor_trace,expected_phases))
    binding=report["input_binding"];saved=parent["row"]
    checks["input_bindings"]=binding["episode"]==1 and binding["frames"]==[0,5,10,15] and binding["rgb_sha256"]==saved["rgb_sha256"] and binding["video_length"]==saved["video_length"] and binding["histories_sha256"]=={k:v["sha256"] for k,v in saved["inputs"].items()}
    checks["actual_normalized_center"]=binding["actual_normalized_packet"]==np.asarray(saved["inputs"]["packets"]["values"][-1]).reshape(5,2).tolist()
    checks["preprocessing_gate"]=math.isfinite(binding["preprocess_max_abs"]) and 0<=binding["preprocess_max_abs"]<=parent["protocol"]["gates"]["preprocess_max_abs"]
    checks["v7_features"]=binding["native_features_sha256"]==parent["expected_v7_features"]["official_features_sha256"] and binding["target_sha256"]==parent["expected_v7_features"]["target_features_sha256"]
    artifact_dir=Path(str(report_path.with_suffix(""))+"_assets")
    if not artifact_dir.is_relative_to(root):raise ValueError("artifact directory outside project")
    if set(report["audit_arrays"])!=set(LAYOUT):raise ValueError("complete audit arrays required")
    arrays={}
    for key,(shape,dtype) in LAYOUT.items():
        record=report["audit_arrays"][key];path=Path(record["path"])
        if path!=artifact_dir/(key+".npy"):raise ValueError("wrong array filename")
        arrays[key]=load_array(path,record,shape,dtype,artifact_dir)
    j=arrays["response_columns"];nominal=arrays["nominal_successor"];target=arrays["official_target"];residual=arrays["target_minus_successor"]
    checks["residual_exact"]=bool(np.array_equal(residual,target.astype(np.float64)-nominal.astype(np.float64)))
    checks["target_nominal_hashes"]=array_sha(target)==binding["target_sha256"] and array_sha(nominal)==binding["successor_sha256"]
    mse=float(np.mean(np.asarray(residual)**2));comparisons={"nominal_patch_mse":numeric_compare(report["target_residual"]["global_patch_mse"],mse)}
    checks["v7_mse_gate_preserved"]=abs(mse-parent["v7_nominal_actual_mse"])<=parent["protocol"]["gates"]["nominal_vs_v7_mse_abs"]
    col_rows=report["jvp_columns"]
    if [c["coordinate"] for c in col_rows]!=list(range(10)):raise ValueError("all ten ordered JVP columns required")
    columns=[]
    for i,row in enumerate(col_rows):
        col=np.asarray(j[:,i]);norm=float(np.linalg.norm(col.astype(np.float64)))
        cc=dict(hash_exact=array_sha(col)==row["sha256"],coordinate_axis=row["step"]==i//2 and row["axis"]==("x" if i%2==0 else "y"),
            primal_gate=math.isfinite(row["primal_max_abs"]) and 0<=row["primal_max_abs"]<=parent["protocol"]["gates"]["jvp_primal_vs_nominal_max_abs"])
        nc=dict(norm=numeric_compare(row["l2_norm"],norm),max_abs=numeric_compare(row["max_abs"],float(np.abs(col).max())))
        columns.append(dict(coordinate=i,checks=cc,numeric=nc,consistent=all(cc.values()) and all(x["consistent"] for x in nc.values())))
    checks["all_JVP_columns_bound"]=all(c["consistent"] for c in columns)
    calc=recompute(j,residual);compiled=report["response_compile"]
    checks["compile_layout_damping"]=compiled["response_shape"]==[D,10] and compiled["gram_divisor"]==D and compiled["damping"]==.01 and not compiled["applied"] and not compiled["nonlinear_candidate_evaluated"]
    for field,key in (("gram","gram"),("rhs","rhs"),("eigenvalues_normalized_gram","eigenvalues"),("linear_residual_mse","linear_residual_mse"),("solved_equation_residual_norm","solve_residual_norm")):
        comparisons[field]=numeric_compare(compiled[field],calc[key])
    comparisons["delta_normalized"]=numeric_compare(compiled["delta_normalized"],calc["delta"].reshape(5,2))
    sorted_e=np.maximum(calc["eigenvalues"],0)[::-1]
    comparisons["mse_singular_squared"]=numeric_compare(np.asarray(compiled["singular_values_mse_response"])**2,sorted_e)
    comparisons["raw_singular_squared_div_D"]=numeric_compare(np.asarray(compiled["singular_values_raw_J"])**2/D,sorted_e)
    checks["recomputed_arithmetic"]=all(c["consistent"] for c in comparisons.values())
    fd=audit_fd(report["finite_differences"],j)
    checks["FD_summaries_algebra_only"]=all(row["summaries_consistent"] for row in fd)
    checks["source_and_artifacts_unchanged_after"]=sha(report_path)==before_sha and all(sha(v["path"])==v["sha256"] for v in report["audit_arrays"].values())
    check_seals(manifest["seals"])
    finite_summary={str(eps):{k:max(row["recorded"][k] for row in fd if row["epsilon"]==eps) for k in ("rmse","max_abs","relative_l2")} for eps in (.01,.005)}
    return dict(schema=NAME,status="saved_array_arithmetic_consistent" if all(checks.values()) else "audit_inconsistent",
        protocol=protocol(),checks=checks,comparisons=comparisons,columns=columns,FD_summary_checks=fd,
        FD_recorded_maxima=finite_summary,FD_accuracy_verdict="not_assessed_no_threshold",FD_vectors_recomputed=False,
        recomputed={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in calc.items()},
        singular_values_report_vs_direct_svd=dict(raw_max_abs=float(np.max(np.abs(np.asarray(compiled["singular_values_raw_J"])-calc["raw_singular_direct"]))),
            mse_normalized_max_abs=float(np.max(np.abs(np.asarray(compiled["singular_values_mse_response"])-calc["normalized_singular_direct"])))),
        bindings=dict(parent_manifest_sha256=PARENT_MANIFEST_SHA,parent_report_path=str(report_path),parent_report_sha256=before_sha,audit_arrays=report["audit_arrays"]),
        limitation="Independent saved-array algebra/hash audit, not independent model/JVP/FD recomputation; FD vectors are unavailable; no FD pass or P5 efficacy claim.",
        neural_execution=False,P5_admission=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--audit",action="store_true");parser.add_argument("--manifest",type=Path)
    parser.add_argument("--report",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.output.exists():raise FileExistsError(args.output)
    if args.audit:
        if args.manifest is None:parser.error("--manifest required")
        manifest=json.loads(args.manifest.read_text());report=args.report or Path(manifest["expected_report"])
        result=audit(root,manifest,report)
    else:result=prepare(root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(result,f,sort_keys=True,indent=2,allow_nan=False)
    print(result["status"])
    if result["status"]=="audit_inconsistent":raise SystemExit(1)


if __name__=="__main__":main()
