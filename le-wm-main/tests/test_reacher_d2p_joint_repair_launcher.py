from pathlib import Path


def test_joint_repair_launcher_is_cuda_only_and_uses_fresh_root_ranges() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    source = (root / "run_reacher_d2p_joint_repair_cuda.sh").read_text(encoding="utf-8")
    assert "train_reacher_d2p_joint_repair" in source
    assert "build_reacher_d2p_counterfactual_pair_pool" in source
    assert "--device cuda" in source
    assert "20269100" in source and "20269200" in source and "20269300" in source
    assert "20269400" in source and "20269500" in source and "20269600" in source
    assert "20266100" not in source and "20266200" not in source and "20266300" not in source
    assert "observed_encoded_goal_cost" in source
    assert '--damping "${DAMPING}" --output "${RUN_ROOT}/p1_preflight"' in source
    assert "CPU" not in source
    assert 'RUN_ROOT="$(cd "$(dirname "${RUN_ROOT}")" && pwd)/$(basename "${RUN_ROOT}")"' in source


def test_joint_resume_requires_source_only_checkpoint_and_absolute_policy() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    source = (root / "resume_reacher_d2p_joint_repair_cuda.sh").read_text(encoding="utf-8")
    assert "trained_joint_source_only" in source
    assert "p0_p1_roots_used == false" in source
    assert 'RUN_ROOT="$(cd "$(dirname "${RUN_ROOT_INPUT}")" && pwd)/$(basename "${RUN_ROOT_INPUT}")"' in source
    assert '--policy "${CHECKPOINT}"' in source
    assert "20269300" in source and "20269600" in source


def test_p0_preflight_resolves_local_policy_before_loading() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    source = (root / "preflight_reacher_d2p_p0.py").read_text(encoding="utf-8")
    assert "args.policy = args.policy.resolve()" in source
    assert source.index("args.policy = args.policy.resolve()") < source.index("load_compatible_pretrained(args.policy")


def test_joint_p1_resume_rebinds_preflight_to_selected_damping() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    source = (root / "resume_reacher_d2p_joint_p1_cuda.sh").read_text(encoding="utf-8")
    assert "p1_preflight_selected_damping" in source
    assert "--seed 20269410" in source
    assert '--damping "${DAMPING}" --output "${PREFLIGHT}"' in source
    assert "20269500" in source and "20269600" in source


def test_p1_distribution_repair_uses_selected_packets_and_binds_source_roots() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    source = (root / "run_reacher_d2p_p1_distribution_repair_cuda.sh").read_text(encoding="utf-8")
    assert "build_reacher_d2p_p1_proposal_pair_pool" in source
    assert "--training-source-manifest \"${RUN_ROOT}/checkpoint/repair_manifest.json\"" in source
    assert "--seed 20268100" in source and "--seed 20270100" in source and "--seed 20270200" in source
    assert "--device cuda" in source
    assert "CPU" not in source


def test_damping_matched_repair_regenerates_source_then_refuses_a_changed_deployment_damping() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    source = (root / "run_reacher_d2p_damping_matched_repair_cuda.sh").read_text(encoding="utf-8")
    assert "build_reacher_d2p_p1_proposal_pair_pool" in source
    assert '--damping "${D}"' in source
    assert '[[ "${D_NEXT}" == "${D}" ]]' in source
    assert "20271600" in source and "20272000" in source and "20272400" in source
    assert "--device cuda" in source
    assert "CPU" not in source


def test_damping10_continuation_binds_to_the_newly_selected_damping_and_uses_fresh_roots() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    source = (root / "run_reacher_d2p_damping10_repair_cuda.sh").read_text(encoding="utf-8")
    assert '[[ "${D}" == "10" || "${D}" == "10.0" ]]' in source
    assert '--damping "${D}"' in source
    assert '[[ "${D_NEXT}" == "${D}" ]]' in source
    assert "20272500" in source and "20272900" in source and "20273300" in source
    assert "--device cuda" in source
    assert "CPU" not in source


def test_multidamping_repair_covers_all_predeclared_deployment_dampings() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments" / "spectral_reachability"
    source = (root / "run_reacher_d2p_multidamping_repair_cuda.sh").read_text(encoding="utf-8")
    assert 'DS=(0.001 0.01 0.1 1 10)' in source
    assert "merge_reacher_d2p_p1_proposal_pair_pools" in source
    assert "audit_reacher_d2p_p1_selection_stability" in source
    assert "SELECTOR_RECEIPTS" in source
    assert "20273400" in source and "20273800" in source and "20274600" in source
    assert '"${MANIFESTS[@]}"' in source
    assert "--device cuda" in source
    assert "CPU" not in source
