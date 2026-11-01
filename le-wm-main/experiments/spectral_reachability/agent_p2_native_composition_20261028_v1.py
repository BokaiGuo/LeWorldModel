"""Direct reuse surface, NOT another composition runner or saved consumer.

The sealed native implementation already performs the minimal two-program
intervention: prefix actions produce a full intermediate context; suffix_cuda
compiles at that context; run_arm executes and witnesses the second program.
The control is a root-compiled joint-H2 SRA proposal with one full10 edge.

Production entry is execute_known, owned by MAIN and CUDA-only. Low-level native
functions are exposed by identity for integration, not standalone authorization.
run_arm(fixture=True) is only a RAM callback seam, not neural execution evidence.
No physical release entrypoint is provided. P1/freshness remain unadmitted.

This is conditional recompilation, NOT reuse of two independently precompiled
programs, a waypoint planner, arbitrary-depth search, or equal-FLOPs evidence.
"""
from experiments.spectral_reachability import agent_p2_runtime_guard_20261028_v1 as runtime
from experiments.spectral_reachability import p2_composition_predict_frontier_cuda_20261027_v2 as native
from experiments.spectral_reachability.d2p_composition_witness_v2 import compose as compose_action_witness

# Direct identities: do not wrap, fork, replace globals, or reseal old kernels.
prepare_known = runtime.prepare
verify_known = runtime.verify
execute_known = runtime.execute_known
run_known_root = runtime.run_known_root
run_native_arm = native.run_arm
compile_root_cuda = native.compile_cuda
compile_suffix_cuda = native.suffix_cuda
candidate_bank = native.candidate_bank
resource_allocation = runtime.contract.allocation

# Read-only integration guidance. These are missing prerequisites, not a new
# authority schema, fabricated certificates, an executable queue, or measurements.
UNMET_CONFIRMATORY_DEPENDENCIES = (
    'independent_P1_real_capability_change_and_task_benefit_MAIN_authority',
    'fresh_root_history_goal7_blind_and_complete_exclusion_certification',
    'full_resource_cost_instrumentation_and_MAIN_approved_common_stopping_rule',
    'exact_physical_restore_controller_success_and_renderer_MAIN_release',
)
UNIMPLEMENTED_GENERALIZATIONS = (
    'independently_precompiled_reusable_program_library_composition',
    'arbitrary_depth_or_non_greedy_multi_program_search',
)

__all__ = [
    'prepare_known','verify_known','execute_known','run_known_root','run_native_arm',
    'compile_root_cuda','compile_suffix_cuda','candidate_bank','resource_allocation',
    'compose_action_witness','UNMET_CONFIRMATORY_DEPENDENCIES','UNIMPLEMENTED_GENERALIZATIONS',
]
