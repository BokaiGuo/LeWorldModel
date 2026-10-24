#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "${ROOT}"
OUT="${ROOT}/results/decision_spectral_stage_gates_20261023_cuda_identity_audited.json"
[[ ! -e "${OUT}" ]] || { echo "refusing to overwrite stage-gate audit" >&2; exit 3; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi unavailable; refusing CPU fallback" >&2; exit 4; }
bash experiments/run_with_torch_cudnn.sh .venv/bin/python -m experiments.gpu_preflight --device cuda
.venv/bin/python -m experiments.decision_benchmark.audit_remaining_decision_spectral_stages \
  --v6r2-report results/pointmaze_counterfactual_endpoint_v6r2_decision_spectral_endpoint_20261023_cuda/report.json \
  --p1-audit results/pointmaze_counterfactual_endpoint_v6r2_decision_spectral_endpoint_20261023_cuda/decision_spectral_mismatch_binding_audit_20261023.json \
  --p1-identity-audit results/pointmaze_counterfactual_endpoint_v6r2_decision_spectral_endpoint_20261023_cuda/endpoint_execution_identity_audit_20261023.json \
  --p1-prime-audit results/pointmaze_v6r3d_pre_outcome_routing_audit_20261023.json \
  --p2-report results/decision_spectral_synthetic_p2_20261023_cuda/report.json \
  --p3-report results/spectral_repair_synthetic_p3_20261023_cuda/report.json \
  --p3-replication-report results/spectral_repair_synthetic_p3_20261023_cuda_seed202610232/report.json \
  --p3-replication-report results/spectral_repair_synthetic_p3_20261023_cuda_seed202610233/report.json \
  --p3v2-sweep results/spectral_repair_synthetic_p3_v2_task_metric_sweep_r2_20261023_cuda/report.json \
  --real-repair-preflight results/pointmaze_endpoint_repairability_router_v10_r6_static_preflight_20261023.json \
  --output "${OUT}"
