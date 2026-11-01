#!/usr/bin/env bash
# Host-CUDA-only low-drift repair.  The selected-packet source is constructed
# from the *same* checkpoint from which we fine-tune, then its selection law is
# audited before any P1 calibration can consume the new checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
BASE="${1:?usage: $0 BASE_RUN SELECTOR_MATCHED_POOL RUN_ROOT}"
POOL="${2:?usage: $0 BASE_RUN SELECTOR_MATCHED_POOL RUN_ROOT}"
RUN="${3:?usage: $0 BASE_RUN SELECTOR_MATCHED_POOL RUN_ROOT}"
BASE="$(cd "${BASE}" && pwd)/$(basename "${BASE}")"
POOL="$(cd "${POOL}" && pwd)/$(basename "${POOL}")"
RUN="$(cd "$(dirname "${RUN}")" && pwd)/$(basename "${RUN}")"
CHECKPOINT="${BASE}/checkpoint"
STATS="${BASE}/stats.json"
SOURCE_MANIFEST="${POOL}/manifest.json"
RETENTION_POOL="${BASE}/../reacher_d2p_joint_repair_v1_20261024_cuda/counterfactual_pool"
DATASET="$(jq -er '.transition_dataset' "${CHECKPOINT}/repair_manifest.json")"

[[ ! -e "${RUN}" ]] || { echo "refusing to overwrite immutable run ${RUN}" >&2; exit 2; }
[[ -f "${CHECKPOINT}/weights.pt" && -f "${STATS}" && -f "${SOURCE_MANIFEST}" && -f "${RETENTION_POOL}/manifest.json" && -f "${DATASET}" ]] || {
  echo "selector-matched repair inputs are incomplete" >&2; exit 2;
}
jq -e '(.schema == "reacher_d2p_p1_proposal_pair_pool.v1") and (.status == "completed") and (.proposal_contract.damping == 1)' "${SOURCE_MANIFEST}" >/dev/null
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

mkdir -p "${RUN}"
cp "${STATS}" "${RUN}/stats.json"
.venv/bin/python -m experiments.spectral_reachability.train_reacher_d2p_joint_repair \
  --dataset "${DATASET}" \
  --pair-pool "${POOL}/pairs.h5" --pair-manifest "${SOURCE_MANIFEST}" \
  --retention-pair-pool "${RETENTION_POOL}/pairs.h5" --retention-pair-manifest "${RETENTION_POOL}/manifest.json" \
  --adapter-artifact "${RUN}/stats.json" --base-checkpoint "${CHECKPOINT}" --output "${RUN}/checkpoint" \
  --seed 20272101 --epochs 2 --transition-batch-size 32 --pair-batch-size 8 --steps-per-epoch 64 \
  --learning-rate 1e-6 --counterfactual-weight 1 --device cuda
.venv/bin/python -m experiments.spectral_reachability.audit_reacher_d2p_p1_selection_stability \
  --policy "${RUN}/checkpoint" --adapter-artifact "${RUN}/stats.json" --source-manifest "${SOURCE_MANIFEST}" \
  --device cuda --output "${RUN}/selector_stability"
jq -e '(.status == "completed") and (.selection.all_source_selections_stable == true)' "${RUN}/selector_stability/report.json" >/dev/null || {
  echo "low-drift repair changed its selected-packet deployment law; P1 remains sealed" >&2; exit 4;
}
