#!/usr/bin/env bash
# Run the only admissible learned PushT comparison after a matched feasibility
# ceiling.  The two stages are deliberately serial: the learned loop cannot
# see the oracle until calibration-only authorization has been frozen.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PREFLIGHT="${PUSHT_CANDIDATE_HORIZON_PREFLIGHT_OUTPUT:-results/pusht_candidate_horizon_preflight_20261022_cuda}"
OUTPUT="${PUSHT_MODE_AWARE_MATCHED_OUTPUT:-results/pusht_mode_aware_preflight_matched_20261022_cuda}"

PUSHT_CANDIDATE_HORIZON_PREFLIGHT_OUTPUT="$PREFLIGHT" \
  bash experiments/run_pusht_candidate_horizon_preflight_cuda.sh

authorized="$(.venv/bin/python -c '
import json
from pathlib import Path
report = json.loads((Path("'"$PREFLIGHT"'") / "preflight.json").read_text(encoding="utf-8"))
print("true" if report["calibration_authorization"]["authorized"] else "false")
')"
if [[ "$authorized" != "true" ]]; then
  echo "Preflight is candidate_or_horizon_limited; learned ranker will not run." >&2
  exit 3
fi

exec .venv/bin/python -m experiments.run_pusht_mode_aware_shared_candidate \
  --output "$OUTPUT" \
  --device cuda \
  --root-protocol boundary_contact \
  --oracle-feasibility-preflight "$PREFLIGHT/preflight.json" \
  --train-roots 48 \
  --held-out-roots 32 \
  --pulse-steps 2 \
  --macro-pulse-steps 2 \
  --closure-steps 18 \
  --lookahead 2 \
  --inward-magnitude 0.08 \
  --tangent-magnitudes 0.06 0.10 0.16 \
  --contact-normal-magnitudes 0.02 0.04 0.08 \
  --focused-three-arm
