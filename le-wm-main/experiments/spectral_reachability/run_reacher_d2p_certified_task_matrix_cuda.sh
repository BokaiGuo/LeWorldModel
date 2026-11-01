#!/usr/bin/env bash
set -euo pipefail
cd /home/gbk/code/LeWorldModel/le-wm-main
# Three fixed pre-existing actuator conditions on the same fresh metadata-only
# admitted source roots. Fail closed; never skip a failed/partial reservation.
for task in full_drive weak_shoulder weak_elbow; do
  output="results/reacher_d2p_certified_task_${task}_fresh8_20261025_cuda.json"
  if [[ -f "$output" ]]; then
    .venv/bin/python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "completed" and all(p["gates"].values()); assert p["sealed_protocol"]["compiler_variant"] == "geometry_certified"; assert p["sealed_protocol"]["task"] == sys.argv[2]; assert p["sealed_protocol"]["root_seeds"] == list(range(202633000,202633008)); print("Validated completed task, preserving immutable output:",sys.argv[2])' "$output" "$task"
    continue
  fi
  .venv/bin/python -u -m experiments.spectral_reachability.run_reacher_d2p_refined_dispatch \
    --compiler geometry_certified --task "$task" \
    --policy /home/gbk/.stable_worldmodel/checkpoints/reacher/lewm \
    --adapter-artifact results/reacher_d2p_p1_distribution_repair_v1_20261024_cuda/reacher_training_action_stats.json \
    --disjoint-reference results/reacher_d2p_common_geometry_fresh8_20261025_cuda.json \
    --output "$output" --seed 202633000 --roots 8
done
