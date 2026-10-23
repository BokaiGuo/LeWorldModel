"""Run frozen V17 high-velocity visible C1 evaluation on host CUDA."""
from __future__ import annotations
import json
from experiments.decision_benchmark import run_pointmaze_pairwise_contrast_v11_cuda as runner
from experiments.gdd_wm import preflight_pointmaze_pairwise_contrast_v16_visible_radial_velocity_alias as preflight

BASE_DATASET = runner.ROOT / "data" / "point_maze_v60_goal_radial_visible_high_velocity_history_20261023"
DATASET = BASE_DATASET
PREFLIGHT = runner.ROOT / "results" / "pointmaze_pairwise_contrast_v17_visible_high_velocity_static_preflight_20261023.json"
OUTPUT = runner.ROOT / "results" / "pointmaze_pairwise_contrast_v17_visible_high_velocity_20261023_cuda"
ROOT_RANGE=(24712,25544); SPLIT_COUNTS={"train":256,"calibration":256,"outcome":256}
ROOT_SEED,HELD_ACTION_SEED,TRAIN_ACTION_SEED=202611272,202611273,202611274
EPOCHS=160; RUN_LABEL="V17"; RUN_SCHEMA="pointmaze_pairwise_contrast_v17_visible_high_velocity_runtime.v1"; RUNTIME_FILENAME="v17_cuda_runtime.json"; CANDIDATE_DESIGN="velocity_matched_braking_pairs"
def _pool():
 m=json.loads((DATASET/'velocity_matched_manifest.json').read_text()); expected={"seed":202611271,"new_root_id_range":[24712,25543],"velocity_magnitude":1.5,"target_radius":0.3,"history_qpos_bounds":[0.05,3.95]}
 if any(m.get(k)!=v for k,v in expected.items()): raise RuntimeError("V17 pool contract mismatch")
 return {"resumed_v17_pool":True,"manifest":str((DATASET/'velocity_matched_manifest.json').resolve())}
def main():
 runner.BASE_DATASET=BASE_DATASET; runner.DATASET=DATASET; runner.PREFLIGHT=PREFLIGHT; runner.OUTPUT=OUTPUT; runner.ROOT_RANGE=ROOT_RANGE; runner.SPLIT_COUNTS=SPLIT_COUNTS; runner.EXTENSION_SEED=202611271; runner.ROOT_SEED,runner.HELD_ACTION_SEED,runner.TRAIN_ACTION_SEED=ROOT_SEED,HELD_ACTION_SEED,TRAIN_ACTION_SEED; runner.EPOCHS=EPOCHS; runner.RUN_LABEL=RUN_LABEL; runner.RUN_SCHEMA=RUN_SCHEMA; runner.RUNTIME_FILENAME=RUNTIME_FILENAME; runner.PREFLIGHT_MODULE=preflight; runner.CANDIDATE_DESIGN=CANDIDATE_DESIGN; runner._ensure_extension=_pool
 return runner.main()
if __name__=="__main__": raise SystemExit(main())
