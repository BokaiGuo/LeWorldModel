"""Create an explicitly reduced, single-GPU JEPA runner preflight config."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/jepa-wms/configs/evals/simu_env_planning/pt/jepa-wm/pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
DEST = ROOT / "results/pc_jepa_official_runner_smoke_config_20261105_codex.yaml"

with SOURCE.open() as f:
    cfg = yaml.safe_load(f)
cfg["folder"] = str(ROOT / "results/pc_jepa_official_runner_smoke_20261105_codex")
cfg["model_kwargs"]["checkpoint"] = str(ROOT / "data/jepa_wm_checkpoints/jepa_wm_pusht.pth.tar")
cfg["model_kwargs"]["pretrain_kwargs"]["heads_cfg"]["pretrain_dec_path"] = None
cfg["meta"]["eval_episodes"] = 1
cfg["logging"]["optional_plots"] = False
cfg["planner"]["iterations"] = 2
cfg["planner"]["num_samples"] = 8
cfg["planner"]["num_elites"] = 2
cfg["planner"]["decode_each_iteration"] = False
cfg["model_kwargs"]["data"]["loader"]["num_workers"] = 0
cfg["model_kwargs"]["data"]["loader"]["persistent_workers"] = False
with DEST.open("x") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Wrote reduced preflight config to {DEST}; not a paper benchmark.")
