"""Use the strict paired comparator with the verified one-pulse reference."""
from . import codex_p1_rgb_pusher_ablation_compare_20261030_main as comparison

comparison.NAME = "codex_p1_rgb_five_vs_single_20261030_main"
comparison.OLD_DIR = comparison.ROOT / "results/codex_p1_rgb_pusher_ablation_20261030_main_run_RUN01"
comparison.OLD_REPORT_SHA = "9a3afed2ba7abc964dc69c4f8498a8e6cd8256b0ec7439661a4c56b0bd4017e7"
comparison.OLD_SAVED = comparison.ROOT / "results/codex_p1_rgb_pusher_ablation_20261030_main_saved_RUN01.json"
comparison.OLD_SAVED_SHA = "7d3a232b63e25faa00e59735e8049737075e343f80e5671068214b019e43c40a"

if __name__ == "__main__":
    comparison.main()
