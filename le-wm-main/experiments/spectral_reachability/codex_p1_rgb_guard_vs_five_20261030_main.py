"""Run strict paired comparison against the sealed five-pulse experiment."""
from . import codex_p1_rgb_pusher_ablation_compare_20261030_main as comparison

comparison.NAME = "codex_p1_rgb_guard_vs_five_20261030_main"
comparison.OLD_DIR = comparison.ROOT / "results/codex_p1_rgb_pusher_fivepulse_20261030_main_run_RUN01"
comparison.OLD_REPORT_SHA = "443a08e1c93f5d792a57a2b7b584abb32575d1e7f393a17fa1471128733a54a6"
comparison.OLD_SAVED = comparison.ROOT / "results/codex_p1_rgb_pusher_fivepulse_20261030_main_saved_RUN01.json"
comparison.OLD_SAVED_SHA = "20af958e010126256b18d33967c0e33a2ab814a3b93e241a9dc0e7a6e5841ff6"

if __name__ == "__main__":
    comparison.main()
