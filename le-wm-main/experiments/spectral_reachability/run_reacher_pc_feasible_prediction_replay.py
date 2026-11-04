"""Replay every sealed feasible-prefix physical arm at matching time points."""
from experiments.spectral_reachability import run_reacher_pc_guarded_prediction_replay as replay


if __name__ == "__main__":
    replay.ARMS = ("selected", "zero", "same_bank_greedy", "refined_0", "refined_1")
    replay.main()
