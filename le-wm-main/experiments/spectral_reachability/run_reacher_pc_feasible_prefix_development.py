"""Use the known-root development runner with the new backtracking compiler."""
from experiments.spectral_reachability import reacher_pc_feasible_prefix_backtrack as refinement
from experiments.spectral_reachability import run_reacher_pc_prefix_refinement_development as runner


if __name__ == "__main__":
    runner.refinement = refinement
    runner.main()
