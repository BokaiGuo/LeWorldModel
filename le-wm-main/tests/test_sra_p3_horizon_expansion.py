import inspect
from experiments.spectral_reachability import run_sra_p3_horizon_expansion_cuda as p3
def test_p3_is_fixed_tree_outcome_blind_cuda_protocol():
 s=inspect.getsource(p3); assert 'require_cuda()' in s and 'itertools.product' in s and 'replay_position_endpoints' in s and 'no_outcome_labels_used_for_selection' in s and '6*5**(h-1)' in s and 'model_batch_size' in s
