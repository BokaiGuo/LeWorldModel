# New Wall panel: saved-plan coverage diagnosis

On the already stopped four-root plan-then-feasibility panel, analyze only
the saved v13 and global-CEM prediction batches. No new candidate,
optimizer, simulator rollout, or physical witness is allowed. First verify
the source plan audit and result/trace hashes. Reconstruct the stored
full-horizon actions with the existing alternate line-segment geometry.

For every root and arm report total full-chain samples, samples ever
crossing `x>0`, samples with a blocked step, successful samples under the
existing 0.05 threshold, and best crossing terminal distance. For the two
roots where the pre-existing global-CEM selected chain succeeds, compare
its action prefix to each recorded v13 candidate and retained set at each
search depth by coordinatewise action RMS. Record minimum distances and
nearest-retained status, without imposing a distance threshold.

Check task/arm pairing, selected action bounds, and all trace hashes. Count
duplicates as scored samples, not independent tasks. Report that Euclidean
action proximity may be misleading across collision branches. No result
here can prove that a retained prefix has a useful continuation, establish
causality, or change PC2/PC3. Do not reopen the source panel or use the
post-hoc reference as a planner input.
