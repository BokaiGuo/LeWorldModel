# Fixed-pair physical replay protocol

Development-only attribution on the two exposed roots in the sealed
normalized-direction experiment. Replay every one of the 13 fixed prefixes
per root with both previously executed tails, in root/prefix/L2/normalized
order. No candidate or root selection is allowed.

The replay uses the saved 15 raw actions (five prefix, ten suffix), not the
additional zero-action probe. It restores the original root and goal, records
every action and RGB frame, and retains stop/error events. Require exact root,
goal, warmup and cp5 RGB identity with the source; compare final wrapped
joint distance against the recorded source at absolute tolerance 1e-6. Any
mismatch prevents use for the subsequent latent attribution. Source receipt
and journal hashes and the full roster are checked before execution.

The replay has no neural calls and does not itself establish P1, independent
generalization, or an advantage. Subsequent encoding must use host CUDA and
actual cp5/cp10/cp15 frames with matching action history. Simulator state is
used only for restoring and evaluating the physical experiment, never as
planning input.
