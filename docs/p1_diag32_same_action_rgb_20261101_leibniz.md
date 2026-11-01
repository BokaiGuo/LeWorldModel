# diag32 fixed same-action RGB diagnostic

Fixed known root0, candidate 4 / iteration 13 and all 15 exact actions from
the sealed diag32 plan and physical replay. No selection based on these
outcomes. Verify all physical dispatches, images, hashes, sources and native
stop before inference. Encode actual RGB at steps 0, 5, 10, 15, then perform
one open-loop 3-packet model rollout from the original saved latent/action
history using those same 15 normalized actions. Step 0 prediction denotes the
saved root latent, not a new rollout. Compare each latent to the fixed goal
with identity-192 MSE and predicted/actual latent RMSE; pair with saved actual
physical metrics. Recheck the original candidate q0/q1/q3.

Host CUDA only, fixed model and math SDPA, no training, nonblocking shared GPU
lock, source hashes checked before and after. Budget: four RGB encoder calls,
three action encoder calls, three predictions; ledger is `closed_unverified`,
not independently certified total compute. Prior optimization and physical
costs are excluded. Failures and partial receipts must not be promoted to
results. Known-root exploratory diagnostic only; no P1 admission.
