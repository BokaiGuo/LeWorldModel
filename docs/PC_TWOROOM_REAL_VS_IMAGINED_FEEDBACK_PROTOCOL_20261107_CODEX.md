# Real versus imagined feedback, fixed-prefix development cases

Previously exposed cases 38 and 39: use the same initial failed chain and
execute its first five actions in both arms. Subsequently run the same
680-candidate baseline search, seed, remaining horizon and old-tail policy
each packet. `real_rgb` re-encodes actual current RGB; `imagined` propagates
the model's predicted latent after its selected action packet and never reads
actual successor images for planning. The initial imagined latent is the
sealed predicted first successor of the same original chain. Both use a
single-latent current context at each replanning boundary and the same fixed
goal representation; hence the imagined arm deliberately does not carry a
multi-frame latent history. Record all queries, packet steps, synchronized
planning time, total wall time and actual actions/termination.

Before executing, verify the latent-context rollout and goal cost against
the ordinary RGB model on real initial observations and fixed candidate
actions; compare costs and predicted features, fail if max error exceeds
0.001. Verify frozen model, source hashes and original initial prefix. The
environment may stop each arm on true success; this stopping signal is not a
score or input to candidate choice. Because termination can occur at different
times, report both per-decision budgets and full episode costs. Search
candidates can diverge after the different state inputs change the CEM update.
These two posthoc failures are development diagnostics, not an independent
benchmark, descriptor test, or gate confirmation.
