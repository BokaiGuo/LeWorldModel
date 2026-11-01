# Live PushT and CUDA visual Handler: bounded local smoke

This is a precommitted local integration diagnostic, not a P1 planner run or
fairness certification. It uses the source-bound fixed pretrained model,
host CUDA only, the cooperative GPU lock, a new output directory, and one
source encoder invocation. No training or CPU neural inference.

Fixed schedule: construct the real installed registered PushT, reset seed 0,
issue five literal `[0, 0]` actions, stop immediately on native termination,
render the actual fifth returned state, encode that frame through the real
visual Handler/Sink on CUDA, and close. The Handler requires boundary label 5;
the program checks five actual returned steps and actual actions first and
never pads a stopped trajectory. The rendered PNG, request, latent, returned
states/actions, termination flags, model hashes, and separate physical and
total guard journals are retained. Output is exclusive and failures retained.

Before execution a separate preparation binds source files including the
native PushT source, model/source manifest, this document, test and runner,
as well as fixed per-resource caps. It does not establish that every native
library or Python dependency is bound. Constructor/close/goal/state are
independent physical-vector costs, not existing total-guard axes. GPU wall
and memory, functional calls, captured methods, uncooperative processes,
unmeasured initialization and direct native calls remain incomplete. The
cooperative GPU lock is advisory.

This local path is not structured/CEM host execution, a task-performance
comparison or a known-root replay. A stopped rollout or failed callback is
reported as partial; no retry or favorable selection. Existing full-host
readiness remains closed and `closed_unverified` is not certification.
