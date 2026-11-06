# Fixed-bank tasks: post-seal feasibility diagnostic

The four-task retention panel is stopped. This separate post-hoc diagnostic
does not rerun planners, change tasks, tune a selector, or feed witnesses
to a planner. After verifying its sealed manifest, trajectory audit and
probe-input audit, apply the existing exact-Wall geometric feasibility
constructor at its pre-existing candidate heights, under the unchanged
15-step horizon and action box. Save all attempts, including failures.
For any eligible witness, replay with both exact dynamics and the alternate
same-environment geometry, saving actions, path and terminal distance.
The fixed success criterion is `<0.05`.

Success shows only that an action chain exists for that task. Failure to
construct a witness does not establish infeasibility. This privileged
post-outcome diagnostic is not method performance, source independence,
PC2 mechanism, or independent replication. Run once and preserve the
result; no adaptive heights, retries, or panel expansion.
