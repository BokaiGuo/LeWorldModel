# v22 fixed-bank tasks: post-seal feasibility diagnostic

The four-task v22 retention panel is stopped. This separate post-outcome
diagnostic does not rerun planners, change tasks, tune a selector, or feed
witnesses to any planner. After verifying the sealed manifest, trajectory
audit and probe-input audit, apply the existing exact-Wall geometric
feasibility constructor at its existing candidate heights, under the
unchanged 15-step horizon and action box. Record all attempts. Replay
eligible actions using both exact dynamics and alternate same-environment
geometry, saving actions, path and terminal distance. The unchanged
success criterion is `<0.05`.

Success proves existence only; failure does not prove infeasibility.
This privileged post-outcome diagnostic is not planner performance,
source independence, PC2 mechanism or independent replication. Run once;
no adaptive heights, retries or panel expansion.
