# P3 exploratory dual-branch feedback contract

The opened R8 MAIN01 decision journal has SHA-256
`7eed6f6b1e55b07a0e509f6dc721c8b2d949421f91f198fb26686053bfd5319b`.
All 24,117 journal records were checked against their index, predecessor and
record hash, and the file hash matched `end.json`. At indices 4017, 8036,
12055, 16074, 20093 and 24112, the six original stage-zero decisions all say
continue, `progress/progress`, four candidates, horizon 3. All six slot
statuses are failed, with cleanup returned; the old run remains failed.
These records do not establish the hypothetical later decisions.

This new post-observation exploratory contract accepts the original root
scheduler output unchanged: `progress/progress/4` or
`reconfiguration/reconfiguration/8`, horizon 1 or 3, continue true. All other
decisions, including composition and stop, fail closed. At every decision,
both H1 and H3 panels use the *original branch and width*: progress uses the
original progress bank generator, four actual native replays and four immediate
model costs per horizon; R uses `evaluate_R`, zero plus eight proposals,
postcontext signals and suffixes. Each horizon's winner is selected by
immediate model cost and index from eligible rows and receives separate whole
and segmented replays. R without both witnesses fails without dispatch. These
are matched horizon-control panels, not the original full tree.

Before scheduling, the ledger holds the componentwise maximum of the two
branch budgets; immediately after the actual scheduler result it tightens to
the exact branch vector, before dispatching either group. Both groups require
pre-admission with remaining verification work. Completion requires attempted
and returned vectors exactly equal to the branch vector; failures retain
attempted counts and do not dispatch a packet. The principal overlapping axes:

| Per completed decision | progress | reconfiguration |
|---|---:|---:|
| predict API | 32 | 106 |
| VJP directions | 384 | 3840 |
| Jacobian API | 2 | 20 |
| replay | 12 | 40 |
| program compile | 8 | 20 |
| array SVD | 4 | 22 |
| solve | 2 | 18 |
| metric | 8 | 36 |

These are not FLOPs. Same-branch, completed decisions have matching recorded
vectors; if actual feedback changes the scheduler branch across arms, the
arms do not have equal work. The design cannot claim unconditional equal
budget, independent roots, P1 passage, full-tree equivalence, physical
certification, or a scientific pass.

The independent host driver binds this seam to the existing CUDA adapter,
UUID bridge, two prepared roots, six slots, feedback callbacks, durable
stores, unique stage-scoped capsules, resource guards and failure handling.
The durable journal starts with the inherited transport's legacy contract
header; dual-specific events and terminal identify the actual contract.
Its old saved consumer is incompatible. The exact dual-branch native path,
source closure and full saved cross-log certification remain unverified;
the 32 MiB individual journal-record limit and engineering caps can stop a
real run. A hard kill can leave partials without a terminal. Preserve partials
and never retry an existing output. No GPU, physical run or training was
performed in preparing this version.
