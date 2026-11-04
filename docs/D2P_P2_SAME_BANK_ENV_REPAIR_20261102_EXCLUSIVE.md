# P2 known-root environment-repair development protocol

This is a separately authorized development attempt, not an automatic retry
under the original no-retry plan. The old technical plan remains the immutable
specification of the fixed computation; the new SHA-bound repair plan supplies
new authorization, environment preflight and a different exclusive output.
The old failed attempt remains a failure regardless of the new outcome.

## Observations and source verification

RUN_01 report SHA-256 is
`69454fe8fb16c9a210f131ac00f6ff041a1aa0be1db94ae781747ed8652d058a`,
journal SHA-256 `4c299c40b1a4b5e79b5545f2ca9ed2f4ed8cb98dd4c55685c4dadabd9fc0e3d0`,
binding SHA-256 `e664443b39c66dbe81b59ef7d40b1e24fdc2aa4f7de9f50ddd909f18309ef53e`.
The journal has only `ledger_open`: it records no objective or candidate
operations; this does not prove the loader was never entered. The user reports
that a prefixed command saw no CUDA in the sandbox while an approved direct
interpreter command saw CUDA on the host. The recorded diagnostic also
reports a host tensor allocation. No CUDA diagnosis was repeated while
preparing this protocol. The evidence does not establish that the BLAS
variable caused the visibility difference.

CPU-only verification of technical plan SHA
`2cf0b49d0769d735a8ac82b8c6e19b6f942a56f7f893a3c87f356a91a5df645f`
passed against its explicitly listed source and input bindings. The new plan
is `le-wm-main/results/p2_same_bank_env_repair_20261102_exclusive_PLAN_01.json`,
SHA `860d6c4a03bc32b23aa12b1dba9aba046c7de9d3c9b12619fdedeb628d6347ea`.
The repair wrapper SHA is
`2acc9108069419aecbf0aef8b67392319eefd71e3aae3e74209af1d2c148474b`.
Neither manifest establishes the complete transitive runtime or hardware
identity.

## Fixed scope and cost

Same known B0 bank, context, goal and model, 16 candidates per arm, shared
MSE objective and projected-gradient schedule: at most 144 objective scores,
128 gradients and 288 transition-packet calls per arm, plus the other
instrumented module-call caps in the technical plan. Attempts are journaled.
Model load, preflight, FLOPs, energy, upstream preparation and total wall time
are not fully metered or matched. The 900-second check is not preemptive and
the 200 MiB disk floor is checked at dispatch boundaries, not a storage quota.
This does not authenticate a composition mechanism or equal total budget.

The wrapper sets `OPENBLAS_NUM_THREADS=1` inside Python before numerical
imports; the direct shell command has no environment prefix. It records the
inherited and effective values. This preserves the intended thread setting,
not a proof of identical BLAS runtime, numerical behavior, or CUDA access.
No CUDA probing was done in plan preparation.

## Future coordinated host execution

Only after the operator coordinates host GPU use with other tasks, in
`/home/gbk/code/LeWorldModel/le-wm-main`, execute directly on the authorized
host interpreter (not a sandbox lacking GPU access):

```sh
.venv/bin/python -B -m experiments.spectral_reachability.p2_same_bank_env_repair_20261102_exclusive verify --plan-sha256 860d6c4a03bc32b23aa12b1dba9aba046c7de9d3c9b12619fdedeb628d6347ea
.venv/bin/python -B -m experiments.spectral_reachability.p2_same_bank_env_repair_20261102_exclusive run --plan-sha256 860d6c4a03bc32b23aa12b1dba9aba046c7de9d3c9b12619fdedeb628d6347ea --authorization P2_ENV_REPAIR_NEW_KNOWN_ROOT_DEVELOPMENT_ONLY
```

The output is exclusively
`le-wm-main/results/p2_same_bank_host_development_20261102_exclusive_RUN_ENV_REPAIR_01/`.
The wrapper obtains the existing advisory cooperative lock for CUDA
availability/device-count preflight and writes `environment.json`; it releases
that lock and the native session reacquires it nonblocking before model load.
Another task can acquire the lock in the gap, in which case this attempt fails
and is retained; a passed probe does not guarantee later CUDA execution.
The native lock covers model load through model postcheck. No cooperating
lock-holder is displaced; noncooperators are outside the guarantee. On any
failure preserve all partial files and stop: no retry, replacement or imputed
result. The wrapper verifies the repair plan again after execution; if that
postcheck fails, any inner success report is invalid for this protocol.

If and only if the new report says `completed_development_control` and the
wrapper and postcheck return successfully, record its actual SHA-256 using
`sha256sum` and substitute it for `<REPORT_SHA256>` below. Independently verify
that the crosscheck source SHA is
`86acb5c4ec6b91791ed18c80995136b54b80bd01285e98db543c2b9390618494`
before executing the CPU-only consumer:

```sh
.venv/bin/python -B -m experiments.spectral_reachability.p2_same_bank_winner_ledger_crosscheck_20261102_exclusive --output /home/gbk/code/LeWorldModel/le-wm-main/results/p2_same_bank_host_development_20261102_exclusive_RUN_ENV_REPAIR_01 --report-sha256 <REPORT_SHA256>
```

This checks recorded winners against optimizer-return events in addition to
the existing consumer checks; it does not independently authenticate the
native runtime. No physical execution is authorized by this plan, and neither
P1 nor P2 admission, scientific success, generalization to independent roots,
or physical improvement can be inferred.

## CPU contract evidence

On preparation, the repair, existing host runner, and winner crosscheck test
files ran with `-B -m pytest -q -p no:cacheprovider`: 11 passed. Mock probes
exercise failure, authorization, lock sequencing, and collision behavior;
they are not CUDA or experimental results.
