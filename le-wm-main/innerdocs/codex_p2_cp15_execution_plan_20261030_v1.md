# P2 cp15 physical plan bridge: status

This saved-input bridge reconstructs each of four existing physical bindings
from the original native task and independently consumed CUDA panel, then
joins the two literal CUDA winner suffixes to the original raw15 prefix.
Its output is an in-memory plan digest, not a physical prepare or run.
The parent native witness is retained only as parent provenance; the new
winner reference is explicitly a different schema.

Rebuild without GPU, neural inference, or physics:

```
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m experiments.spectral_reachability.codex_p2_cp15_execution_plan_20261030_v1 --base 0
```

Repeat with bases 1, 2, 3. Test:

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider tests/test_codex_p2_cp15_execution_plan_20261030_v1.py tests/test_codex_p2_cp15_physical_binding_20261030_v1.py
```

The bridge does not yet supply the complete prepare schema, durable lifecycle,
independent physical saved consumer, or partial-frame failure consumer.
The old consumer validates a different candidate witness and cannot certify
these new plans. Runtime must check the complete 106-event physical prefix
and durably record its guard before suffix dispatch, preserve early stops,
failures and resource counts, and validate actual actions and RGB. No new
physical execution is authorized merely by this source bridge. The old
per-job caps remain unmodified; actual new physical resource use is unknown.

The previously recorded CUDA search has matched operation counts, not proof
of equal total compute. It also releases the cooperative lock before process
exit and explicit CUDA context cleanup. Known roots, inherited acquisition
and unmeasured costs prevent independent confirmation or scientific admission.
Suffix freezing is sequential spectral planning, not complete program
composition; P1 and P2 gates remain unchanged.
