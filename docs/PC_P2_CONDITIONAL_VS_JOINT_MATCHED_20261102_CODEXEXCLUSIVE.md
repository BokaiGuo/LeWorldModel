# P2 conditional versus joint initialization: matched-operation development wiring

The historical B0 conditional comparison charged 16 conditional prefix
transitions, 32 suffix transitions, 16 Jacobians and 16 solves only to
composition. Its physical result is not a matched-total-compute result.
This new wiring is a separate development design; it does not change that
report, its original SRA comparator or any admission gate.

Both methods use every one of the same 16 authenticated saved initial
two-packet candidates and the same frozen model, input latent history, goal,
raw action bounds, eight-update optimizer and terminal latent MSE. Each uses
an actual model-based initializer, not dummy calls:

* Conditional: transition the first packet, detach the predicted cp5 context,
  differentiate the terminal latent with respect to the 10 suffix coordinates,
  and apply the 10-dimensional bounded ridge update to the suffix.
* Joint control: differentiate the terminal latent with respect to all 20
  coordinates of the two-packet plan through both transitions, then apply a
  20-dimensional bounded ridge update to the whole plan. This is a joint
  no-conditional-initializer control, **not** historical 40-candidate SRA.

For each initializer, the nominal terminal is captured during the Jacobian's
forward evaluation, not by an additional uncharged inference. Each candidate
and arm requests two transition forwards, one 192-output Jacobian and one
bounded solve with the same ridge and L2 radius. The conditional branch's
first transition is detached; the joint branch differentiates through both.
Thus Jacobians have shapes 192×10 and 192×20, respectively. They do **not**
have equal FLOPs, backward graph, solve cost, wall time, energy or memory.
The joint control has a larger update space; this is a substantive stronger
comparison, not a pure single-factor ablation. Requested 192 directions are
not an independent measurement of autograd engine backward operations.

For 16 candidates, the initializer meter expects per arm 32 transition
forwards, 16 Jacobian requests and 16 bounded solves, plus 32 calls to each
hooked model entrypoint. The common optimizer expects per arm 144 score
callbacks, 128 gradient callbacks and 288 packet transitions, plus 288 calls
to each hooked model entrypoint. Existing ledger category
`sra_initialization` is reused for the joint control solely because the old
ledger has a closed category vocabulary; the meter records its actual role as
`joint_gauss_newton_not_historical_sra`. Never label its result historical
SRA. Shared inherited bank/Q/model-load costs, unhooked operations, hardware
cost and source closure remain outside this count.

## Host integration boundary

`run_native(fixed, transition, meter, device)` calls the real native
objective, Jacobian/solver and common optimizer. It requires a CUDA device,
the full inherited fixed B0 input, and the new `MatchedMeter`. The caller
must separately bind and freeze code, dependencies, model/checkpoint,
normalization, fixed root roster, candidate bank, wall/memory limits, stopping
rules, and durable output; acquire the existing cooperative GPU lock;
authenticate frozen CUDA model and input; create the transition with
`pusht_full_packet_compiler.make_transition`; and use exactly one model-hook
scope, for example:

```python
meter = matched.MatchedMeter(journal)
with objective.native_model_scope(model, meter):
    result = matched.run_native(fixed, transition, meter, device)
summary = meter.finish()
```

The underlying existing `Ledger` must be new and exclusively owned; verify
its chain and absence of outstanding attempts after closing. A failed,
partial, rejected, nested-failure or incomplete schedule does not count as
matched success. Preserve all failures, and do not retry selectively or
substitute candidates. The API and nominal caps cannot prevent uninstrumented
operations by an untrusted caller. The legacy native session installs
different hooks and must not be nested here.

The current `native.verified_input()` deliberately admits only the previously
exposed B0 source; this implementation does not choose roots by outcome but
cannot establish root-disjoint or multi-root evidence. No host launcher,
frozen new multi-root plan, executed model result, physical handoff, saved
consumer for the new report, or fair total-compute claim is provided. Tests
use analytic non-neural functions and existing optimizer/ledger code; they
are not model or physical evidence. GPU, physical execution and training are
not performed.
