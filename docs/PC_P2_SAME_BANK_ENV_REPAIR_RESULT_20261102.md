# P2 same-bank host environment-repair development result

After preserving the failed pre-model attempt, the separately sealed
environment-repair plan ran to `completed_development_control` on host CUDA.
The saved ledger consumer and independent winner/return-event crosscheck
passed. The first failed run is not overwritten and remains an engineering
failure, not a negative scientific result.

The two labels each processed all 16 identical initial action sequences with
the same frozen model, context, goal and eight-update optimizer. Per label,
the ledger reports 144 score and 128 gradient callbacks and 288 packet
transitions. Both labels selected candidate 12 at iteration 8 with exactly
the same saved raw-action SHA-256
`abc456ddf36ecbb2cb150bc3e358f5c400d4df7701a1a9eb2e9138f89c9550d4`
and model objective 0.7764005065. The event crosscheck bound all 32
candidate-arm winners to 288 optimizer return events.

This is a deliberate same-input, same-optimizer repeatability control, not a
genuine composition-versus-no-composition ablation: the arm labels do not
create different programs. It includes neither physical execution nor
independent roots, and its explicit counters do not certify equal total
compute, full source closure or model-load cost. P1 and P2 admission remain
false.

Report SHA-256:
`6cf83f5b4a90ec2c125fa8c5cbb2bb0191d27a81dc2b5dffdd900099366a9584`.
Journal SHA-256:
`341634abb251a90674405a8417722f85bf62a83bfd39ead251ce39c61d4f676a`.
