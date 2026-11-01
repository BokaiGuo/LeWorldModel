# E3 continuous episode transport: candidate partition

This RAM-only integer envelope adds to the frozen three-node H3 numeric bound:
32 PNG at 256 KiB each, 32 MiB physical journal, 16 MiB physical metadata,
8 MiB two-image encoder arrays, 8 MiB encoder event/metadata. The 64 MiB final
reserve is counted once. Two encodings each retain seven full arrays, exact
NumPy NPY headers plus conservative incompressible gzip bounds, including the
duplicate preprocessed/encoder input and full 257-token hidden state.

This is a proposed hard-cap partition, not proof the unfinished real writer or
all repeated native metadata fits: an implementation must actually enforce each
category before each write and test full physical/encoder event, partial failure,
report and completion serialization without deletion or truncation. Any category
overflow is partial/fail-closed, not permission to expand or shrink data. Early
stop retains actual counts. Wall and CUDA peak remain runtime checks. No physical,
NN, production or fullstrict executed; no scientific gate changes.
