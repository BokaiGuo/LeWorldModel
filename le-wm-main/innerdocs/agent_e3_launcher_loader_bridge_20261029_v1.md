# E3 loader journal bridge (partial)

This adapter preserves the existing loader's four buffer events and receipt in
the unified journal; its saved check delegates receipt semantics to the original
loader validator and checks the event order and exact buffer joins. The disk
fixture does not load a model. It does not authenticate source bindings, run a
launcher, certify native semantics, or grant scientific admission. Its caller
must independently authenticate the prepared inputs and transport before use.
This is an intermediate integration component, not the requested complete E3
launcher or strict consumer.

The temporary-disk test also exercises the actual old loader receipt validator
with synthetic receipt values, including a same-buffer corruption rejection.
It does not prove that real checkpoint bytes were consumed or that CUDA ran.
