# Dynamic cp10 physical inputs: inherited receipt adapter

`assemble(specs, groups, source_class)` accepts five literal candidate bridge row specs and two independently SHA-supplied physical groups `C`, `ZP`, each containing `prepare`, `report`, `completion`, `receipt`, `source` descriptors `{path,sha256}`. It reads actual bytes, validates the known original physical producer schemas, all four physical report/receipt plan rows and source/prepare/report/completion receipt bindings, then checks each fixed slot against its original plan, root/goal/context, causal raw10/prefix, 51-frame guard, cp10 checkpoint, three PNG files and pixel hashes and full state/snapshot identity. Original physical full strict is inherited, not reconstructed. No caller-selected rows, missing checkpoint fallback, or new task choice is allowed.

Source class is explicitly `historical`, `fixture`, or `future_candidate`. Historical known root SHA values are rejected in `future_candidate`, even if descriptors are different; passing that check is **not** proof of new roots, genuinely future data, outcome blindness, producer authority or trusted time. A separately reviewed external source-selection/identity policy and producer full saved receipt remain necessary. This adapter currently recognizes only the actual original C/ZP physical ABI, not an imaginary future schema; a future producer with different schemas requires a reviewed explicit adapter. The returned `execution_authorized`, `fresh_roots_certified` and `source_authenticity_independently_verified` are always false.

This is not the requested CUDA writer or a complete saved consumer. Those remain unimplemented; the old bridge is hardcoded to historical sources/schema, so treating it as a dynamic executor would spoof evidence. The fixture tests include read-only historical ABI and historical-to-future spoof rejection. No production, CUDA or physics is run.

```sh
cd /home/gbk/code/LeWorldModel/le-wm-main
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_agent_p3_dynamic_cp10_physical_inputs_20261029_v1.py
```
