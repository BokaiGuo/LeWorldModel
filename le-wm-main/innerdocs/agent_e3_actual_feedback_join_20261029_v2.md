# E3 actual feedback join v2

This fixture-only numerical join adds a bit-exact check that normalizing the
actual prior raw5 in float32 produces the last incoming history token. It rejects
wrongly joined history and signed-zero bit changes. The rest delegates to v1;
it does not authenticate RGB, model, native lifecycle, sources or receipts.
Not a production dispatcher or saved strict consumer; no scientific admission.
