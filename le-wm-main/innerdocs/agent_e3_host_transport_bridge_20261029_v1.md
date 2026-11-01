# E3 HOST numerical / unified transport bridge

The HOST backend receives its unchanged four-field array descriptors. The bridge
adds numerical origin and host_native role only when writing unified events;
the same SHA, filename, shape, dtype and actual NPY bytes are preserved. It
rejects unknown or altered descriptors before emitting an event. Saved numerical
events strip only the two added fields; original HOST SavedBackend then rebuilds
all candidate/gradient/replay math and module chronology. Temporary disk tests
cover K1 Dxy and K3 Dx with interleaved physical events using fake CUDA algebra.

This bridge does not authenticate actual source, model, RGB or task, does not
persist the whole numerical result row, and is not a production dispatcher or
whole-directory scientific strict. Unified transport's own consumer checks its
referenced directory; no NN/physics or production executed. Existing callbacks,
cleanup, encoder and final publication integration remain necessary.
