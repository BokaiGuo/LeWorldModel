# E3 later HOST backend and numerical saved replay

Connects the new control10/15 fixed GD solver to the frozen native HOST backend,
preserving live raw leaf graph, full native module IO/AD and state-seal check.
Independent saved backend reruns all candidates, 32 scalar gradients, selection,
three replay and 27 resource axes against actual stored arrays/events. On exception
it attempts to retain the partial numerical row; exception is not hidden.

Tests use the old in-memory fake CUDA algebra and real temporary HOST-ABI Store;
they execute no real CUDA or NN. This is not an authenticated actual RGB input,
full source/model/precision validation, complete directory census, combined
transport adapter, physical dispatcher, or production CLI. Existing HOST Store
uses descriptors incompatible with the new unified transport; direct mixing is
not yet implemented or certified. Science gates remain false.
