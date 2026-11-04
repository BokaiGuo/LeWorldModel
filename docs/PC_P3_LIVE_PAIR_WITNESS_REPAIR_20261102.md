# P3 live pair: action witness repair

The live signal/reverse paired entry previously returned winning candidate
indices and costs without the actual commands or predicted paths. An index
alone is not a sufficient standalone action witness for subsequent physical
execution, and the original signal event omitted the terminal and Jacobian
needed to audit the allocation.

The entry now includes a deep copy of each candidate's commands and predicted
path in its completion event and winner, and includes nominal contexts,
terminal and serialized Jacobian in the signal event. Its deterministic
quota and comparison rules are unchanged. A regression test checks the
winner's actions against the input bank for both budgets and roots.

The pair and scheduler contract tests passed (23 tests). These are CPU
callback fixtures, not real model or physical results. The entry still needs
an identity-bound host CUDA launcher, durable journal and independent saved
consumer before a real run is adequately authenticated. It also does not
measure full encoder, wall, physical or total compute, and no claim about P3
or C3 performance or admission follows from this repair.
