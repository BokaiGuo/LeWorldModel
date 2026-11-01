# Known-root diag32 same-bank fixed-tail intervention

Development-only post-outcome study on cp10 root0, not P1 admission or
independent confirmation. Fixed roster is selected, zero, and model-greedy
prefix, each crossed with the selected candidate 4 iteration 13 plan's final
ten actions and a ten-action zero tail. All six complete command sequences
and their input hashes are saved before any branch executes.

The comparison pool consists of all 264 recorded plans (eight initial
candidates times iterations 0 through 32) from the same root0 diag32
optimization; this includes the selected iteration 13. Greedy minimizes the
recorded one-packet model Q over this pool, with candidate then iteration as
tie breakers. It does not consult physical outcomes. This definition is not
the different comparison restricted to the eight final iterations.
Physical outcomes from earlier studies, including the selected plan's
success, were known before this development study was designed.

Each branch uses the real environment and the same root and goal. Root and
previously recorded selected/zero cp5 observations are checked exactly;
the greedy cp5 is checked against the first fresh greedy replay. All outputs,
actual action returns, RGB, physical accounting and stops are recorded; native
success, truncation, and out-of-domain censoring end a branch without padding.
Fresh dynamics are newly collected receipts, not historical backfill. The
project cooperative lock covers the physical run; there are no neural calls.
