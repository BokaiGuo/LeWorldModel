# Diag32 known-root fixed-tail physical result

Six branches executed under the cooperative lock, without neural calls.
The saved report SHA-256 is
`f87f9c6c207076108c158c6c2335f05b2afa688eff9debacab793f92096da0d3`.
The separate saved checker revalidates the 264-plan selection pool, source
closure, physical action journal and resource counts, root and cp5 identities,
and each saved frame and state. It is not an independent simulator replay.

| Prefix | Tail | Last step | JointXY | Native success |
| --- | --- | ---: | ---: | --- |
| selected | common | 15 | 16.656736 | yes |
| selected | zero | 15 | 176.161748 | no |
| zero | common | 15 | 21.106422 | no |
| zero | zero | 15 | 175.176818 | no |
| greedy | common | 10 | 10.976097 | yes |
| greedy | zero | 15 | 92.649570 | no |

Both native-success branches terminated normally; no truncation or domain
censoring occurred. The greedy/common branch has no step-15 endpoint; it was
not padded and its step-10 jointXY is not a same-time comparison.

Greedy is the first Q1 minimum over all 8 x 33 saved root0 diag32 optimization
evaluations, candidate 4 iteration 0, Q1 `0.8473501205444336`. The selected
plan is candidate 4 iteration 13. This exact pool includes the selected plan;
it is not the alternative pool of eight final iterations. Common tail means
the selected plan's last ten actions. Earlier physical outcomes were already
known before this intervention was designed, so this remains a known-root
development result, not independent confirmation or P1 admission.
