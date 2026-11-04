# Two-packet optimized future-value selector: development result

The [fixed protocol](PC_POINTMAZE_NONPROGRESS_LOOKAHEAD_PROTOCOL_20261103.md)
gave each of nine bank prefixes 50 two-packet model evaluations with the same
seed and CEM settings, for 450 two-packet samples per root. The first-packet
predictions matched the sealed one-packet bank scores within the specified
tolerance. The model was frozen. This is 900 predicted packet transitions
per root, in addition to the original bank-generation cost.

The selector requires predicted non-improvement at the first packet and a
strictly lower optimized terminal cost than both same-bank greedy and zero.
It returned **no candidate on roots 8343, 8345 and 8351**. On root 8349 it
selected bank candidate 4, the same candidate as the prior immediate-only
rule. Thus the new rule made no changed executable choice on these roots.
The selected prefix was physically executed and its step-five RGB matched
the sealed bank; its physical distance rose from 1.591247 to 1.604257.

This result establishes neither an improved closed-loop outcome nor P1.
The 8349 prefix's previously completed conditional feedback experiment
ended farther from the goal than zero, but that earlier result was exposed
before this exploratory selector was designed and is not a fresh test.
The present lookahead optimizes only one subsequent five-action packet in
imagined rollout, not the requested full remaining 20-action feedback
horizon. Its chosen suffix was not executed. It also lacks a random-prefix
arm, physical same-action model-error decomposition, and equal-total-cost
strong baselines. Root, checkpoint training and normalizer provenance remain
uncertified for independent confirmation.

Receipt [DEV01](../le-wm-main/results/pc_pointmaze_nonprogress_lookahead_20261103_DEV01.json)
SHA-256 `fef972f1527f14b6439bb8fc6a7f60061b1be37a0324756d0c3baa92f256259a`.
