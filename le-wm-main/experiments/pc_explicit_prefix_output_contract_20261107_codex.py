"""Fail-closed outputs for the historical scored-incumbent prefix solver."""

from experiments.pc_explicit_prefix_scored_incumbent_20261107_codex import (
    ExplicitPrefixScoredIncumbent,
)


class ExplicitPrefixSelectedActionOnly(ExplicitPrefixScoredIncumbent):
    """Return scored selected actions without inventing distribution statistics.

    Historical sampling and selection are unchanged. The selected chain is not
    necessarily the mean of the final CEM branch and the historical `var` may
    belong to a different branch or later round. Consumers needing a proposal
    distribution must use a separately defined solver with explicit provenance.
    """

    def solve(self, info_dict, init_action=None):
        result = super().solve(info_dict, init_action=init_action)
        return {
            "actions": result["actions"],
            "costs": result["costs"],
            "selected_action_distribution": "unavailable",
        }
