from experiments.pc_post_prefix_opportunity_coverage_20261107_codex import summarize


def test_summary_aggregates_decisions_by_episode_and_excludes_initial_success():
    def row(episode, initial, endpoint, full, post):
        return {"episode": episode, "episode_initial_success": initial,
                "coverage": {k: {"ever": v, "tail": v} for k, v in
                             zip(("endpoint", "full", "post"), (endpoint, full, post), strict=True)}}
    result = summarize([row(1, False, False, False, True), row(1, False, False, False, True),
                        row(2, False, True, True, False), row(3, True, True, True, True)])
    assert result["decisions"] == 4
    assert result["non_initial_episodes"] == 2
    assert result["episode_tail_non_initial"] == {"endpoint": 1, "full": 1, "post": 1}
    assert result["post_only_vs_full"] == result["full_only_vs_post"] == 1
