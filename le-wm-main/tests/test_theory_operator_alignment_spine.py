from experiments.theory_spectral_transfer import (
    TransferConfig,
    operator_alignment_anisotropy_sweep_report,
    operator_alignment_decoupling_report,
    operator_alignment_law_card,
    operator_alignment_metric_robustness_report,
    operator_alignment_null_sweep_report,
    operator_alignment_report,
    operator_alignment_theorem_report,
)
from experiments.theory_spectral_transfer.run_operator_alignment_sweep import operator_alignment_paper_spine
from experiments.theory_spectral_transfer.run_operator_alignment_sweep import operator_alignment_oral_object_card
from experiments.theory_spectral_transfer.run_operator_alignment_sweep import operator_alignment_paper_draft


def test_operator_alignment_paper_spine_is_manuscript_form():
    config = TransferConfig()
    report = operator_alignment_report(config=config, angles=(0, 15, 30, 45, 60, 75, 90), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=32, seed=0)
    theorem = operator_alignment_theorem_report(config=config, angles=(0, 15, 30, 45, 60, 75, 90))
    law_card = operator_alignment_law_card(config=config, angles=(0, 15, 30, 45, 60, 75, 90), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=32, seed=0)

    spine = operator_alignment_paper_spine(report, theorem, law_card)

    assert spine.startswith("# Same Spectrum Is Not Enough: Noncommutative Operator Alignment Governs Spectral Transfer")
    assert "## Abstract" in spine
    assert "spectral transfer is under-identified without operator alignment" in spine
    assert "It does not replace the companion empirical paper" in spine


def test_operator_alignment_oral_object_card_is_law_first():
    config = TransferConfig()
    report = operator_alignment_report(config=config, angles=(0, 15, 30, 45, 60, 75, 90), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=32, seed=0)
    theorem = operator_alignment_theorem_report(config=config, angles=(0, 15, 30, 45, 60, 75, 90))
    law_card = operator_alignment_law_card(config=config, angles=(0, 15, 30, 45, 60, 75, 90), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=32, seed=0)

    card = operator_alignment_oral_object_card(report, theorem, law_card)

    assert card.startswith("# Same Spectrum Is Not Enough")
    assert "not about controllability" in card
    assert "spectrum-only explanation" in card
    assert "row/series unit: rotation angle × k band" in card


def test_operator_alignment_paper_draft_is_full_manuscript_form():
    config = TransferConfig()
    report = operator_alignment_report(config=config, angles=(0, 15, 30, 45, 60, 75, 90), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=32, seed=0)
    theorem = operator_alignment_theorem_report(config=config, angles=(0, 15, 30, 45, 60, 75, 90))
    law_card = operator_alignment_law_card(config=config, angles=(0, 15, 30, 45, 60, 75, 90), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=32, seed=0)

    draft = operator_alignment_paper_draft(report, theorem, law_card)

    assert draft.startswith("# Same Spectrum Is Not Enough: Noncommutative Operator Alignment Governs Spectral Transfer")
    assert "## Related Work" in draft
    assert "## Method" in draft
    assert "## Results" in draft
    assert "## Claim Ledger" in draft
    assert "under-identified without operator alignment" in draft


def test_operator_alignment_null_sweep_strengthens_falsifiers():
    config = TransferConfig()
    report = operator_alignment_null_sweep_report(config=config, angles=(0, 15, 30, 45, 60, 75, 90), ks=(4, 8, 16, 32), random_trials=16, seed=0)

    assert report["oblique"]["commutator_norm"] > report["permutation_preserving_null"]["commutator_norm"]["mean"]
    assert report["oblique"]["commutator_norm"] > report["spectrum_matched_random_basis_null"]["commutator_norm"]["mean"]


def test_operator_alignment_anisotropy_sweep_is_monotone_ready():
    config = TransferConfig()
    report = operator_alignment_anisotropy_sweep_report(config=config, input_exponents=(0.0, 0.5, 1.0, 1.5, 2.0), angles=(0, 15, 30, 45, 60, 75, 90), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=8, seed=0)

    assert len(report["rows"]) == 5
    assert report["rows"][0]["input_exponent"] == 0.0
    assert report["rows"][-1]["input_exponent"] == 2.0


def test_operator_alignment_metric_robustness_reports_multiple_proxies():
    config = TransferConfig()
    report = operator_alignment_metric_robustness_report(config=config, angles=(0, 15, 30, 45, 60, 75, 90), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=8, seed=0)

    assert report["rows"][0]["h1_top_band_lift"] > 0.0
    assert "hmax_residual_band_lift" in report["rows"][0]


def test_operator_alignment_decoupling_separates_axes():
    config = TransferConfig()
    report = operator_alignment_decoupling_report(config=config, alignment_angles=(0, 45, 90), spectrum_exponents=(0.0, 1.0, 2.0), horizons=(1, 2, 4, 8), ks=(4, 8, 16, 32), random_trials=8, seed=0)

    assert len(report["fixed_spectrum_alignment_rows"]) == 3
    assert len(report["fixed_alignment_spectrum_rows"]) == 3
    assert report["summary"]["alignment_axis_commutator_range"] > 0.0
    assert report["summary"]["spectrum_axis_commutator_range"] > 0.0
