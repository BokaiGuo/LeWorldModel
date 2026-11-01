import hashlib
import json

import numpy as np

from experiments.spectral_reachability import codex_p1_native_rollout_equivalence_20261030_main as audit


def test_saved_vectors_scores_and_bindings():
    path = audit.ROOT / "results/codex_p1_native_rollout_equivalence_20261030_main_RUN01.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "7152974b335328940b0630313d6d049874a5f72a5ea9e20621f47bf304991855"
    data = json.loads(path.read_bytes())
    plan_path, plan_hash = audit.sources.SOURCES["plan"]
    plan, _ = audit.adam.prior.gap.read(audit.ROOT / plan_path, plan_hash)
    assert data["status"] == "complete" and len(data["rows"]) == len(plan["rows"]) == 24
    assert data["model_before"] == data["model_after"] and not data["P1_admission"]
    latent_errors, gradient_errors = [], []
    for row, original in zip(data["rows"], plan["rows"], strict=True):
        assert row["index"] == original["index"] and row["action_sha256"] == original["selected_action_sha256"]
        a, b = np.asarray(row["manual_latent"]), np.asarray(row["native_latent"])
        ga, gb = np.asarray(row["manual_gradient"]), np.asarray(row["native_gradient"])
        assert a.shape == b.shape == (192,) and ga.shape == gb.shape == (20,)
        assert all(np.isfinite(x).all() for x in (a, b, ga, gb))
        le, ge = np.max(np.abs(a-b)), np.max(np.abs(ga-gb))
        assert le == row["latent_max_abs"] and ge == row["gradient_max_abs"]
        assert abs(row["manual_score"] - original["candidate_scores"][original["selected_index"]]) < 1e-5
        assert abs(row["manual_score"] - row["native_score"]) < 1e-6
        latent_errors.append(le)
        gradient_errors.append(ge)
    assert max(latent_errors) == data["summary"]["max_latent_abs"]
    assert max(gradient_errors) == data["summary"]["max_gradient_abs"]
