"""Audit actual joint-CEM batch boundary in the sealed v3 RGB decisions."""
import json

from experiments import pc_response_temporal_v3_rgb_live_20261104_codex as wrapper

OUT = wrapper.PREFIX.with_name(wrapper.PREFIX.name + "_refinement_audit.json")


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    wrapper.configure()
    sha = wrapper.live.old.prior.sha
    previous = json.loads(wrapper.verifier.AUDIT.read_text())
    if previous["manifest_sha256"] != sha(wrapper.live.MANIFEST):
        raise ValueError("RGB audit seal mismatch")
    paths, _, _ = wrapper.live.paths("response")
    findings = []
    for step, path in enumerate(paths):
        data = json.loads(path.read_text())
        result, trace = data["result"], data["trace"]
        generations = result["refinement"]
        boundary = len(trace) - len(generations) - 1
        if boundary < 1 or sum(g["size"] for g in generations) != result["refine_samples"]:
            raise ValueError("invalid refinement boundary")
        for generation, batch in zip(generations, trace[boundary:-1], strict=True):
            if generation["size"] != len(batch["scores"]):
                raise ValueError("generation batch mismatch")
        horizon = 15 - 5 * step
        before = [(chain, score) for batch in trace[:boundary]
                  if len(batch["requested"][0]) == horizon
                  for chain, score in zip(batch["requested"], batch["scores"], strict=True)]
        start = result["refinement_start"]
        if (len(before) != start["full_samples_seen"] or not before
                or start["terminal_score"] != min(score for _, score in before)
                or (start["chain"], start["terminal_score"]) not in before):
            raise ValueError("incorrect refinement start")
        findings.append({"step": step, "boundary_batch": boundary, "complete_prior_samples": len(before),
                         "score": start["terminal_score"], "decision_sha256": sha(path)})
    wrapper.live.old.write_json(OUT, {"manifest_sha256": sha(wrapper.live.MANIFEST),
                                    "audit_sha256": sha(wrapper.verifier.AUDIT), "rows": findings,
                                    "limitation": "uses recorded model predictions; no model recomputation"})
    print("Verified all three RGB response refinement boundaries.", flush=True)


if __name__ == "__main__":
    main()
