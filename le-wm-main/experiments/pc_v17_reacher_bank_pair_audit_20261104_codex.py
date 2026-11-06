"""Audit shared proposal generation and final paired decisions."""
import json
from pathlib import Path

import numpy as np

from experiments import pc_v17_reacher_score_response_dev_20261104_codex as source
from experiments.spectral_reachability import pc_v3_dependency_seal as seal


def main():
    output = source.OUT / "bank_pair_audit.json"
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source.OUT / "manifest.json").read_text())
    audit = json.loads((source.OUT / "audit.json").read_text())
    replay = json.loads((source.OUT / "physical_replay.json").read_text())
    seal.verify(manifest["dependencies"])
    if (audit["status"] != "passed" or replay["audit_sha256"] != seal.digest(source.OUT / "audit.json")
            or any(x["status"] != "passed" for x in replay["rows"])):
        raise ValueError("source audit or physical replay missing")
    records = []
    for step in range(3):
        fixed_parts = []
        for arm in source.ARMS:
            path = source.OUT / f"{arm}_decision{step}.json"
            data = json.loads(path.read_text())
            carry = None if step == 0 else np.asarray(data["carry_in"], np.float32)
            nominals = tuple(tuple(tuple(float(x) for x in row) for row in nominal)
                             for nominal in source.old.temporal.NOMINALS)
            config = source.old.v13.Config(action_dim=2, segment_length=5, segments=3-step,
                model_packet_length=5, nominals=nominals, token_cap=source.CAP,
                seed=2027110401 + step * 10000)
            bank = np.asarray(data["trace"][0]["raw"], np.float32)
            np.testing.assert_array_equal(bank, source.old.v13.proposal_bank(config, carry))
            fixed_parts.append(np.concatenate((bank[:1], bank[1 if carry is None else 2:]), axis=0))
            final = data["trace"][-1]
            pair, scores = np.asarray(final["raw"], np.float32), np.asarray(final["scores"], np.float64)
            if pair.shape != (2, 15 - 5 * step, 2) or scores.shape != (2,):
                raise ValueError("paired comparison shape")
            np.testing.assert_array_equal(pair[1, :5], np.zeros((5, 2), np.float32))
            np.testing.assert_array_equal(pair[0, 5:], pair[1, 5:])
            decision = data["decision"]
            if arm == "global_cem":
                np.testing.assert_allclose(scores, decision["pair_scores"], atol=0, rtol=0)
            else:
                np.testing.assert_allclose(scores, decision["compiler"]["paired_scores"], atol=0, rtol=0)
            selected = pair[int(np.argmin(scores))]
            np.testing.assert_array_equal(selected, np.asarray(decision["selected"], np.float32))
            records.append({"arm": arm, "step": step, "bank_samples": len(bank),
                            "pair_scores": scores.tolist(), "carry_present": carry is not None})
        for bank in fixed_parts[1:]:
            np.testing.assert_array_equal(bank, fixed_parts[0])
    seal.verify(manifest["dependencies"])
    source.old.host.write_new(output, {"status": "passed", "records": records,
        "audit_sha256": seal.digest(source.OUT / "audit.json"),
        "replay_sha256": seal.digest(source.OUT / "physical_replay.json"),
        "audit_script_sha256": seal.digest(Path(__file__)),
        "scope": "saved bank/pair consistency; not statistical confirmation"})
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
