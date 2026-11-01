"""Join verified full-bank CUDA replay to native source identities."""
import argparse
import json
from pathlib import Path

from . import p1_full_bank_cuda_replay_20261030_main as full
from . import p1_diverse_physical_source_join_20261030_main as source

NAME = "p1_full_bank_physical_source_20261030_main"
ROOT = full.ROOT
RESULT = ROOT / "results/p1_full_bank_cuda_replay_20261030_main_RUN01.json"
RESULT_SHA = "a86fc9b1a283e1001a173ef3bc0ca6f3249b219a1152dc762441b8abb930b2b6"
PREPARE = ROOT / "results/p1_full_bank_cuda_replay_20261030_main_prepare_RUN01.json"
PREPARE_SHA = "37488b452ba5e3e0c64428c9d30dabde3ed975803191b6ac43ed1bfd2b65c462"


def configure():
    full.configure()
    source.RESULT, source.RESULT_SHA = RESULT, RESULT_SHA
    source.PREPARE, source.PREPARE_SHA = PREPARE, PREPARE_SHA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    out = Path(parser.parse_args().output)
    if (not out.is_absolute() or out.parent != ROOT / "results" or out.resolve() != out
            or not out.name.startswith(NAME + "_") or out.exists()):
        raise ValueError("exclusive output")
    configure()
    result = source.build()
    result["full_bank_selection_sha256"] = full.SELECTED_SHA
    result["full_bank_source_code_sha256"] = full.replay.prior.digest(Path(__file__))
    print(json.dumps({"sha256": full.replay.prior.one.saved.write(out, result),
                      "rows": len(result["rows"]), "physical_authorized": False}))


if __name__ == "__main__":
    main()
