"""Seal independent per-prefix P2 search inputs without loading a model."""
import argparse
import json
import os
from pathlib import Path

from experiments.spectral_reachability import codex_p2_program_library_accounting_20261030_v1 as original
from experiments.spectral_reachability import codex_p2_program_independent_budget_20261030_v1 as independent
from experiments.spectral_reachability import codex_p3_quota_source_identity_20261030_v1 as identity

NAME="codex_p2_program_independent_prepare_20261030_v1"
ROOT=original.ROOT


def material():
    base=original.material()
    paths=[Path(__file__).resolve(),Path(independent.__file__).resolve(),
           ROOT/"tests/test_codex_p2_program_independent_budget_20261030_v1.py",
           ROOT/"tests/test_codex_p2_program_independent_prepare_20261030_v1.py",
           ROOT/"experiments/spectral_reachability/codex_p2_program_independent_saved_20261030_v1.py",
           ROOT/"tests/test_codex_p2_program_independent_saved_20261030_v1.py",
           ROOT/"innerdocs/codex_p2_program_independent_budget_20261030_v1.md"]
    sources={row["path"]:row for row in base["sources"]}
    for row in identity.inventory(paths):
        sources[row["path"]]=row
    return dict(schema=NAME+".prepare",protocol=independent.contract(),cases=base["cases"],
        panel_path=base["panel_path"],panel_sha256=base["panel_sha256"],
        policy=base["policy"],expected_model_hash=base["expected_model_hash"],gpu_uuid=base["gpu_uuid"],
        lock_path=base["lock_path"],sources=sorted(sources.values(),key=lambda x:x["path"]),
        source_closure="dynamic closure unknown",prepare_only=True,scientific_pass=False)


def prepare(path):
    path=Path(path)
    independent.kernel.require(path.is_absolute() and path.parent==ROOT/"results" and
        path.name.startswith(NAME+"_PREP_") and not os.path.lexists(path),"exclusive prepare")
    return original.previous.write_new(path,material())


def verify(path,expected):
    path=Path(path)
    independent.kernel.require(path.is_absolute() and not path.is_symlink() and
        original.previous.sha(path)==expected,"prepare SHA")
    value=json.loads(path.read_bytes())
    identity.verify(value["sources"])
    independent.kernel.require(value==material(),"fixed source and contract reconstruction")
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    print(json.dumps(dict(status="prepared_no_cuda_or_physics",path=args.output,
                          sha256=prepare(args.output)),sort_keys=True))


if __name__=="__main__":
    main()
