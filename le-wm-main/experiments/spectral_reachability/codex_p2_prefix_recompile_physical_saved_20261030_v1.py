"""Saved-only prefix-recompile physical consumer in a separate process."""
import argparse
import json
from experiments.spectral_reachability import codex_p2_prefix_recompile_physical_20261030_v1 as lifecycle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('output','report-sha','completion-sha','receipt'):
        parser.add_argument('--'+field,required=True)
    args=parser.parse_args()
    receipt=lifecycle.consume(args.output,args.report_sha,args.completion_sha)
    print(json.dumps(lifecycle.write_new(args.receipt,receipt),sort_keys=True))


if __name__=='__main__':
    main()
