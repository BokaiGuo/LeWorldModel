"""Read-only cell03 HOST authority and saved physical-input precheck; no GPU/physics."""
import argparse
import json

import numpy as np

from experiments.spectral_reachability import agent_e3_chunked_identity_host_20261029_v1 as host

ROOT_SHA = 'a3bd09470de75652506544b8bea0ff3c5fbae0c0d3b28775da47ee3e1adbb6f7'
CP5_PIXEL_SHA = '5b5f3fb1b83fa832bbbc663d9eb2efcb4171b5d093090a4296e172f3deb2e327'


def check(authority, authority_sha):
    binding = host.authority(authority, authority_sha)
    need = host.need
    need(binding['cell'] == 3, 'fixed cell03')
    old = host.launch.episode.old
    control = old.authority.load(*binding['args'])
    body = control['body']
    need(body['root_sha256'] == ROOT_SHA and body['horizon_packets'] == 3
         and body['domain']['name'] == 'Dx' and body['feedback_boundary'] == 5,
         'fixed known-root H3 Dx control5 body')
    first = control['first5']
    need(first.dtype == np.float32 and first.shape == (5, 2)
         and np.all(first[:, 1].view(np.uint32) == 0), 'control5 selected Dx positive zero')
    setup = old.setup_authority.resolve(binding['resolver']['path'], binding['resolver']['sha256'], 3, body)
    cp5 = setup['expected_cp5']
    need(setup['original_root_sha'] == ROOT_SHA and cp5['step'] == 5
         and cp5['pixel_sha256'] == CP5_PIXEL_SHA and setup['prefix5'].shape == (5, 2),
         'saved native root and cp5 identity')
    need(host.launch.host.sha(authority) == authority_sha, 'authority stable after saved join')
    return dict(cell=3, task='r0.g0.h3.dDx', root_sha256=ROOT_SHA,
                cp5_pixel_sha256=CP5_PIXEL_SHA, horizon_packets=3, domain='Dx',
                first5_positive_zero_y=True, saved_setup_inputs_joined=True,
                runtime_replay=False, neural_calls=0, physics_calls=0,
                full_saved_strict=False, freshness=False, scientific_admission=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--authority-sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.authority, args.authority_sha256), sort_keys=True))


if __name__ == '__main__':
    main()
