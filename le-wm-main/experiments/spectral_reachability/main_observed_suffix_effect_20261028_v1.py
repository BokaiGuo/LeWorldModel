"""Independent saved-array arithmetic after the entire observed compiler strict.

Model-only intervention: never infer physical benefit from optimized predictions.
"""
import argparse
from pathlib import Path
import numpy as np
from experiments.spectral_reachability import agent_nonlinear_suffix_observed_recompile_20261028_v2 as source

NAME = 'main_observed_suffix_effect_20261028_v1'


def qvalue(goal, context):
    g = np.asarray(goal, dtype=np.float64)
    z = np.asarray(context['latent_history'], dtype=np.float64)
    source.require(g.shape == (192,) and z.shape == (3, 192)
                   and np.isfinite(g).all() and np.isfinite(z).all(), 'finite original goal/full3')
    return float(np.mean((z[-1] - g) ** 2))


def analyze(spec, row):
    source.require(spec['index'] == row['index'] and row['status'] == 'sealed', 'same sealed slot')
    request = spec['request']; numeric = row['result']['numeric']; plan = row['result']['plan']
    goal = request['task']['goal']; incumbent = request['incumbent']
    values = numeric['iterates']
    source.require(len(values) == 9 and [v['iteration'] for v in values] == list(range(9)), 'all nine ordered scores')
    costs = [qvalue(goal, v['context']) for v in values]
    source.require(costs == [v['Q'] for v in values], 'independent float64 original meanQ')
    selected = min(range(9), key=lambda i: (costs[i], i))
    source.require(type(numeric['winner_iteration']) is int and selected == numeric['winner_iteration']
                   == plan['winner_iteration'], 'stable best of all nine, no candidate removal')
    prefix = np.asarray(incumbent['prefix'], dtype=np.float32)
    old = np.asarray(incumbent['commands'], dtype=np.float32)
    commands = np.asarray(plan['commands'], dtype=np.float32)
    source.require(prefix.shape == old.shape == (5, 2) and commands.shape == (10, 2)
                   and np.isfinite(commands).all(), 'exact raw action axes')
    source.require(prefix.tobytes() == commands[:5].tobytes()
                   and old.tobytes() == np.asarray(values[0]['commands'], np.float32).tobytes(),
                   'prefix and incumbent rawfloat32 bits unchanged')
    source.require(np.asarray(values[selected]['commands'], np.float32).tobytes() == commands[5:].tobytes()
                   and costs[0] == spec['observed_incumbent_Q'] == numeric['observed_incumbent_Q']
                   and costs[selected] == numeric['new_suffix_Q'] == plan['new_suffix_Q']
                   == qvalue(goal, plan['new_terminal_context'])
                   and plan['new_terminal_context'] == values[selected]['context'], 'indivisible new actions/context/Q')
    delta = commands[5:].astype(np.float64) - old.astype(np.float64)
    return dict(index=spec['index'], root_slot=spec['root_slot'], condition=spec['condition'],
                observed_incumbent_Q=costs[0], optimized_Q=costs[selected],
                model_Q_delta=costs[selected]-costs[0], selected_iteration=selected,
                all_nine_Q=costs, suffix_raw_L2_change=float(np.linalg.norm(delta)),
                suffix_changed=commands[5:].tobytes() != old.tobytes(), prefix_exact=True,
                original_goal_sha256=source.digest(goal), physical_benefit=None,
                caveat='Two known roots; model-only suffix optimization, not P1 or a fresh confirmation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('producer-output', 'report-sha', 'receipt', 'receipt-sha', 'output'):
        p.add_argument('--'+key, required=True)
    a = p.parse_args(); directory = Path(a.producer_output).absolute()
    source.require(source.sha(a.receipt) == a.receipt_sha, 'external persisted receipt SHA')
    strict = source.consume(directory, a.report_sha)
    source.require(strict['complete'] is True and source.raw(strict) == Path(a.receipt).read_bytes(),
                   'entire compiler strict rebuilt and persisted bytes equal')
    report = source.signal.read(directory/'report.json', 32*1024**2)
    prepared = source.signal.read(report['prepared'], 32*1024**2)
    result = dict(schema=NAME, report_sha256=a.report_sha, receipt_sha256=a.receipt_sha,
                  full_upstream_rebuilt_here=True,
                  rows=[analyze(s, r) for s, r in zip(prepared['rows'], report['rows'], strict=True)],
                  known_roots=2, fixed_conditions=4, new_neural_calls=0, new_physics_calls=0,
                  P1_admission=False, scientific_admission=False, freshness=False)
    out = Path(a.output).absolute()
    source.require(out.parent == source.ROOT/'results' and out.name.startswith(NAME+'_')
                   and not out.exists() and not out.is_symlink(), 'new exclusive saved analysis')
    with out.open('xb') as f:
        f.write(source.raw(result))
    print(source.raw(dict(path=str(out), sha256=source.sha(out), rows=result['rows'])).decode())


if __name__ == '__main__':
    main()
