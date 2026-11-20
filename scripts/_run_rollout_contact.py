# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2026 Mark Higgins
"""Helper: run a rollout contact benchmark in a clean subprocess."""
import os
import sys
import time
import json
import argparse

build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, build_dir)

python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python'))
sys.path.insert(0, python_dir)

import bgbot_cpp
from neongammon.data import load_benchmark_file
from neongammon.weights import WeightConfig

bgbot_cpp.init_escape_tables()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, required=True)
    parser.add_argument('--truncation-depth', type=int, required=True)
    parser.add_argument('--decision-ply', type=int, required=True)
    parser.add_argument('--late-ply', type=int, default=-1)
    parser.add_argument('--late-threshold', type=int, default=20)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--step', type=int, default=1,
                        help='Subsample every Nth scenario (default: 1 = all)')
    args = parser.parse_args()

    w = WeightConfig.default()
    w.validate()

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    contact_file = os.path.join(data_dir, 'contact.bm')
    scenarios = load_benchmark_file(contact_file, step=args.step)
    n = scenarios.size()
    print(f'Loaded {n} scenarios (step={args.step})', file=sys.stderr, flush=True)

    kwargs = dict(
        n_trials=args.n_trials,
        truncation_depth=args.truncation_depth,
        decision_ply=args.decision_ply,
        n_threads=args.threads,
    )
    if args.late_ply >= 0:
        kwargs['late_ply'] = args.late_ply
        kwargs['late_threshold'] = args.late_threshold

    strat = bgbot_cpp.create_rollout_5nn(*w.weight_args, **kwargs)
    print('Rollout strategy created, scoring...', file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    r = bgbot_cpp.score_benchmarks_rollout(scenarios, strat, 1)
    elapsed = time.perf_counter() - t0
    er = r.total_error / r.count * 1000 if r.count > 0 else 0
    print(json.dumps({'er': er, 'elapsed': elapsed, 'count': r.count}))


if __name__ == '__main__':
    main()
