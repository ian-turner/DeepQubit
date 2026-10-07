"""Prints a per-goal summary of a deepxube solve run and writes it to <results_dir>/summary.txt.

Goal names come from the goals .pkl's 'names' (set by goals_from_txt.py); goals without names are listed by index.

Usage: python scripts/solve_summary.py --results <results_dir> [--goals <goals.pkl>]
"""
import os
import pickle
from argparse import ArgumentParser

import numpy as np


T_GATES = ('t', 'tdg')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='directory holding the results.pkl from deepxube solve')
    parser.add_argument('--goals', type=str, default=None, help='goals .pkl that was solved (for goal names)')
    args = parser.parse_args()

    results = pickle.load(open(os.path.join(args.results, 'results.pkl'), 'rb'))
    num_goals = len(results['goals'])
    names, skipped = [str(i) for i in range(num_goals)], []
    if args.goals is not None:
        goals_data = pickle.load(open(args.goals, 'rb'))
        if 'names' in goals_data:
            # names label goals by position, so the solved goals must be these goals in this order
            goals = goals_data['goals']
            if len(goals) != num_goals or any(a != b for a, b in zip(goals, results['goals'])):
                raise SystemExit(f"goals in {args.goals} do not match {args.results}/results.pkl")
            names = goals_data['names']
        skipped = goals_data.get('skipped', [])

    width = max([len(x) for x in names + [name for name, _ in skipped]] + [4])
    lines = [f"{'goal':<{width}}  {'solved':>6}  {'gates':>5}  {'T':>4}  {'time (s)':>8}  {'nodes gen':>11}"]
    for i, name in enumerate(names):
        if i >= len(results['solved']):
            lines.append(f"{name:<{width}}  {'not run':>6}")
            continue
        solved = results['solved'][i]
        time_s, nodes = results['times'][i], results['num_nodes_generated'][i]
        if solved:
            acts = results['actions'][i]
            t_count = sum(getattr(a, 'name', None) in T_GATES for a in acts)
            gates, t_str = str(len(acts)), str(t_count)
        else:
            gates, t_str = '-', '-'
        lines.append(f"{name:<{width}}  {'yes' if solved else 'no':>6}  {gates:>5}  {t_str:>4}  {time_s:>8.2f}  {nodes:>11,}")
    for name, reason in skipped:
        lines.append(f"{name:<{width}}  skipped: {reason}")

    solved = results['solved']
    times = [t for t, s in zip(results['times'], solved) if s]
    lines.append('')
    lines.append(f"solved {sum(solved)}/{num_goals}" + (f" ({len(skipped)} skipped)" if skipped else '')
                 + (f", mean time (solved) {np.mean(times):.2f}s" if times else ''))

    summary = '\n'.join(lines)
    print(summary)
    with open(os.path.join(args.results, 'summary.txt'), 'w') as fp:
        print(summary, file=fp)
