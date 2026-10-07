"""Compares our solve results with the trasyn baseline, goal by goal, on the same goals (e.g. random_1000).

Reads the per-goal trasyn CSV from trasyn_bench.py and one or more solve summaries (the summary.txt solve.sh writes,
or a copy of it in data/results/), pairs the goals by name, and prints:
- per method: goals solved, mean/median T-count and gate count (over the goals it solved), mean/median time per goal
- per run vs trasyn, on the goals both solved: how many goals have fewer/equal/more T gates and gates, and the mean
  difference (ours - trasyn; negative = ours is shorter)
and draws the number of goals at each T-count and gate-count difference, as two figures (the counts differ in scale):
paper/images/compare_<trasyn csv stem>_t_count.png and ..._gate_count.png.

Both must be at the same epsilon (trasyn's solved column is error <= epsilon, as QCircuit.is_solved) and trasyn's
default gate set 'tshxyz' is our default CliffT, so gate counts compare directly. Times come from different programs
(trasyn per call, ours deepxube's search time per goal), so compare them only for runs on the same hardware.

Usage: python scripts/trasyn_compare.py [summary ...] [--trasyn <csv>] [--labels <label> ...] [--fig <prefix>]
       (defaults: data/results/1qubit/e0.01_L10_random_100B_0.0W.txt vs
                  data/baselines/trasyn_n1_random_1000_e0.01_30T.csv)
"""
import csv
import os
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']  # train_report.py


def read_trasyn(csv_file):
    """goal -> (solved, t_count, gate_count, time)"""
    return {r['goal']: (bool(int(r['solved'])), int(r['t_count']), int(r['gate_count']), float(r['time']))
            for r in csv.DictReader(open(csv_file))}


def read_summary(summary_file):
    """goal -> (solved, t_count, gate_count, time) from a solve_summary.py table; unsolved goals have no counts.
    Goals listed as 'not run' or 'skipped' are left out."""
    rows = {}
    lines = open(summary_file).read().splitlines()
    for line in lines[1:]:
        if not line.strip():  # the totals line follows a blank line
            break
        parts = line.split()
        if parts[1] == 'yes':
            rows[parts[0]] = (True, int(parts[3]), int(parts[2]), float(parts[4]))
        elif parts[1] == 'no':
            rows[parts[0]] = (False, None, None, float(parts[4]))
    return rows


def print_table(header, rows):
    """header and rows are lists of cells; the first column is left-aligned, the rest right-aligned"""
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print('  '.join(f"{str(c):<{w}}" if i == 0 else f"{str(c):>{w}}" for i, (c, w) in enumerate(zip(r, widths))))


def method_row(label, rows, goals):
    solved = [rows[g] for g in goals if rows[g][0]]
    stats = lambda x: [f"{np.mean(x):.2f}", f"{np.median(x):.1f}"] if len(x) else ['-', '-']
    times = [rows[g][3] for g in goals]
    return ([label, f"{len(solved)}/{len(goals)}"] + stats([r[1] for r in solved]) + stats([r[2] for r in solved])
            + [f"{np.mean(times):.3f}", f"{np.median(times):.3f}"])


def diff_cells(diffs):
    """fewer / equal / more goals and the mean difference"""
    return [(diffs < 0).sum(), (diffs == 0).sum(), (diffs > 0).sum(), f"{diffs.mean():+.2f}" if len(diffs) else '-']


def plot_diffs(runs, idx, metric, fig_file):
    """runs: [(label, color, T-count diffs, gate-count diffs)]; bars of the number of paired goals at each difference
    in runs[k][idx]"""
    fig, ax = plt.subplots()
    width = 0.8 / len(runs)
    values = np.arange(min(min(r[idx].min() for r in runs), 0), max(max(r[idx].max() for r in runs), 0) + 1)
    for k, run in enumerate(runs):
        counts = np.array([(run[idx] == v).sum() for v in values])
        bars = ax.bar(values + (k - (len(runs) - 1) / 2) * width, counts, width, color=run[1], label=run[0])
        if len(runs) == 1:  # one run: label each bar with its count (the small bars are barely visible)
            ax.bar_label(bars, labels=[str(n) if n else '' for n in counts])
            ax.margins(y=0.08)  # room for the tallest bar's label
    ax.set_xticks(values)
    ax.set_xlabel(f"{metric} difference (ours − trasyn)")
    ax.set_ylabel('Goals')
    ax.grid(True)
    ax.set_axisbelow(True)
    if len(runs) > 1:
        ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(fig_file) or '.', exist_ok=True)
    fig.savefig(fig_file, dpi=300)
    plt.close(fig)


def main():
    parser = ArgumentParser()
    parser.add_argument('summaries', type=str, nargs='*',
                        default=['data/results/1qubit/e0.01_L10_random_100B_0.0W.txt'],
                        help='solve summary tables (solve_summary.py output)')
    parser.add_argument('--trasyn', type=str, default='data/baselines/trasyn_n1_random_1000_e0.01_30T.csv')
    parser.add_argument('--labels', type=str, nargs='*', default=None,
                        help='one label per summary (default: the summary file stems)')
    parser.add_argument('--fig', type=str, default=None,
                        help='figure path prefix; writes <prefix>_t_count.png and <prefix>_gate_count.png '
                             '(default: paper/images/compare_<trasyn csv stem>)')
    args = parser.parse_args()

    labels = args.labels or [os.path.splitext(os.path.basename(s))[0] for s in args.summaries]
    if len(labels) != len(args.summaries):
        raise SystemExit('--labels needs one label per summary')
    if len(args.summaries) > len(CATEGORICAL):
        raise SystemExit(f"at most {len(CATEGORICAL)} summaries per figure")
    trasyn = read_trasyn(args.trasyn)
    ours = [read_summary(s) for s in args.summaries]

    # goals in all the files; a partial run (e.g. an interrupted trasyn_bench.py) narrows the comparison
    goals = sorted(set(trasyn).intersection(*ours), key=lambda g: (not g.isdigit(), int(g) if g.isdigit() else 0, g))
    for name, rows in [(args.trasyn, trasyn)] + list(zip(args.summaries, ours)):
        if len(rows) != len(goals):
            print(f"warning: {name} has {len(rows)} goals; comparing on the {len(goals)} goals in every file")
    if not goals:
        raise SystemExit('no goals in common')

    print_table(['method', 'solved', 'T mean', 'T med', 'gates mean', 'gates med', 'time mean (s)', 'time med (s)'],
                [method_row('trasyn', trasyn, goals)] + [method_row(l, r, goals) for l, r in zip(labels, ours)])
    print('(T-count and gate count over the goals each method solved; time over all goals)')

    print('\nvs trasyn on the goals both solved (fewer/equal/more = goals where ours has fewer/equal/more; '
          'diff = ours - trasyn):')
    runs, table = [], []
    for k, (label, rows) in enumerate(zip(labels, ours)):
        paired = [g for g in goals if rows[g][0] and trasyn[g][0]]
        t_diffs = np.array([rows[g][1] - trasyn[g][1] for g in paired])
        gate_diffs = np.array([rows[g][2] - trasyn[g][2] for g in paired])
        table.append([label, len(paired)] + diff_cells(t_diffs) + diff_cells(gate_diffs))
        if paired:
            runs.append((label, CATEGORICAL[k], t_diffs, gate_diffs))
    print_table(['method', 'paired', 'T fewer', 'equal', 'more', 'mean diff',
                 'gates fewer', 'equal', 'more', 'mean diff'], table)

    if runs:
        stem = os.path.splitext(os.path.basename(args.trasyn))[0]
        prefix = args.fig or f"paper/images/compare_{stem}"
        print()
        for idx, metric, suffix in ((2, 'T-count', 't_count'), (3, 'Gate count', 'gate_count')):
            fig_file = f"{prefix}_{suffix}.png"
            plot_diffs(runs, idx, metric, fig_file)
            print(f"Wrote {fig_file}")


if __name__ == '__main__':
    main()
