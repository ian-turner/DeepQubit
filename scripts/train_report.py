"""Graphs the % of training states solved per update for every deepxube training run under tmp/.

A run is any directory holding a heur_train_summary.pkl (deepxube adds one entry per update and keeps it across resumes).
% solved is the number deepxube logs after each update: the mean over random-walk lengths of the % of that update's
finished searches that were solved.

For a run at tmp/<DOMAIN>/<HEUR>/ this writes data/training/<DOMAIN>_<HEUR>.csv (update, itr, per_solved) and
paper/images/<DOMAIN>_<HEUR>.png.

Usage: python scripts/train_report.py [--dir tmp] [--smooth <updates>]
"""
import csv
import os
import pickle
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, 'data', 'training')
FIG_DIR = os.path.join(REPO, 'paper', 'images')
SUMMARY_FILE = 'heur_train_summary.pkl'

LINE = '#2a78d6'
TEXT, TEXT_SECONDARY = '#0b0b0b', '#52514e'
GRID, AXIS = '#e1e0d9', '#c3c2b7'


def find_runs(root):
    return sorted(dirpath for dirpath, _, files in os.walk(root) if SUMMARY_FILE in files)


def per_update_solved(run_dir):
    """[(itr, per_solved)] in update order; itr is the training iteration at the start of the update"""
    summary = pickle.load(open(os.path.join(run_dir, SUMMARY_FILE), 'rb'))
    stats = summary.itr_to_steps_to_pathfindstats
    # equal weight per random-walk length, as in deepxube's get_eq_weighted_perf
    return [(itr, float(np.mean([s['per_solved'] for s in stats[itr].values()]))) for itr in sorted(stats)]


def trailing_mean(x, window):
    csum = np.concatenate([[0.0], np.cumsum(x)])
    starts = np.maximum(np.arange(len(x)) - window + 1, 0)
    return (csum[1:] - csum[starts]) / (np.arange(len(x)) + 1 - starts)


def plot(per_solved, fig_file, smooth):
    plt.rcParams.update({'font.size': 9})
    fig, ax = plt.subplots(figsize=(5, 3))
    updates = np.arange(len(per_solved))
    marker = 'o' if len(per_solved) == 1 else None  # a lone point draws no line
    if smooth > 1:
        ax.plot(updates, per_solved, color=LINE, lw=1, alpha=0.25)
        ax.plot(updates, trailing_mean(per_solved, smooth), color=LINE, lw=1.5, marker=marker)
    else:
        ax.plot(updates, per_solved, color=LINE, lw=1.5, marker=marker)

    ax.set_xlim(0, max(len(per_solved) - 1, 1))
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Update', color=TEXT)
    ax.set_ylabel('% solved', color=TEXT)
    ax.grid(axis='y', color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(fig_file, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=os.path.join(REPO, 'tmp'), help='directory searched for training runs')
    parser.add_argument('--smooth', type=int, default=1, help='also draw a trailing mean over this many updates')
    args = parser.parse_args()

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    runs = find_runs(args.dir)
    if not runs:
        raise SystemExit(f"no {SUMMARY_FILE} found under {args.dir}")

    for run_dir in runs:
        name = os.path.relpath(run_dir, args.dir).replace(os.sep, '_')
        rows = per_update_solved(run_dir)
        if not rows:
            print(f"{name}: no updates yet, skipped")
            continue

        csv_file = os.path.join(CSV_DIR, f"{name}.csv")
        with open(csv_file, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['update', 'itr', 'per_solved'])
            for update, (itr, per_solved) in enumerate(rows):
                writer.writerow([update, itr, f"{per_solved:.4f}"])

        fig_file = os.path.join(FIG_DIR, f"{name}.png")
        plot(np.array([p for _, p in rows]), fig_file, args.smooth)
        print(f"{name}: {len(rows)} updates, last %solved {rows[-1][1]:.2f} -> "
              f"{os.path.relpath(csv_file, REPO)}, {os.path.relpath(fig_file, REPO)}")
