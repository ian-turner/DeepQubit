"""Compares the % of training states solved per update across deepxube training runs.

A run is any directory under tmp/ holding a heur_train_summary.pkl (deepxube adds one entry per update and keeps it across
resumes). % solved is the number deepxube logs after each update: the mean over random-walk lengths of the % of that
update's finished searches that were solved.

Every run found gets data/training/<DOMAIN>_<HEUR>.csv (update, itr, per_solved), unless that CSV already has more
updates (e.g. it came from the cluster and tmp/ holds a short local run of the same name). The graphs are drawn from
these CSVs, so a run trained elsewhere only needs its CSV. Each comparison in COMPARISONS gets paper/images/<name>.png:
its runs (named by config, so tmp/$DOMAIN/$HEUR) averaged over bins of consecutive updates, about --bins points per
run, once at least two of them have a CSV.

Usage: python scripts/train_report.py [--dir tmp] [--bins 100]
"""
import csv
import math
import os
import pickle
import subprocess
from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(REPO, 'data', 'training')
FIG_DIR = os.path.join(REPO, 'paper', 'images')
SUMMARY_FILE = 'heur_train_summary.pkl'

CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
ORDINAL = ['#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b']  # one blue, light -> dark

# name -> (legend title, colors, [(label, config)]); a run's color is its position in the list, so it stays the same
# when other runs are missing (NeRF levels line up across the NeRF lists for the same reason)
COMPARISONS = {
    'n1_e0.01_encodings': ('Encoding', CATEGORICAL, [
        ('M', 'n1_e0.01'), ('H', 'n1_e0.01_H'), ('Q', 'n1_e0.01_Q'), ('H+M', 'n1_e0.01_H+M'),
        ('Q+M', 'n1_e0.01_Q+M'), ('H+Q', 'n1_e0.01_H+Q'), ('H+M+Q', 'n1_e0.01_H+M+Q')]),
    'n1_e0.01_nerf_M': ('NeRF dim', ORDINAL, [
        ('none', 'n1_e0.01'), ('5', 'n1_e0.01_L5'), ('10', 'n1_e0.01_L10'), ('15', 'n1_e0.01_L15'),
        ('20', 'n1_e0.01_L20')]),
    'n1_e0.01_nerf_H': ('NeRF dim', ORDINAL, [
        ('none', 'n1_e0.01_H'), ('5', 'n1_e0.01_H_L5'), ('10', 'n1_e0.01_H_L10'), ('15', 'n1_e0.01_H_L15')]),
    'n1_e0.01_nerf_H+M+Q': ('NeRF dim', ORDINAL, [
        ('none', 'n1_e0.01_H+M+Q'), ('5', 'n1_e0.01_H+M+Q_L5'), ('10', 'n1_e0.01_H+M+Q_L10'),
        ('15', 'n1_e0.01_H+M+Q_L15')]),
    'n3_exact_encodings': ('Encoding', CATEGORICAL, [('M (float)', 'n3_exact'), ('B9+C2 (ring)', 'n3_exact_ring'),
                                                    ('B9 (ring)', 'n3_exact_ring_0C')]),
}


def find_runs(root):
    return sorted(dirpath for dirpath, _, files in os.walk(root) if SUMMARY_FILE in files)


def per_update_solved(run_dir):
    """[(itr, per_solved)] in update order; itr is the training iteration at the start of the update"""
    summary = pickle.load(open(os.path.join(run_dir, SUMMARY_FILE), 'rb'))
    stats = summary.itr_to_steps_to_pathfindstats
    # equal weight per random-walk length, as in deepxube's get_eq_weighted_perf
    return [(itr, float(np.mean([s['per_solved'] for s in stats[itr].values()]))) for itr in sorted(stats)]


def csv_file_of(run):
    """data/training CSV of the run in tmp/<run>"""
    return os.path.join(CSV_DIR, f"{run.replace(os.sep, '_')}.csv")


def read_csv(csv_file):
    """[(itr, per_solved)] in update order"""
    with open(csv_file, newline='') as fp:
        return [(int(row['itr']), float(row['per_solved'])) for row in csv.DictReader(fp)]


def config_run(config):
    """$DOMAIN/$HEUR of a config (train.sh trains it in tmp/$DOMAIN/$HEUR)"""
    out = subprocess.run(['bash', '-c', 'source "$1" && echo "$DOMAIN/$HEUR"', '_', os.path.join('configs', config)],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def binned(per_solved, bin_size):
    """middle update and mean of each run of bin_size consecutive updates"""
    bins = [per_solved[i:i + bin_size] for i in range(0, len(per_solved), bin_size)]
    mid = np.array([i * bin_size + (len(b) - 1) / 2 for i, b in enumerate(bins)])
    return mid, np.array([b.mean() for b in bins])


def plot_comparison(runs, legend_title, fig_file, num_bins):
    """runs: [(label, color, per-update % solved)]"""
    fig, ax = plt.subplots()
    num_updates = max(len(per_solved) for _, _, per_solved in runs)
    bin_size = math.ceil(num_updates / num_bins)
    for label, color, per_solved in runs:
        mid, mean = binned(per_solved, bin_size)
        ax.plot(mid, mean, color=color, label=label, marker='o' if len(mid) == 1 else None)

    ax.set_xlim(0, max(num_updates - 1, 1))
    ax.set_ylim(top=min(ax.get_ylim()[1], 100))
    ax.set_xlabel('Update')
    ax.set_ylabel('% solved')
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(title=legend_title)
    fig.tight_layout()
    fig.savefig(fig_file, dpi=300)
    plt.close(fig)
    return bin_size


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default=os.path.join(REPO, 'tmp'), help='directory searched for training runs')
    parser.add_argument('--bins', type=int, default=100, help='about how many points each graph shows')
    args = parser.parse_args()

    os.makedirs(CSV_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    for run_dir in find_runs(args.dir):
        name = os.path.relpath(run_dir, args.dir)
        rows = per_update_solved(run_dir)
        csv_file = csv_file_of(name)
        if not rows:
            print(f"{name}: no updates yet, skipped")
            continue
        if os.path.isfile(csv_file) and len(read_csv(csv_file)) > len(rows):
            print(f"{name}: {len(rows)} updates in {args.dir} but more in {os.path.relpath(csv_file, REPO)}, kept the CSV")
            continue

        with open(csv_file, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['update', 'itr', 'per_solved'])
            for update, (itr, per_solved) in enumerate(rows):
                writer.writerow([update, itr, f"{per_solved:.4f}"])
        print(f"{name}: {len(rows)} updates -> {os.path.relpath(csv_file, REPO)}")

    for comp_name, (legend_title, colors, members) in COMPARISONS.items():
        runs, missing = [], []
        for idx, (label, config) in enumerate(members):
            csv_file = csv_file_of(config_run(config))
            if os.path.isfile(csv_file):
                runs.append((label, colors[idx], np.array([p for _, p in read_csv(csv_file)])))
            else:
                missing.append(config)
        fig_file = os.path.join(FIG_DIR, f"{comp_name}.png")
        if len(runs) >= 2:
            bin_size = plot_comparison(runs, legend_title, fig_file, args.bins)
            print(f"{comp_name}: {len(runs)}/{len(members)} runs, {bin_size} updates per point -> "
                  f"{os.path.relpath(fig_file, REPO)}")
        else:
            print(f"{comp_name}: {len(runs)}/{len(members)} runs, no graph")
        if missing:
            print(f"    not trained: {', '.join(missing)}")
