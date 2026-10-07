"""Runs the trasyn baseline synthesizer on a goals .pkl and writes a per-goal CSV.

CSV columns: goal (the pkl's 'names' if present, else the goal index), time (s), t_count, gate_count, error,
solved (error <= epsilon). error is unitary_distance, the metric QCircuit.is_solved uses (for one qubit it equals
trasyn's own error). Default output: data/baselines/trasyn_n<N>_<goals stem>_e<epsilon>_<t_budget>T.csv

Usage: python scripts/trasyn_bench.py [goals.pkl] [--epsilon 0.01] [--t_budget 30] [--output <csv>]
       (default goals: data/targets/1qubit/random_1000.pkl)
"""
import os
import csv
import pickle
from time import time
from argparse import ArgumentParser
import trasyn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from utils.matrix_utils import *


def main():
    parser = ArgumentParser()
    parser.add_argument('goals', type=str, nargs='?', default='data/targets/1qubit/random_1000.pkl')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--t_budget', type=int, default=30)
    parser.add_argument('--output', type=str, default=None,
                        help='default: data/baselines/trasyn_n<N>_<goals stem>_e<epsilon>_<t_budget>T.csv')
    args = parser.parse_args()

    # loading goal matrices
    data = pickle.load(open(args.goals, 'rb'))
    Us = [x.unitary for x in data['goals']]
    names = data.get('names', [str(i) for i in range(len(Us))])
    N = int(np.log2(Us[0].shape[0]))

    stem = os.path.splitext(os.path.basename(args.goals))[0]
    output = args.output or f"data/baselines/trasyn_n{N}_{stem}_e{args.epsilon}_{args.t_budget}T.csv"
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    print('Running Trasyn benchmark for epsilon=%.2e on %d goals' % (args.epsilon, len(Us)))
    rows = []
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['goal', 'time', 't_count', 'gate_count', 'error', 'solved'])
        for i, (name, U) in enumerate(zip(names, Us)):
            start_time = time()
            if N > 1:
                qc = QuantumCircuit(N)
                qc.unitary(U, list(range(N)))
                qc_synth, _, _ = trasyn.synthesize_qiskit_circuit(qc, error_threshold=args.epsilon,
                                                                  nonclifford_budget=args.t_budget)
                err = unitary_distance(U, Operator(qc_synth).data)
                t_count = sum(1 for x in qc_synth.data if x.name == 't')
                gate_count = len(qc_synth)
            else:
                seq, mat, _ = trasyn.synthesize(U, error_threshold=args.epsilon, nonclifford_budget=args.t_budget)
                err = unitary_distance(U, mat)
                t_count = seq.count('t')
                gate_count = len(seq)
            synth_time = time() - start_time

            row = (name, synth_time, t_count, gate_count, float(err), bool(err <= args.epsilon))
            rows.append(row)
            # written as it goes, so an interrupted run keeps its finished goals
            writer.writerow([name, '%.4f' % synth_time, t_count, gate_count, '%.6e' % err, int(row[-1])])
            f.flush()
            print('Goal %s (%i) | time: %.3f | T-count: %i | gate count: %i | error: %.3e' %
                  (name, i, synth_time, t_count, gate_count, err))

    times, t_counts, gate_counts, _, solved = [np.array(x) for x in list(zip(*rows))[1:]]
    print('Solved %.1f%% (%d/%d) | mean time: %.3f | mean T-count: %.2f | mean gate count: %.2f' %
          (100 * solved.mean(), solved.sum(), len(solved), times.mean(), t_counts.mean(), gate_counts.mean()))
    print('Wrote %s' % output)


if __name__ == '__main__':
    main()
