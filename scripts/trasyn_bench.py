"""Runs the trasyn baseline synthesizer on a goals .pkl and writes a per-goal CSV.

CSV columns: goal (the pkl's 'names' if present, else the goal index), time (s), t_count, gate_count, error,
solved (error <= epsilon). error is unitary_distance, the metric QCircuit.is_solved uses (for one qubit it equals
trasyn's own error). Default output: data/baselines/trasyn_n<N>_<goals stem>_e<epsilon>_<t_budget>T.csv

The first goal is synthesized once, untimed, before the timed loop: trasyn's first call in a process pays one-time
costs (on GPU: CUDA/cupy startup and kernel loading; cold reads of its ~60 MB lookup tables) that took ~4 s on the
cluster vs ~0.2 s per goal after. trasyn keeps no search state between calls, so this does not speed up later goals.

Usage: python scripts/trasyn_bench.py [goals.pkl] [--epsilon 0.01] [--t_budget 30] [--output <csv>]
       (default goals: data/targets/1qubit/random_1000.pkl)
"""
import os
import csv
import pickle
from time import time
from typing import Tuple
from argparse import ArgumentParser
import trasyn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from utils.matrix_utils import *


def synthesize(U: np.ndarray, num_qubits: int, epsilon: float, t_budget: int) -> Tuple[int, int, float]:
    """Synthesizes U with trasyn; returns (T-count, gate count, unitary_distance error)"""
    if num_qubits > 1:
        qc = QuantumCircuit(num_qubits)
        qc.unitary(U, list(range(num_qubits)))
        qc_synth, _, _ = trasyn.synthesize_qiskit_circuit(qc, error_threshold=epsilon, nonclifford_budget=t_budget)
        err = unitary_distance(U, Operator(qc_synth).data)
        return sum(1 for x in qc_synth.data if x.name == 't'), len(qc_synth), err
    seq, mat, _ = trasyn.synthesize(U, error_threshold=epsilon, nonclifford_budget=t_budget)
    return seq.count('t'), len(seq), unitary_distance(U, mat)


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
    # untimed warm-up on the first goal, so its time does not include trasyn's one-time startup costs
    start_time = time()
    synthesize(Us[0], N, args.epsilon, args.t_budget)
    print('Warm-up (untimed): %.3f' % (time() - start_time))

    rows = []
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['goal', 'time', 't_count', 'gate_count', 'error', 'solved'])
        for i, (name, U) in enumerate(zip(names, Us)):
            start_time = time()
            t_count, gate_count, err = synthesize(U, N, args.epsilon, args.t_budget)
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
