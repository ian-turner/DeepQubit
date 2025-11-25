import os
import sys
from time import time
from argparse import ArgumentParser
import trasyn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from utils.matrix_utils import *


def main():
    # running benchmark on rz gates
    parser = ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--t_budget', type=int, default=30)
    args = parser.parse_args()

    N = None # size of matrix
    Us = [] # loading matrices
    filenames = []
    for file in os.listdir(args.directory):
        if file.endswith('.txt'):
            filepath = os.path.join(args.directory, file)
            filenames.append(file)
            N, U = load_matrix_from_file(filepath)
            Us.append(U)

    print('Running Trasyn benchmark for epsilon=%.2e' % args.epsilon)
    for i, U in enumerate(Us):
        start_time = time()
        if N > 1:
            qc = QuantumCircuit(N)
            qc.unitary(U, list(range(N)))
            qc_synth = trasyn.synthesize_qiskit_circuit(qc, error_threshold=args.epsilon, nonclifford_budget=args.t_budget)
            err = unitary_distance(U, Operator(qc_synth).data)
            t_count = 0
            for x in qc_synth.data:
                if x.name == 't':
                    t_count += 1
            gate_count = len(qc_synth)

        else:
            seq, _, err = trasyn.synthesize(U, error_threshold=args.epsilon, nonclifford_budget=args.t_budget)
            t_count = seq.count('t')
            gate_count = len(seq)

        synth_time = time() - start_time
        print('%s | time: %.3f | T-count: %i | gate count: %i | error: %.3e' % \
            (filenames[i], synth_time, t_count, gate_count, err)) 


if __name__ == '__main__':
    main()
