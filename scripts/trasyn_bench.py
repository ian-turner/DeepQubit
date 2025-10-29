import os
from time import time
import trasyn

from utils.matrix_utils import *


def main():
    # running benchmark on rz gates
    gates = ['rz3', 'rz4', 'rz5', 'rz6', 'rz7']
    rz_dir = './data/targets/1qubit'

    Us = [] # loading matrices
    for g in gates:
        filepath = os.path.join(rz_dir, g + '.txt')
        _, U = load_matrix_from_file(filepath)
        Us.append(U)

    eps_l = [1e-2, 5e-3, 1e-3]
    for eps in eps_l:
        print('Running Trasyn rz benchmark for epsilon=%.2e' % eps)
        for i, U in enumerate(Us):
            start_time = time()
            seq, _, err = trasyn.synthesize(U, error_threshold=eps, nonclifford_budget=30)
            synth_time = time() - start_time
            print('Rz%i | time: %.3f | T-count: %i | gate count: %i | error: %.3e' % \
                (3+i, synth_time, seq.count('t'), len(seq), err)) 


if __name__ == '__main__':
    main()
