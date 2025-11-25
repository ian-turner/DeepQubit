import os
import sys
from time import time
from argparse import ArgumentParser
import trasyn

from utils.matrix_utils import *


def main():
    # running benchmark on rz gates
    parser = ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--t_budget', type=int, default=30)
    args = parser.parse_args()

    Us = [] # loading matrices
    for file in os.listdir(args.directory):
        if file.endswith('.txt'):
            filepath = os.path.join(args.directory, file)
            _, U = load_matrix_from_file(filepath)
            Us.append(U)

    print('Running Trasyn benchmark for epsilon=%.2e' % args.epsilon)
    for i, U in enumerate(Us):
        start_time = time()
        seq, _, err = trasyn.synthesize(U, error_threshold=args.epsilon, nonclifford_budget=args.t_budget)
        synth_time = time() - start_time
        print('U%d | time: %.3f | T-count: %i | gate count: %i | error: %.3e' % \
            (i, synth_time, seq.count('t'), len(seq), err)) 


if __name__ == '__main__':
    main()
