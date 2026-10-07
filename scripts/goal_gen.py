"""Generates random target unitaries (Haar-random, from qiskit's random_unitary) as a goals .pkl.

Each goal is a random 2^n x 2^n unitary with the identity as its start state, in the same {'states', 'goals'}
layout deepxube reads. The same --seed always gives the same goals.

Usage: python scripts/goal_gen.py [--num_qubits 1] [--num 1000] [--seed 0] [--output <pkl>]
       (default output: data/targets/<n>qubit/random_<num>.pkl)
"""
import os
import pickle
from argparse import ArgumentParser

import numpy as np
from qiskit.quantum_info import random_unitary

from domains.qcircuit import QState, QGoal
from utils.matrix_utils import I, tensor_product, hash_unitary, hash_unitary_batch


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--num_qubits', type=int, default=1)
    parser.add_argument('--num', type=int, default=1000, help='number of goals')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', type=str, default=None,
                        help='default: data/targets/<num_qubits>qubit/random_<num>.pkl')
    args = parser.parse_args()
    output = args.output or f"data/targets/{args.num_qubits}qubit/random_{args.num}.pkl"

    # sampling unitaries
    rng = np.random.default_rng(args.seed)
    Us = np.array([random_unitary(2 ** args.num_qubits, seed=rng).data for _ in range(args.num)])

    # creating state/goal pairs: identity start, random unitary goal
    In = tensor_product([I] * args.num_qubits)
    In_hash = hash_unitary(In)
    data = {'states': [QState(In, In_hash) for _ in range(args.num)],
            'goals': [QGoal(U, h) for U, h in zip(Us, hash_unitary_batch(Us).tolist())]}

    # saving using pickle
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    pickle.dump(data, open(output, 'wb'), protocol=-1)
    print(f"wrote {args.num} random {args.num_qubits}-qubit goals (seed {args.seed}) to {output}")
