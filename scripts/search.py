"""
Script that uses A* search to synthesize a
unitary matrix from an arbitrary gate set
"""
import os
from time import time
from argparse import ArgumentParser
import numpy as np
import torch
from typing import List
from deepxube.search.astar import AStar, get_path
from deepxube.nnet import nnet_utils
from environments.qcircuit import *
from utils.matrix_utils import *


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-n', '--nnet_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-m', '--max_steps', type=int, default=1e4)
    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-e', '--epsilon', type=float, default=1e-2)
    parser.add_argument('-l', '--path_weight', type=float, default=0.2)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    args = parser.parse_args()

    # loading goal data
    num_qubits: int
    goal_matrix: np.ndarray[np.complex128]
    with open(args.input, 'r') as f:
        lines = [x.strip() for x in list(f)]
        num_qubits = int(lines[1])
        N = 2**(num_qubits)
        goal_matrix = np.zeros((N, N), dtype=np.complex128)
        for i in range(N):
            row = lines[2+i]
            cols = row.split(' ')
            for j, col in enumerate(cols):
                left, right = col.split(',')
                real = float(left[1:])
                imag = float(right[:-1])
                goal_matrix[i][j] = real + imag*1j

    # environment setup
    env: QCircuit = QCircuit(num_qubits=num_qubits, epsilon=args.epsilon)
    goal_states: List[QGoal] = [QGoal(torch.tensor(goal_matrix, dtype=torch.complex64, device=t_device))]
    start_states: List[QState] = [QState(tensor_product([I] * num_qubits))]
    weights: List[float] = [args.path_weight]

    # loading heuristic function
    device, devices, on_gpu = nnet_utils.get_device()
    nnet_weights_file: str = os.path.join(args.nnet_dir, 'target.pt')
    heuristic_fn = nnet_utils.load_heuristic_fn(nnet_weights_file, device, \
                                                on_gpu, env.get_v_nnet(), env)

    # setup A* search
    astar = AStar(env)
    astar.add_instances(start_states, goal_states, weights, heuristic_fn)
    start_time = time()

    # running search
    step: int = 0
    while np.any([not x.finished for x in astar.instances]) and step < args.max_steps:
        astar.step(heuristic_fn, args.batch_size, verbose=args.verbose)
        step += 1
    
    # getting path
    search_time = time() - start_time
    if astar.instances[0].finished:
        _, path_actions, _ = get_path(astar.instances[0].goal_node)
        # getting gate count and T-count
        gate_count: int = len(path_actions)
        t_count: int = 0
        for x in path_actions:
            if isinstance(x, TGate) or isinstance(x, TdgGate):
                t_count += 1

        # converting circuit to OpenQASM 2.0
        with open(args.output, 'w') as f:
            f.write('// Gate-count: %d, T-count: %d, time: %.3fs\n' % \
                    (gate_count, t_count, search_time))
            f.write('OPENQASM 2.0;\n')
            f.write('include "qelib1.inc";\n')
            f.write('qreg qubits[%d];\n' % num_qubits)
            for x in path_actions:
                f.write('%s ' % x.asm_name)
                if isinstance(x, OneQubitGate):
                    f.write('qubits[%d]' % x.qubit)
                elif isinstance(x, ControlledGate):
                    f.write('qubits[%d], qubits[%d]' % (x.control, x.target))
                f.write(';\n')
        
        print('Found circuit with gate count %d and T count %d in %.3f seconds' % \
              (gate_count, t_count, search_time))

    else:
        print('Could not find circuit in %d steps' % args.max_steps)