"""
Script that uses A* search to synthesize a
unitary matrix from an arbitrary gate set
"""
import os
import yaml
from time import time
from argparse import ArgumentParser
import numpy as np
from typing import List
from deepxube.search.astar import AStar, get_path
from deepxube.nnet import nnet_utils
from environments.qcircuit import *
from utils.matrix_utils import load_matrix_from_file
from utils.qasm_utils import *


config = {
    'max_steps': 1000,
    'batch_size': 1000,
    'epsilon': 0.01,
    'nerf_dim': 0,
    'path_weight': 0.2,
    'encoding': 'matrix',
    'gateset': 't,s,h,x,y,z',
}


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-n', '--nnet_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-m', '--max_steps', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-e', '--epsilon', type=float)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-L', '--nerf_dim', type=int)
    parser.add_argument('--path_weight', type=float)
    parser.add_argument('--encoding', type=str,
                        choices=['matrix', 'hurwitz', 'quaternion', 'discrete'],
                        help='Encoding method of unitary matrix before passing to NNet')
    parser.add_argument('--gateset', type=str)
    args = vars(parser.parse_args())

    # overriding default config options from config file
    if 'config' in args:
        with open(args['config'], 'r') as f:
            config_yaml = yaml.safe_load(f.read())
            for x in config_yaml:
                config[x] = config_yaml[x]
        del args['config']

    # overriding default config options with command line arguments
    for x in args:
        y = args[x]
        if y != None:
            config[x] = y
 
    start_time = time()

    # loading goal data
    num_qubits, goal_matrix = load_matrix_from_file(args['input'])

    # environment setup
    env: QCircuit = QCircuit(
        num_qubits=num_qubits,
        epsilon=config['epsilon'],
        encoding=config['encoding'],
        gateset=config['gateset'],
        L=config['nerf_dim'],
    )
    goal_states: List[QGoal] = [QGoal(goal_matrix)]
    start_states: List[QState] = [QState(tensor_product([I] * num_qubits))]
    weights: List[float] = [config['path_weight']]

    # loading heuristic function
    device, devices, on_gpu = nnet_utils.get_device()
    nnet_weights_file: str = os.path.join(config['nnet_dir'], 'target.pt')
    heuristic_fn = nnet_utils.load_heuristic_fn(nnet_weights_file, device, \
                                                on_gpu, env.get_v_nnet(), env)

    # setup A* search
    astar = AStar(env)
    astar.add_instances(start_states, goal_states, weights, heuristic_fn)
    print('Setup took %.3f seconds' % (time() - start_time))

    start_time = time()

    # running search
    step: int = 0
    while np.any([not x.finished for x in astar.instances]) and step < config['max_steps']:
        astar.step(heuristic_fn, config['batch_size'], verbose=args['verbose'])
        step += 1
    
    # getting path
    search_time = time() - start_time
    if astar.instances[0].finished:
        _, path_actions, _ = get_path(astar.instances[0].goal_node)
        # getting gate count and T-count
        gate_count: int = len(path_actions)
        t_count: int = 0
        for x in path_actions:
            if x.name == 't' or x.name == 'tdg':
                t_count += 1

        # converting circuit to OpenQASM 2.0
        qasm_str = path_to_qasm(path_actions, num_qubits)
        error = unitary_distance(qasm_to_matrix(qasm_str), goal_matrix)
        with open(args['output'], 'w') as f:
            f.write('// Gate-count: %d, T-count: %d, time: %.3fs, error: %.3e\n' % \
                    (gate_count, t_count, search_time, error))
            f.write(qasm_str)
        
        print('Found circuit with gate count %d and T count %d in %.3f seconds' % \
              (gate_count, t_count, search_time))

    else:
        print('Could not find circuit in %d steps' % config['max_steps'])
