"""
Script that uses A* search to synthesize a
unitary matrix from an arbitrary gate set
"""
import os
import yaml
from time import time
from argparse import ArgumentParser
import numpy as np
from typing import List, cast
from deepxube.pathfinding.v.bwas import BWASEnum, InstanceBWAS
from deepxube.base.pathfinding import get_path
from deepxube.base.heuristic import HeurNNet
from deepxube.nnet import nnet_utils
from environments.qcircuit import *
from utils.matrix_utils import load_matrix_from_file
from utils.qasm_utils import *


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('-n', '--nnet_dir', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-m', '--max_steps', type=int, default=1000)
    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-e', '--epsilon', type=float, default=0.01)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('-L', '--nerf_dim', type=int, default=0)
    parser.add_argument('--path_weight', type=float, default=0.2)
    parser.add_argument('--search_eps', type=float, default=0.0)
    parser.add_argument('--encoding', type=str, default='matrix',
                        choices=['matrix', 'hurwitz', 'quaternion', 'discrete'],
                        help='Encoding method of unitary matrix before passing to NNet')
    parser.add_argument('--gateset', type=str, default='t,s,h,x,y,z,cx')
    config = vars(parser.parse_args())

    # overriding default config options from config file
    if 'config_file' in config and config['config_file'] != None:
        with open(config['config_file'], 'r') as f:
            config_yaml = yaml.safe_load(f.read())
            for x in config_yaml:
                config[x] = config_yaml[x]

    # overriding default config options with command line arguments
    for x in config:
        y = config[x]
        if y != None:
            config[x] = y
 
    start_time = time()

    # loading goal data
    num_qubits, goal_matrix = load_matrix_from_file(config['input'])

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
    nnet_weights_file: str = os.path.join(config['nnet_dir'], 'heur_targ.pt')
    heur_fn = QCircuitNNetParV(n=num_qubits,
                               L=config['nerf_dim'],
                               encoding=config['encoding'])
    nnet = nnet_utils.load_nnet(nnet_weights_file, heur_fn.get_nnet(), device)
    nnet.to(device)

    # setup A* search
    astar = BWASEnum(env=env,
                     heur_fn=heur_fn.get_nnet_fn(nnet=nnet,
                                                 batch_size=config['batch_size'],
                                                 device=device,
                                                 update_num=None))
    root_node = astar.create_root_nodes(start_states, goal_states, compute_init_heur=False)[0]
    astar.add_instances([InstanceBWAS(root_node=root_node,
                                      batch_size=config['batch_size'],
                                      weight=config['path_weight'],
                                      eps=config['search_eps'],
                                      inst_info=None)])
    print('Setup took %.3f seconds' % (time() - start_time))

    start_time = time()

    # running search
    step: int = 0
    while np.any([not x.finished() for x in astar.instances]) and step < config['max_steps']:
        astar.step(verbose=config['verbose'])
        step += 1
    
    # getting path
    search_time = time() - start_time
    if astar.instances[0].finished:
        _, path_actions, _ = get_path(astar.instances[0].goal_node)
        # getting gate count and T-count
        gate_count: int = len(path_actions)
        t_count: int = 0
        for x in path_actions:
            if x.name in ['t', 'tdg', 't10', 't100']:
                t_count += 1

        # converting circuit to OpenQASM 2.0
        qasm_str = path_to_qasm(path_actions, num_qubits)
        error = unitary_distance(qasm_to_matrix(qasm_str), goal_matrix)
        with open(config['output'], 'w') as f:
            f.write('// Gate-count: %d, T-count: %d, time: %.3fs, error: %.3e\n' % \
                    (gate_count, t_count, search_time, error))
            f.write(qasm_str)
        
        print('Found circuit with gate count %d and T count %d in %.3f seconds' % \
              (gate_count, t_count, search_time))

    else:
        print('Could not find circuit in %d steps' % config['max_steps'])
