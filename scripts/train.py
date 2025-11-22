import os
import torch
from argparse import ArgumentParser
from deepxube.training import avi
from environments.qcircuit import QCircuit
import yaml


config = {
    'num_qubits': 1,
    'encoding': 'matrix',
    'epsilon': 0.01,
    'gateset': 't,s,h,x,y,z',
    'step_max': 30,
    'batch_size': 1000,
    'itrs_per_update': 1000,
    'max_itrs': 1e5,
    'greedy_update_step_max': 100,
    'num_update_procs': 10,
    'perturb': False,
    'nerf_dim': 0,
}


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--nnet_dir', type=str, required=True)
    parser.add_argument('--num_qubits', type=int)
    parser.add_argument('--encoding', type=str,
                        choices=['matrix', 'hurwitz', 'quaternion', 'discrete'],
                        help='Encoding method of unitary matrix before passing to NNet')
    parser.add_argument('--epsilon', type=float,
                        help='Tolerance value for solved condition')
    parser.add_argument('--gateset', type=str)
    parser.add_argument('--step_max', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--itrs_per_update', type=int)
    parser.add_argument('--max_itrs', type=int)
    parser.add_argument('--greedy_update_step_max', type=int)
    parser.add_argument('--num_update_procs', type=int)
    parser.add_argument('--perturb', action='store_true')
    parser.add_argument('-L', '--nerf_dim', type=int)
    args = vars(parser.parse_args())

    # overriding default config options from config file
    if 'config' in args and args['config'] != None:
        with open(args['config'], 'r') as f:
            config_yaml = yaml.safe_load(f.read())
            for x in config_yaml:
                config[x] = config_yaml[x]
        del args['config']

    # overriding default config options with command line arguments
    for x in args:
        y = args[x]
        if y != None and y != False:
            config[x] = y

    # environment setup
    env = QCircuit(
        num_qubits=config['num_qubits'],
        epsilon=config['epsilon'],
        L=config['nerf_dim'],
        perturb=config['perturb'],
        encoding=config['encoding'],
    )

    # copying config options
    if not os.path.exists(config['nnet_dir']):
        os.mkdir(config['nnet_dir'])
    config_file_copy = os.path.join(config['nnet_dir'], 'config.yaml')
    with open(config_file_copy, 'w') as f:
        yaml.safe_dump(config, f)

    # running approximate value iteration
    avi.train(
        env=env,
        nnet_dir=config['nnet_dir'],
        step_max=config['step_max'],
        batch_size=config['batch_size'],
        itrs_per_update=config['itrs_per_update'],
        max_itrs=config['max_itrs'],
        greedy_update_step_max=config['greedy_update_step_max'],
        num_update_procs=config['num_update_procs'],
    )
