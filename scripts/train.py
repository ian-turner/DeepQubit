import os
import yaml
import torch
from argparse import ArgumentParser
from deepxube.training.train_utils import TrainArgs
from deepxube.training.train_heur import train
from deepxube.base.updater import UpdateHeur, UpArgs, UpHeurArgs
from deepxube.updater.updaters import UpdateHeurBWASEnum, UpdateHeurBWQSEnum, \
                                      UpdateHeurGrPolQEnum, UpdateHeurStepLenSup, UpBWASArgs

from environments.qcircuit import QCircuit, QCircuitNNetParV


config = {
    'num_qubits': 1,
    'heur_type': 'V',
    'encoding': 'matrix',
    'epsilon': 0.01,
    'gateset': 't,s,h,x,y,z',
    'step_max': 30,
    'batch_size': 1000,
    'itrs_per_update': 1000,
    'max_itrs': 1e5,
    'greedy_update_step_max': 100,
    'num_test_per_step': 30,
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
    parser.add_argument('--num_test_per_step', type=int)
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

    # updater setup
    up_args = UpArgs(config['step_max'], config['itrs_per_update'], config['itrs_per_update'],
                     config['num_update_procs'], config['greedy_update_step_max'], config['batch_size'],
                     config['batch_size'], False)
    up_heur_args = UpHeurArgs(up_args, False, 1)
    up_bwas_args = UpBWASArgs(up_heur_args, 0.2, 0.1)
    updater = UpdateHeurBWASEnum(env, up_bwas_args, QCircuitNNetParV(config['num_qubits'], \
                                                    config['nerf_dim'], config['encoding']))

    # running training
    train_args: TrainArgs = TrainArgs(config['batch_size'], 0.001, 0.9999993, config['max_itrs'],
                                      False, 1, 0, 0.02, -1)
    train(updater, config['nnet_dir'], train_args, debug=True)
