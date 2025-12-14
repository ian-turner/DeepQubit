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


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str)
    parser.add_argument('--nnet_dir', type=str, required=True)
    parser.add_argument('--num_qubits', type=int, default=1)
    parser.add_argument('--encoding', type=str, default='matrix',
                        choices=['matrix', 'hurwitz', 'quaternion', 'discrete'],
                        help='Encoding method of unitary matrix before passing to NNet')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='Tolerance value for solved condition')
    parser.add_argument('--gateset', type=str, default='t,s,h,x,y,z,cx')
    parser.add_argument('--step_max', type=int, default=100)
    parser.add_argument('--up_itrs', type=int, default=100)
    parser.add_argument('--up_gen_itrs', type=int, default=100)
    parser.add_argument('--up_procs', type=int, default=1)
    parser.add_argument('--up_search_itrs', type=int, default=100)
    parser.add_argument('--up_batch_size', type=int, default=100)
    parser.add_argument('--up_nnet_batch_size', type=int, default=1000)
    parser.add_argument('--up_v', action='store_true')
    parser.add_argument('--sync_main', action='store_true')
    parser.add_argument('--ub_heur_solns', action='store_true')
    parser.add_argument('--backup', type=int, default=1)
    parser.add_argument('--perturb', action='store_true')
    parser.add_argument('--up_bwas_weight', type=float, default=0.2)
    parser.add_argument('--up_bwas_eps', type=float, default=0.1)
    parser.add_argument('--train_batch_size', type=int, default=1000)
    parser.add_argument('--train_lr', type=float, default=0.001)
    parser.add_argument('--train_lr_d', type=float, default=0.9999993)
    parser.add_argument('--train_max_itrs', type=int, default=1e6)
    parser.add_argument('--train_balance_steps', action='store_true')
    parser.add_argument('--train_rb', type=int, default=1)
    parser.add_argument('--train_display', type=int, default=-1)
    parser.add_argument('-L', '--nerf_dim', type=int, default=0)
    config = vars(parser.parse_args())

    # overriding default config options from config file
    if 'config_file' in config and config['config_file'] != None:
        with open(config['config_file'], 'r') as f:
            config_yaml = yaml.safe_load(f.read())
            for x in config_yaml:
                config[x] = config_yaml[x]
        del config['config_file']

    # overriding default config options with command line arguments
    for x in config:
        y = config[x]
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
    up_args = UpArgs(step_max=config['step_max'],
                     up_itrs=config['up_itrs'],
                     up_gen_itrs=config['up_gen_itrs'],
                     up_procs=config['up_procs'],
                     up_search_itrs=config['up_search_itrs'],
                     up_batch_size=config['up_batch_size'],
                     up_nnet_batch_size=config['up_nnet_batch_size'],
                     sync_main=config['sync_main'],
                     up_v=config['up_v'])
    up_heur_args = UpHeurArgs(up_args=up_args,
                              ub_heur_solns=config['ub_heur_solns'],
                              backup=config['backup'])
    up_bwas_args = UpBWASArgs(up_heur_args=up_heur_args,
                              weight=config['up_bwas_weight'],
                              eps=config['up_bwas_eps'])
    updater = UpdateHeurBWASEnum(env=env,
                                 up_bwas_args=up_bwas_args,
                                 heur_nnet=QCircuitNNetParV(n=config['num_qubits'],
                                                            L=config['nerf_dim'],
                                                            encoding=config['encoding']))

    # running training
    train_args: TrainArgs = TrainArgs(batch_size=config['train_batch_size'],
                                      lr=config['train_lr'],
                                      lr_d=config['train_lr_d'],
                                      max_itrs=config['train_max_itrs'],
                                      balance_steps=config['train_balance_steps'],
                                      rb=config['train_rb'],
                                      targ_up_searches=0,
                                      loss_thresh=0.02,
                                      display=config['train_display'])
    train(updater, config['nnet_dir'], train_args, debug=True)
