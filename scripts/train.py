import torch
from argparse import ArgumentParser
from deepxube.training import avi
from environments.qcircuit import QCircuit


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--nnet_dir', type=str, required=True)
    parser.add_argument('--num_qubits', type=int, required=True)
    parser.add_argument('--encoding', type=str, default='matrix')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--step_max', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--itrs_per_update', type=int, default=1000)
    parser.add_argument('--max_itrs', type=int, default=1e5)
    parser.add_argument('--greedy_update_step_max', type=int, default=5)
    parser.add_argument('--num_update_procs', type=int, default=5)
    parser.add_argument('--perturb', action='store_true')
    parser.add_argument('-L', '--nerf_dim', type=int, default=0)
    args = parser.parse_args()
    
    # environment setup
    env = QCircuit(
        num_qubits=args.num_qubits,
        epsilon=args.epsilon,
        L=args.nerf_dim,
        perturb=args.perturb,
        encoding=args.encoding,
    )

    # running approximate value iteration
    avi.train(
        env=env,
        nnet_dir=args.nnet_dir,
        step_max=args.step_max,
        batch_size=args.batch_size,
        itrs_per_update=args.itrs_per_update,
        max_itrs=args.max_itrs,
        greedy_update_step_max=args.greedy_update_step_max,
        num_update_procs=args.num_update_procs,
    )
