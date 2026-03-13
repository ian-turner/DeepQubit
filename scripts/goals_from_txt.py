import pickle
from argparse import ArgumentParser

from domains.qcircuit import *
from utils.matrix_utils import *


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, nargs='+', required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    # loading matrices from .txt files
    Us = []
    n: int
    for x in args.input:
        n, U = load_matrix_from_file(x)
        Us.append(U)

    # creating state/goal pairs from matrices
    In = tensor_product([I] * n)
    states = [QState(In) for _ in range(len(Us))]
    goals = [QGoal(x) for x in Us]
    data = {'states': states, 'goals': goals}

    # saving using pickle
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
