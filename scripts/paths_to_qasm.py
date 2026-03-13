import os
import pickle
from argparse import ArgumentParser

from domains.qcircuit import *
from utils.matrix_utils import *


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    # loading results.pkl file
    data = pickle.load(open(args.input, 'rb'))

    # writing paths to files
    try:
        os.mkdir(args.output)
    except FileExistsError as e:
        pass
    for i, is_solved in enumerate(data['solved']):
        if is_solved:
            filename = os.path.join(args.output, '%d.qasm' % i)
            with open(filename, 'w') as fp:
                # OPENQASM header
                print('OPENQASM 3.0;', file=fp)
                print('include "stdgates.inc";', file=fp)
                
                # qubit initialization
                n = data['actions'][i][0].num_qubits
                print('qubit[%d] qs;' % n, file=fp)

                # gate actions
                for act in data['actions'][i]:
                    print(act.__repr__() + ';', file=fp)
