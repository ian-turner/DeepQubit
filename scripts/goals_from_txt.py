"""Builds a goals .pkl from .txt target unitaries (files and/or directories of *.txt files).

The pkl holds {'states', 'goals', 'names', 'skipped'}: 'names' are the .txt file stems, aligned with 'goals';
'skipped' lists (name, reason) for targets left out. deepxube only reads 'states'/'goals'.

Usage: python scripts/goals_from_txt.py --input data/targets/3qubit --output goals.pkl
                                        [--domain qcircuit_exact] [--reachable_only]
"""
import os
import glob
import pickle
from argparse import ArgumentParser
from typing import List

from domains.qcircuit import *
from utils.matrix_utils import *
from check_goals import det_omega_power, is_reachable


def expand_inputs(inputs: List[str]) -> List[str]:
    """directories expand to their *.txt files (sorted by name); files are kept in the given order"""
    files: List[str] = []
    for x in inputs:
        if os.path.isdir(x):
            files.extend(sorted(glob.glob(os.path.join(x, '*.txt'))))
        else:
            files.append(x)
    return files


if __name__ == '__main__':
    # parsing command line arguments
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, nargs='+', required=True, help='.txt files and/or directories of them')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--domain', type=str, default='qcircuit', choices=['qcircuit', 'qcircuit_exact'],
                        help='qcircuit_exact makes ring goals, skipping targets that are not in Z[omega, 1/sqrt2]')
    parser.add_argument('--reachable_only', action='store_true',
                        help='skip targets whose determinant rules out exact Clifford+T synthesis (see check_goals.py)')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='qcircuit_exact: entry tolerance when fitting targets onto the ring (.txt files may carry ~8 digits)')
    args = parser.parse_args()

    files = expand_inputs(args.input)
    if len(files) == 0:
        raise SystemExit(f"no .txt targets found in {args.input}")

    # loading matrices from .txt files
    names, Us = [], []
    num_qubits = set()
    for x in files:
        n, U = load_matrix_from_file(x)
        names.append(os.path.splitext(os.path.basename(x))[0])
        Us.append(U)
        num_qubits.add(n)
    if len(num_qubits) != 1:
        raise SystemExit(f"targets have different qubit counts {sorted(num_qubits)}: {files}")
    n = num_qubits.pop()

    # creating state/goal pairs from matrices
    In = tensor_product([I] * n)
    if args.domain == 'qcircuit_exact':
        from domains.qcircuit_exact import QStateExact, QGoalExact
        def make_state(U): return QStateExact.from_complex(U, args.tol)
        def make_goal(U): return QGoalExact.from_complex(U, args.tol)
    else:
        make_state, make_goal = QState, QGoal

    data = {'states': [], 'goals': [], 'names': [], 'skipped': []}
    for name, U in zip(names, Us):
        if args.reachable_only and not is_reachable(U):
            data['skipped'].append((name, f"not exactly reachable (det = omega^{det_omega_power(U)})"))
            continue
        try:
            state, goal = make_state(In), make_goal(U)
        except ValueError as e:
            data['skipped'].append((name, f"not in the Clifford+T ring ({e})"))
            continue
        data['states'].append(state)
        data['goals'].append(goal)
        data['names'].append(name)

    for name, reason in data['skipped']:
        print(f"skipped {name}: {reason}")

    # saving using pickle
    pickle.dump(data, open(args.output, 'wb'))
    print(f"wrote {len(data['goals'])}/{len(names)} goals ({n} qubits) to {args.output}: {' '.join(data['names'])}")
