"""Converts a float goals .pkl (QState/QGoal with complex unitaries) into exact ring goals for qcircuit_exact.

Goals that are not in Z[omega, 1/sqrt2] (i.e. not exactly synthesizable) are reported and skipped.

Usage: python scripts/goals_to_exact.py --input tmp/n3_goals.pkl --output tmp/n3_goals_exact.pkl
"""
import pickle
from argparse import ArgumentParser

from domains.qcircuit_exact import QStateExact, QGoalExact


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--tol', type=float, default=1e-9)
    args = parser.parse_args()

    data = pickle.load(open(args.input, 'rb'))
    states, goals = [], []
    for i, (state, goal) in enumerate(zip(data['states'], data['goals'])):
        try:
            state_ex = QStateExact.from_complex(state.unitary, args.tol)
            goal_ex = QGoalExact.from_complex(goal.unitary, args.tol)
        except ValueError as e:
            print(f"{i}: skipped ({e})")
            continue
        states.append(state_ex)
        goals.append(goal_ex)
        print(f"{i}: ok (goal exponent k={goal_ex.k})")
    pickle.dump({'states': states, 'goals': goals}, open(args.output, 'wb'))
    print(f"wrote {len(goals)}/{len(data['goals'])} goals to {args.output}")
