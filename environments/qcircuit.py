import numpy as np
from abc import ABC
from typing import Self, Tuple, List, Dict
from deepxube.base.env import (EnvGrndAtoms, State, Action, Goal, EnvSupportsPDDL, EnvStartGoalRW, EnvEnumerableActs,
                               EnvVizable)
from deepxube.base.heuristic import HeurNNetModule, HeurNNetV, HeurNNetQ, HeurNNetQFixOut, HeurNNetQIn

from utils.pytorch_models import ResnetModel
from utils.matrix_utils import *
from utils.perturb import perturb_unitary_random_batch_strict
from environments.gates import get_gate_set


class QState(State):
    # tolerance for comparing unitaries between states
    epsilon: float = 1e-6

    def __init__(self, unitary: np.ndarray[np.complex128]):
        self.unitary = unitary
    
    def __hash__(self):
        return hash_unitary(self.unitary)

    def __eq__(self, other: Self):
        return unitary_distance(self.unitary, other.unitary) <= self.epsilon


class QGoal(Goal):
    # tolerance for comparing unitaries between goals
    epsilon: float = 1e-6

    def __init__(self, unitary: np.ndarray[np.complex128]):
        self.unitary = unitary
    
    def __hash__(self):
        return hash_unitary(self.unitary)

    def __eq__(self, other: Self):
        return unitary_distance(self.unitary, other.unitary) <= self.epsilon
    

class QAction(Action, ABC):
    epsilon: float = 1e-6

    def apply_to(self, state: QState) -> QState:
        new_state_unitary = np.matmul(self.full_gate_unitary, state.unitary).astype(np.complex128)
        return QState(new_state_unitary)

    def __eq__(self, other):
        return unitary_distance(self.full_gate_unitary, other.full_gate_unitary) <= self.epsilon

    def __hash__(self):
        return hash_unitary(self.full_gate_unitary)


class OneQubitGate(QAction, ABC):
    def __init__(self,
                 num_qubits: int,
                 qubit: int,
                 unitary: np.ndarray,
                 name: str,
                 cost: float):
        self.num_qubits = num_qubits
        self.qubit = qubit
        self.unitary = unitary
        self.name = name
        self.cost = cost
        self._generate_full_unitary(num_qubits)
    
    def _generate_full_unitary(self, num_qubits: int):
        mats = [I] * num_qubits
        mats[self.qubit] = self.unitary
        self.full_gate_unitary = tensor_product(mats)

    def __repr__(self) -> str:
        return '%s(qubit=%d)' % (self.name, self.qubit)
    

class ControlledGate(QAction, ABC):
    def __init__(self,
                 num_qubits: int,
                 control: int,
                 target: int,
                 unitary: np.ndarray,
                 name: str,
                 cost: float):
        self.unitary = unitary
        self.name = name
        self.control = control
        self.target = target
        self.cost = cost
        self._generate_full_unitary(num_qubits)

    def _generate_full_unitary(self, num_qubits: int):
        p0_mats = [I] * num_qubits
        p1_mats = [I] * num_qubits

        p0_mats[self.control] = P0
        p1_mats[self.control] = P1
        p1_mats[self.target] = self.unitary
        
        p0_full = tensor_product(p0_mats)
        p1_full = tensor_product(p1_mats)
        self.full_gate_unitary = p0_full + p1_full

    def __repr__(self) -> str:
        return '%s(control=%d, target=%d)' % (self.name, self.target, self.control)


class QCircuitNNetParV(HeurNNetV[QState, QGoal]):
    def __init__(self, n: int, L: int = 0, encoding: str = 'matrix'):
        self.n = n
        self.L = L
        self.encoding = encoding

        match (encoding):
            case 'hurwitz':
                self.N = 2**(2 * n) - 1
            case 'matrix':
                self.N = 2**(2 * n + 1)
            case 'quaternion':
                self.N = 4
            case _:
                raise Exception('Invalid encoding `%s`' % self.encoding)

    def get_nnet(self) -> HeurNNetModule:
        return ResnetModel(self.N, self.L, [2000, 2000, 4000][self.n], 1000, 4, 1, True)

    def to_np(self, states: List[QState], goals: List[QGoal]) -> List[np.ndarray]:
        # calculating overall transformation from start to goal unitary
        total_unitaries = np.array([y.unitary @ invert_unitary(x.unitary) for (x, y) in zip(states, goals)])

        # converting to nnet input based on encoding
        return [unitaries_to_nnet_input(total_unitaries, encoding=self.encoding)]


class QCircuit(EnvStartGoalRW[QState, QAction, QGoal],
               EnvEnumerableActs[QState, QAction, QGoal]):
    def __init__(self,
                 num_qubits: int,
                 epsilon: float = 0.01,
                 perturb: bool = False,
                 encoding: str = 'matrix',
                 gateset: str = 't,s,h,x,y,z',
                 L: int = 15):
        super(QCircuit, self).__init__()
        
        self.L = L
        self.perturb = perturb
        self.num_qubits = num_qubits
        self.epsilon = epsilon
        self.encoding = encoding

        self._generate_actions(gateset)

    def _generate_actions(self, gateset: str):
        """
        Generates the action set for n qubits given a specific gate set
        by looping over each possible gate at each qubit
        """
        gates = get_gate_set(gateset)
        self.actions: List[QAction] = []
        for gate in gates:
            # looping over each gate in the gate set
            for i in range(self.num_qubits):
                # looping over each qubit
                if 'controlled' in gate and gate['controlled']:
                    # if the gate is a controlled gate,
                    # loop over each possible pair of qubits
                    for j in range(self.num_qubits):
                        if i != j:
                            new_gate = ControlledGate(self.num_qubits, i, j, name=gate['name'],
					  	      cost=gate['cost'], unitary=gate['unitary'])
                            self.actions.append(new_gate)
                else:
                    # if the gate only acts on one qubit,
                    # add gate to all qubits once
                    self.actions.append(OneQubitGate(self.num_qubits, i, **gate))

    def get_start_states(self, num_states: int) -> List[QState]:
        """
        Generates a set of states with the identity as their unitary

        @param num_states: Number of states to generate
        @returns: Generated states
        """
        return [QState(tensor_product([I] * self.num_qubits)) for _ in range(num_states)]

    def get_state_actions(self, states: List[QState]) -> List[List[QAction]]:
        return [[x for x in self.actions] for _ in states]

    def next_state(self, states: List[QState], actions: List[QAction]) -> Tuple[List[QState], List[float]]:
        next_states = []
        for state, action in zip(states, actions):
            next_state = action.apply_to(state)
            next_states.append(next_state)

        transition_costs = [x.cost for x in actions]
        return next_states, transition_costs

    def sample_goal(self, states_start: List[QState], states_goal: List[QState]) -> List[QGoal]:
        """
        Creates goal objects from state-goal pairs
        """
        U_b = np.array([y.unitary @ invert_unitary(x.unitary) for (x, y) in zip(states_start, states_goal)])
        if self.perturb:
            U_pt = perturb_unitary_random_batch_strict(U_b, (1/np.sqrt(2)) * self.epsilon)
            return [QGoal(x) for x in U_pt]
        else: return [QGoal(x) for x in U_b]
    
    def is_solved(self, states: List[QState], goals: List[QGoal]) -> List[bool]:
        """
        Checks whether each state is solved by comparing their unitaries (within a tolerance)

        @param states: List of quantum circuit states
        @param goals: List of goals to check against
        @returns: List of bools representing solved/not-solved
        """
        return [unitary_distance(state.unitary, goal.unitary) <= self.epsilon \
                for (state, goal) in zip(states, goals)]

    def states_goals_to_nnet_input(self, states: List[QState], goals: List[QGoal]) -> List[np.ndarray[float]]:
        """
        Converts quantum state class objects to numpy arrays that can be
        converted to tensors for neural network training

        Also inverts the state matrix and multiplies it to the goal matrix,
        just passing the resulting unitary to the network, since all that
        matters is the 'distance' between the two unitaries

        @param states: List of quantum circuit states
        @param goals: List of quantum circuit goals
        @returns: List of numpy arrays of flattened state and unitaries (in float format)
        """
        # calculating overall transformation from start to goal unitary
        total_unitaries = np.array([y.unitary @ invert_unitary(x.unitary) for (x, y) in zip(states, goals)])

        # converting to nnet input based on encoding
        return [unitaries_to_nnet_input(total_unitaries, encoding=self.encoding)]
        
    def get_v_nnet(self) -> HeurNNetV:
        # calculating nnet input size based on encoding
        match (self.encoding):
            case 'hurwitz':
                N = 2**(2 * self.num_qubits) - 1
            case 'matrix':
                N = 2**(2 * self.num_qubits + 1)
            case 'quaternion':
                N = 4
        return ResnetModel(N, self.L, [2000, 2000, 4000][self.num_qubits-1], 1000, 4, 1, True)

    # ------------------- NOT IMPLEMENTED -------------------

    def get_q_nnet(self) -> HeurNNetQ:
        raise NotImplementedError()
    
    def get_pddl_domain(self):
        raise NotImplementedError()
    
    def state_goal_to_pddl_inst(self, state, goal):
        raise NotImplementedError()
    
    def pddl_action_to_action(self, pddl_action):
        raise NotImplementedError()
    
    def visualize(self, states, goals):
        raise NotImplementedError()
