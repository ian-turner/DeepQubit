import re
import numpy as np
from numpy.typing import NDArray
from abc import ABC
from typing import Self, Tuple, List, Dict, Any
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from qiskit import qasm2
from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StringToAct, StateGoalVizable
from deepxube.base.nnet_input import StateGoalIn, HasFlatSGIn, StateGoalActFixIn, HasFlatSGActsEnumFixedIn
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input

from utils.matrix_utils import *
from utils.perturb import perturb_unitary_random_batch_strict


class QState(State):
    # tolerance for comparing unitaries between states
    epsilon: float = 1e-6

    def __init__(self, unitary: np.ndarray[np.complex128]):
        self.unitary = unitary
        self.path = []
    
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
        new_state_unitary = np.matmul(self._full_unitary, state.unitary).astype(np.complex128)
        new_state = QState(new_state_unitary)
        return new_state


class OneQubitGate(QAction, ABC):
    def __init__(self, num_qubits: int, qubit: int):
        self.num_qubits = num_qubits
        self.qubit = qubit
        self._generate_full_unitary()
    
    def _generate_full_unitary(self):
        mats = [I] * self.num_qubits
        mats[self.qubit] = self.unitary
        self._full_unitary = tensor_product(mats).astype(np.complex128)

    def __repr__(self) -> str:
        return '%s(qubit=%d)' % (type(self).__name__, self.qubit)

    def __eq__(self, other):
        return (unitary_distance(self._full_unitary, other._full_unitary) <= self.epsilon) \
               and (self.qubit == other.qubit)

    def __hash__(self):
        return hash((self.qubit, hash_unitary(self._full_unitary)))
    

class ControlledGate(QAction, ABC):
    def __init__(self, num_qubits: int, control: int, target: int):
        self.control = control
        self.target = target
        self._generate_full_unitary()

    def _generate_full_unitary(self):
        p0_mats = [I] * self.num_qubits
        p1_mats = [I] * self.num_qubits

        p0_mats[self.control] = P0
        p1_mats[self.control] = P1
        p1_mats[self.target] = self.unitary
        
        p0_full = tensor_product(p0_mats)
        p1_full = tensor_product(p1_mats)
        self._full_unitary = (p0_full + p1_full).astype(np.complex128)

    def __repr__(self) -> str:
        return '%s(control=%d, target=%d)' % (type(self).__name__, self.target, self.control)

    def __eq__(self, other):
        return (unitary_distance(self._full_unitary, other._full_unitary) <= self.epsilon) \
               and (self.control == other.control) and (self.target == other.target)

    def __hash__(self):
        return hash((self.control, self.target, hash_unitary(self._full_unitary)))


class HGate(OneQubitGate):
    unitary = (1/np.sqrt(2)) * np.array([[1, 1],
                                         [1, -1]])
    cost = 1.0


class SGate(OneQubitGate):
    unitary = np.array([[1, 0],
                        [0, 1j]])
    cost = 1.0


class ZGate(OneQubitGate):
    unitary = np.array([[1, 0],
                        [0, -1]])
    cost = 1.0


class TGate(OneQubitGate):
    unitary = np.array([[1, 0],
                        [0, np.exp(1j*np.pi/4)]])
    cost = 1.0


class XGate(OneQubitGate):
    unitary = np.array([[0, 1],
                        [1, 0]])
    cost = 1.0


class YGate(OneQubitGate):
    unitary = np.array([[0, -1j],
                        [1j, 0]])
    cost = 1.0


class CNOTGate(ControlledGate):
    unitary = np.array([[1, 0],
                        [0, 1]])
    cost = 1.0


def get_gate_set(gateset: str) -> List[QAction]:
    match gateset:
        case 'CliffT':
            return [HGate, SGate, YGate, TGate, XGate, ZGate, CNOTGate]


@domain_factory.register_class('qcircuit')
class QCircuit(ActsEnumFixed[QState, QAction, QGoal],
               StartGoalWalkable[QState, QAction, QGoal],
               StringToAct[QState, QAction, QGoal],
               HasFlatSGActsEnumFixedIn[QState, QAction, QGoal]):
    def __init__(self,
                 num_qubits: int,
                 epsilon: float = 0.01,
                 perturb: bool = False,
                 encoding: str = 'matrix',
                 gateset: str = 'CliffT',
                 nerf_dim: int = 15):
        super().__init__()
        
        self.nerf_dim = nerf_dim
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
        k = 0
        for gate in gates:
            # looping over each gate in the gate set
            for i in range(self.num_qubits):
                # looping over each qubit
                    if issubclass(gate, ControlledGate):
                        for j in range(self.num_qubits):
                            # if the gate is a controlled gate,
                            # loop over each possible pair of qubits
                            if i != j:
                                _gate = gate(self.num_qubits, i, j)
                                _gate.action = k
                                self.actions.append(_gate)
                                k += 1
                    elif issubclass(gate, OneQubitGate):
                        # if the gate only acts on one qubit,
                        # add gate to all qubits once
                        _gate = gate(self.num_qubits, i)
                        _gate.action = k
                        self.actions.append(_gate)
                        k += 1
    
    def actions_to_indices(self, actions: List[QAction]) -> List[int]:
        return [self.actions.index(x) for x in actions]

    def sample_start_states(self, num_states: int) -> List[QState]:
        """
        Generates a set of states with the identity as their unitary

        @param num_states: Number of states to generate
        @returns: Generated states
        """
        return [QState(tensor_product([I] * self.num_qubits)) for _ in range(num_states)]

    def get_actions_fixed(self) -> List[List[QAction]]:
        return [x for x in self.actions]

    def string_to_action_help(self) -> str:
        return "index of gate action (actions: %s)" % (str(self.actions))

    def next_state(self, states: List[QState], actions: List[QAction]) -> Tuple[List[QState], List[float]]:
        next_states = []
        for state, action in zip(states, actions):
            next_state = action.apply_to(state)
            next_states.append(next_state)

        transition_costs = [x.cost for x in actions]
        return next_states, transition_costs

    def sample_goal_from_state(self, states_start: List[QState], states_goal: List[QState]) -> List[QGoal]:
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

    def string_to_action(self, act_str: str) -> QAction:
        return self.actions[int(act_str)]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        N: int
        match self.encoding:
            case 'matrix':
                N = 2 ** (2 + self.num_qubits)

        return [2 * self.nerf_dim * N if self.nerf_dim > 0 else N], [self.num_qubits]

    def to_np_flat_sg(self, states: List[QState], goals: List[QGoal]) -> List[np.ndarray[float]]:
        # calculating overall transformation from start to goal unitary
        total_unitaries = np.array([y.unitary @ invert_unitary(x.unitary) for (x, y) in zip(states, goals)])

        # converting to nnet input based on encoding
        return [unitaries_to_nnet_input(total_unitaries, encoding=self.encoding, nerf_dim=self.nerf_dim)]


@domain_factory.register_parser('qcircuit')
class QCircuitParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args = args_str.split('_')
        args_dict = {}
        for arg in args:
            num_qubits = re.search(r'n(\d+)', arg)
            epsilon = re.search(r'e(\d+)\.(\d+)', arg)
            nerf_dim = re.search(r'L(\d+).*', arg)
            encoding = re.search('[HQ]', arg)
            if num_qubits is not None:
                args_dict['num_qubits'] = int(num_qubits.group(1))
            if epsilon is not None:
                args_dict['epsilon'] = float(epsilon.group()[1:])
            if nerf_dim is not None:
                args_dict['nerf_dim'] = int(nerf_dim.group(1))
            if encoding is not None:
                match encoding.group():
                    case 'M':
                        args_dict['encoding'] = 'matrix'
                    case 'H':
                        args_dict['encoding'] = 'hurwitz'
                    case 'Q':
                        args_dict['encoding'] = 'quaternion'
            if arg == 'P':
                args_dict['perturb'] = True
        return args_dict

    def help(self) -> str:
        return 'An integer for the number of qubits. E.g. \'qcircuit.3\''


@register_nnet_input('qcircuit', 'qcircuit_nnet_input')
class QCircutNNetInput(StateGoalIn[QCircuit, QState, QGoal]):
    def get_input_info(self) -> int:
        return self.domain.num_qubits

    def to_np(self, states: List[QState], goals: List[QGoal]) -> List[NDArray]:
        # calculating overall transformation from start to goal unitary
        total_unitaries = np.array([y.unitary @ invert_unitary(x.unitary) for (x, y) in zip(states, goals)])

        # converting to nnet input based on encoding
        return [unitaries_to_nnet_input(total_unitaries, encoding=self.encoding)]


@register_nnet_input('qcircuit', 'qcircuit_nnet_input_fix_act')
class QCircutNNetInput(StateGoalActFixIn[QCircuit, QState, QGoal, QAction]):
    def get_input_info(self) -> int:
        return self.domain.num_qubits

    def to_np(self, states: List[QState], goals: List[QGoal], actions_l: List[List[QAction]]) -> List[NDArray]:
        # calculating overall transformation from start to goal unitary
        total_unitaries = np.array([y.unitary @ invert_unitary(x.unitary) for (x, y) in zip(states, goals)])

        # converting to nnet input based on encoding
        return [unitaries_to_nnet_input(total_unitaries, encoding=self.encoding)]
