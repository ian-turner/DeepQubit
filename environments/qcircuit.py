import numpy as np
from abc import ABC
from typing import Self, Tuple, List, Dict
from deepxube.environments.environment_abstract import Environment, State, Action, Goal, HeurFnNNet
from utils.pytorch_models import ResnetModel
from utils.matrix_utils import *


class QState(State):
    # tolerance for comparing unitaries between states
    epsilon: float = 1e-6

    def __init__(self, unitary: np.ndarray[np.complex128]):
        self.unitary = unitary
        self.prev_actions = [None]
    
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
    unitary: np.ndarray[np.complex128]
    full_gate_unitary: np.ndarray[np.complex128]
    cost: float
    
    def apply_to(self, state: QState) -> QState:
        new_state_unitary = np.matmul(self.full_gate_unitary, state.unitary).astype(np.complex128)
        return QState(new_state_unitary)


class OneQubitGate(QAction, ABC):
    qubit: int
    unitary: np.ndarray[np.complex128]
    full_gate_unitary: np.ndarray[np.complex128]
    
    def __init__(self, num_qubits: int, qubit: int):
        self.qubit = qubit
        self._generate_full_unitary(num_qubits)
    
    def _generate_full_unitary(self, num_qubits: int):
        mats = [I] * num_qubits
        mats[self.qubit] = self.unitary
        self.full_gate_unitary = tensor_product(mats)

    def __repr__(self) -> str:
        return '%s(qubit=%d)' % (type(self).__name__, self.qubit)
    

class ControlledGate(QAction, ABC):
    control: int
    target: int
    unitary: np.ndarray[np.complex128]
    full_gate_unitary: np.ndarray[np.complex128]

    def __init__(self, num_qubits: int, control: int, target: int):
        self.control = control
        self.target = target
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
        return '%s(control=%d, target=%d)' % (type(self).__name__, self.target, self.control)


class HGate(OneQubitGate):
    unitary = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    cost = 0.01
    asm_name = 'h'

class SGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    cost = 0.01
    asm_name = 's'

class SdgGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    cost = 0.01
    asm_name = 'sdg'
    
class TGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=np.complex128)
    cost = 1.0
    asm_name = 't'

class TdgGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, np.exp(-1j*np.pi/4)]], dtype=np.complex128)
    cost = 1.0
    asm_name = 'tdg'

class CNOTGate(ControlledGate):
    unitary = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    cost = 0.1
    asm_name = 'cx'


class QCircuit(Environment):
    gate_set = [HGate, SGate, SdgGate, TGate, TdgGate, CNOTGate]

    def __init__(self, num_qubits: int, epsilon: float = 0.01):
        super(QCircuit, self).__init__(env_name='qcircuit')
        
        self.num_qubits: int = num_qubits
        self.epsilon: float = epsilon
        self._generate_actions()

    def _generate_actions(self):
        """
        Generates the action set for n qubits given a specific gate set
        by looping over each possible gate at each qubit
        """
        # self.actions: List[QAction] = []
        # for gate in self.gate_set:
        #     # looping over each gate in the gate set
        #     for i in range(self.num_qubits):
        #         # looping over each qubit
        #         if issubclass(gate, OneQubitGate):
        #             # if the gate only acts on one qubit,
        #             # add gate to all qubits once
        #             self.actions.append(gate(self.num_qubits, i))
        #         elif issubclass(gate, ControlledGate):
        #             # if the gate is a controlled gate,
        #             # loop over each possible pair of qubits
        #             for j in range(self.num_qubits):
        #                 if i != j: self.actions.append(gate(self.num_qubits, i, j))

        # temporary fix for one qubit environment ONLY
        self.actions: Dict[str, QAction] = dict()
        self.actions['h'] = HGate(1, 0)
        self.actions['s'] = SGate(1, 0)
        self.actions['sdg'] = SdgGate(1, 0)
        self.actions['t'] = TGate(1, 0)
        self.actions['tdg'] = TdgGate(1, 0)

    def get_start_states(self, num_states: int) -> List[QState]:
        """
        Generates a set of states with the identity as their unitary

        @param num_states: Number of states to generate
        @returns: Generated states
        """
        return [QState(tensor_product([I] * self.num_qubits)) for _ in range(num_states)]

    def get_state_actions(self, states: List[QState]) -> List[List[QAction]]:
        # return [[x for x in self.actions] for _ in states]

        # temporary fix for one qubit environment ONLY
        actions = []
        for x in states:
            prev_action = x.prev_actions[-1]
            match prev_action:
                case TGate() | TdgGate():
                    actions.append([
                        self.actions['h'],
                        self.actions['s'],
                        self.actions['sdg'],
                    ])
                case HGate():
                    actions.append([
                        self.actions['s'],
                        self.actions['sdg'],
                        self.actions['t'],
                        self.actions['tdg'],
                    ])
                case SGate() | SdgGate():
                    actions.append([
                        self.actions['h'],
                        self.actions['t'],
                        self.actions['tdg'],
                    ])
                case _:
                    actions.append([
                        self.actions['h'],
                        self.actions['s'],
                        self.actions['sdg'],
                        self.actions['t'],
                        self.actions['tdg'],
                    ])
        return actions

    def next_state(self, states: List[QState], actions: List[QAction]) -> Tuple[List[QState], List[float]]:
        next_states = []
        for state, action in zip(states, actions):
            next_state = action.apply_to(state)
            next_state.prev_actions.append(action)
            next_states.append(next_state)

        transition_costs = [x.cost for x in actions]
        return next_states, transition_costs

    def sample_goal(self, states_start: List[QState], states_goal: List[QState]) -> List[QGoal]:
        """
        Creates goal objects from state-goal pairs

        TODO: add 'noise' to goal unitaries to make model
        more robust to dealing with approximation
        """
        return [QGoal(x.unitary) for x in states_goal]
    
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
        total_unitaries = [np.matmul(y.unitary, invert_unitary(x.unitary)) for (x, y) in zip(states, goals)]
        return [np.vstack([unitary_to_nnet_input(x) for x in total_unitaries]).astype(float)]

    def get_v_nnet(self) -> HeurFnNNet:
        input_size: int = 2**(2*self.num_qubits + 1)
        nnet = ResnetModel(input_size, 0, 5000, 1000, 4, 1, True)
        return nnet

    # ------------------- NOT IMPLEMENTED -------------------

    def get_q_nnet(self) -> HeurFnNNet:
        raise NotImplementedError()
    
    def get_pddl_domain(self):
        raise NotImplementedError()
    
    def state_goal_to_pddl_inst(self, state, goal):
        raise NotImplementedError()
    
    def pddl_action_to_action(self, pddl_action):
        raise NotImplementedError()
    
    def visualize(self, states, goals):
        raise NotImplementedError()
