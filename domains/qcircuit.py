import numpy as np
from abc import ABC
from typing import Self, Tuple, List, Dict, Any
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, \
                                 StringToAct, DomainParser, StateGoalVizable
from deepxube.base.nnet_input import StateGoalIn, HasFlatSGIn
from deepxube.factories.domain_factory import register_domain, register_domain_parser
from deepxube.factories.nnet_input_factory import register_nnet_input
from utils.matrix_utils import *
from utils.perturb import perturb_unitary_random_batch_strict
from domains.gates import get_gate_set
from numpy.typing import NDArray
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from qiskit import qasm2
from qiskit.visualization import circuit_drawer


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
        new_state_unitary = np.matmul(self.full_gate_unitary, state.unitary).astype(np.complex128)
        new_state = QState(new_state_unitary)
        new_state.path = [*state.path, self]
        return new_state

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


def path_to_qasm(path: List[QAction], num_qubits) -> str:
    qasm_str = ''
    qasm_str += 'OPENQASM 2.0;\n'
    qasm_str +='include "qelib1.inc";\n'
    qasm_str += 'qreg qubits[%d];\n' % num_qubits
    for x in path:
        name = x.name
        if x.name == 't10' or x.name == 't100':
            name = 't'
        qasm_str += '%s ' % name
        if isinstance(x, OneQubitGate):
            qasm_str += 'qubits[%d]' % x.qubit
        elif isinstance(x, ControlledGate):
            qasm_str += 'qubits[%d], qubits[%d]' % (x.control, x.target)
        qasm_str += ';\n'
    return qasm_str


@register_domain('qcircuit')
class QCircuit(ActsEnumFixed[QState, QAction, QGoal],
               StartGoalWalkable[QState, QAction, QGoal],
               StateGoalVizable[QState, QAction, QGoal],
               StringToAct[QState, QAction, QGoal],
               HasFlatSGIn[QState, QAction, QGoal]):
    def __init__(self,
                 num_qubits: int,
                 epsilon: float = 0.01,
                 perturb: bool = False,
                 encoding: str = 'matrix',
                 gateset: str = 't,s,h,x,y,z,cx',
                 L: int = 15):
        super().__init__()
        
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
        k = 0
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
                            new_gate.action = k
                            self.actions.append(new_gate)
                            k += 1
                else:
                    # if the gate only acts on one qubit,
                    # add gate to all qubits once
                    new_gate = OneQubitGate(self.num_qubits, i, **gate)
                    new_gate.action = k
                    self.actions.append(new_gate)
                    k += 1

    def get_start_states(self, num_states: int) -> List[QState]:
        """
        Generates a set of states with the identity as their unitary

        @param num_states: Number of states to generate
        @returns: Generated states
        """
        return [QState(tensor_product([I] * self.num_qubits)) for _ in range(num_states)]

    def _get_actions_fixed(self) -> List[List[QAction]]:
        return [x for x in self.actions]

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

    def visualize_state_goal(self, state: QState, goal: QGoal, fig: Figure) -> None:
        # creating circuit from path
        qasm_str = path_to_qasm(state.path, self.num_qubits)
        qc = qasm2.loads(qasm_str)

        # drawing circuit
        ax = fig.add_axes([0.0, 0.5, 1.0, 0.5])
        ax.set_title('Quantum Circuit', fontsize=20)
        qc.draw('mpl', ax=ax)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(3)
            spine.set_linestyle('--')

        # drawing state matrix
        ax = fig.add_axes([0.0, 0.0, 0.5, 0.5])
        ax.set_title('State Matrix', fontsize=20)
        data = state.unitary
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                real = np.real(data[i][j])
                imag = np.imag(data[i][j])
                numstr = ''
                numstr += '%.2f' % real
                if imag < 0.:
                    numstr += '\n%.2fi' % imag
                else:
                    numstr += '\n+%.2fi' % imag

                ax.text(j, i, '%s' % numstr,
                        horizontalalignment='center',
                        verticalalignment='center')
        ax.imshow(np.abs(state.unitary), alpha=0.3, cmap='viridis', interpolation='nearest')
        plt.axis('off')

        # drawing goal matrix
        ax = fig.add_axes([0.5, 0.0, 0.5, 0.5])
        ax.set_title('Goal Matrix', fontsize=20)
        data = goal.unitary
        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                real = np.real(data[i][j])
                imag = np.imag(data[i][j])
                numstr = ''
                numstr += '%.2f' % real
                if imag < 0.:
                    numstr += '\n%.2fi' % imag
                else:
                    numstr += '\n+%.2fi' % imag

                ax.text(j, i, '%s' % numstr,
                        horizontalalignment='center',
                        verticalalignment='center')
        ax.imshow(np.abs(goal.unitary), alpha=0.3, cmap='viridis', interpolation='nearest')
        plt.axis('off')

    def string_to_action(self, act_str: str) -> QAction:
        return self.actions[int(act_str)]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [2**(2+self.num_qubits)], [self.num_qubits]

    def to_np_flat_sg(self, states: List[QState], goals: List[QGoal]) -> List[np.ndarray[float]]:
        # calculating overall transformation from start to goal unitary
        total_unitaries = np.array([y.unitary @ invert_unitary(x.unitary) for (x, y) in zip(states, goals)])

        # converting to nnet input based on encoding
        return [unitaries_to_nnet_input(total_unitaries, encoding=self.encoding)]


@register_domain_parser('qcircuit')
class QCircuitParser(DomainParser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        return {'num_qubits': int(args_str)}

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
