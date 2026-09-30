import re
import time
import itertools
import numpy as np
from numpy.typing import NDArray
from abc import ABC
from typing import Self, Tuple, List, Dict, Any, Optional, Type
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from qiskit import qasm2
from qiskit.quantum_info import random_unitary
from deepxube.base.factory import Parser
from deepxube.base.domain import State, Action, Goal, ActsEnumFixed, StartGoalWalkable, StringToAct, StateGoalVizable
from deepxube.base.nnet_input import StateGoalIn, HasFlatSGIn, StateGoalActFixIn, HasFlatSGActsEnumFixedIn
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input
from deepxube.utils.timing_utils import Times

from utils.matrix_utils import *
from utils.perturb import perturb_unitary_givens_batch


def _cached_hash(obj: Any) -> int:
    """The hash of a QState/QGoal, computed on first use unless the domain passed a batch-computed one
    (getattr: objects unpickled from before the cache existed have no `_hash`)"""
    if getattr(obj, '_hash', None) is None:
        obj._hash = hash_unitary(obj.unitary)
    return obj._hash


class QState(State):
    # tolerance for comparing unitaries between states
    epsilon: float = 1e-6

    def __init__(self, unitary: np.ndarray[np.complex128], hash_val: Optional[int] = None):
        self.unitary = unitary
        self._hash = hash_val

    def __hash__(self):
        return _cached_hash(self)

    def __eq__(self, other: Self):
        return unitary_distance(self.unitary, other.unitary) <= self.epsilon


class QGoal(Goal):
    # tolerance for comparing unitaries between goals
    epsilon: float = 1e-6

    def __init__(self, unitary: np.ndarray[np.complex128], hash_val: Optional[int] = None):
        self.unitary = unitary
        self._hash = hash_val

    def __hash__(self):
        return _cached_hash(self)

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
        return '%s qs[%d]' % (self.name, self.qubit)

    def __eq__(self, other):
        return (unitary_distance(self._full_unitary, other._full_unitary) <= self.epsilon) \
               and (self.qubit == other.qubit)

    def __hash__(self):
        return hash((self.qubit, hash_unitary(self._full_unitary)))
    

class ControlledGate(QAction, ABC):
    def __init__(self, num_qubits: int, control: int, target: int):
        self.control = control
        self.target = target
        self.num_qubits = num_qubits
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
        return '%s qs[%d], qs[%d]' % (self.name, self.control, self.target)

    def __eq__(self, other):
        return (unitary_distance(self._full_unitary, other._full_unitary) <= self.epsilon) \
               and (self.control == other.control) and (self.target == other.target)

    def __hash__(self):
        return hash((self.control, self.target, hash_unitary(self._full_unitary)))


class HGate(OneQubitGate):
    unitary = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    cost = 1.0
    name = 'h'

class SGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, 1j]])
    cost = 1.0
    name = 's'

class SdgGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, -1j]])
    cost = 1.0
    name = 'sdg'

class ZGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, -1]])
    cost = 1.0
    name = 'z'

class TGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])
    cost = 1.0
    name = 't'

class TdgGate(OneQubitGate):
    unitary = np.array([[1, 0], [0, np.exp(-1j*np.pi/4)]])
    cost = 1.0
    name = 'tdg'

class XGate(OneQubitGate):
    unitary = np.array([[0, 1], [1, 0]])
    cost = 1.0
    name = 'x'

class YGate(OneQubitGate):
    unitary = np.array([[0, -1j], [1j, 0]])
    cost = 1.0
    name = 'y'

class CNOTGate(ControlledGate):
    unitary = np.array([[0, 1], [1, 0]])
    cost = 1.0
    name = 'cx'

class CZGate(ControlledGate):
    unitary = np.array([[1, 0], [0, -1]])
    cost = 1.0
    name = 'cz'

class CHGate(ControlledGate):
    unitary = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    cost = 1.0
    name = 'ch'


def get_gate_set(gateset: str) -> List[QAction]:
    match gateset:
        case 'CliffT':
            return [HGate, SGate, YGate, TGate, XGate, ZGate, CNOTGate]
        case 'CliffT_S':
            return [HGate, SGate, SdgGate, TGate, TdgGate, CNOTGate]
        case 'CliffT_inv':
            return [HGate, SGate, SdgGate, TGate, TdgGate, CNOTGate]


# ---------------------------------------------------------------------------
# Macro gates: known exact Clifford+T words for structured multi-qubit gates.
# Each macro maps distinct qubit indices to a circuit-order list of
# (gate_class, qubits) pairs (first pair is applied first). They are used to
# generate structured training goals (domain flag `G`) that random walks from
# the identity essentially never produce (Toffoli-like permutations, CCZ-like
# sign diagonals, ...). Every word is composed of gate-set gates, so every
# prefix is exactly reachable by construction.
# ---------------------------------------------------------------------------
MacroWord = List[Tuple[Type[QAction], Tuple[int, ...]]]


def _toffoli(a: int, b: int, t: int) -> MacroWord:
    """15-gate Toffoli (controls a, b; target t), T-count 7 (Nielsen & Chuang Fig. 4.9)"""
    return [(HGate, (t,)), (CNOTGate, (b, t)), (TdgGate, (t,)), (CNOTGate, (a, t)), (TGate, (t,)),
            (CNOTGate, (b, t)), (TdgGate, (t,)), (CNOTGate, (a, t)), (TGate, (b,)), (TGate, (t,)),
            (HGate, (t,)), (CNOTGate, (a, b)), (TGate, (a,)), (TdgGate, (b,)), (CNOTGate, (a, b))]


def _ccz(a: int, b: int, c: int) -> MacroWord:
    """13-gate CCZ: the Toffoli word without the two H gates on the target"""
    return [g for g in _toffoli(a, b, c) if g != (HGate, (c,))]


def _fredkin(c: int, a: int, b: int) -> MacroWord:
    """17-gate controlled-SWAP (control c, swaps a and b)"""
    return [(CNOTGate, (b, a))] + _toffoli(c, a, b) + [(CNOTGate, (b, a))]


def _swap(a: int, b: int) -> MacroWord:
    return [(CNOTGate, (a, b)), (CNOTGate, (b, a)), (CNOTGate, (a, b))]


def _cz(a: int, b: int) -> MacroWord:
    return [(HGate, (b,)), (CNOTGate, (a, b)), (HGate, (b,))]


def _cs(a: int, b: int) -> MacroWord:
    """controlled-S = (T x T) CNOT (I x Tdg) CNOT"""
    return [(TGate, (a,)), (TGate, (b,)), (CNOTGate, (a, b)), (TdgGate, (b,)), (CNOTGate, (a, b))]


def _ch(a: int, b: int) -> MacroWord:
    """controlled-H (qelib1.inc definition)"""
    return [(HGate, (b,)), (SdgGate, (b,)), (CNOTGate, (a, b)), (HGate, (b,)), (TGate, (b,)), (CNOTGate, (a, b)),
            (TGate, (b,)), (HGate, (b,)), (SGate, (b,)), (XGate, (b,)), (SGate, (a,))]


# name -> (word builder, number of distinct qubits it takes)
MACROS: Dict[str, Tuple[Any, int]] = {
    'toffoli': (_toffoli, 3), 'ccz': (_ccz, 3), 'fredkin': (_fredkin, 3),
    'swap': (_swap, 2), 'cz': (_cz, 2), 'cs': (_cs, 2), 'ch': (_ch, 2),
}

# exact replacements (in preference order) for gates missing from a gate set
_GATE_SUBS: Dict[Type[QAction], List[List[Type[QAction]]]] = {
    TdgGate: [[ZGate, SGate, TGate], [SGate, SGate, SGate, TGate]],   # T^7
    SdgGate: [[ZGate, SGate], [SGate, SGate, SGate]],                 # S^3
    ZGate: [[SGate, SGate]],
    XGate: [[HGate, ZGate, HGate], [HGate, SGate, SGate, HGate]],
}


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
                 random_goal: bool = False,
                 nerf_dim: int = 0,
                 macro_frac: float = 0.0):
        super().__init__()
        
        self.nerf_dim = nerf_dim
        self.macro_frac = macro_frac
        self.perturb = perturb
        self.num_qubits = num_qubits
        self.epsilon = epsilon
        self.encoding = encoding
        self.random_goal = random_goal
        self.gateset = gateset

        self._identity = tensor_product([I] * num_qubits)
        self._identity_hash = hash_unitary(self._identity)
        self._generate_actions(gateset)
        self._macro_words: List[List[QAction]] = self._generate_macro_words() if macro_frac > 0 else []
        if macro_frac > 0 and len(self._macro_words) == 0:
            raise ValueError(f"macro goals (flag G) need at least 2 qubits, got {num_qubits}")

    def __repr__(self) -> str:
        return 'QCircuit(gateset=%s, num_qubits=%d, epsilon=%f, nerf_dim=%d, perturb=%s, encoding=%s, macro_frac=%s)' % \
               (self.gateset, self.num_qubits, self.epsilon, self.nerf_dim, str(self.perturb), self.encoding, self.macro_frac)

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
    
    def _lookup_action(self, gate: Type[QAction], qubits: Tuple[int, ...]) -> Optional[QAction]:
        for act in self.actions:
            if type(act) is not gate:
                continue
            if isinstance(act, OneQubitGate) and (act.qubit,) == qubits:
                return act
            if isinstance(act, ControlledGate) and (act.control, act.target) == qubits:
                return act
        return None

    def _expand_macro(self, word: MacroWord) -> List[QAction]:
        """Turns a macro word into gate-set actions, substituting exact replacements for missing gates"""
        acts: List[QAction] = []
        for gate, qubits in word:
            act = self._lookup_action(gate, qubits)
            if act is not None:
                acts.append(act)
                continue
            for sub in _GATE_SUBS.get(gate, []):
                sub_acts = [self._lookup_action(g, qubits) for g in sub]
                if all(a is not None for a in sub_acts):
                    acts.extend(sub_acts)  # type: ignore[arg-type]
                    break
            else:
                raise ValueError(f"Gate set {self.gateset!r} has no way to build {gate.__name__} on qubits {qubits}")
        return acts

    def _generate_macro_words(self) -> List[List[QAction]]:
        """All macros over all assignments of distinct qubits, expanded to gate-set actions"""
        words: List[List[QAction]] = []
        for _, (builder, arity) in MACROS.items():
            if arity > self.num_qubits:
                continue
            for qubits in itertools.permutations(range(self.num_qubits), arity):
                words.append(self._expand_macro(builder(*qubits)))
        return words

    def _macro_goal_states(self, states_start: List[QState], num_steps_l: List[int]) -> List[QState]:
        """Structured goals: concatenate random macro words and keep a prefix of at most `num_steps` gates,
        so the goal is reachable in at most num_steps gates. Half of the time the prefix is cut at exactly
        num_steps (partial macros, i.e. the dense intermediate states of a decomposition); otherwise it is
        cut at the last macro boundary <= num_steps, so products of whole macros (Toffoli, CCZ, ...) are
        frequent goals rather than only appearing when num_steps happens to equal a word length"""
        Us: List[NDArray] = []
        for state, num_steps in zip(states_start, num_steps_l):
            U = state.unitary
            for act in self._macro_prefix(num_steps):
                U = np.matmul(act._full_unitary, U)
            Us.append(U.astype(np.complex128))
        return self._make_states(np.array(Us))

    def _macro_prefix(self, num_steps: int) -> List[QAction]:
        """Random macro words concatenated and cut to at most num_steps gates (see _macro_goal_states)"""
        acts: List[QAction] = []
        boundary: int = 0
        while len(acts) < num_steps:
            word = self._macro_words[np.random.randint(len(self._macro_words))]
            if len(acts) + len(word) <= num_steps:
                boundary = len(acts) + len(word)
            acts.extend(word)
        cut: int = num_steps if (np.random.uniform() < 0.5 or boundary == 0) else boundary
        return acts[:cut]

    def sample_problem_instances(self, num_steps_l: List[int], times: Optional[Times] = None) -> Tuple[List[QState], List[QGoal]]:
        """As the base class (identity start, random walk of num_steps, relative goal), except that a fraction
        `macro_frac` of the instances get a structured goal built from macro words instead of a random walk"""
        if (self.macro_frac <= 0) or self.random_goal:
            return super().sample_problem_instances(num_steps_l, times=times)
        if times is None:
            times = Times()

        start_time = time.time()
        states_start: List[QState] = self.sample_start_states(len(num_steps_l))
        times.record_time("sample_start_states", time.time() - start_time)

        start_time = time.time()
        num_steps = np.array(num_steps_l)
        use_macro = (np.random.uniform(size=len(num_steps_l)) < self.macro_frac) & (num_steps > 0)
        states_goal: List[QState] = list(states_start)
        idx_walk = np.where(~use_macro)[0]
        if len(idx_walk) > 0:
            walked = self.random_walk([states_start[i] for i in idx_walk], [int(num_steps[i]) for i in idx_walk])[0]
            for i, state in zip(idx_walk, walked):
                states_goal[i] = state
        idx_macro = np.where(use_macro)[0]
        if len(idx_macro) > 0:
            structured = self._macro_goal_states([states_start[i] for i in idx_macro], [int(num_steps[i]) for i in idx_macro])
            for i, state in zip(idx_macro, structured):
                states_goal[i] = state
        times.record_time("random_walk", time.time() - start_time)

        start_time = time.time()
        goals: List[QGoal] = self.sample_goal_from_state(states_start, states_goal)
        times.record_time("sample_goal", time.time() - start_time)

        return states_start, goals

    def actions_to_indices(self, actions: List[QAction]) -> List[int]:
        return [x.action for x in actions]

    def sample_start_states(self, num_states: int) -> List[QState]:
        """
        Generates a set of states with the identity as their unitary

        @param num_states: Number of states to generate
        @returns: Generated states
        """
        return [QState(self._identity, self._identity_hash) for _ in range(num_states)]

    def get_actions_fixed(self) -> List[List[QAction]]:
        return [x for x in self.actions]

    def string_to_action_help(self) -> str:
        return "index of gate action (actions: %s)" % (str(self.actions))

    def next_state(self, states: List[QState], actions: List[QAction]) -> Tuple[List[QState], List[float]]:
        A = np.array([a._full_unitary for a in actions])
        B = np.array([s.unitary for s in states])
        new_unitaries = np.matmul(A, B).astype(np.complex128)
        return self._make_states(new_unitaries), [a.cost for a in actions]

    @staticmethod
    def _make_states(Us: NDArray) -> List[QState]:
        """States for [B, N, N] unitaries, with hashes computed in one batch"""
        return [QState(u, h) for u, h in zip(Us, hash_unitary_batch(Us).tolist())]

    def sample_goal_from_state(self, states_start: List[QState], states_goal: List[QState]) -> List[QGoal]:
        """
        Creates goal objects from state-goal pairs
        """
        if self.random_goal:
            return [QGoal(random_unitary(2 ** self.num_qubits).data) for _ in range(len(states_start))]

        S = np.array([s.unitary for s in states_start])
        G = np.array([s.unitary for s in states_goal])
        U_b = np.matmul(G, np.conj(S).transpose(0, 2, 1))
        if self.perturb:
            U_b = perturb_unitary_givens_batch(U_b, self.epsilon)
        return [QGoal(x, h) for x, h in zip(U_b, hash_unitary_batch(U_b).tolist())]
    
    def is_solved(self, states: List[QState], goals: List[QGoal]) -> List[bool]:
        """
        Checks whether each state is solved by comparing their unitaries (within a tolerance)

        @param states: List of quantum circuit states
        @param goals: List of goals to check against
        @returns: List of bools representing solved/not-solved
        """
        Us = np.array([s.unitary for s in states])
        Cs = np.array([g.unitary for g in goals])
        return list(unitary_distance_batch(Us, Cs) <= self.epsilon)

    def string_to_action(self, act_str: str) -> QAction:
        return self.actions[int(act_str)]

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        N = encoding_size(self.encoding, self.num_qubits)
        return [2 * self.nerf_dim * N if self.nerf_dim > 0 else N], [1]

    def to_np_flat_sg(self, states: List[QState], goals: List[QGoal]) -> List[np.ndarray[float]]:
        S = np.array([s.unitary for s in states])
        G = np.array([g.unitary for g in goals])
        total_unitaries = np.matmul(G, np.conj(S).transpose(0, 2, 1))
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
            encoding = re.search(r'[MHQ](?:\+[MHQ])*', arg)
            if num_qubits is not None:
                args_dict['num_qubits'] = int(num_qubits.group(1))
            if epsilon is not None:
                args_dict['epsilon'] = float(epsilon.group()[1:])
            if nerf_dim is not None:
                args_dict['nerf_dim'] = int(nerf_dim.group(1))
            if encoding is not None:
                enc_names = {'M': 'matrix', 'H': 'hurwitz', 'Q': 'quaternion'}
                args_dict['encoding'] = '+'.join(enc_names[c] for c in encoding.group().split('+'))
            elif arg == 'P':
                args_dict['perturb'] = True
            elif arg == 'R':
                args_dict['random_goal'] = True
            elif arg == 'S':
                args_dict['gateset'] = 'CliffT_S'
            elif arg == 'I':
                args_dict['gateset'] = 'CliffT_inv'
            elif re.fullmatch(r'G(\d*\.?\d*)', arg):
                # structured (macro-word) goals; `G` = half of the instances, `G0.3` = 30%
                args_dict['macro_frac'] = float(arg[1:]) if len(arg) > 1 else 0.5
        return args_dict

    def help(self) -> str:
        return 'An integer for the number of qubits. E.g. \'qcircuit.3\''


@register_nnet_input('qcircuit', 'qcircuit_nnet_input_fix_act')
class QCircuitNNetInputFix(StateGoalActFixIn[QCircuit, QState, QGoal, QAction]):
    def get_input_info(self) -> int:
        return self.domain.num_qubits

    def to_np(self, states: List[QState], goals: List[QGoal], actions_l: List[List[QAction]]) -> List[NDArray]:
        S = np.array([s.unitary for s in states])
        G = np.array([g.unitary for g in goals])
        total_unitaries = np.matmul(G, np.conj(S).transpose(0, 2, 1))
        return [unitaries_to_nnet_input(total_unitaries, encoding=self.encoding)]
