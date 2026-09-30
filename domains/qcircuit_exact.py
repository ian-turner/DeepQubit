"""Exact Clifford+T synthesis domain.

States and goals are unitaries over the ring D[omega] = Z[omega, 1/sqrt2] stored as integer
coefficient arrays plus an exponent (see utils/ring.py), always in canonical form, so hashing and
equality are exact and there is no epsilon. Gates are the same action objects as domains/qcircuit.py
(applied as integer row operations), and the parser, gate sets and macro-word goals are reused.
The network input is built from residues of the exact coefficients (binary) rather than floats.

Domain string: qcircuit_exact.n<N>[_I|_S][_G[<frac>]][_<encoding>][_K<cap>], e.g. qcircuit_exact.n3_I_G_B9+C2
Encodings (joined by '+'):
  B<m>  residues mod 2^m of the 4 ring coefficients of every entry of G S^dagger (m bits each; default 9),
        plus a one-hot of the exponent k. Lossless while every coefficient is < 2^(m-1), i.e. k <= 2m-2.
  C<m>  channel representation: for each generator X_i, Z_i the coefficients of R P R^dagger in the Pauli
        basis, as residues mod 2^m of their Z[sqrt2] numerators (default m=2), plus a one-hot exponent per
        generator row. For a Clifford this is exactly the stabilizer tableau with signs.
  M     float view (real/imag of the canonical matrix), the old matrix encoding, for ablations.
"""
import re
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from numpy.typing import NDArray

from deepxube.base.domain import State, Goal
from deepxube.base.factory import Parser
from deepxube.base.nnet_input import StateGoalActFixIn
from deepxube.factories.domain_factory import domain_factory
from deepxube.factories.nnet_input_factory import register_nnet_input

from domains.qcircuit import (QCircuit, QAction, HGate, SGate, SdgGate, ZGate, TGate, TdgGate, XGate, YGate,
                              CNOTGate, CZGate, CHGate)
from utils import ring


class QStateExact(State):
    def __init__(self, coeffs: NDArray, k: int):
        self.coeffs: NDArray = coeffs  # (N, N, 4) int64, canonical
        self.k: int = int(k)

    def __hash__(self) -> int:
        return hash((self.k, self.coeffs.tobytes()))

    def __eq__(self, other: Any) -> bool:
        return self.k == other.k and np.array_equal(self.coeffs, other.coeffs)

    @property
    def unitary(self) -> NDArray:
        """complex128 view, for interop with scripts written for the float domain"""
        return ring.to_complex(self.coeffs, np.array(self.k))

    @classmethod
    def from_complex(cls, U: NDArray, tol: float = 1e-9) -> "QStateExact":
        return cls(*ring.from_complex(U, tol))


class QGoalExact(Goal):
    def __init__(self, coeffs: NDArray, k: int):
        self.coeffs: NDArray = coeffs
        self.k: int = int(k)

    def __hash__(self) -> int:
        return hash((self.k, self.coeffs.tobytes()))

    def __eq__(self, other: Any) -> bool:
        return self.k == other.k and np.array_equal(self.coeffs, other.coeffs)

    @property
    def unitary(self) -> NDArray:
        return ring.to_complex(self.coeffs, np.array(self.k))

    @classmethod
    def from_complex(cls, U: NDArray, tol: float = 1e-9) -> "QGoalExact":
        return cls(*ring.from_complex(U, tol))


# gate class -> power of omega applied to the rows with the qubit set
_PHASE_GATES: Dict[type, int] = {TGate: 1, SGate: 2, ZGate: 4, SdgGate: 6, TdgGate: 7}
_ENC_DEFAULT_BITS: Dict[str, int] = {'B': 9, 'C': 2, 'M': 0}


@domain_factory.register_class('qcircuit_exact')
class QCircuitExact(QCircuit):
    def __init__(self, num_qubits: int, gateset: str = 'CliffT', macro_frac: float = 0.0, encoding: str = 'B9+C2',
                 k_cap: int = 20):
        super().__init__(num_qubits=num_qubits, epsilon=0.0, perturb=False, encoding='matrix', gateset=gateset,
                         random_goal=False, nerf_dim=0, macro_frac=macro_frac)
        self.encoding = encoding
        self.k_cap = k_cap  # exponent one-hots are capped here (channel rows at 2 * k_cap)
        self.N = 1 << num_qubits
        self._parts: List[Tuple[str, int]] = self._parse_encoding(encoding)
        # channel representation tables
        self._sigma, self._phase = ring.pauli_tables(num_qubits)  # (4^n, N)
        self._gen_comp: NDArray = np.stack([ring.companion(ring.pauli_coeffs(num_qubits, p))
                                            for p in ring.pauli_generator_indices(num_qubits)])  # (2n, 4N, 4N)
        # trace(Q R P R^dagger) = sum_i omega^phase[Q,i] * M[sigma[Q,i], i]; the coefficients of omega^t * z are
        # column t of z's companion block (t < 4; omega^(t+4) = -omega^t), so the ring coefficients j of the trace
        # are signed sums of entries [4 sigma + j, 4 i + (phase mod 4)] of the (4N x 4N) companion form of M
        N = self.N
        i = np.arange(N)[None, :, None]
        j = np.arange(2)[None, None, :]  # only a (j=0) and b (j=1) of the real value a + b*sqrt2 are needed
        self._chan_idx: NDArray = (4 * self._sigma[:, :, None] + j) * (4 * N) + (4 * i + self._phase[:, :, None] % 4)  # (4^n, N, 2)
        self._chan_sign: NDArray = np.where(self._phase >= 4, -1.0, 1.0)[None, :, :, None]  # (1, 4^n, N, 1)

    def __repr__(self) -> str:
        return 'QCircuitExact(gateset=%s, num_qubits=%d, encoding=%s, k_cap=%d, macro_frac=%s)' % \
               (self.gateset, self.num_qubits, self.encoding, self.k_cap, self.macro_frac)

    @staticmethod
    def _parse_encoding(encoding: str) -> List[Tuple[str, int]]:
        parts: List[Tuple[str, int]] = []
        for part in encoding.split('+'):
            m = re.fullmatch(r'([BCM])(\d*)', part)
            if m is None:
                raise ValueError(f"Unknown exact encoding part {part!r}")
            parts.append((m.group(1), int(m.group(2)) if m.group(2) else _ENC_DEFAULT_BITS[m.group(1)]))
        return parts

    # ---- states and transitions -------------------------------------------------------------
    def sample_start_states(self, num_states: int) -> List[QStateExact]:
        c, k = ring.identity(self.N)
        return [QStateExact(c.copy(), k) for _ in range(num_states)]

    def _apply_group(self, c: NDArray, k: NDArray, action: QAction) -> Tuple[NDArray, NDArray]:
        """Apply one gate to a batch of coefficient arrays (B, N, N, 4) with exponents (B,); not normalized"""
        n = self.num_qubits
        t = type(action)
        if t is HGate:
            return ring.apply_h(c, k, n, action.qubit)
        if t in _PHASE_GATES:
            return ring.apply_phase_rows(c, n, action.qubit, _PHASE_GATES[t]), k
        if t is XGate:
            return ring.apply_x(c, n, action.qubit), k
        if t is YGate:
            return ring.apply_y(c, n, action.qubit), k
        if t is CNOTGate:
            return ring.apply_x(c, n, action.target, control=action.control), k
        if t is CZGate:
            return ring.apply_z(c, n, action.target, control=action.control), k
        if t is CHGate:
            return ring.apply_h(c, k, n, action.target, control=action.control)
        raise ValueError(f"No exact row operation for {t.__name__}")

    def _next_arrays(self, states: List[QStateExact], actions: List[QAction]) -> Tuple[NDArray, NDArray]:
        c = np.stack([s.coeffs for s in states])
        k = np.array([s.k for s in states], dtype=np.int64)
        out_c = np.empty_like(c)
        out_k = np.empty_like(k)
        ids = np.array([a.action for a in actions])
        for aid in np.unique(ids):
            idx = np.where(ids == aid)[0]
            out_c[idx], out_k[idx] = self._apply_group(c[idx], k[idx], self.actions[aid])
        return ring.normalize_batch(out_c, out_k)

    def next_state(self, states: List[QStateExact], actions: List[QAction]) -> Tuple[List[QStateExact], List[float]]:
        c, k = self._next_arrays(states, actions)
        return [QStateExact(ci, ki) for ci, ki in zip(c, k)], [a.cost for a in actions]

    def _relative(self, states: List[QStateExact], goals: List[QGoalExact]) -> Tuple[NDArray, NDArray]:
        """Canonical G S^dagger for each pair"""
        S = np.stack([s.coeffs for s in states])
        G = np.stack([g.coeffs for g in goals])
        k = np.array([g.k + s.k for s, g in zip(states, goals)], dtype=np.int64)
        return ring.normalize_batch(ring.matmul_dagger(G, S), k)

    def sample_goal_from_state(self, states_start: List[QStateExact], states_goal: List[QStateExact]) -> List[QGoalExact]:
        c, k = self._relative(states_start, states_goal)
        return [QGoalExact(ci, ki) for ci, ki in zip(c, k)]

    def is_solved(self, states: List[QStateExact], goals: List[QGoalExact]) -> List[bool]:
        return [s.k == g.k and np.array_equal(s.coeffs, g.coeffs) for s, g in zip(states, goals)]

    def _macro_goal_states(self, states_start: List[QStateExact], num_steps_l: List[int]) -> List[QStateExact]:
        states_goal: List[QStateExact] = []
        for state, num_steps in zip(states_start, num_steps_l):
            c, k = state.coeffs[None], np.array([state.k], dtype=np.int64)
            for act in self._macro_prefix(num_steps):
                c, k = ring.reduce_batch(*self._apply_group(c, k, act))
            c, k = ring.canonicalize_batch(c, k)
            states_goal.append(QStateExact(c[0], k[0]))
        return states_goal

    # ---- neural network input ---------------------------------------------------------------
    def _part_dim(self, kind: str, m: int) -> int:
        n, N = self.num_qubits, self.N
        if kind == 'B':
            return N * N * 4 * m + (self.k_cap + 1)
        if kind == 'C':
            return 2 * n * (4 ** n) * 2 * m + 2 * n * (2 * self.k_cap + 1)
        return 2 * N * N  # 'M'

    def get_input_info_flat_sg(self) -> Tuple[List[int], List[int]]:
        return [sum(self._part_dim(kind, m) for kind, m in self._parts)], [1]

    @staticmethod
    def _bits(x: NDArray, m: int) -> NDArray:
        """Residues mod 2^m of int array (B, ...) as m bits each (least significant first) -> (B, prod * m) uint8"""
        assert 1 <= m <= 16
        r = np.mod(x, 1 << m).reshape(x.shape[0], -1).astype(np.uint16)
        bits = np.unpackbits(r.view(np.uint8).reshape(r.shape + (2,)), axis=-1, bitorder='little')  # (B, P, 16)
        return bits[:, :, :m].reshape(x.shape[0], -1)

    @staticmethod
    def _onehot(k: NDArray, cap: int) -> NDArray:
        """(B,) or (B, R) integers -> (B, R * (cap + 1)) one-hots, values above cap clipped to cap"""
        kk = np.minimum(k, cap).reshape(k.shape[0], -1)
        return (kk[:, :, None] == np.arange(cap + 1)).reshape(k.shape[0], -1)

    def _channel(self, c: NDArray, k: NDArray) -> Tuple[NDArray, NDArray]:
        """Channel representation of R = c / sqrt2^k on the generators X_i, Z_i.
        Returns Z[sqrt2] numerators (B, 2n, 4^n, 2) meaning a + b*sqrt2, and per-row exponents (B, 2n) such that
        coefficient = (a + b sqrt2) / sqrt2^exponent, reduced per row to the minimal exponent."""
        B, n, N = c.shape[0], self.num_qubits, self.N
        R = ring.companion(c)
        Rt = np.swapaxes(R, 1, 2)
        rows: List[NDArray] = []
        for gen in self._gen_comp:
            M = (R @ gen @ Rt).reshape(B, -1)  # companion form of R P R^dagger (numerators, exponent 2k)
            rows.append((M[:, self._chan_idx] * self._chan_sign).sum(axis=2))  # (B, 4^n, 2): trace(Q M) = a + b*sqrt2
        coef = np.rint(np.stack(rows, axis=1)).astype(np.int64)  # (B, 2n, 4^n, 2)
        ex = np.repeat((2 * k + 2 * n)[:, None], 2 * n, axis=1).astype(np.int64)  # trace(Q R P R^dag) / N
        return ring.reduce_zsqrt2_rows(coef, ex)

    def _encode_part(self, kind: str, m: int, c: NDArray, k: NDArray) -> NDArray:
        if kind == 'B':
            return np.concatenate([self._bits(c, m), self._onehot(k, self.k_cap)], axis=1)
        if kind == 'C':
            coef, ex = self._channel(c, k)
            return np.concatenate([self._bits(coef, m), self._onehot(ex, 2 * self.k_cap)], axis=1)
        U = ring.to_complex(c, k).reshape(c.shape[0], -1)
        return np.concatenate([U.real, U.imag], axis=1)

    def to_np_flat_sg(self, states: List[QStateExact], goals: List[QGoalExact]) -> List[NDArray]:
        c, k = self._relative(states, goals)
        feats = [self._encode_part(kind, m, c, k) for kind, m in self._parts]
        return [np.concatenate(feats, axis=1).astype(np.float32)]


@domain_factory.register_parser('qcircuit_exact')
class QCircuitExactParser(Parser):
    def parse(self, args_str: str) -> Dict[str, Any]:
        args_dict: Dict[str, Any] = {}
        for arg in args_str.split('_'):
            num_qubits = re.fullmatch(r'n(\d+)', arg)
            k_cap = re.fullmatch(r'K(\d+)', arg)
            if num_qubits is not None:
                args_dict['num_qubits'] = int(num_qubits.group(1))
            elif k_cap is not None:
                args_dict['k_cap'] = int(k_cap.group(1))
            elif arg == 'S':
                args_dict['gateset'] = 'CliffT_S'
            elif arg == 'I':
                args_dict['gateset'] = 'CliffT_inv'
            elif re.fullmatch(r'G(\d*\.?\d*)', arg):
                args_dict['macro_frac'] = float(arg[1:]) if len(arg) > 1 else 0.5
            elif re.fullmatch(r'[BCM]\d*(\+[BCM]\d*)*', arg):
                args_dict['encoding'] = arg
            else:
                raise ValueError(f"Unexpected qcircuit_exact argument {arg!r}")
        return args_dict

    def help(self) -> str:
        return ("n<N> qubits, I/S gate set, G[<frac>] macro goals, encoding B<m>/C<m>/M joined by '+', K<cap> exponent cap. "
                "E.g. 'qcircuit_exact.n3_I_G_B9+C2'")


@register_nnet_input('qcircuit_exact', 'qcircuit_exact_nnet_input_fix_act')
class QCircuitExactNNetInputFix(StateGoalActFixIn[QCircuitExact, QStateExact, QGoalExact, QAction]):
    def get_input_info(self) -> Tuple[List[int], List[int]]:
        return self.domain.get_input_info_flat_sg()

    def to_np(self, states: List[QStateExact], goals: List[QGoalExact], actions_l: List[List[QAction]]) -> List[NDArray]:
        return self.domain.to_np_flat_sg(states, goals)
