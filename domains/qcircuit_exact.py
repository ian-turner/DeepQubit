"""Exact Clifford+T synthesis domain.

States and goals are unitaries over the ring D[omega] = Z[omega, 1/sqrt2] stored as integer
coefficient arrays plus an exponent (see utils/ring.py), always in canonical form, so hashing and
equality are exact and there is no epsilon. Gates are the same action objects as domains/qcircuit.py
(applied as integer row operations), and the parser conventions and gate sets are reused.
The network input is built from residues of the exact coefficients (binary) rather than floats.

Domain string: qcircuit_exact.n<N>[_I|_S][_<encoding>][_K<cap>], e.g. qcircuit_exact.n3_I_B9+C2
Encodings (joined by '+'):
  B<m>  residues mod 2^m of the 4 ring coefficients of every entry of G S^dagger (m bits each; default 9),
        plus a one-hot of the exponent k. Lossless while every coefficient is < 2^(m-1), i.e. k <= 2m-2.
  C<m>  channel representation: for each generator X_i, Z_i the coefficients of R P R^dagger in the Pauli
        basis, as residues mod 2^m of their Z[sqrt2] numerators (default m=2), plus a one-hot exponent per
        generator row. For a Clifford this is exactly the stabilizer tableau with signs.
  M     float view (real/imag of the canonical matrix), the old matrix encoding, for ablations.
  Z<m>  compact form of B<m> for nnets/resnet_fc_ring: the int16 coefficients of G S^dagger plus k (4 N^2 + 1 values,
        18x fewer bytes than the B9 floats); the network expands the m residue bits on the device. Must be the only part.

Speed notes: coefficients are stored as int16 (|coefficient| <= sqrt2^k, see utils/ring.py), and the relative
unitary of a generated state is derived from its parent's by one column operation (see `_relative`).
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


_STORE_LIMIT: int = 1 << 15  # coefficients are stored as int16 when they fit (always in practice: k <= 29)


def _compact(c: NDArray) -> NDArray:
    """Storage form of one coefficient array: int16 if every coefficient fits, else int64. The dtype depends only on
    the values, so equal matrices always get equal bytes (and hashes)."""
    if c.dtype == np.int16:
        return c
    c = np.asarray(c, dtype=np.int64)
    return c.astype(np.int16) if (np.abs(c) < _STORE_LIMIT).all() else c


def _compact_batch(c: NDArray) -> List[NDArray]:
    """`_compact` for every item of a batch (B, N, N, 4); int16 items are views of one batch array"""
    fits = (np.abs(c) < _STORE_LIMIT).reshape(c.shape[0], -1).all(axis=1)
    c16 = c.astype(np.int16)
    if fits.all():
        return list(c16)
    return [c16[i] if fits[i] else c[i] for i in range(c.shape[0])]


class _RingMatrix:
    """Canonical exact unitary c / sqrt2^k (shared by states and goals). Hashing and equality are exact; the hash is
    computed once. Only coeffs and k are pickled."""
    coeffs: NDArray
    k: int

    def _init_ring(self, coeffs: NDArray, k: int) -> None:
        self.coeffs = _compact(coeffs)  # (N, N, 4), canonical
        self.k = int(k)
        self._hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self.k, self.coeffs.tobytes()))
        return self._hash

    def __eq__(self, other: Any) -> bool:
        return self.k == other.k and np.array_equal(self.coeffs, other.coeffs)

    @property
    def unitary(self) -> NDArray:
        """complex128 view, for interop with scripts written for the float domain"""
        return ring.to_complex(self.coeffs, np.array(self.k))

    @classmethod
    def from_complex(cls, U: NDArray, tol: float = 1e-9) -> Any:
        return cls(*ring.from_complex(U, tol))

    def __getstate__(self) -> Dict[str, Any]:
        return {'coeffs': self.coeffs, 'k': self.k}

    def __setstate__(self, d: Dict[str, Any]) -> None:
        self.__init__(d['coeffs'], d['k'])  # type: ignore[misc]


class QStateExact(_RingMatrix, State):
    def __init__(self, coeffs: NDArray, k: int, parent: Optional["QStateExact"] = None, action: Optional[QAction] = None):
        self._init_ring(coeffs, k)
        # transient, not pickled: the state/action next_state generated this state from, and a cached canonical
        # relative unitary (goal, coeffs, k) -- see QCircuitExact._relative
        self._parent: Optional[QStateExact] = parent
        self._action: Optional[QAction] = action
        self._rel: Optional[Tuple[Any, NDArray, int]] = None


class QGoalExact(_RingMatrix, Goal):
    def __init__(self, coeffs: NDArray, k: int):
        self._init_ring(coeffs, k)


# gate class -> power of omega applied to the rows with the qubit set
_PHASE_GATES: Dict[type, int] = {TGate: 1, SGate: 2, ZGate: 4, SdgGate: 6, TdgGate: 7}
_ENC_DEFAULT_BITS: Dict[str, int] = {'B': 9, 'C': 2, 'M': 0, 'Z': 9}
# row operations of these gates leave row 0 and the exponent alone, so canonical input stays canonical
_ROW0_FIXED: Tuple[type, ...] = (TGate, SGate, ZGate, SdgGate, TdgGate, CNOTGate, CZGate)
_RAISES_K: Tuple[type, ...] = (HGate, CHGate)


@domain_factory.register_class('qcircuit_exact')
class QCircuitExact(QCircuit):
    def __init__(self, num_qubits: int, gateset: str = 'CliffT', encoding: str = 'B9+C2', k_cap: int = 20):
        super().__init__(num_qubits=num_qubits, epsilon=0.0, perturb=False, encoding='matrix', gateset=gateset,
                         random_goal=False, nerf_dim=0)
        self.encoding = encoding
        self.k_cap = k_cap  # exponent one-hots are capped here (channel rows at 2 * k_cap)
        self.N = 1 << num_qubits
        self._parts: List[Tuple[str, int]] = self._parse_encoding(encoding)
        if any(kind == 'Z' for kind, _ in self._parts) and len(self._parts) > 1:
            raise ValueError(f"encoding part Z must be used alone, got {encoding!r}")
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
        return 'QCircuitExact(gateset=%s, num_qubits=%d, encoding=%s, k_cap=%d)' % \
               (self.gateset, self.num_qubits, self.encoding, self.k_cap)

    @staticmethod
    def _parse_encoding(encoding: str) -> List[Tuple[str, int]]:
        parts: List[Tuple[str, int]] = []
        for part in encoding.split('+'):
            m = re.fullmatch(r'([BCMZ])(\d*)', part)
            if m is None:
                raise ValueError(f"Unknown exact encoding part {part!r}")
            parts.append((m.group(1), int(m.group(2)) if m.group(2) else _ENC_DEFAULT_BITS[m.group(1)]))
        return parts

    # ---- states and transitions -------------------------------------------------------------
    def sample_start_states(self, num_states: int) -> List[QStateExact]:
        c, k = ring.identity(self.N)
        c = _compact(c)  # shared read-only; each start state is its own object (it carries its own caches)
        return [QStateExact(c, k) for _ in range(num_states)]

    def _apply_group(self, c: NDArray, k: NDArray, action: QAction, conj: bool = False) -> Tuple[NDArray, NDArray]:
        """Apply one gate (its complex conjugate with conj) as row operations to a batch (B, N, N, 4) with exponents
        (B,); not normalized. conj(Y) = -Y, a global phase that canonicalization removes."""
        n = self.num_qubits
        t = type(action)
        if t is HGate:
            return ring.apply_h(c, k, n, action.qubit)
        if t in _PHASE_GATES:
            return ring.apply_phase_rows(c, n, action.qubit, -_PHASE_GATES[t] if conj else _PHASE_GATES[t]), k
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

    def _apply_actions(self, c: NDArray, k: NDArray, actions: List[QAction], cols: bool = False) -> Tuple[NDArray, NDArray]:
        """Canonical A_b c_b (rows) or c_b A_b^dagger (cols) for canonical int64 inputs, grouped by action. Only gates
        that can change the exponent are reduced, and only gates that can change row 0 are re-canonicalized (row
        ops of phase gates / CNOT / CZ fix row 0; any column op can change it)."""
        out_c = np.empty_like(c)
        out_k = k.copy()
        ids = np.fromiter((a.action for a in actions), dtype=np.int64, count=len(actions))
        reduce = np.zeros(len(actions), dtype=bool)
        canon = np.full(len(actions), cols)
        for aid in np.unique(ids):
            idx = np.flatnonzero(ids == aid)
            act = self.actions[aid]
            if cols:  # c A^dagger = (conj(A) c^T)^T
                sub, sub_k = self._apply_group(np.swapaxes(c[idx], 1, 2), k[idx], act, conj=True)
                sub = np.swapaxes(sub, 1, 2)
            else:
                sub, sub_k = self._apply_group(c[idx], k[idx], act)
            out_c[idx], out_k[idx] = sub, sub_k
            reduce[idx] = isinstance(act, _RAISES_K)
            canon[idx] |= not isinstance(act, _ROW0_FIXED)
        for mask, fn in ((reduce, ring.reduce_batch), (canon, ring.canonicalize_batch)):
            if mask.any():
                idx = np.flatnonzero(mask)
                out_c[idx], out_k[idx] = fn(out_c[idx], out_k[idx])
        return out_c, out_k

    def next_state(self, states: List[QStateExact], actions: List[QAction]) -> Tuple[List[QStateExact], List[float]]:
        c = np.stack([s.coeffs for s in states]).astype(np.int64)
        k = np.array([s.k for s in states], dtype=np.int64)
        c, k = self._apply_actions(c, k, actions)
        states_next = [QStateExact(ci, ki, s, a) for ci, ki, s, a in zip(_compact_batch(c), k.tolist(), states, actions)]
        return states_next, [a.cost for a in actions]

    def _relative_scratch(self, states: List[Any], goals: List[Any]) -> Tuple[NDArray, NDArray]:
        """Canonical G S^dagger for each pair by a full product (anything with .coeffs and .k)"""
        S = np.stack([s.coeffs for s in states])
        G = np.stack([g.coeffs for g in goals])
        k = np.array([g.k + s.k for s, g in zip(states, goals)], dtype=np.int64)
        return ring.normalize_batch(ring.matmul_dagger(G, S), k)

    @staticmethod
    def _cached_rel(state: QStateExact, goal: QGoalExact) -> Optional[Tuple[Any, NDArray, int]]:
        rel = state._rel
        return rel if (rel is not None and rel[0] is goal) else None

    def _relative_direct(self, states: List[QStateExact], goals: List[QGoalExact]) -> Tuple[NDArray, NDArray]:
        """G S^dagger from the state's own cache, else by a column op on the parent's cached one, else from scratch
        (cached on the state: these are roots and relabeled goals, which are few)"""
        B, N = len(states), self.N
        c = np.empty((B, N, N, 4), dtype=np.int64)
        k = np.empty(B, dtype=np.int64)
        hit: List[int] = []
        inc: List[int] = []
        scratch: List[int] = []
        for i, (s, g) in enumerate(zip(states, goals)):
            if self._cached_rel(s, g) is not None:
                hit.append(i)
            elif s._parent is not None and self._cached_rel(s._parent, g) is not None:
                inc.append(i)
            else:
                scratch.append(i)
        if hit:
            c[hit] = np.stack([states[i]._rel[1] for i in hit])  # type: ignore[index]
            k[hit] = [states[i]._rel[2] for i in hit]  # type: ignore[index]
        if inc:
            rels = [states[i]._parent._rel for i in inc]  # type: ignore[union-attr]
            pc = np.stack([rel[1] for rel in rels]).astype(np.int64)  # type: ignore[index]
            pk = np.array([rel[2] for rel in rels], dtype=np.int64)  # type: ignore[index]
            c[inc], k[inc] = self._apply_actions(pc, pk, [states[i]._action for i in inc], cols=True)  # type: ignore[misc]
        if scratch:
            sc, sk = self._relative_scratch([states[i] for i in scratch], [goals[i] for i in scratch])
            c[scratch], k[scratch] = sc, sk
            for j, i in enumerate(scratch):
                states[i]._rel = (goals[i], _compact(sc[j]), int(sk[j]))
        return c, k

    def _relative(self, states: List[QStateExact], goals: List[QGoalExact]) -> Tuple[NDArray, NDArray]:
        """Canonical G S^dagger for each pair (int64 coefficients, exponents).
        G (A S)^dagger = (G S^dagger) A^dagger, so for a state that next_state generated from a parent the relative
        unitary is one column operation on the parent's: no 4N x 4N product and no long sqrt2 reduction. Relatives
        are cached only on parents (the expanded ~1/|A| of the search tree): a parent whose own parent is cached (it
        was expanded earlier) gets its relative here first by a column op. Everything else -- roots, relabeled
        goals -- is computed from scratch."""
        parents: List[QStateExact] = []
        parent_goals: List[QGoalExact] = []
        seen = set()
        for s, g in zip(states, goals):
            p = s._parent
            if (p is not None) and (p._parent is not None) and (self._cached_rel(s, g) is None) \
                    and (self._cached_rel(p, g) is None) and (self._cached_rel(p._parent, g) is not None) \
                    and ((id(p), id(g)) not in seen):
                seen.add((id(p), id(g)))
                parents.append(p)
                parent_goals.append(g)
        if parents:
            pc, pk = self._relative_direct(parents, parent_goals)
            for p, g, ci, ki in zip(parents, parent_goals, pc, pk.tolist()):
                p._rel = (g, _compact(ci), ki)
        return self._relative_direct(states, goals)

    def sample_goal_from_state(self, states_start: List[QStateExact], states_goal: List[QStateExact]) -> List[QGoalExact]:
        c, k = self._relative_scratch(states_start, states_goal)
        return [QGoalExact(ci, ki) for ci, ki in zip(_compact_batch(c), k.tolist())]

    def is_solved(self, states: List[QStateExact], goals: List[QGoalExact]) -> List[bool]:
        return [s.k == g.k and np.array_equal(s.coeffs, g.coeffs) for s, g in zip(states, goals)]

    # ---- neural network input ---------------------------------------------------------------
    def _part_dim(self, kind: str, m: int) -> int:
        n, N = self.num_qubits, self.N
        if kind == 'B':
            return N * N * 4 * m + (self.k_cap + 1)
        if kind == 'C':
            return 2 * n * (4 ** n) * 2 * m + 2 * n * (2 * self.k_cap + 1)
        if kind == 'Z':
            return N * N * 4 + 1
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
        if self._parts[0][0] == 'Z':  # int16 wrap-around keeps the residues mod 2^16 (exact while k <= 29)
            k16 = np.minimum(k, _STORE_LIMIT - 1)[:, None]
            return [np.concatenate([c.reshape(c.shape[0], -1), k16], axis=1).astype(np.int16)]
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
            elif re.fullmatch(r'[BCMZ]\d*(\+[BCMZ]\d*)*', arg):
                args_dict['encoding'] = arg
            else:
                raise ValueError(f"Unexpected qcircuit_exact argument {arg!r}")
        return args_dict

    def help(self) -> str:
        return ("n<N> qubits, I/S gate set, encoding B<m>/C<m>/M joined by '+' (or Z<m> alone), "
                "K<cap> exponent cap. "
                "E.g. 'qcircuit_exact.n3_I_B9+C2'")


@register_nnet_input('qcircuit_exact', 'qcircuit_exact_nnet_input_fix_act')
class QCircuitExactNNetInputFix(StateGoalActFixIn[QCircuitExact, QStateExact, QGoalExact, QAction]):
    def get_input_info(self) -> Tuple[List[int], List[int]]:
        return self.domain.get_input_info_flat_sg()

    def to_np(self, states: List[QStateExact], goals: List[QGoalExact], actions_l: List[List[QAction]]) -> List[NDArray]:
        return self.domain.to_np_flat_sg(states, goals)
