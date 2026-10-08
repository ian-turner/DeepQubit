"""Writes the named exact benchmark targets as .txt files in data/targets/<n>qubit/, and copies Synthetiq's benchmark
specs (3-qubit permutation classes, 4-qubit comparison operators, cciswap, carry) next to them.

Named gates are built from qiskit circuits and stored big-endian (qubit 0 = most significant bit, the domain's order):
Operator(qc).reverse_qargs(). Each is checked against an independent construction (a hand-written matrix or bit-level
permutation) and must be unitary, in Z[omega, 1/sqrt2] as written (no global phase fix, so the determinant test is
meaningful) and pass the determinant reachability test (check_goals.is_reachable). The Synthetiq files are copied
verbatim; they are in qiskit's little-endian order (Synthetiq's ccx.txt is Operator(ccx(0, 1, 2)).data, checked
below), so in the domain they read as the qubit-reversed operator, which has the same T-count, T-depth and CNOT count.
They pass the same checks. Every written file is re-read with load_matrix_from_file and compared.

Only the files listed here are written: the older targets (ch, cs, cz, crz_2, cch, ccrz_2, cct, ccz, csqrtiswap,
csqrtswap, fredkin, toffoli) are left alone. The 3-qubit "AND" gate of Rietsch et al. 2024 Table II is not a target:
no source defines it (TODO). Gates that no circuit without ancillas can reach exactly (controlled-T,
CCS, CC-sqrtX, C-sqrtSWAP, CCT, QFT3, C3X, C3Z, C3-sqrtX) are not targets; see notes/data.md.

Usage: python scripts/make_targets.py [--synthetiq ~/research/synthetiq] [--out data/targets] [--check]
       (--check verifies the existing files instead of writing them)
"""
import os
import shutil
from argparse import ArgumentParser
from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import iSwapGate
from qiskit.quantum_info import Operator

from utils import ring
from utils.matrix_utils import load_matrix_from_file, save_matrix_to_file
from check_goals import det_omega_power, is_reachable


def big_endian(qc: QuantumCircuit) -> NDArray:
    """qiskit's matrix of qc with qubit 0 as the most significant bit"""
    return Operator(qc).reverse_qargs().data


def snap(U: NDArray) -> NDArray:
    """U with every entry recomputed from its exact Z[omega, 1/sqrt2] form, so the files read 0.5 rather than
    0.4999999999999999 and 0 rather than 6e-17 (raises ValueError if U is not in the ring as written)"""
    c, k = ring.from_complex(U, 1e-9, fix_phase=False)  # canonical: U times some power of omega

    def part(x, y):  # (x + y/sqrt2) / sqrt2^k with exact powers of two
        m = k // 2
        if k % 2 == 0:
            return x * 0.5 ** m + y * np.sqrt(0.5) * 0.5 ** m
        return y * 0.5 ** (m + 1) + x * np.sqrt(0.5) * 0.5 ** m
    for t in range(8):
        a, b, cc, d = np.moveaxis(ring.rotate(c, t), -1, 0)  # entry = (a + b w + cc w^2 + d w^3) / sqrt2^k
        V = part(a, b - d) + 1j * part(cc, b + d)
        if np.allclose(V, U, atol=1e-9):
            return V
    raise AssertionError("no power of omega maps the ring form back to U")


def circuit(n: int, *ops) -> QuantumCircuit:
    """n-qubit circuit from (QuantumCircuit method, *args) tuples, e.g. ('ccx', 0, 1, 2), ('append', gate, [0, 1])"""
    qc = QuantumCircuit(n)
    for op, *args in ops:
        getattr(qc, op)(*args)
    return qc


def permutation(n: int, f: Callable[..., Tuple[int, ...]]) -> NDArray:
    """|x> -> |f(x)>, with f taking and returning the n bits of a basis state, qubit 0 first"""
    M = np.zeros((2 ** n, 2 ** n), dtype=np.complex128)
    for x in range(2 ** n):
        y = f(*[(x >> (n - 1 - q)) & 1 for q in range(n)])
        M[sum(b << (n - 1 - q) for q, b in enumerate(y)), x] = 1
    return M


def controlled(U: NDArray, num_controls: int = 1) -> NDArray:
    """U on the last qubits when the first num_controls qubits are all 1"""
    M = np.eye(U.shape[0] * 2 ** num_controls, dtype=np.complex128)
    M[-U.shape[0]:, -U.shape[0]:] = U
    return M


def relative_phase_of(U: NDArray, V: NDArray) -> bool:
    """U = D V for a diagonal D of 4th roots of unity (U is V up to relative phases, as a relative-phase Toffoli is)"""
    D = U @ V.conj().T
    d = np.diag(D)
    return np.allclose(D, np.diag(d), atol=1e-12) and np.allclose(d ** 4, 1, atol=1e-12)


r2 = 1 / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
SX = np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2  # V = sqrt(X)
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
TOFFOLI = permutation(3, lambda a, b, c: (a, b, c ^ (a & b)))
CCZ = np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(np.complex128)


def rzx(theta: float) -> NDArray:
    """exp(-i theta/2 Z0 X1)"""
    return np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * np.kron(Z, X)


# (name, circuit, independent matrix, description); the matrices must agree exactly (not just up to a global phase)
Target = Tuple[str, QuantumCircuit, NDArray, str]
NAMED: Dict[int, List[Target]] = {
    2: [
        ('csx', circuit(2, ('csx', 0, 1)), controlled(SX),
         'controlled-V, V = sqrt(X) (qiskit CSXGate; Amy et al. 2013 Fig. 5c); control 0'),
        ('cy', circuit(2, ('cy', 0, 1)), controlled(Y), 'controlled-Y; control 0'),
        ('swap', circuit(2, ('swap', 0, 1)), permutation(2, lambda a, b: (b, a)), 'SWAP'),
        ('w', circuit(2, ('cx', 0, 1), ('ch', 1, 0), ('cx', 0, 1)),
         np.array([[1, 0, 0, 0], [0, r2, r2, 0], [0, r2, -r2, 0], [0, 0, 0, 1]]),
         'W gate of Amy et al. 2013 (Sec. 6.1, Fig. 6; wire 1 = qubit 0): Hadamard on span{|01>, |10>}'),
        ('iswap', circuit(2, ('iswap', 0, 1)), ISWAP, 'iSWAP'),
        ('dcx', circuit(2, ('dcx', 0, 1)), permutation(2, lambda a, b: (b, a ^ b)),
         'double CNOT (qiskit DCXGate): CX(0,1) then CX(1,0)'),
        ('ecr', circuit(2, ('ecr', 0, 1)), rzx(-np.pi / 4) @ np.kron(X, np.eye(2)) @ rzx(np.pi / 4),
         'echoed cross-resonance (qiskit ECRGate): RZX(pi/4), X on 0, RZX(-pi/4)'),
        ('bell', circuit(2, ('h', 0), ('cx', 0, 1)),
         np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0]]) * r2,
         'Bell-basis change H(0) then CX(0,1): |00>,|01>,|10>,|11> -> Phi+, Psi+, Phi-, Psi-'),
        ('sqrt_iswap', circuit(2, ('append', iSwapGate().power(0.5), [0, 1])),
         np.array([[1, 0, 0, 0], [0, r2, 1j * r2, 0], [0, 1j * r2, r2, 0], [0, 0, 0, 1]]), 'square root of iSWAP'),
        ('qft2', circuit(2, ('h', 0), ('cp', np.pi / 2, 1, 0), ('h', 1), ('swap', 0, 1)),
         np.array([[1j ** (j * k) for k in range(4)] for j in range(4)]) / 2,
         '2-qubit QFT (DFT matrix in the big-endian basis; = Operator(QFTGate(2)).data)'),
    ],
    3: [
        ('peres', circuit(3, ('ccx', 0, 1, 2), ('cx', 0, 1)),
         permutation(3, lambda a, b, c: (a, a ^ b, c ^ (a & b))),
         'Peres gate: Toffoli(0,1;2) then CX(0,1) (Peres 1985; Amy et al. 2013 Fig. 7d)'),
        ('toffoli_neg1', circuit(3, ('x', 0), ('ccx', 0, 1, 2), ('x', 0)),
         permutation(3, lambda a, b, c: (a, b, c ^ ((1 - a) & b))),
         'Toffoli with control 0 negated (Amy et al. 2013 Fig. 7b)'),
        ('toffoli_neg2', circuit(3, ('x', 0), ('x', 1), ('ccx', 0, 1, 2), ('x', 0), ('x', 1)),
         permutation(3, lambda a, b, c: (a, b, c ^ ((1 - a) & (1 - b)))),
         'Toffoli with both controls negated (Rietsch et al. 2024 Table II gives no definition; assumed)'),
        ('qor', circuit(3, ('x', 0), ('x', 1), ('ccx', 0, 1, 2), ('x', 0), ('x', 1), ('x', 2)),
         permutation(3, lambda a, b, c: (a, b, c ^ (a | b))), 'quantum OR: target 2 ^= (0 OR 1) (Amy et al. 2013 Fig. 7c)'),
        ('rccx', circuit(3, ('rccx', 0, 1, 2)), TOFFOLI,
         'Margolus relative-phase Toffoli (qiskit RCCXGate); controls 0, 1, target 2; checked as Toffoli up to '
         'relative phases'),
        ('ciswap', circuit(3, ('append', iSwapGate().control(1), [0, 1, 2])), controlled(ISWAP),
         'controlled-iSWAP; control 0'),
        ('tr', circuit(3, ('x', 1), ('ccx', 0, 1, 2), ('cx', 0, 1), ('x', 1)),
         permutation(3, lambda a, b, c: (a, a ^ b, c ^ (a & (1 - b)))),
         'Thapliyal-Ranganathan gate (A, B, C) -> (A, A^B, A.notB ^ C) (Thapliyal & Ranganathan 2009) = X(1) Peres X(1)'),
        ('maj', circuit(3, ('cx', 2, 1), ('cx', 2, 0), ('ccx', 0, 1, 2)),
         permutation(3, lambda c, b, a: (c ^ a, b ^ a, (a & b) ^ (a & c) ^ (b & c))),
         'Cuccaro et al. 2004 MAJ (Fig. 1) on (c, b, a) = qubits (0, 1, 2): CX(a;b), CX(a;c), Toffoli(c,b;a); '
         'a -> MAJ(a, b, c)'),
        ('uma', circuit(3, ('ccx', 0, 1, 2), ('cx', 2, 0), ('cx', 0, 1)),
         permutation(3, lambda c, b, a: (c ^ a ^ (b & c), b ^ c ^ a ^ (b & c), a ^ (b & c))),
         'Cuccaro et al. 2004 2-CNOT UMA (Fig. 2a) on (c, b, a) = qubits (0, 1, 2): Toffoli(c,b;a), CX(a;c), CX(c;b)'),
    ],
    4: [
        ('rcccx', circuit(4, ('rcccx', 0, 1, 2, 3)), permutation(4, lambda a, b, c, d: (a, b, c, d ^ (a & b & c))),
         'relative-phase C3X (qiskit RC3XGate, Maslov 2016); controls 0, 1, 2, target 3; checked as C3X up to '
         'relative phases'),
        ('toffoli', circuit(4, ('ccx', 0, 1, 2)), np.kron(TOFFOLI, np.eye(2)), 'Toffoli(0,1;2), qubit 3 idle'),
        ('ccz', circuit(4, ('ccz', 0, 1, 2)), np.kron(CCZ, np.eye(2)), 'CCZ on 0, 1, 2, qubit 3 idle'),
    ],
}
# gates matched only up to relative phases (the independent matrix is the gate they are a relative-phase version of)
RELATIVE_PHASE = {'rccx', 'rcccx'}

# (Synthetiq file under data/input, destination under --out); copied verbatim
SYNTHETIQ: List[Tuple[str, str]] = (
    [(f"64/permutations/{i}.txt", f"3qubit_perms/{i}.txt") for i in range(30)] +
    [(f"64/comparison/{x}.txt", f"4qubit/{x}.txt") for x in ['U1', 'U1_var', 'U2', 'adder']] +
    [('62/cciswap.txt', '4qubit/cciswap.txt'), ('66/carry/carry.txt', '4qubit/carry.txt')]
)


def check(name: str, U: NDArray) -> None:
    """unitary, in the ring as written, and exactly reachable; raises ValueError otherwise"""
    N = U.shape[0]
    if not np.allclose(U @ U.conj().T, np.eye(N), atol=1e-12):
        raise ValueError(f"{name}: not unitary")
    ring.from_complex(U, 1e-9, fix_phase=False)  # raises ValueError
    if not is_reachable(U):
        raise ValueError(f"{name}: not exactly reachable (det = omega^{det_omega_power(U)})")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--synthetiq', type=str, default=os.path.expanduser('~/research/synthetiq'),
                        help='Synthetiq checkout (its data/input holds the copied specs)')
    parser.add_argument('--out', type=str, default='data/targets')
    parser.add_argument('--check', action='store_true', help='only verify the existing files')
    args = parser.parse_args()
    src = os.path.join(args.synthetiq, 'data', 'input')

    # Synthetiq's .txt specs are little-endian (qiskit's order)
    _, ccx = load_matrix_from_file(os.path.join(src, '64/comparison/ccx.txt'))
    assert np.array_equal(ccx, Operator(circuit(3, ('ccx', 0, 1, 2))).data), "Synthetiq ccx.txt is not little-endian"

    for n, targets in NAMED.items():
        for name, qc, V, desc in targets:
            U = big_endian(qc)
            same = relative_phase_of(U, V) if name in RELATIVE_PHASE else np.allclose(U, V, atol=1e-12)
            if not same:
                raise SystemExit(f"{name}: qiskit circuit and independent construction differ")
            check(name, U)
            U = snap(U)
            filename = os.path.join(args.out, f"{n}qubit", f"{name}.txt")
            if not args.check:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                save_matrix_to_file(U, filename, name)
            n_file, U_file = load_matrix_from_file(filename)
            assert n_file == n and np.array_equal(U_file, U), f"{filename} does not match {name}"
            print(f"{filename}: det = omega^{det_omega_power(U)}; {desc}")

    for x, dest in SYNTHETIQ:
        filename = os.path.join(args.out, dest)
        if not args.check:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            shutil.copyfile(os.path.join(src, x), filename)
        n, U = load_matrix_from_file(filename)
        check(dest, U)
        print(f"{filename}: det = omega^{det_omega_power(U)}; Synthetiq data/input/{x}")
