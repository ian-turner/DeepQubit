import numpy as np
from numpy import trace, log, exp, diag, diagonal
from numpy.linalg import det, eig, inv, svd
from numpy.typing import NDArray
from typing import List
from qiskit import qasm2
from qiskit.quantum_info import Operator
from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.circuit.library import U3Gate
from utils.hurwitz import su_encode_to_features_np


# identity matrix on one qubit
I = np.eye(2, dtype=np.complex128)
# 'zero' project for one qubit
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
# 'one' projector for one qubit
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)


def load_matrix_from_file(filename: str) -> NDArray:
    if filename.endswith('.txt'):
        num_qubits: int
        matrix: NDArray
        with open(filename, 'r') as f:
            lines = [x.strip() for x in list(f)]
            num_qubits = int(lines[1])
            N = 2**(num_qubits)
            matrix = np.zeros((N, N), dtype=np.complex128)
            for i in range(N):
                row = lines[2+i]
                cols = row.split(' ')
                for j, col in enumerate(cols):
                    left, right = col.split(',')
                    real = float(left[1:])
                    imag = float(right[:-1])
                    matrix[i][j] = real + imag*1j
        return num_qubits, matrix
    
    elif filename.endswith('.npy'):
        matrix = np.load(filename)
        num_qubits = int(np.log2(matrix.shape[0]))
        return num_qubits, matrix

    else:
        raise Exception('Invalid file format')


def save_matrix_to_file(matrix: NDArray, filename: str):
    num_qubits = int(np.log2(matrix.shape[0]))
    with open(filename, 'w') as f:
        f.write('matrix\n%s' % num_qubits)
        for row in matrix:
            row_str = '\n'
            for x in row:
                row_str += '(%s,%s) ' % (np.real(x), np.imag(x))
            f.write(row_str)


def tensor_product(mats: List[NDArray]) -> NDArray:
    """Computes the tensor product (Kronecker product) of a list of matrices"""
    current = 1
    for mat in mats:
        current = np.kron(current, mat)
    return current


def phase_align_batch(Us: NDArray, tol: float = 1e-6) -> NDArray:
    """Removes the global phase of each unitary in [B, N, N] by making its largest-magnitude
    entry real and positive. Ties within tol resolve to the first entry in row-major order,
    so the choice is stable under floating-point noise (Clifford+T matrices are full of exact
    ties and exact zeros, which made the old sum(U**2) / sign-of-U[0,0] rule ill-conditioned)."""
    flat = Us.reshape(len(Us), -1)
    mags = np.abs(flat)
    k = np.argmax(mags >= mags.max(axis=1, keepdims=True) - tol, axis=1)
    e = flat[np.arange(len(Us)), k]
    return Us * np.conj(e / np.abs(e))[:, None, None]


def phase_align(U: NDArray) -> NDArray:
    """Single-matrix version of phase_align_batch"""
    return phase_align_batch(U[None])[0]


def hash_unitary(unitary: NDArray, tolerance: float = 0.001) -> int:
    """Creates fixed-length representation of unitary operator"""
    return hash(tuple(np.round(phase_align(unitary).flatten() / tolerance)))


def unitary_distance(U: NDArray, C: NDArray) -> float:
    """Computes the distance between two unitaries"""
    # from paper 'Synthetiq: Fast and Versatile Quantum Circuit Synthesis'
    tr_cu = np.trace(np.matmul(invert_unitary(C), U))
    if tr_cu == 0.: tr_cu = 1.
    num = np.linalg.norm(U - (tr_cu / np.abs(tr_cu)) * C)
    return num / np.sqrt(U.shape[0])


def unitary_distance_batch(Us: NDArray, Cs: NDArray) -> NDArray:
    """Batched unitary_distance for arrays of shape [B, N, N]."""
    tr_cu = np.trace(np.matmul(np.conj(Cs).transpose(0, 2, 1), Us), axis1=1, axis2=2)
    tr_cu = np.where(tr_cu == 0., 1., tr_cu)
    phase = tr_cu / np.abs(tr_cu)
    diff = Us - phase[:, None, None] * Cs
    norms = np.linalg.norm(diff.reshape(len(Us), -1), axis=1)
    return norms / np.sqrt(Us.shape[1])


def invert_unitary(U: NDArray) -> NDArray:
    """Inverts a unitary matrix"""
    return U.conj().T


def phase_align_quaternion(Us: NDArray) -> NDArray:
    """Aligns phase so that quaternion encoding works properly"""
    theta1 = np.angle(Us[:, 0, 0])
    theta2 = np.angle(Us[:, 1, 1])
    theta_tilde = theta1 + theta2
    exp_theta = np.exp(-1j * theta_tilde / 2)
    return exp_theta[:, None, None] * Us


def unitaries_to_quaternions(Us: NDArray) -> NDArray[float]:
    """Converts U(2) -> SU(2) -> Quaternions based on representation in
    `Quantum logic gate synthesis as a Markov decision process` (Alam, 2023)"""
    assert Us.shape[1] == 2 # only U(2) is accepted
    Us = phase_align_quaternion(Us)
    a = np.real(Us[:, 0, 0]) # select real element of top-left entry
    b = np.imag(Us[:, 0, 0]) # imag element of top-left entry
    c = np.real(Us[:, 0, 1]) # real element of top-right entry
    d = np.imag(Us[:, 0, 1]) # imag element of top-right entry
    Q = np.array([a, b, c, d]).T # quaternion vectors for each U
    return Q


def quaternions_to_unitaries(Q: NDArray[float]) -> NDArray:
    """Converts quaternion representation back to unitary
    for debugging and testing quaternion encoding"""
    Us = np.zeros((Q.shape[0], 2, 2), dtype=np.complex64)
    Us[:, 0, 0] = Q[:, 0] + 1j * Q[:, 1]
    Us[:, 0, 1] = Q[:, 2] + 1j * Q[:, 3]
    Us[:, 1, 0] = -Q[:, 2] + 1j * Q[:, 3]
    Us[:, 1, 1] = Q[:, 0] - 1j * Q[:, 1]
    return Us


def nerf_embedding(xs: NDArray, dim: int) -> NDArray:
    t = 2 ** np.arange(dim)
    ys = xs[:,:,None] * t[None,None,:]
    _sin = np.sin(ys)
    _cos = np.cos(ys)
    ys = np.stack([_sin, _cos], axis=-1)
    a, b, c, _ = ys.shape
    ys = ys.reshape(a, b * c * 2)
    return ys


def encoding_size(encoding: str, num_qubits: int) -> int:
    """Feature size of an encoding, which may be several encodings joined by '+'"""
    sizes = {'matrix': 2 ** (2 * num_qubits + 1),
             'hurwitz': (2 ** num_qubits) ** 2 - 1,
             'quaternion': 4}
    return sum(sizes[enc] for enc in encoding.split('+'))


def unitaries_to_nnet_input(Us: NDArray, encoding: str = 'matrix', nerf_dim: int = 0) -> NDArray:
    """Converts a batch of unitaries to nnet input format.
    Encodings joined by '+' (e.g. 'hurwitz+quaternion') are concatenated feature-wise."""
    parts: List[NDArray] = []
    for enc in encoding.split('+'):
        match enc:
            case 'matrix':
                flat = phase_align_batch(Us).reshape(len(Us), -1)
                parts.append(np.hstack([np.real(flat), np.imag(flat)]))
            case 'hurwitz':
                parts.append(su_encode_to_features_np(Us))
            case 'quaternion':
                parts.append(unitaries_to_quaternions(Us))
    inps_flat = parts[0] if len(parts) == 1 else np.hstack(parts)
    if nerf_dim > 0:
        return nerf_embedding(inps_flat, nerf_dim)
    return inps_flat
