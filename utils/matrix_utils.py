import scipy
from scipy.linalg import schur, expm
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


def phase_align(U: NDArray) -> NDArray:
    """Aligns the global phase of a unitary so that
       phase_align(U) == phase_align(W) if U=e^(i*theta)W"""
    N = U.shape[0]
    mu = (1/(N**2)) * np.sum(U ** 2)
    if mu == 0.0:
        mu = 1.0
    mu_norm = mu / np.abs(mu)
    mu_half = mu_norm * np.exp(-1j*np.angle(mu_norm)/2)
    mu_conj = np.conj(mu_half)
    W = mu_conj * U
    if np.real(W[0][0]) < 0:
        W = np.exp(1j*np.pi) * W
    return W


def hash_unitary(unitary: NDArray, tolerance: float = 0.001) -> int:
    """Creates fixed-length representation of unitary operator"""
    return hash(tuple(np.round(phase_align(unitary).flatten() / tolerance)))


def unitary_distance(U: NDArray, C: NDArray) -> float:
    """Computes the distance between two unitaries"""
    # from paper 'Synthetiq: Fast and Versatile Quantum Circuit Synthesis'
    M = np.ones(U.shape, dtype=np.complex128)
    tr_cu = np.trace(np.matmul(invert_unitary(M * C), M * U))
    if tr_cu == 0.: tr_cu = 1.
    num = np.linalg.norm(M * U - (tr_cu / np.abs(tr_cu)) * M * C)
    d_sc = num / np.sqrt(np.linalg.norm(M))
    return d_sc


def random_unitary(dim: int) -> NDArray:
    """Generates a randomly distributed set of unitary matrices"""
    return scipy.stats.unitary_group.rvs(dim)


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


def perturb_unitary(Us: NDArray, epsilon: float) -> NDArray:
    """Adds a random perturbation to a unitary U to get a unitary V
    such that d(U,V) < epsilon"""
    N = Us.shape[0]
    n = Us.shape[1]
    X = np.random.rand(N, n, n) + np.random.rand(N, n, n) * 1j
    X_hat = np.transpose(X.conj(), axes=(0, 2, 1))
    H0 = X + X_hat
    H = 2 * epsilon * H0 * (1 / np.linalg.matrix_norm(H0))[:, None, None]
    W = expm(1j*H)
    Us_hat = Us @ W
    return Us_hat


def nerf_embedding(xs: NDArray, dim: int) -> NDArray:
    t = 2 ** np.arange(dim)
    ys = xs[:,:,None] * t[None,None,:]
    a, b, c = ys.shape
    return ys.reshape(a, b * c)


def unitaries_to_nnet_input(Us: NDArray, encoding: str = 'matrix', nerf_dim: int = 0) -> NDArray:
    """Converts a batch of unitaries to nnet input format"""
    inps_flat: NDArray
    match encoding:
        case 'matrix':
            for i, U in enumerate(Us):
                Us[i] = phase_align(U)
            u_flat = [x.flatten() for x in Us]
            u_final = [np.hstack([np.real(x), np.imag(x)]) for x in u_flat]
            inps_flat = np.vstack(u_final)
        case 'hurwitz':
            inps_flat = su_encode_to_features_np(Us)
        case 'quaternion':
            inps_flat = unitaries_to_quaternions(Us)
    if nerf_dim > 0:
        return nerf_embedding(inps_flat, nerf_dim)
    return inps_flat
