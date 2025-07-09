import torch
import scipy
import numpy as np
from typing import List
from deepxube.nnet.nnet_utils import get_device


# getting gpu device
t_device, _, _ = get_device()


# identity matrix on one qubit
I = torch.eye(2, dtype=torch.complex64, device=t_device)
# 'zero' project for one qubit
P0 = torch.tensor([[1, 0], [0, 0]], dtype=torch.complex64, device=t_device)
# 'one' projector for one qubit
P1 = torch.tensor([[0, 0], [0, 1]], dtype=torch.complex64, device=t_device)


def tensor_product(mats: List[torch.Tensor]) -> torch.Tensor:
    """
    Computes the tensor product (Kronecker product) of a list of matrices

    @param mats: List of numpy complex matrices
    @returns: Numpy complex matrix result of tensor product
    """
    current = torch.tensor(1, dtype=torch.complex64, device=t_device)
    for mat in mats:
        current = torch.kron(current, mat)
    return current


def phase_align(unitary: torch.Tensor) -> torch.Tensor:
    """
    Aligns the global phase of a unitary so that the top left
    element has complex phase 0

    @param unitary: n x n unitary matrix to align
    @returns: n x n re-aligned unitary matrix
    """
    phs = torch.angle(unitary[0][0])
    return torch.exp(-1j * phs) * unitary


def hash_unitary(unitary: torch.Tensor, tolerance: float = 1e-6) -> int:
    """
    Creates fixed-length representation of unitary operator

    @param unitary: n x n unitary matrix
    @param tolerance: Level of discretization for matrix values
    @returns: Integer uniquely representing matrix up to tolerance
    """
    t = phase_align(unitary).flatten() / tolerance
    rounded_real = torch.round(t.real)
    rounded_imag = torch.round(t.imag)
    t_rounded = torch.complex(rounded_real, rounded_imag)
    return hash(tuple(t_rounded))


def unitary_to_nnet_input(unitary: torch.Tensor) -> torch.Tensor:
    """
    Converts a complex-valued unitary matrix into real-valued
    flat numpy arrays that can be converted to tensors easily

    @param unitary: Unitary matrix to convert
    @returns: Numpy vector of real and imaginary values of matrix
    """
    unitary_aligned = phase_align(unitary)
    unitary_flat = unitary_aligned.flatten()
    unitary_real = torch.real(unitary_flat)
    unitary_imag = torch.imag(unitary_flat)
    unitary_nnet = torch.hstack((unitary_real, unitary_imag)).float()
    return unitary_nnet


def unitary_distance(U: torch.Tensor, C: torch.Tensor) -> float:
    """
    Computes the distance between two matrices using the operator norm

    @param mat1: First unitary
    @param mat2: Second unitary
    @returns: Distance as floating point number
    """
    
    # from paper 'Synthetiq: Fast and Versatile Quantum Circuit Synthesis'
    M = torch.ones(U.shape, dtype=torch.complex64, device=t_device)
    tr_cu = torch.trace(torch.matmul(invert_unitary(M * C), M * U))
    if tr_cu == 0.: tr_cu = torch.tensor(1., dtype=torch.complex64, device=t_device)
    num = torch.norm(M * U - (tr_cu / torch.abs(tr_cu)) * M * C)
    d_sc = num / torch.sqrt(torch.norm(M))
    return d_sc


def random_unitary(dim: int) -> np.ndarray[np.complex128]:
    """
    Generates a randomly distributed set of unitary matrices

    @param dim: Dimension of unitary group to generate
    @return: Numpy complex array of unitary matrix
    """
    return scipy.stats.unitary_group.rvs(dim)


def invert_unitary(unitary: torch.Tensor) -> torch.Tensor:
    """
    Inverts a unitary matrix

    @param unitary: Numpy complex matrix to invert
    @returns: Inverted numpy complex matrix
    """
    return torch.conj(unitary.T)