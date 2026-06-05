#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batched random perturbations of unitary matrices under Hilbert-Schmidt distance.

Guarantee: for each sample b,  0 <= d(U_b, U_tilde_b) <= epsilon_b,
where d is unitary_distance (Frobenius norm divided by sqrt(n), matching
matrix_utils.unitary_distance_batch).

Since right-multiplication by a unitary is an isometry,
  d(U, WU) = ||WU - U||_F / sqrt(n) = ||W - I||_F / sqrt(n) = d(I, W),
so the distance depends only on W, not on U itself.

Perturbation is applied as W_{ij}(alpha, phi) @ U, a Givens rotation in a
random 2D subspace (i, j) by angle alpha with phase phi:
  new_row_i =  cos(alpha) * U[i] - e^{-i*phi} sin(alpha) * U[j]
  new_row_j =  e^{i*phi}  sin(alpha) * U[i] + cos(alpha) * U[j]

The distance-to-angle mapping is closed-form (no eigendecomposition, no bisection):
  d = 2*sqrt(2) |sin(alpha/2)| / sqrt(n)   (normalized)
  alpha = 2 arcsin(d * sqrt(n) / (2*sqrt(2)))

Constraint: eps <= 2*sqrt(2)/sqrt(n) per sample.
"""

import math
import numpy as np
from utils.matrix_utils import unitary_distance_batch


# ----------------------------- public utilities -----------------------------

def is_unitary_batch(U_B):
    B, n, _ = U_B.shape
    G = np.matmul(U_B.conj().transpose(0, 2, 1), U_B)
    diff = G - np.eye(n, dtype=np.complex128)[None]
    return np.linalg.norm(diff.reshape(B, -1), axis=1)


# ----------------------------- internal helpers -----------------------------

def _givens_scale(n, normalized):
    """Scale factor s such that sin(alpha/2) = d * s for a Givens rotation.

    ||I - W_ij||_F = 2*sqrt(2) |sin(alpha/2)|
    d_normalized    = ||I - W_ij||_F / sqrt(n) = 2*sqrt(2) |sin(alpha/2)| / sqrt(n)
    => s = sqrt(n) / (2*sqrt(2))
    """
    return math.sqrt(n) / (2.0 * math.sqrt(2.0)) if normalized else 1.0 / (2.0 * math.sqrt(2.0))


def _givens_d_to_alpha(d, n, normalized):
    """Closed-form distance -> Givens angle: alpha = 2 arcsin(d * s)."""
    return 2.0 * np.arcsin(np.clip(np.asarray(d, dtype=np.float64) * _givens_scale(n, normalized), 0.0, 1.0))


def _givens_alpha_to_d(alpha, n, normalized):
    """Givens angle -> distance: d = |sin(alpha/2)| / s."""
    return np.abs(np.sin(np.asarray(alpha, dtype=np.float64) / 2.0)) / _givens_scale(n, normalized)


# --------------------- main perturbation (batched) ---------------------

def perturb_unitary_givens_batch(U_B,
                                 epsilon,
                                 *,
                                 normalized=True,
                                 rng=None,
                                 uniform_in="distance",
                                 return_info=False):
    """Randomly perturb each U_b by a Givens rotation in a random 2D subspace.

    Guarantees d in [0, eps] using unitary_distance convention (/ sqrt(n)).
    No eigendecomposition, no bisection — O(n) application cost per sample.

    Args:
      U_B        : [B, n, n] complex128 unitaries.  n >= 2.
      epsilon    : scalar or [B] per-sample eps.
      normalized : use normalized distance (/ sqrt(n)) if True.
      rng        : numpy.random.Generator (optional).
      uniform_in : 'distance' -> d_target ~ Uniform(0, eps);
                   't'        -> alpha ~ Uniform(0, alpha_max).
      return_info: if True, also return dict with 'alpha', 'phi', 'i', 'j', 'd'.

    Returns:
      U_tilde_B  : [B, n, n] complex128 perturbed unitaries.
      (optional) dict with keys 'alpha', 'phi', 'i', 'j', 'd'.
    """
    U_B = np.asarray(U_B, dtype=np.complex128)
    assert U_B.ndim == 3 and U_B.shape[1] == U_B.shape[2], "U_B must be [B, n, n]"
    B, n, _ = U_B.shape
    assert n >= 2, "Givens rotation requires n >= 2"

    rng = np.random.default_rng() if rng is None else rng

    eps_arr = np.asarray(epsilon, dtype=np.float64)
    if eps_arr.ndim == 0:
        eps_arr = np.full((B,), float(eps_arr))
    else:
        assert eps_arr.shape == (B,), "epsilon must be scalar or shape [B]"

    d_max = 1.0 / _givens_scale(n, normalized)  # 2*sqrt(2)/sqrt(n) or 2*sqrt(2)
    if np.any(eps_arr > d_max + 1e-10):
        raise ValueError(
            f"epsilon {eps_arr.max():.6g} exceeds the maximum distance reachable by a "
            f"single Givens rotation ({d_max:.6g} for n={n}, "
            f"{'normalized' if normalized else 'unnormalized'})."
        )

    if uniform_in == "distance":
        d_target = rng.uniform(0.0, eps_arr)
        alpha_B = _givens_d_to_alpha(d_target, n, normalized)
    elif uniform_in == "t":
        alpha_max = _givens_d_to_alpha(eps_arr, n, normalized)
        alpha_B = rng.uniform(0.0, alpha_max)
    else:
        raise ValueError("uniform_in must be 'distance' or 't'.")

    phi_B = rng.uniform(0.0, 2.0 * math.pi, size=B)
    i_B = rng.integers(0, n, size=B)
    offsets = rng.integers(1, n, size=B)  # in [1, n-1], guarantees j != i
    j_B = (i_B + offsets) % n

    cos_a = np.cos(alpha_B)
    sin_a = np.sin(alpha_B)
    e_phi = np.exp(1j * phi_B)

    idx = np.arange(B)
    rows_i = U_B[idx, i_B, :]   # read before any writes
    rows_j = U_B[idx, j_B, :]

    U_tilde = U_B.copy()
    U_tilde[idx, i_B, :] = cos_a[:, None] * rows_i - (np.conj(e_phi) * sin_a)[:, None] * rows_j
    U_tilde[idx, j_B, :] = (e_phi * sin_a)[:, None] * rows_i + cos_a[:, None] * rows_j

    if return_info:
        d_actual = _givens_alpha_to_d(alpha_B, n, normalized)
        return U_tilde, {"alpha": alpha_B, "phi": phi_B, "i": i_B, "j": j_B, "d": d_actual}
    return U_tilde


# ----------------------------- testing -----------------------------

def _random_unitary_batched_np(B, n, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((B, n, n)) + 1j * rng.standard_normal((B, n, n))
    Q, R = np.linalg.qr(Z)
    diag = np.diagonal(R, axis1=1, axis2=2)
    phase = diag / np.clip(np.abs(diag), 1e-30, None)
    idx = np.arange(n)
    Dphase = np.zeros((B, n, n), dtype=np.complex128)
    Dphase[:, idx, idx] = np.conj(phase)
    Q = Q @ Dphase
    lam0 = 2 * math.pi * rng.random((B, n))
    D = np.zeros((B, n, n), dtype=np.complex128)
    D[:, idx, idx] = np.exp(1j * lam0)
    return Q @ D


def _sanity_all():
    import time
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(123)

    B, n = 64, 4
    U_B = _random_unitary_batched_np(B, n, seed=42)
    eps = 5e-2
    U_tilde, _ = perturb_unitary_givens_batch(
        U_B, eps, normalized=True, rng=rng, uniform_in="distance", return_info=True
    )
    d = unitary_distance_batch(U_B, U_tilde)
    print("=== Givens: random U(4), eps=5e-2 ===")
    print(f"d min/mean/max = {d.min():.3e} / {d.mean():.3e} / {d.max():.3e}  (<= eps? {np.all(d <= eps + 1e-12)})")
    uni = is_unitary_batch(U_tilde)
    print(f"unitarity ||U^H U - I||_F mean/max = {np.mean(uni):.3e} / {np.max(uni):.3e}")

    B2, n2 = 2000, 2
    U_I = np.tile(np.eye(n2, dtype=np.complex128), (B2, 1, 1))
    eps2 = 1e-4
    U_tilde2, _ = perturb_unitary_givens_batch(
        U_I, eps2, normalized=True, rng=rng, uniform_in="distance", return_info=True
    )
    d2 = unitary_distance_batch(U_I, U_tilde2)
    print(f"\n=== Givens: U=I_2 stress, eps=1e-4 ===")
    print(f"d min/mean/max = {d2.min():.3e} / {d2.mean():.3e} / {d2.max():.3e}  (<= eps? {np.all(d2 <= eps2 + 1e-12)})")
    print(f"violations: {np.sum(d2 > eps2 + 1e-12)} (should be 0)")
    uni2 = is_unitary_batch(U_tilde2)
    print(f"unitarity ||U^H U - I||_F mean/max = {np.mean(uni2):.3e} / {np.max(uni2):.3e}")


if __name__ == "__main__":
    _sanity_all()
