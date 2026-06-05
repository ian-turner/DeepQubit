#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict batched random perturbations of unitary matrices under Hilbert–Schmidt distance.

Guarantee (numerical): for each sample b,
    0 <= d(U_b, U_tilde_b) <= epsilon_b
with d = ||U - V||_F / sqrt(2n) if normalized=True, else plain Frobenius.

We sample a random Hermitian direction H_b (||H_b||_F = 1), pick a target distance
in [0, epsilon_b] (or a t in [0, t_max]), build W_b = exp(i t H_b), and set U_tilde_b = W_b U_b.

To enforce the bound strictly at very small epsilons, we:
  - measure the actual matrix distance after construction,
  - if d > eps, clamp 't' by bisection on the measured distance until d <= eps - margin,
    where margin = max(1e-15, 1e-6 * eps),
  - finally apply a tiny multiplicative backoff on t and re-measure.

Includes tests, notably the U=I_2 case with epsilon=1e-4.
"""

import math
import numpy as np

# ----------------------------- utilities -----------------------------

def hs_distance_batch(U_B, V_B, *, normalized=True):
    diff = U_B - V_B
    fro2 = np.sum(np.abs(diff)**2, axis=(1, 2))
    fro = np.sqrt(fro2)
    if normalized:
        n = U_B.shape[1]
        fro /= math.sqrt(2*n)
    return fro

def is_unitary_batch(U_B):
    B, n, _ = U_B.shape
    G = np.matmul(U_B.conj().transpose(0, 2, 1), U_B)
    diff = G - np.eye(n, dtype=np.complex128)[None, :, :]
    return np.linalg.norm(diff.reshape(B, -1), axis=1)

def _rand_hermitian_batch(B, n, rng):
    X = rng.standard_normal((B, n, n)) + 1j * rng.standard_normal((B, n, n))
    H = (X + np.transpose(np.conj(X), (0, 2, 1))) / 2.0
    norms = np.linalg.norm(H.reshape(B, -1), axis=1) + 1e-300
    H /= norms[:, None, None]
    return H

def _d_from_eigs_I(lam, t, normalized, n):
    # ||I - exp(i t H)||_F^2 = 2 sum_k (1 - cos(t*lam_k))
    s = np.sum(1.0 - np.cos(t * lam))
    fro = math.sqrt(2.0 * s)
    return fro / math.sqrt(2*n) if normalized else fro

def _solve_t_for_distance(lam, target_d, normalized, n):
    # Largest t in [0, π/λ_max] with d(I, exp(i t H)) <= target_d (monotone region)
    if target_d <= 0.0:
        return 0.0
    lam_abs_max = float(np.max(np.abs(lam))) + 1e-300
    t_hi = math.pi / lam_abs_max
    d_hi = _d_from_eigs_I(lam, t_hi, normalized, n)
    if d_hi <= target_d:
        return t_hi
    t_lo = 0.0
    for _ in range(70):  # a bit tighter than before
        t_mid = 0.5 * (t_lo + t_hi)
        d_mid = _d_from_eigs_I(lam, t_mid, normalized, n)
        if d_mid <= target_d:
            t_lo = t_mid
        else:
            t_hi = t_mid
    return t_lo

def _exp_iH_from_eigh(lam, V, t):
    return (V * np.exp(1j * (t * lam))[None, :]) @ V.conj().T

def _measured_dist(U, lam, V, t, normalized):
    n = U.shape[0]
    W = _exp_iH_from_eigh(lam, V, t)
    d = np.linalg.norm(U - W @ U, 'fro')
    if normalized:
        d /= math.sqrt(2*n)
    return float(d)

def _strict_clamp_t(U, lam, V, t_init, eps, normalized, *, margin):
    """
    Ensure measured d(U, exp(i t H) U) <= eps - margin via bisection on t in [0, t_init].
    Then apply a tiny multiplicative backoff and re-check.
    """
    if eps <= 0.0:
        return 0.0

    # If already within safe margin, keep t
    d0 = _measured_dist(U, lam, V, t_init, normalized)
    if d0 <= eps - margin:
        return t_init

    # Bisection using measured distance
    lo, hi = 0.0, t_init
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        dm = _measured_dist(U, lam, V, mid, normalized)
        if dm <= eps - margin:
            lo = mid
        else:
            hi = mid
    t = lo

    # Final tiny backoff to avoid equality drift and re-check
    t *= 0.999999  # 1 - 1e-6
    d_final = _measured_dist(U, lam, V, t, normalized)
    if d_final > eps:
        # One more conservative shrink if somehow still above
        t *= 0.999
    return t

# --------------------- main perturbation (batched) ---------------------

def perturb_unitary_random_batch_strict(U_B,
                                        epsilon,
                                        *,
                                        normalized=True,
                                        rng=None,
                                        directions=None,
                                        uniform_in="distance",
                                        return_info=False):
    """
    Randomly perturb each unitary U_b within the ε-neighborhood (0 ≤ d ≤ ε), with strict clamp.

    Args:
      U_B        : [B, n, n] complex128 array of unitaries.
      epsilon    : scalar ε or [B] array of per-sample ε.
      normalized : use normalized HS distance if True; else Frobenius.
      rng        : numpy.random.Generator (optional).
      directions : optional [B, n, n] Hermitian directions (will be symmetrized & normalized).
      uniform_in : 'distance' → sample d ~ Uniform(0, ε),
                   't'        → sample t ~ Uniform(0, t_max).
      return_info: if True, return dict with per-sample 't_raw','t','d','H'.

    Returns:
      U_tilde_B  : [B, n, n] complex128.
      (optional) info dict.
    """
    U_B = np.asarray(U_B, dtype=np.complex128)
    assert U_B.ndim == 3 and U_B.shape[1] == U_B.shape[2], "U_B must be [B, n, n]"
    B, n, _ = U_B.shape

    rng = np.random.default_rng() if rng is None else rng

    eps_arr = np.asarray(epsilon, dtype=np.float64)
    if eps_arr.ndim == 0:
        eps_arr = np.full((B,), float(eps_arr))
    else:
        assert eps_arr.shape == (B,), "epsilon must be scalar or shape [B]"

    if directions is None:
        H_B = _rand_hermitian_batch(B, n, rng)
    else:
        H_B = np.asarray(directions, dtype=np.complex128)
        assert H_B.shape == (B, n, n)
        H_B = (H_B + np.transpose(np.conj(H_B), (0, 2, 1))) / 2.0
        norms = np.linalg.norm(H_B.reshape(B, -1), axis=1) + 1e-300
        H_B /= norms[:, None, None]

    U_tilde = np.empty_like(U_B)
    t_raw = np.empty(B, dtype=np.float64)
    t_used = np.empty(B, dtype=np.float64)
    d_used = np.empty(B, dtype=np.float64)

    for b in range(B):
        lam, V = np.linalg.eigh(H_B[b])  # lam real
        eps_b = float(eps_arr[b])

        # Target selection
        t_raw_b = None
        if uniform_in == "distance":
            d_target = rng.uniform(0.0, eps_b)
            t_raw_b = _solve_t_for_distance(lam, d_target, normalized, n)
        elif uniform_in == "t":
            t_max = _solve_t_for_distance(lam, eps_b, normalized, n)
            t_raw_b = rng.uniform(0.0, t_max)
        else:
            raise ValueError("uniform_in must be 'distance' or 't'.")

        # Strict clamp with safety margin
        margin = max(1e-15, 1e-7 * eps_b)
        t_clamped = _strict_clamp_t(U_B[b], lam, V, t_raw_b, eps_b, normalized, margin=margin)

        W = _exp_iH_from_eigh(lam, V, t_clamped)
        U_out = W @ U_B[b]
        d_out = float(np.linalg.norm(U_B[b] - U_out, 'fro') / (math.sqrt(2*n) if normalized else 1.0))

        U_tilde[b] = U_out
        t_raw[b] = t_raw_b
        t_used[b] = t_clamped
        d_used[b] = d_out

    if return_info:
        return U_tilde, {"t_raw": t_raw, "t": t_used, "d": d_used, "H": H_B}
    return U_tilde

# ----------------------------- testing -----------------------------

def _random_unitary_batched_np(B, n, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((B, n, n)) + 1j * rng.standard_normal((B, n, n))
    Q, R = np.linalg.qr(Z)  # [B, n, n]
    diag = np.diagonal(R, axis1=1, axis2=2)
    phase = diag / np.clip(np.abs(diag), 1e-30, None)
    Dphase = np.zeros((B, n, n), dtype=np.complex128)
    idx = np.arange(n)
    Dphase[:, idx, idx] = np.conj(phase)
    Q = Q @ Dphase
    lam0 = 2 * math.pi * rng.random((B, n))
    D = np.zeros((B, n, n), dtype=np.complex128)
    D[:, idx, idx] = np.exp(1j * lam0)
    return Q @ D

def _sanity_all():
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(123)

    # 1) Random U(4) batch, ε=5e-2
    B, n = 64, 4
    U_B = _random_unitary_batched_np(B, n, seed=42)
    eps = 5e-2
    U_tilde, info = perturb_unitary_random_batch_strict(
        U_B, eps, normalized=True, rng=rng, uniform_in="distance", return_info=True
    )
    d = hs_distance_batch(U_B, U_tilde, normalized=True)
    print("=== Random U(4) batch ===")
    print(f"d min/mean/max = {d.min():.3e} / {d.mean():.3e} / {d.max():.3e}  (<= ε? {np.all(d <= eps + 1e-12)})")
    uni = is_unitary_batch(U_tilde)
    print(f"unitarity ||U^H U - I||_F mean/max = {np.mean(uni):.3e} / {np.max(uni):.3e}")

    # 2) Stress: U = I_2, tiny ε = 1e-4
    B2, n2 = 2000, 2
    U_I = np.tile(np.eye(n2, dtype=np.complex128), (B2, 1, 1))
    eps2 = 1e-4
    U_tilde2, info2 = perturb_unitary_random_batch_strict(
        U_I, eps2, normalized=True, rng=rng, uniform_in="distance", return_info=True
    )
    d2 = hs_distance_batch(U_I, U_tilde2, normalized=True)
    print("\n=== U = I_2 stress (ε=1e-4) ===")
    print(f"d min/mean/max = {d2.min():.3e} / {d2.mean():.3e} / {d2.max():.3e}  (<= ε? {np.all(d2 <= eps2 + 1e-12)})")
    # Show how close we run to the boundary
    over = np.sum(d2 > eps2 + 1e-12)
    print(f"violations over ε: {over} (should be 0)")
    uni2 = is_unitary_batch(U_tilde2)
    print(f"unitarity ||U^H U - I||_F mean/max = {np.mean(uni2):.3e} / {np.max(uni2):.3e}")

if __name__ == "__main__":
    _sanity_all()

