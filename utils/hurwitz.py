#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batched Hurwitz (complex Givens) parameterization for SU(n) in NumPy (complex).

Encode:
  U_B [B, n, n] (complex128) -> theta_B [B, k], phi_B [B, k], lam_B [B, n]
  where k = n(n-1)/2 and sum(lam_B[b]) ≡ 0 (mod 2π) for every batch element.

Decode:
  (theta_B [B, k], phi_B [B, k], lam_head_B [B, n-1]) -> U_B [B, n, n]

Also included:
  su_pack_features/su_unpack_features for [B, n^2-1] vectors up to global phase.
"""

import math
import functools
import numpy as np

TWO_PI = 2.0 * math.pi
PI_OVER_2 = 0.5 * math.pi

# -------------------------- utils --------------------------

def _wrap_to_2pi(x):
    return np.mod(x, TWO_PI)

def _project_to_su_batch(U_B):
    """Remove global phase per batch: U <- U * exp(-i * arg(det(U))/n)."""
    B, n, _ = U_B.shape
    det = np.linalg.det(U_B)
    phi = np.angle(det) / n                   # [B]
    return U_B * np.exp(-1j * phi)[:, None, None]

def _left_apply_givens_inplace(A, i, j, c, s, eiph, conj=False):
    """
    Apply G (conj=False) or G^H (conj=True) to rows (i, j) of A for all batches.
    G = [[ c,        e^{iφ}s],
         [-e^{-iφ}s,  c     ]]
    c, s, eiph are shape [B].
    """
    Ai = A[:, i, :].copy()
    Aj = A[:, j, :].copy()
    ph = -eiph if conj else eiph
    A[:, i, :] = c[:, None] * Ai + (ph * s)[:, None] * Aj
    A[:, j, :] = (-np.conj(ph) * s)[:, None] * Ai + c[:, None] * Aj

@functools.lru_cache(maxsize=16)
def _givens_pairs(n):
    """Ordered (i, j) pairs for Givens QR reduction of an n×n matrix (column-major, bottom-up)."""
    return [(i, j) for i in range(n - 1) for j in range(n - 1, i, -1)]

# -------------------------- ENCODE --------------------------

def su_encode_batched_np(U_B, eps=1e-30, tiny=1e-300):
    """
    Encode a batch of unitaries (complex128) into Hurwitz angles for SU(n).
    Inputs:
      U_B: [B, n, n] complex128 (if 2D, treated as B=1)
    Returns:
      theta_B: [B, k], phi_B: [B, k], lam_B: [B, n] with sum(lam_B[b]) ≡ 0 (mod 2π)
    """
    U_B = np.asarray(U_B)
    if U_B.ndim == 2:
        U_B = U_B[None, ...]
    assert U_B.ndim == 3 and U_B.shape[1] == U_B.shape[2]
    B, n, _ = U_B.shape
    k = n * (n - 1) // 2

    A = _project_to_su_batch(U_B).astype(np.complex128, copy=True)
    theta_B = np.zeros((B, k), dtype=np.float64)
    phi_B   = np.zeros((B, k), dtype=np.float64)

    for idx, (i, j) in enumerate(_givens_pairs(n)):
        a = A[:, i, i]
        b = A[:, j, i]
        aa, bb = np.abs(a)**2, np.abs(b)**2
        r = np.sqrt(aa + bb + eps)

        theta = np.zeros(B, dtype=np.float64)
        phi   = np.zeros(B, dtype=np.float64)

        m_b_big   = bb >= tiny
        m_a_small = aa < tiny

        m_general = m_b_big & ~m_a_small
        if np.any(m_general):
            ag, bg, rg = a[m_general], b[m_general], r[m_general]
            phi[m_general]   = _wrap_to_2pi(np.angle(ag) - np.angle(bg))
            c = np.clip(np.abs(ag) / rg, 0.0, 1.0)
            s = np.clip(np.abs(bg) / rg, 0.0, 1.0)
            theta[m_general] = np.arctan2(s, c)

        # a≈0, b≠0  ->  θ = π/2, φ = -arg(b)
        m_a_zero = m_b_big & m_a_small
        if np.any(m_a_zero):
            theta[m_a_zero] = PI_OVER_2
            phi[m_a_zero]   = _wrap_to_2pi(-np.angle(b[m_a_zero]))

        # b≈0  ->  θ=0 (already zero), φ arbitrary (0)
        theta_B[:, idx] = theta
        phi_B[:, idx]   = phi
        _left_apply_givens_inplace(A, i, j, np.cos(theta), np.sin(theta), np.exp(1j * phi))

    # Only the first n-1 diagonal phases are independent; enforce sum ≡ 0 for the last.
    lam_head = _wrap_to_2pi(np.angle(np.diagonal(A, axis1=1, axis2=2)[:, :n-1]))
    lam_last = _wrap_to_2pi(-np.sum(lam_head, axis=1, keepdims=True))
    lam_B    = np.concatenate([lam_head, lam_last], axis=1)
    return theta_B, phi_B, lam_B

# -------------------------- DECODE --------------------------

def su_decode_batched_np(theta_B, phi_B, lam_head_B, n):
    """
    Decode a batch to SU(n) (complex128).
    Inputs:
      theta_B: [B, k], phi_B: [B, k], lam_head_B: [B, n-1]
    Returns:
      U_B: [B, n, n] complex128, det ≈ 1, round-trip ~1e-15 in float64.
    """
    theta_B    = np.asarray(theta_B,    dtype=np.float64)
    phi_B      = np.asarray(phi_B,      dtype=np.float64)
    lam_head_B = np.asarray(lam_head_B, dtype=np.float64)
    B, k = theta_B.shape
    assert phi_B.shape == (B, k) and lam_head_B.shape == (B, n - 1)

    lam_last = _wrap_to_2pi(-np.sum(lam_head_B, axis=1, keepdims=True))
    lam = np.concatenate([lam_head_B, lam_last], axis=1)

    U = np.zeros((B, n, n), dtype=np.complex128)
    diag_idx = np.arange(n)
    U[:, diag_idx, diag_idx] = np.exp(1j * lam)

    # Replay G^H in reverse order
    for angle_col, (i, j) in zip(range(k - 1, -1, -1), reversed(_givens_pairs(n))):
        th = theta_B[:, angle_col]
        ph = phi_B[:,  angle_col]
        _left_apply_givens_inplace(U, i, j, np.cos(th), np.sin(th), np.exp(1j * ph), conj=True)

    return U

# -------------------------- pack/unpack to n^2-1 features --------------------------

def su_pack_features(theta_B, phi_B, lam_B):
    """
    (θ, φ, λ) -> [B, n^2-1] by concatenating θ, φ, and the first n-1 phases.
    """
    B, k = theta_B.shape
    n = lam_B.shape[1]
    feats = np.concatenate([theta_B, phi_B, lam_B[:, :n-1]], axis=1)
    assert feats.shape[1] == n*n - 1
    return feats

def su_unpack_features(feats, n):
    """
    [B, n^2-1] -> (theta_B, phi_B, lam_head_B).
    """
    B, D = feats.shape
    k = n * (n - 1) // 2
    assert D == 2*k + (n-1)
    return feats[:, :k], feats[:, k:2*k], feats[:, 2*k:]

def su_encode_to_features_np(U_B):
    th, ph, lam = su_encode_batched_np(U_B)
    return su_pack_features(th, ph, lam)

def su_decode_from_features_np(feats, n=None):
    """
    [B, n^2-1] -> [B, n, n] SU(n). n is inferred from feature dimension if not given.
    """
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats[None, :]
    if n is None:
        n = int(round(math.sqrt(feats.shape[1] + 1)))
        assert n * n - 1 == feats.shape[1], \
            f"Feature dim {feats.shape[1]} is not n²-1 for any integer n"
    th, ph, lamh = su_unpack_features(feats, n)
    return su_decode_batched_np(th, ph, lamh, n)

# -------------------------- quick test --------------------------

def _random_unitary_batched_np(B, n, seed=0):
    """Generate a random U(n) batch, shape [B, n, n]."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((B, n, n)) + 1j * rng.standard_normal((B, n, n))
    Q, R = np.linalg.qr(Z)
    diag = np.diagonal(R, axis1=1, axis2=2)
    phase = diag / np.clip(np.abs(diag), 1e-30, None)
    Dp = np.zeros((B, n, n), dtype=np.complex128)
    idx = np.arange(n)
    Dp[:, idx, idx] = np.conj(phase)
    Q = Q @ Dp
    lam0 = TWO_PI * rng.random((B, n))
    D = np.zeros((B, n, n), dtype=np.complex128)
    D[:, idx, idx] = np.exp(1j * lam0)
    return Q @ D

def _rel_err(U, V):
    return float(np.linalg.norm(U - V) / np.linalg.norm(V))

if __name__ == "__main__":
    B, n = 8, 4
    U = _random_unitary_batched_np(B, n, seed=123)
    U_su = _project_to_su_batch(U)

    theta, phi, lam = su_encode_batched_np(U)
    U_rec = su_decode_batched_np(theta, phi, lam[:, :n-1], n)

    rel = _rel_err(U_rec, U_su)
    dets = np.linalg.det(U_rec)
    print(f"[B={B}, n={n}] rel. error: {rel:.3e}")
    print(f"[B={B}, n={n}] |det| mean/min/max: {np.mean(np.abs(dets)):.6f} / "
          f"{np.min(np.abs(dets)):.6f} / {np.max(np.abs(dets)):.6f}")

    feats = su_encode_to_features_np(U)            # [B, n^2-1]
    U_rt  = su_decode_from_features_np(feats)      # n inferred
    print(f"[features] rel. error: {_rel_err(U_rt, U_su):.3e}, feats.shape={feats.shape}")
