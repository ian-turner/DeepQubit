"""Exact integer arithmetic for matrices over D[omega] = Z[omega, 1/sqrt2], omega = exp(i*pi/4).

A matrix U is stored as int64 coefficients `c` of shape (..., N, N, 4) -- the coefficients of
1, omega, omega^2, omega^3 for each entry -- and an integer exponent k, meaning U = c / sqrt2**k.
Every exact Clifford+T unitary has such a representation; with k minimal and the global phase fixed
by `canonicalize` it is unique, so equality and hashing are exact array comparisons.

Products go through the companion form: each ring element z becomes the 4x4 integer matrix whose
column t holds the coefficients of z*omega^t. The companion form is a ring homomorphism and
conj(z) maps to the transpose, so matrix products and daggers are ordinary (block) matmuls. They are
done in float64 for BLAS speed, which is exact while all intermediate values stay below 2**53.

Row operations for the gates are pure int64 ops. Qubit 0 is the most significant bit of the basis
index (same big-endian convention as domains/qcircuit.py).

Coefficient size: the Galois automorphisms omega -> omega^j (j odd) commute with conjugation, so they map a
unitary to a unitary, and every coefficient is an average of embedded entries; hence |coefficient| <= sqrt2^k.
"""
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray

SQRT2: float = np.sqrt(2.0)

# coefficient vectors are rows; new = c @ M
# multiplication by omega: (a, b, c, d) -> (-d, a, b, c)
_ROT1: NDArray = np.array([[0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1],
                           [-1, 0, 0, 0]], dtype=np.int64)
# ROT[t] = multiplication by omega^t
ROT: NDArray = np.stack([np.linalg.matrix_power(_ROT1, t) for t in range(8)]).astype(np.int64)
# complex conjugation: (a, b, c, d) -> (a, -d, -c, -b)
CONJ: NDArray = np.array([[1, 0, 0, 0],
                          [0, 0, 0, -1],
                          [0, 0, -1, 0],
                          [0, -1, 0, 0]], dtype=np.int64)
# ROT[t] and CONJ are signed permutations: c @ M == c[..., IDX] * SGN, which numpy does faster than an int matmul
ROT_IDX: NDArray = np.abs(ROT).argmax(axis=1)                       # (8, 4)
ROT_SGN: NDArray = np.take_along_axis(ROT, ROT_IDX[:, None, :], axis=1)[:, 0, :]
CONJ_IDX: NDArray = np.abs(CONJ).argmax(axis=0)
CONJ_SGN: NDArray = CONJ[CONJ_IDX, np.arange(4)]
# companion columns as one (4, 16) float matrix: [i, 4*j + t] = ROT[t][i, j]  (see `companion`)
_ROT4F: NDArray = np.stack([ROT[t] for t in range(4)], axis=-1).reshape(4, 16).astype(np.float64)
# multiplication by sqrt2 = omega - omega^3: (a, b, c, d) -> (b - d, a + c, b + d, c - a)
MUL_SQRT2: NDArray = np.array([[0, 1, 0, -1],
                               [1, 0, 1, 0],
                               [0, 1, 0, 1],
                               [-1, 0, 1, 0]], dtype=np.int64)


# ---------------------------------------------------------------------------
# element-wise ring operations on coefficient arrays (..., 4)
# ---------------------------------------------------------------------------
def rotate(c: NDArray, t: int) -> NDArray:
    """Multiply every entry by omega^t"""
    t = t % 8
    return c[..., ROT_IDX[t]] * ROT_SGN[t]


def rotate_per_item(c: NDArray, ts: NDArray) -> NDArray:
    """Multiply the entries of batch item b by omega^ts[b]. c: (B, ..., 4), ts: (B,)"""
    ts = np.mod(ts, 8)
    out = c.copy()
    for t in np.unique(ts):
        if t != 0:
            idx = np.flatnonzero(ts == t)
            out[idx] = c[idx][..., ROT_IDX[t]] * ROT_SGN[t]
    return out


def conj(c: NDArray) -> NDArray:
    return c[..., CONJ_IDX] * CONJ_SGN


def dagger(c: NDArray) -> NDArray:
    """Conjugate transpose of (..., N, N, 4)"""
    return np.swapaxes(conj(c), -3, -2)


def mul_sqrt2(c: NDArray) -> NDArray:
    """(a, b, c, d) -> (b - d, a + c, b + d, c - a), i.e. c @ MUL_SQRT2"""
    a, b, cc, d = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    out = np.empty_like(c)
    np.subtract(b, d, out=out[..., 0])
    np.add(a, cc, out=out[..., 1])
    np.add(b, d, out=out[..., 2])
    np.subtract(cc, a, out=out[..., 3])
    return out


def divisible_by_sqrt2(c: NDArray) -> NDArray:
    """Per batch item: are all entries of (B, ..., 4) divisible by sqrt2 in Z[omega]? z * sqrt2 = (b - d, a + c, b + d,
    c - a) is divisible by 2 iff a = c and b = d (mod 2)"""
    odd = ((c[..., 0] ^ c[..., 2]) | (c[..., 1] ^ c[..., 3])) & 1
    return ~odd.reshape(c.shape[0], -1).any(axis=1)


def reduce_batch(c: NDArray, k: NDArray) -> Tuple[NDArray, NDArray]:
    """Divide out sqrt2 while every entry allows it (and k > 0), giving the minimal exponent. (B, N, N, 4), (B,)
    Each round only looks at the items that were still divisible in the previous one."""
    c = c.copy()
    k = k.copy()
    act = np.flatnonzero(k > 0)
    while act.size > 0:
        sub = c[act]
        div = divisible_by_sqrt2(sub)
        act = act[div]
        if act.size == 0:
            break
        c[act] = mul_sqrt2(sub[div]) >> 1  # exact: every entry is even
        k[act] -= 1
        act = act[k[act] > 0]
    return c, k


def canonicalize_batch(c: NDArray, k: NDArray) -> Tuple[NDArray, NDArray]:
    """Fix the global phase: multiply each matrix by the power of omega that makes its first nonzero entry
    (row-major) lexicographically largest (so the identity stays the identity). All 8 rotations of a nonzero
    entry are distinct, so this is unique. Assumes minimal k (see reduce_batch)."""
    B = c.shape[0]
    flat = c.reshape(B, -1, 4)
    first = (flat != 0).any(axis=-1).argmax(axis=1)  # index of first nonzero entry
    z = flat[np.arange(B), first]  # (B, 4)
    rots = z[:, ROT_IDX] * ROT_SGN  # (B, 8, 4): z * omega^t
    mask = np.ones((B, 8), dtype=bool)
    for j in range(4):
        vals = np.where(mask, rots[:, :, j], np.iinfo(np.int64).min)
        mask &= vals == vals.max(axis=1, keepdims=True)
    best = mask.argmax(axis=1)
    return rotate_per_item(c, best), k


def normalize_batch(c: NDArray, k: NDArray) -> Tuple[NDArray, NDArray]:
    return canonicalize_batch(*reduce_batch(c, k))


# ---------------------------------------------------------------------------
# companion form and products
# ---------------------------------------------------------------------------
def companion(c: NDArray) -> NDArray:
    """(..., N, N, 4) -> (..., 4N, 4N) float64 block matrix; block (i, j) is the companion matrix of entry (i, j)"""
    lead = c.shape[:-3]
    N = c.shape[-2]
    comp = (c.reshape(-1, 4).astype(np.float64) @ _ROT4F).reshape(lead + (N, N, 4, 4))  # [j, t] = coeff j of z*omega^t
    return np.moveaxis(comp, -2, -3).reshape(lead + (4 * N, 4 * N))


def from_companion(m: NDArray) -> NDArray:
    """Inverse of `companion` (reads column 0 of every block); exact after rounding"""
    lead = m.shape[:-2]
    N = m.shape[-1] // 4
    blocks = m.reshape(lead + (N, 4, N, 4))[..., :, :, :, 0]  # (..., N, 4, N)
    return np.rint(np.moveaxis(blocks, -2, -1)).astype(np.int64)


def matmul(a: NDArray, b: NDArray) -> NDArray:
    """Ring matrix product of coefficient arrays (broadcast over leading dims); exponents add"""
    return from_companion(companion(a) @ companion(b))


def matmul_dagger(a: NDArray, b: NDArray) -> NDArray:
    """a @ b^dagger"""
    return from_companion(companion(a) @ np.swapaxes(companion(b), -1, -2))


# ---------------------------------------------------------------------------
# conversions
# ---------------------------------------------------------------------------
def to_complex(c: NDArray, k: NDArray) -> NDArray:
    """(..., N, N, 4), (...) -> complex128 (..., N, N)"""
    c = np.asarray(c, dtype=np.float64)  # stored coefficients may be int16
    a, b, cc, d = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    re = a + (b - d) / SQRT2
    im = cc + (b + d) / SQRT2
    scale = SQRT2 ** np.asarray(k, dtype=np.float64)
    return (re + 1j * im) / scale[..., None, None]


def from_complex(U: NDArray, tol: float = 1e-9, k_max: int = 24, fix_phase: bool = True) -> Tuple[NDArray, int]:
    """Fit a complex matrix onto the ring lattice. Returns canonical (coefficients (N, N, 4), k) or raises
    ValueError if no representation with exponent <= k_max reproduces U within tol.
    With fix_phase, U may carry an arbitrary global phase: a ring-valued unitary has det in {omega^l}, so the
    phases e^{i phi} with e^{i N phi} det(U) an 8th root of unity are tried (8N candidates)."""
    try:
        return _from_complex(U, tol, k_max)
    except ValueError:
        if not fix_phase:
            raise
    N = U.shape[0]
    arg_det = np.angle(np.linalg.det(U))
    for r in range(N):
        for l in range(8):
            phi = (-arg_det + l * np.pi / 4 + 2 * np.pi * r) / N
            try:
                return _from_complex(U * np.exp(1j * phi), tol, k_max)
            except ValueError:
                continue
    raise ValueError("matrix is not (within tol, up to global phase) in Z[omega, 1/sqrt2] with exponent <= %d" % k_max)


def _from_complex(U: NDArray, tol: float, k_max: int) -> Tuple[NDArray, int]:
    N = U.shape[0]
    for k in range(k_max + 1):
        M = U * SQRT2 ** k
        S = int(np.ceil(2 * SQRT2 ** k)) + 1
        s = np.arange(-S, S + 1)
        parts: List[NDArray] = []
        ok = True
        for x in (M.real.ravel(), M.imag.ravel()):
            resid = x[:, None] - s[None, :] / SQRT2  # x = integer + s/sqrt2
            hit = np.abs(resid - np.rint(resid)) < tol
            if not hit.any(axis=1).all():
                ok = False
                break
            s_best = s[hit.argmax(axis=1)]
            parts.append(np.rint(x - s_best / SQRT2).astype(np.int64))
            parts.append(s_best.astype(np.int64))
        if not ok:
            continue
        a, sx, cc, ty = parts  # re = a + sx/sqrt2, im = cc + ty/sqrt2 with sx = b - d, ty = b + d
        if (np.mod(sx - ty, 2) != 0).any():
            continue
        b = (sx + ty) // 2
        d = (ty - sx) // 2
        c = np.stack([a, b, cc, d], axis=-1).reshape(N, N, 4)
        if np.abs(to_complex(c, np.array(k)) - U).max() > 1e-6:
            continue
        c_n, k_n = normalize_batch(c[None], np.array([k]))
        return c_n[0], int(k_n[0])
    raise ValueError("matrix is not (within tol) in Z[omega, 1/sqrt2] with exponent <= %d" % k_max)


def identity(N: int) -> Tuple[NDArray, int]:
    c = np.zeros((N, N, 4), dtype=np.int64)
    c[np.arange(N), np.arange(N), 0] = 1
    c, k = canonicalize_batch(c[None], np.array([0]))
    return c[0], int(k[0])


# ---------------------------------------------------------------------------
# gate row operations on a batch (B, N, N, 4) -- all states get the same gate
# ---------------------------------------------------------------------------
def _bit(num_qubits: int, qubit: int) -> int:
    return 1 << (num_qubits - 1 - qubit)


def _row_pairs(num_qubits: int, qubit: int, control: int = -1) -> Tuple[NDArray, NDArray]:
    """Rows with the qubit bit clear / set (restricted to rows where the control bit is set, if given)"""
    N = 1 << num_qubits
    idx = np.arange(N)
    mask = _bit(num_qubits, qubit)
    r0 = idx[(idx & mask) == 0]
    if control >= 0:
        r0 = r0[(r0 & _bit(num_qubits, control)) != 0]
    return r0, r0 | mask


def apply_phase_rows(c: NDArray, num_qubits: int, qubit: int, t: int) -> NDArray:
    """Multiply the rows with the qubit set by omega^t (T: t=1, S: 2, Z: 4, Sdg: 6, Tdg: 7)"""
    _, r1 = _row_pairs(num_qubits, qubit)
    out = c.copy()
    out[:, r1] = rotate(c[:, r1], t)
    return out


def apply_x(c: NDArray, num_qubits: int, qubit: int, control: int = -1) -> NDArray:
    """X on a qubit (CNOT when a control is given): swap row pairs"""
    r0, r1 = _row_pairs(num_qubits, qubit, control)
    out = c.copy()
    out[:, r0], out[:, r1] = c[:, r1], c[:, r0]
    return out


def apply_y(c: NDArray, num_qubits: int, qubit: int) -> NDArray:
    """Y = [[0, -i], [i, 0]]: new r0 = -i*r1 = omega^6 r1, new r1 = i*r0 = omega^2 r0"""
    r0, r1 = _row_pairs(num_qubits, qubit)
    out = c.copy()
    out[:, r0] = rotate(c[:, r1], 6)
    out[:, r1] = rotate(c[:, r0], 2)
    return out


def apply_z(c: NDArray, num_qubits: int, qubit: int, control: int = -1) -> NDArray:
    """Z on a qubit (CZ when a control is given): negate rows with the bit set"""
    _, r1 = _row_pairs(num_qubits, qubit, control)
    out = c.copy()
    out[:, r1] = -c[:, r1]
    return out


def apply_h(c: NDArray, k: NDArray, num_qubits: int, qubit: int, control: int = -1) -> Tuple[NDArray, NDArray]:
    """H on a qubit (CH when a control is given): rows (r0, r1) -> (r0 + r1, r0 - r1) / sqrt2, so k += 1"""
    r0, r1 = _row_pairs(num_qubits, qubit, control)
    out = c.copy()
    out[:, r0] = c[:, r0] + c[:, r1]
    out[:, r1] = c[:, r0] - c[:, r1]
    return out, k + 1


# ---------------------------------------------------------------------------
# Pauli tables for the channel representation
# ---------------------------------------------------------------------------
def val2(x: NDArray) -> NDArray:
    """2-adic valuation of int64 array entries (a large number for 0)"""
    lowbit = x & -x
    return np.where(x == 0, 10 ** 6, np.log2(np.maximum(lowbit, 1)).astype(np.int64))


def reduce_zsqrt2_rows(coef: NDArray, ex: NDArray) -> Tuple[NDArray, NDArray]:
    """coef (..., R, P, 2) are numerators a + b*sqrt2 over sqrt2**ex (..., R). Divide every row by the largest power
    of sqrt2 that divides all of its entries (at most ex): the sqrt2-adic valuation of a + b*sqrt2 is
    min(2*val2(a), 2*val2(b) + 1). Dividing by sqrt2 maps (a, b) -> (b, a/2)."""
    a, b = coef[..., 0], coef[..., 1]
    v = np.minimum(2 * val2(a), 2 * val2(b) + 1).min(axis=-1)  # (..., R)
    r = np.minimum(v, ex)
    half = r // 2
    a2 = a >> half[..., None]
    b2 = b >> half[..., None]
    odd = (r % 2 == 1)[..., None]
    out = np.stack([np.where(odd, b2, a2), np.where(odd, a2 >> 1, b2)], axis=-1)
    return out, ex - r


def pauli_tables(num_qubits: int) -> Tuple[NDArray, NDArray]:
    """For all 4^n Paulis P (index = base-4 digits per qubit, qubit 0 most significant, digit 0/1/2/3 = I/X/Y/Z):
    P[i, sigma[p, i]] = omega^phase[p, i] and all other entries are 0. Returns sigma, phase of shape (4^n, N)."""
    N = 1 << num_qubits
    num_paulis = 4 ** num_qubits
    idx = np.arange(N)
    sigma = np.zeros((num_paulis, N), dtype=np.int64)
    phase = np.zeros((num_paulis, N), dtype=np.int64)
    for p in range(num_paulis):
        xmask = 0
        ph = np.zeros(N, dtype=np.int64)
        for q in range(num_qubits):
            digit = (p // 4 ** (num_qubits - 1 - q)) % 4
            bit = _bit(num_qubits, q)
            has_bit = (idx & bit) != 0
            if digit in (1, 2):
                xmask |= bit
            if digit == 3:
                ph += np.where(has_bit, 4, 0)
            if digit == 2:  # Y: -i on the row with bit clear, +i on the row with bit set
                ph += np.where(has_bit, 2, 6)
        sigma[p] = idx ^ xmask
        phase[p] = np.mod(ph, 8)
    return sigma, phase


def pauli_coeffs(num_qubits: int, p: int) -> NDArray:
    """Coefficient array (N, N, 4) of Pauli p (exponent 0)"""
    sigma, phase = pauli_tables(num_qubits)
    N = 1 << num_qubits
    c = np.zeros((N, N, 4), dtype=np.int64)
    for i in range(N):
        c[i, sigma[p, i]] = ROT[phase[p, i]][0]  # coefficients of omega^t = row 0 of ROT[t]
    return c


def pauli_generator_indices(num_qubits: int) -> List[int]:
    """Pauli indices of X_0..X_{n-1}, Z_0..Z_{n-1}"""
    xs = [1 * 4 ** (num_qubits - 1 - q) for q in range(num_qubits)]
    zs = [3 * 4 ** (num_qubits - 1 - q) for q in range(num_qubits)]
    return xs + zs
