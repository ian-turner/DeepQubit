# Utility Modules

## matrix_utils.py

Core unitary math used throughout the domain.

**Constants:** `I` (2×2 identity), `P0`/`P1` (|0⟩⟨0|, |1⟩⟨1| projectors).

| Function | Purpose |
|----------|---------|
| `tensor_product(mats)` | Kronecker product of a list of matrices |
| `invert_unitary(U)` | Conjugate transpose U† |
| `unitary_distance(U, C)` | Distance metric from Synthetiq paper; phase-invariant |
| `phase_align(U)`, `phase_align_batch(Us)` | Removes global phase by making the largest-magnitude entry real and positive; ties within `tol` pick the first entry in row-major order, so the result is stable under fp noise on Clifford+T matrices |
| `hash_unitary(U, tol=0.001)` | Phase-aligns, rounds, hashes |
| `load_matrix_from_file(f)` | Loads `.txt` or `.npy`; returns `(num_qubits, matrix)` |
| `save_matrix_to_file(M, f)` | Saves to `.txt` format |
| `unitaries_to_nnet_input(Us, encoding, nerf_dim)` | Dispatch to encoding — see [encodings](encodings.md) |

**Text file format** (`.txt`):
```
matrix
<num_qubits>
(re,im) (re,im) ...   <- row 0
...
```

## perturb.py

Randomly perturbs a batch of unitaries within an ε-ball under Hilbert–Schmidt distance.

Two functions are provided:

| Function | Direction | Cost | Notes |
|----------|-----------|------|-------|
| `perturb_unitary_random_batch_strict` | Isotropic (full Lie algebra) | O(n³) — eigendecomp + vectorized bisection | Default; used by domain |
| `perturb_unitary_givens_batch` | Single 2D plane (non-isotropic) | O(n) — closed-form distance inversion | ~20× faster; ε ≤ 2/√n |

Both exploit `d(U, WU) = d(I, W)` (right-multiply isometry) — distance never depends on U. Both use the same normalization as `unitary_distance_batch` (Frobenius / √n), so epsilon is directly comparable to `self.epsilon` in the domain.

**Lie algebra approach**: samples random unit-Frobenius Hermitian H, finds t via vectorized bisection (53 iters, float64 exact) on `[0, π/λ_max]` where d(t) is provably monotone.

**Givens approach**: picks a random (i,j) plane and phase φ, computes `α = 2 arcsin(d√n / (2√2))` directly (no bisection). Updates only 2 rows of U. Constraint: ε ≤ 2√2/√n (≈ 1.41 for n=4).

Used in `QCircuit.sample_goal_from_state` when `perturb=True`.

## ring.py

Exact integer arithmetic over D[ω] = Z[ω, 1/√2] for the [exact domain](exact.md): coefficient arrays (…, N, N, 4) plus an exponent.

| Function | Purpose |
|----------|---------|
| `rotate`, `conj`, `dagger`, `mul_sqrt2` | element-wise ring ops (multiply by ωᵗ, conjugate, conj-transpose, ×√2) |
| `reduce_batch`, `canonicalize_batch`, `normalize_batch` | minimal exponent; fix global phase; both |
| `companion`, `from_companion`, `matmul`, `matmul_dagger` | ring matrix products via the 4N×4N companion form |
| `to_complex`, `from_complex`, `identity` | conversions (`from_complex` fits the lattice up to global phase, raises if not in the ring) |
| `apply_h`, `apply_phase_rows`, `apply_x`, `apply_y`, `apply_z` | batched gate row operations (optional control for CH/CNOT/CZ) |
| `pauli_tables`, `pauli_coeffs`, `pauli_generator_indices` | Pauli bookkeeping for the channel encoding |

## hurwitz.py

Batched Hurwitz parameterization of SU(n) — see [encodings](encodings.md) for the role it plays.

Standalone encode/decode:
- `su_encode_batched_np(U_B)` → `(theta, phi, lam)` angles
- `su_decode_batched_np(theta, phi, lam_head, n)` → unitaries
- `su_encode_to_features_np(U_B)` → `[B, n²-1]` packed features
- `su_decode_from_features_np(feats, n)` → unitaries
