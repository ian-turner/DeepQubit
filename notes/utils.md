# Utility Modules

## matrix_utils.py

Core unitary math used throughout the domain.

**Constants:** `I` (2×2 identity), `P0`/`P1` (|0⟩⟨0|, |1⟩⟨1| projectors).

| Function | Purpose |
|----------|---------|
| `tensor_product(mats)` | Kronecker product of a list of matrices |
| `invert_unitary(U)` | Conjugate transpose U† |
| `unitary_distance(U, C)` | Distance metric from Synthetiq paper; phase-invariant |
| `phase_align(U)` | Removes global phase so equivalent unitaries hash equal |
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

Main function: `perturb_unitary_random_batch_strict(U_B, epsilon, ...)`.

Algorithm: sample a random Hermitian direction H, compute t via bisection so that `‖U - exp(itH)U‖ ≤ ε`, apply strict clamp with safety margin. Guarantees `d ∈ [0, ε]` numerically.

Used in `QCircuit.sample_goal_from_state` when `perturb=True`.

## hurwitz.py

Batched Hurwitz parameterization of SU(n) — see [encodings](encodings.md) for the role it plays.

Standalone encode/decode:
- `su_encode_batched_np(U_B)` → `(theta, phi, lam)` angles
- `su_decode_batched_np(theta, phi, lam_head, n)` → unitaries
- `su_encode_to_features_np(U_B)` → `[B, n²-1]` packed features
- `su_decode_from_features_np(feats, n)` → unitaries
