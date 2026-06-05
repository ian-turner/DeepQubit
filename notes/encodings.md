# Unitary Encodings

`utils/matrix_utils.py`, `utils/hurwitz.py`

Entry point: `unitaries_to_nnet_input(Us, encoding, nerf_dim)` in `matrix_utils.py`.

## Matrix (default)

Phase-aligns each unitary (removes global phase ambiguity), then flattens to real/imag parts:
`[Re(U.flatten()), Im(U.flatten())]`

Size: `2 × (2^N)²` reals for N qubits.

## Hurwitz

Batched complex Givens (Hurwitz) parameterization of SU(n) — `utils/hurwitz.py`.

Encodes a unitary as `(θ, φ, λ)` angles via QR-like elimination, packed into `n²-1` real features. Round-trip error ~1e-15 in float64.

Functions: `su_encode_to_features_np(U_B)` → `[B, n²-1]`, `su_decode_from_features_np(feats, n)`.

Size: `(2^N)² - 1` reals.

## Quaternion

Only valid for 1-qubit (U(2)) unitaries. Phase-aligns to SU(2) then reads off `[Re(a), Im(a), Re(b), Im(b)]` from the top row.

Size: 4 reals. See `unitaries_to_quaternions` / `quaternions_to_unitaries`.

## NeRF Embedding

Optional positional encoding applied on top of any encoding. Controlled by `nerf_dim` (L in domain string).

For each scalar x, produces `[sin(2^0 x), cos(2^0 x), ..., sin(2^(L-1) x), cos(2^(L-1) x)]`.

Final size: `2 × L × base_size`.

## Input Size Formula (`get_input_info_flat_sg`)

```python
match encoding:
    case 'matrix':    N = 2 ** (2 * num_qubits + 1)
    case 'hurwitz':   N = (2 ** num_qubits) ** 2 - 1
    case 'quaternion': N = 4
# With NeRF:
size = 2 * nerf_dim * N if nerf_dim > 0 else N
```
