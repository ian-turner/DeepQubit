# Exact Domain — qcircuit_exact

`domains/qcircuit_exact.py`, arithmetic in `utils/ring.py`, tests in `tests/test_exact.py` (`python tests/test_exact.py`).

Exact Clifford+T synthesis with no tolerance. Every exact Clifford+T unitary has entries in the ring
D[ω] = Z[ω, 1/√2], ω = e^{iπ/4}, so a unitary is stored as an int64 coefficient array of shape (N, N, 4)
(coefficients of 1, ω, ω², ω³ per entry) plus an exponent k, meaning U = coeffs / √2ᵏ. States are kept in
**canonical form**: k minimal (√2 divided out while possible) and global phase fixed by the power of ω that makes
the first nonzero entry lexicographically largest. Canonical forms are unique, so `__eq__` is an exact array
compare and `__hash__` hashes the bytes (computed once). There is no ε; `is_solved` is exact equality.
Coefficients are bounded by √2ᵏ (the Galois automorphisms ω → ωʲ map a unitary to a unitary), so states and goals
store them as **int16** (int64 only if k > 29, never seen: 100-step training goals have k ≤ 9, |c| ≤ 13); the dtype
depends only on the values, so hashes stay consistent. Arithmetic is int64. Only `coeffs`/`k` are pickled.

## Relation to the float domain

`QCircuitExact` subclasses `QCircuit` and reuses its action objects, gate sets and parser
conventions. Only state representation, transitions, goals and the network input are replaced:

| Method | Exact implementation |
|--------|----------------------|
| `next_state` | integer row operations per gate (grouped by action); only H/CH results are √2-reduced and only gates that touch row 0 (H, CH, X, Y) re-canonicalized — phase gates, CNOT and CZ keep a canonical input canonical |
| `sample_goal_from_state` | G·S† via the companion form, canonical |
| `is_solved` | exact equality |
| `to_np_flat_sg` | binary/residue encodings below |

**Relative unitary without products** (`_relative`): G (A S)† = (G S†) A†, so a state that `next_state` generated
(it keeps `_parent`, `_action`) gets its relative unitary by one column operation on its parent's (the row op of
conj(A) on the transpose). Relatives are cached (`_rel = (goal, coeffs, k)`, checked by goal identity) only on
expanded parents, ~1/|A| of the search tree: a parent whose own parent is cached is filled in first by a column op;
roots and relabeled (HER) goals fall back to the full companion product (`_relative_scratch`, also used by
`sample_goal_from_state`). Search encodings never compute the 4N×4N product after the roots.

`QStateExact.unitary` / `QGoalExact.unitary` give a complex128 view for interop (QASM export, distance checks);
`from_complex` fits a float matrix onto the ring lattice (any global phase) and raises `ValueError` if the matrix
is not exactly synthesizable. `scripts/goals_to_exact.py` converts a float goals `.pkl`.

Gate row operations (qubit 0 = most significant bit): T/S/Z/Sdg/Tdg multiply the rows with the qubit set by
ω^t (t = 1, 2, 4, 6, 7); X and CNOT swap row pairs; Y swaps with ±i; H maps row pairs to (r0 + r1, r0 − r1) and
raises k by one; CZ/CH are the controlled variants. Products use the **companion form**: each ring element becomes
the 4×4 integer matrix whose column t holds the coefficients of z·ωᵗ, so a ring matrix is a 4N×4N integer block
matrix, ring products are block matmuls (done in float64 for BLAS, exact below 2⁵³) and conj-transpose is the
transpose.

## Domain string

`qcircuit_exact.n<N>[_I|_S][_<encoding>][_K<cap>]`, e.g. `qcircuit_exact.n3_I_B9+C2`, or
`qcircuit_exact.n3_I_Z9` with `resnet_fc_ring` (recommended).
No `e` (no ε), no `P`/`R` (goals must be exact). `K<cap>` caps the exponent one-hots (default 20).

## Network input

The input is built from the canonical relative unitary R = G·S†. Encoding parts joined by `+`:

| Part | Content | Size (n=3) | Complete? |
|------|---------|-----------|-----------|
| `B<m>` (default m=9) | residues mod 2ᵐ of the 4 coefficients of every entry, m bits each, plus one-hot of k (capped at `k_cap`) | 64·4·m + 21 = 2325 | lossless while every coefficient < 2ᵐ⁻¹, i.e. k ≤ 2m−2 (depth ≤ 4m−4); beyond that it degrades as a ring homomorphism (low-order structure kept, magnitude dropped) |
| `C<m>` (default m=2) | channel representation: for each generator X_i, Z_i the Pauli-basis coefficients of R·P·R†, written as Z[√2] numerators a + b√2 over √2^e with e minimal per row, as m bits of a and b each, plus one-hot of e (capped at 2·`k_cap`) | 6·64·2·m + 6·41 = 1782 | not complete; for a Clifford it is exactly the stabilizer tableau with signs (one ±1 per row, e = 0); any T gate shows as spread and e > 0 |
| `M` | float real/imag of R (the old matrix encoding) | 128 | complete |
| `Z<m>` (default m=9) | compact wire form of `B<m>` for `resnet_fc_ring` only: int16 coefficients of R plus k (int16 wrap keeps residues mod 2¹⁶); the network expands the `B<m>` bits on the device. Must be the only part | 257 (514 bytes vs 9300 for `B9` floats) | lossless for k ≤ 29 |

The channel rows are phase-invariant by construction. Why this matters: in the float encoding CNOT, SWAP and
Toffoli are all 0/1 matrices, and a value net trained on random walks scored all of them like 1–3 gate circuits;
in `C` they separate on support size and exponent (Toffoli spreads X₀, X₁, Z₂ over four Paulis at exponent 2).
`M` is kept for ablations only.

## resnet_fc_ring — the ring algebra on the device

`nnets/resnet_fc_ring.py` (auto-imported by deepxube from the local `nnets/` package). The intended setup is
domain encoding `Z9` with network `resnet_fc_ring.<H>H_<B>B_bn[_<m>C][_fv]` (`B9` input also works and gives the
same features, but ships 18× more bytes through shared memory, the nnet runner, the 1M-row training buffer — 9.3 GB
vs 0.5 GB per update — and every host→GPU copy). Its parameter-free front layer `RingFeatures` gets the integer
coefficients (from the int16 input, or reconstructed from `B` bits, exact while k ≤ 2m−2), expands the `B<m>` bits
and k one-hot for `Z` input, and computes the `C<m>` channel features (and with `fv` the `M` float view) on the
network's device, then feeds `[B bits | channel bits + exponent one-hots | (floats)]` to the usual resnet; the
layout is the same for `B<m>` and `Z<m>`, so checkpoints are interchangeable (the layer's tables are rebuilt from
the domain and not saved; checkpoints from before 2026-10-05, which saved them, still load). The channel uses the two Galois
embeddings ω → ω, ω³: R Pᵍ R† for all 2n generators is one complex (N×N)(N×2nN) product per embedding (Pauli = phased
permutation), the 4ⁿ traces are one GEMM with a fixed ±1 selection matrix (`proj`; every Pauli phase is a power of
i), and the Z[√2] numerator a + b√2 is recovered from its embeddings a ± b√2 by rounding (float64; 16× fewer FLOPs
than the 4N×4N companion form). Rows whose integers are not exact (k > 2m−2 for `B` input, k > 29 for `Z`) get a
zeroed channel/float block. Flags: `<m>C` channel bits (default 2, `0C` disables), `fv` float view. The layer is
tested bit-for-bit against the numpy `C`/`M` encodings (`tests/test_exact.py`). The domain must not include a `C`
part when this network is used (the network refuses).

Speed (local CPU, one search worker, 2100 children per iteration; 2026-10-05): domain `next_state` 20 → 6.5 ms,
`to_np` 47 → 12 ms; retained search memory 2725 → 1291 bytes/node; `RingFeatures` 90 → 60 ms per 2000 states
on CPU. Search trajectories are identical to the old code (same popped states and heuristics).
