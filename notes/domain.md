# Domain — QCircuit

`domains/qcircuit.py`

## Core Types

**`QState`** — wraps a complex128 unitary matrix. Equality is `unitary_distance ≤ ε` (default 1e-6). Hash uses `hash_unitary` (phase-aligned, rounded).

**`QGoal`** — same structure as `QState`; a separate class for the deepxube interface.

**`QAction`** — abstract base. `apply_to(state)` left-multiplies the state unitary: `new_U = gate_U @ state_U`.

## Action Hierarchy

```
QAction (ABC)
├── OneQubitGate (ABC) — acts on one qubit; builds full unitary via tensor product
│   ├── HGate, SGate, SdgGate, ZGate, TGate, TdgGate, XGate, YGate
└── ControlledGate (ABC) — control + target qubit; builds full unitary via P0/P1 projectors
    ├── CNOTGate, CZGate, CHGate
```

## Gates

Each gate class has a `name` attribute that is its OpenQASM identifier (`stdgates.inc`). `__repr__` emits `<name> qs[i]` / `<name> qs[c], qs[t]`, which `scripts/paths_to_qasm.py` writes verbatim, so names must match the QASM standard library exactly.

| Class | `name` | Class | `name` |
|-------|--------|-------|--------|
| HGate | `h` | XGate | `x` |
| SGate | `s` | YGate | `y` |
| SdgGate | `sdg` | ZGate | `z` |
| TGate | `t` | CNOTGate | `cx` |
| TdgGate | `tdg` | CZGate | `cz` |
| | | CHGate | `ch` |

## Gate Sets

Defined in `get_gate_set(gateset: str)`:

| Name | Flag | Gates |
|------|------|-------|
| `CliffT` | (default) | H, S, Y, T, X, Z, CNOT |
| `CliffT_S` | `S` | H, S, Sdg, T, Tdg, CNOT |
| `CliffT_inv` | `I` | H, S, Sdg, T, Tdg, CNOT (same gates as `CliffT_S`) |

`_generate_actions` expands each gate over all valid qubit assignments. One-qubit gates: N instances. Controlled gates: N×(N-1) instances (i≠j).

## QCircuit (the domain)

Registered as `'qcircuit'` with deepxube's `domain_factory`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_qubits` | — | required |
| `epsilon` | 0.01 | solve tolerance |
| `gateset` | `'CliffT'` | see gate sets |
| `encoding` | `'matrix'` | nnet input encoding; `+`-joined names concatenate (e.g. `'hurwitz+quaternion'`) |
| `nerf_dim` | 0 | NeRF embedding dim |
| `perturb` | False | perturb goals during training |
| `random_goal` | False | sample random unitary goals |
| `macro_frac` | 0.0 | fraction of training instances whose goal is built from macro words (flag `G`); see below |

Key methods:
- `sample_start_states` — returns identity unitaries
- `sample_problem_instances` — identity start + random walk of `num_steps` gates → relative goal (deepxube default); with `macro_frac > 0` a fraction of instances get a structured goal instead (see below)
- `sample_goal_from_state` — computes relative transformation `U_goal @ U_start†`; optionally perturbs
- `next_state` — batched matmul: `A @ B` for action matrices A, state matrices B
- `is_solved` — checks `unitary_distance(state, goal) ≤ ε`
- `to_np_flat_sg` — converts (state, goal) pairs to nnet input

## Structured (macro) goals — flag `G`

Random walks from the identity essentially never land on sparse-but-deep unitaries (Toffoli-like permutations, CCZ-like sign diagonals), so a V-net trained only on walks scores such targets like 1–3 gate Clifford circuits and search stalls (see the Sep 2026 n3 diagnosis). With `macro_frac > 0` (`G` = 0.5, `G0.3` = 0.3) that fraction of training instances use `_macro_goal_states` instead of a random walk:

1. Concatenate random words from `_macro_words` (every macro in `MACROS` over every assignment of distinct qubits, expanded to gate-set actions; 42 words for 3 qubits).
2. Keep a prefix of at most `num_steps` gates: with prob. ½ exactly `num_steps` (partial macros → the dense intermediate states of a decomposition), otherwise the last whole-macro boundary ≤ `num_steps` (products of whole macros).

So every goal is reachable in ≤ `num_steps` gates and the step-count curriculum is preserved; labels remain Bellman bootstraps as usual.

| Macro | Word length (CliffT_inv) | Notes |
|-------|-----|-------|
| `toffoli` | 15 | Nielsen & Chuang, T-count 7 |
| `ccz` | 13 | Toffoli word without the target H gates |
| `fredkin` | 17 | CNOT · Toffoli · CNOT |
| `swap`, `cz`, `cs`, `ch` | 3, 3, 5, 14 | 2-qubit macros; `ch` is the qelib1 definition |

Words are `(gate_class, qubits)` lists in circuit order (`MacroWord`). Gates missing from a gate set are replaced by exact words via `_GATE_SUBS` (e.g. `CliffT` has no Tdg → `Z S T`), so the same macros work for `CliffT` (longer words). Macro goals need ≥ 2 qubits. Every word is a product of gate-set gates, so it automatically satisfies the determinant invariant (see [data](data.md#reachability)).

## Parser

`QCircuitParser` parses domain strings like `n1_L15_P_H_e0.001`. See [index](index.md#domain-string-syntax) for flag table.

## NNet Input Classes

Two registered inputs (both named `QCircutNNetInput` — likely a bug):
- `qcircuit_nnet_input` — `StateGoalIn`
- `qcircuit_nnet_input_fix_act` — `StateGoalActFixIn`
