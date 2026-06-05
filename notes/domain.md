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

## Gate Sets

Defined in `get_gate_set(gateset: str)`:

| Name | Gates |
|------|-------|
| `CliffT` | H, S, Y, T, X, Z, CNOT |
| `CliffT_S` | H, S, Sdg, T, Tdg, CNOT |

`_generate_actions` expands each gate over all valid qubit assignments. One-qubit gates: N instances. Controlled gates: N×(N-1) instances (i≠j).

## QCircuit (the domain)

Registered as `'qcircuit'` with deepxube's `domain_factory`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_qubits` | — | required |
| `epsilon` | 0.01 | solve tolerance |
| `gateset` | `'CliffT'` | see gate sets |
| `encoding` | `'matrix'` | nnet input encoding |
| `nerf_dim` | 0 | NeRF embedding dim |
| `perturb` | False | perturb goals during training |
| `random_goal` | False | sample random unitary goals |

Key methods:
- `sample_start_states` — returns identity unitaries
- `sample_goal_from_state` — computes relative transformation `U_goal @ U_start†`; optionally perturbs
- `next_state` — batched matmul: `A @ B` for action matrices A, state matrices B
- `is_solved` — checks `unitary_distance(state, goal) ≤ ε`
- `to_np_flat_sg` — converts (state, goal) pairs to nnet input

## Parser

`QCircuitParser` parses domain strings like `n1_L15_P_H_e0.001`. See [index](index.md#domain-string-syntax) for flag table.

## NNet Input Classes

Two registered inputs (both named `QCircutNNetInput` — likely a bug):
- `qcircuit_nnet_input` — `StateGoalIn`
- `qcircuit_nnet_input_fix_act` — `StateGoalActFixIn`
