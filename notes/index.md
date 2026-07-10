# DeepQubit Wiki

Quantum circuit synthesis using reinforcement learning and search, based on [DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf). Given a target unitary operator, the system learns to find a gate sequence that approximates it within tolerance ε.

## Topics

- [Domain](domain.md) — QCircuit state/action/goal types, gate sets, the deepxube interface
- [Encodings](encodings.md) — How unitaries are converted to neural network inputs (matrix, Hurwitz, quaternion, NeRF)
- [Utils](utils.md) — Unitary math utilities: distances, hashing, tensor products, perturbation
- [Data](data.md) — File formats, directory layout, goal/target conventions
- [Training & Solving](training.md) — Scripts, CLI flags, domain string syntax, output layout

## Quick Reference

| Task | Command |
|------|---------|
| Setup env | `source setup.sh` |
| Generate goals | `bash scripts/goal_gen.sh` |
| Train | `bash scripts/train.sh` |
| Solve | `bash scripts/solve.sh` |
| Goals from .txt targets | `python scripts/goals_from_txt.py --input <files> --output <out.pkl>` |
| Paths to QASM | `python scripts/paths_to_qasm.py --input <results.pkl> --output <dir>` |
| Trasyn benchmark | `python scripts/trasyn_bench.py <goals.pkl> --epsilon 0.01` |

## Domain String Syntax

`qcircuit.n<N>_<flags>` — parsed by `QCircuitParser`:

| Flag | Meaning |
|------|---------|
| `n<N>` | N qubits |
| `e<val>` | epsilon tolerance |
| `L<D>` | NeRF embedding dimension |
| `H` | Hurwitz encoding |
| `Q` | Quaternion encoding |
| `M` | Matrix encoding (default) |
| `H+Q`, `Q+H+M`, ... | Concatenated encodings (any `+`-joined combo of M/H/Q) |
| `P` | Perturb goals |
| `R` | Random goals |
| `S` | CliffT_S gate set |
