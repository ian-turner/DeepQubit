# DeepQubit Wiki

Quantum circuit synthesis using reinforcement learning and search, based on [DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf). Given a target unitary operator, the system learns to find a gate sequence that approximates it within tolerance ε.

## Topics

- [Domain](domain.md) — QCircuit state/action/goal types, gate sets, the deepxube interface
- [Exact Domain](exact.md) — `qcircuit_exact`: integer-ring states (no ε), exact hashing, binary residue/channel encodings, `resnet_fc_ring` network
- [Encodings](encodings.md) — How unitaries are converted to neural network inputs (matrix, Hurwitz, quaternion, NeRF)
- [Utils](utils.md) — Unitary math utilities: distances, hashing, tensor products, perturbation
- [Data](data.md) — File formats, directory layout, goal/target conventions
- [Training & Solving](training.md) — Scripts, CLI flags, domain string syntax, output layout

## Quick Reference

| Task | Command |
|------|---------|
| Setup env | `source setup.sh` |
| Random goals | `python scripts/goal_gen.py [--num_qubits 1] [--num 1000] [--seed 0]` → `data/targets/<n>qubit/random_<num>.pkl` |
| Train | `bash scripts/train.sh` |
| Training comparison graphs | `python scripts/train_report.py [--bins 100]` → `data/training/*.csv` (every run in `tmp/`), `paper/images/<comparison>.png` |
| Solve | `bash scripts/solve.sh [config] [goals .pkl or .txt dir]` (prints/writes a per-goal `summary.txt`) |
| Goals from .txt targets | `python scripts/goals_from_txt.py --input <files or dirs> --output <out.pkl> [--domain qcircuit_exact] [--reachable_only]` |
| Paths to QASM | `python scripts/paths_to_qasm.py --input <results.pkl> --output <dir>` |
| Trasyn benchmark | `python scripts/trasyn_bench.py <goals.pkl> --epsilon 0.01` |
| Check goal reachability | `python scripts/check_goals.py <goals.pkl> [--output <reachable.pkl>]` |
| Float goals → exact goals | `python scripts/goals_to_exact.py --input <goals.pkl> --output <goals_exact.pkl>` |
| Exact-domain tests | `python tests/test_exact.py` |

## Domain String Syntax

`qcircuit.n<N>_<flags>` — parsed by `QCircuitParser` (the exact domain `qcircuit_exact.n<N>_<flags>` takes `n`, `I`/`S`, an encoding `B<m>`/`C<m>`/`M` (or `Z<m>` alone, the compact form for `resnet_fc_ring`) and `K<cap>`; see [exact](exact.md)):

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
| `S` | CliffT_S gate set |
| `I` | CliffT_inv gate set (same gates as `S`) |
