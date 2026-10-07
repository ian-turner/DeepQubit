# DeepQubit

Quantum circuit synthesis with a learned heuristic and search. Given a target unitary, DeepQubit finds a Clifford+T
gate sequence that implements it, either within a tolerance ε (approximate synthesis) or exactly up to global phase
(exact synthesis).

The approach follows [DeepCubeA](https://cse.sc.edu/~foresta/assets/files/SolvingTheRubiksCubeWithDeepReinforcementLearningAndSearch_Final.pdf):
a neural network learns a cost-to-go estimate (gates remaining) by approximate value iteration on randomly generated
problems, and batch-weighted A* uses it to search over circuits. Training and search come from
[deepxube](https://pypi.org/project/deepxube/); this repo supplies the quantum-circuit domains, unitary encodings,
and network front-ends.

## How it works

Synthesis is posed as a shortest-path problem:

| | |
|---|---|
| **State** | the unitary of the circuit built so far (starts at the identity) |
| **Action** | one Clifford+T gate (H, S, S†, T, T†, CNOT, …) on a specific qubit or qubit pair |
| **Goal** | the target unitary |
| **Solved** | `unitary_distance ≤ ε` (approximate), or exact equality up to a power of ω = e^{iπ/4} (exact) |

The network is trained only on goals produced by random walks from the identity. At solve time it is given the
target, and A* searches outward from the identity.

There are two domains:

- **`qcircuit`** — floating-point unitaries with tolerance ε. Supports several network input encodings: the
  phase-aligned matrix (`M`), a Hurwitz/Givens parameterization of SU(n) (`H`), quaternions for one qubit (`Q`),
  any `+`-joined combination, and an optional NeRF-style sinusoidal embedding (`L<dim>`).
- **`qcircuit_exact`** — exact synthesis over the ring Z[ω, 1/√2], which contains every exact Clifford+T unitary.
  States are integer coefficient arrays plus a √2 exponent, kept in a canonical form, so equality and hashing are
  exact and there is no ε. Gates are integer row operations. The network (`resnet_fc_ring`) receives the integer
  coefficients and computes two feature sets on the GPU: binary residues of the coefficients, and the channel
  representation R·P·R† of the relative unitary R over the Pauli generators. For a Clifford, the channel features are
  exactly the stabilizer tableau, and each T gate shows up as spread across Paulis and a higher √2 exponent.

## Getting started

Requires Python 3.10+.

```bash
pip install -r requirements.txt
source setup.sh          # adds the repo root to PYTHONPATH; run in every new shell
```

Training and solving are driven by shell configs in `configs/`. Each config sets the domain string, network, search
settings and benchmark targets:

```bash
bash scripts/train.sh configs/n1_e0.01_L10        # 1 qubit, ε = 0.01, matrix + NeRF L10
bash scripts/solve.sh configs/n1_e0.01_L10        # solve data/targets/1qubit, print a per-goal summary

bash scripts/train.sh configs/n3_exact_ring       # 3 qubits, exact ring domain
bash scripts/solve.sh configs/n3_exact_ring       # solve data/targets/3qubit (reachable targets only)
```

Checkpoints, logs and solve output go to `tmp/<domain>/<network>/`. `solve.sh` accepts either a goals `.pkl` or a
directory of `.txt` target matrices as an optional second argument.
`python scripts/paths_to_qasm.py --input <results.pkl> --output <dir>` exports solutions as OpenQASM 3.

Domain strings combine flags, for example `qcircuit.n1_e0.01_P_L10` or `qcircuit_exact.n3_I_Z9`. They set the number
of qubits, ε, gate set, encoding, NeRF dimension and goal perturbation. The full syntax is in
[`notes/index.md`](notes/index.md#domain-string-syntax).

Run the exact-domain tests with `python tests/test_exact.py`.

## Repository layout

```
domains/      qcircuit.py (float domain), qcircuit_exact.py (exact ring domain)
nnets/        resnet_fc_ring.py — on-device ring/channel feature layer + residual network
utils/        unitary math, encodings (Hurwitz), ε-ball perturbation, Z[ω,1/√2] arithmetic
scripts/      train/solve drivers, goal generation and conversion, baseline benchmarks, training plots
configs/      run configurations (domain, network, search and solve settings)
data/         targets/ (benchmark unitaries), circuits/ (example QASM), results/, baselines/, training/
notes/        project wiki: design notes for every module and script
paper/        figures
```

The wiki in [`notes/`](notes/index.md) covers the domains, encodings, data formats and every script in detail.

## Acknowledgements

The authors gratefully acknowledge the computational resources provided by the Theia high performance computing
cluster at the University of South Carolina which is supported by National Science Foundation Major Research
Instrumentation Grant No. 2320292. We also acknowledge the technical assistance and resources provided by Research
Computing at the University of South Carolina (RRID:SCR_027488).
