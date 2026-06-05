# Data

## Directory Layout

```
data/
├── targets/          # Target unitary matrices in .txt format
│   ├── 1qubit/       # rz3–rz7
│   ├── 2qubit/       # ch, crk_2, crz_2, cs, cz
│   ├── 3qubit/       # cch, ccrz_2, cct, ccz, csqrtiswap, csqrtswap, fredkin, toffoli
│   └── 4qubit/       # rcccx
│   └── goals_n3.pkl  # pre-built goal set for 3-qubit problems
├── benchmarks/       # Benchmark result files and notebooks
│   ├── astar-*.txt   # A* solver results
│   └── trasyn-*.txt  # Trasyn baseline results
├── training/         # Training goal files listing domain configurations
│   ├── 1q*/          # 1-qubit training sets (various encodings/flags)
│   ├── 2q*/          # 2-qubit training sets
│   └── 3q*/          # 3-qubit training sets
├── circuits/         # Target circuits as .qasm files (for verification)
│   ├── 1qubit/, 2qubit/, 3qubit/
└── n2_goals.pkl      # Pre-built goal set for 2-qubit problems

tmp/
└── <domain>/<heur>/  # Training checkpoints and TensorBoard logs
    ├── heur.pt       # Best heuristic weights
    ├── heur_targ.pt  # Target network weights
    ├── status.pkl    # Training status
    └── output.txt    # Training log
```

## File Formats

**`.txt` unitary files** — human-readable matrix format (see `load_matrix_from_file`):
```
matrix
<num_qubits>
(real,imag) (real,imag) ...
```

**`.pkl` goal files** — Python pickle of `{'states': [QState, ...], 'goals': [QGoal, ...]}`.

**`.qasm` files** — OpenQASM 3.0 circuits (output of `paths_to_qasm.py` or existing reference circuits).

**Results `.pkl`** — deepxube solve output with keys `'solved'`, `'actions'`, etc.

## Training File Naming Convention

Files in `data/training/` encode the domain config:
`<nq>q<eps>-<flags>.txt`

Examples:
- `1q1e-2-H-P-L15.txt` — 1 qubit, ε=0.01, Hurwitz encoding, perturb, NeRF dim 15
- `2q5e-1-base.txt` — 2 qubit, ε=0.5, base (matrix) encoding

## Generating Goals

Random goals from deepxube:
```bash
bash scripts/goal_gen.sh   # generates tmp/n1_goals_R_1K.pkl
```

Goals from target `.txt` files:
```bash
python scripts/goals_from_txt.py --input data/targets/1qubit/*.txt --output out.pkl
```
