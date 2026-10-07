# Training & Solving

## Setup

```bash
source setup.sh   # adds repo root to PYTHONPATH; run in every new shell
```

## Training (`scripts/train.sh`)

`bash scripts/train.sh [config]` (default `configs/test`). The config is a shell file defining:

| Variable | Example | Meaning |
|----------|---------|---------|
| `DOMAIN` | `n3_e0.000001` | Domain string (parsed by `QCircuitParser`) |
| `HEUR` | `resnet_fc.1000H_4B_bn` | Network architecture (`--fn <kind>,$HEUR`); `resnet_fc_ring.…` for the exact domain (see [exact](exact.md#resnet_fc_ring--the-ring-algebra-on-the-device)) |
| `PATHFIND` | `graph_v.1B_1W_0.0E` | Search used during training; `*_q` selects a Q-function (`heurq_fixout`/`up_rl_q`), otherwise V (`heurv`/`up_rl_v`) |
| `BATCH_SIZE`, `MAX_ITRS`, `CHECKPOINT` | 10000, 200000, 100 | Trainer args (`--tr tr_h.<bs>bs_<maxit>maxit_<chkpt>chkpt`) |
| `PROCS`, `STEP_MAX`, `SEARCH_ITRS`, `UP_ITRS`, `UP_GEN_ITRS` | 24, 30, 1000, 100, 100 | Updater args (`--up up_rl_v.<p>p_<sm>sm_<sitrs>sitrs_<up>up_<upg>upg`) |

`configs/n3_exact` is the 3-qubit exact run (`n3_e0.000001_I`); `configs/n3_exact_G` is the same with structured macro-word goals (`n3_e0.000001_I_G`, see [domain](domain.md#structured-macro-goals--flag-g)); `configs/n3_exact_ring` runs the integer-ring domain (`DOMAIN_NAME="qcircuit_exact"`, domain `n3_I_Z9` (compact int16 input; same features as `n3_I_B9` with 18× less data per state), network `resnet_fc_ring.1000H_4B_bn_2C` which expands the bits and computes the channel features on the GPU, random-walk goals only for now, solves `data/targets/3qubit` converted to ring goals, see [exact](exact.md)). Runs from before 2026-10-05 used `n3_I_B9` and live in `tmp/n3_I_B9/...`; their checkpoints load unchanged into the Z9 run directory. `configs/n2_exact_ring` is the same ring setup on 2 qubits (`n2_I_Z9`, solves `data/targets/2qubit` with a 100 s limit; all five targets are within 7 gates, and `crk_2` is the same unitary as `cs`).

`DOMAIN_NAME` (default `qcircuit`) selects the domain class passed as `--domain $DOMAIN_NAME.$DOMAIN` in both scripts. No test-set args (`--t_file`, `--t_search_itrs`, `--t_pathfinds`) are passed: deepxube 0.3.2's test-set code is commented out.

deepxube arg strings are `<value><name>` tokens joined by `_`; see `deepxube updater_info`, `trainer_info`, `nnet_info` for the names.

Output goes to `tmp/<DOMAIN>/<HEUR>/`:
- `heur.pt`, `heur_targ.pt` — current and target network
- `heur_status.pkl`, `heur_train_summary.pkl` — training state (resumes if present)
- `output.txt` — stdout log (nothing is printed to the terminal)
- `heur_tboard/` — TensorBoard logs

## Solving (`scripts/solve.sh`)

`bash scripts/solve.sh [config] [goals]` loads `tmp/<DOMAIN>/<HEUR>/heur.pt`, searches the goals with `SOLVE_PATHFIND` (e.g. `graph_v.100B_0.8W`) under `SOLVE_TIME_LIMIT` seconds per goal, and writes to `tmp/<DOMAIN>/<HEUR>/paths/<SOLVE_PATHFIND>/`:
- `results.pkl`, `output.txt` — deepxube solve output
- `summary.txt` — per-goal table (name, solved, gates, T-count, time, nodes generated), also printed; from `scripts/solve_summary.py --results <dir> [--goals <pkl>]`
- `goals.pkl` — only when the goals are a directory (below)

The goals (`SOLVE_GOALS`, or the optional second argument) are either a goals `.pkl` or a directory of `.txt` targets. A directory is first converted with `goals_from_txt.py --domain $DOMAIN_NAME` (so `qcircuit_exact` gets ring goals) into `goals.pkl`, which records the file names so the summary can label each goal; `SOLVE_REACHABLE_ONLY=1` adds `--reachable_only` (drops targets the gate set cannot reach exactly, see [data](data.md#reachability)). A `.pkl` without names is summarized by index.

Configs point at the benchmark directories: `n1_*` → `data/targets/1qubit`, `n2_exact*` → `data/targets/2qubit`, `n3_exact*` (incl. `n3_exact_ring`) → `data/targets/3qubit`; the `n2`/`n3` configs set `SOLVE_REACHABLE_ONLY=1`. `configs/test` still uses `tmp/n1_goals_R_100.pkl`. Example: `bash scripts/solve.sh configs/n1_e0.01 tmp/n1_goals_R_1K.pkl` solves the random 1K set instead.

## Goals from `.txt` Targets (`scripts/goals_from_txt.py`)

```bash
python scripts/goals_from_txt.py --input data/targets/3qubit --output goals.pkl [--domain qcircuit_exact] [--reachable_only] [--tol 1e-6]
```
`--input` takes files and/or directories (directories expand to their `*.txt`, sorted by name; all targets must have the same qubit count). Writes `{'states', 'goals', 'names', 'skipped'}`; `names` are the file stems. With `--domain qcircuit_exact`, targets not in Z[ω,1/√2] (e.g. `rz4`–`rz7`) are skipped; `--tol` is the fitting tolerance (the `.txt` files carry ~8 digits, e.g. `ch`).

## Goal Generation (`scripts/goal_gen.sh`)

```bash
deepxube problem_inst --domain qcircuit.n1_R --step_max 1 --num 1000 \
                      --file tmp/n1_goals_R_1K.pkl --redo
```
Generates 1000 random 1-qubit goals by random walks from identity.

## Converting Goals for the Exact Domain (`scripts/goals_to_exact.py`)

```bash
python scripts/goals_to_exact.py --input tmp/n3_goals.pkl --output tmp/n3_goals_exact.pkl
```
Fits each float unitary onto the ring lattice (any global phase) and skips goals that are not exactly synthesizable.

## Checking Goal Reachability (`scripts/check_goals.py`)

```bash
python scripts/check_goals.py tmp/n3_goals_all8.pkl --output tmp/n3_goals.pkl
```
Prints each goal's determinant as a power of ω and whether it is exactly reachable (see [data](data.md#reachability)); `--output` writes only the reachable goals.

## Trasyn Benchmark (`scripts/trasyn_bench.py`)

Runs the [Trasyn](https://github.com/eth-sri/synthetiq) baseline synthesizer on a goals `.pkl`:
```bash
python scripts/trasyn_bench.py <goals.pkl> --epsilon 0.01 --t_budget 30
```
Reports time, T-count, gate count, and error per goal.

## Converting Results to QASM (`scripts/paths_to_qasm.py`)

```bash
python scripts/paths_to_qasm.py --input <results.pkl> --output <dir>
```
Writes one `<i>.qasm` file per solved goal in OpenQASM 3.0 format (`i` = goal index; names are in the `goals.pkl` the solve used).
