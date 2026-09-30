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
| `HEUR` | `resnet_fc.1000H_4B_bn` | Network architecture (`--fn <kind>,$HEUR`) |
| `PATHFIND` | `graph_v.1B_1W_0.0E` | Search used during training; `*_q` selects a Q-function (`heurq_fixout`/`up_rl_q`), otherwise V (`heurv`/`up_rl_v`) |
| `BATCH_SIZE`, `MAX_ITRS`, `CHECKPOINT` | 10000, 200000, 100 | Trainer args (`--tr tr_h.<bs>bs_<maxit>maxit_<chkpt>chkpt`) |
| `PROCS`, `STEP_MAX`, `SEARCH_ITRS`, `UP_ITRS`, `UP_GEN_ITRS` | 24, 30, 1000, 100, 100 | Updater args (`--up up_rl_v.<p>p_<sm>sm_<sitrs>sitrs_<up>up_<upg>upg`) |
| `TEST_FILE`, `TEST_SEARCH_ITRS` | `tmp/n3_goals.pkl`, 100 | Test-set args (accepted but unused by deepxube 0.3.2) |

`configs/n3_exact` is the 3-qubit exact run (`n3_e0.000001_I`); `configs/n3_exact_G` is the same with structured macro-word goals (`n3_e0.000001_I_G`, see [domain](domain.md#structured-macro-goals--flag-g)); `configs/n3_exact_ring` runs the integer-ring domain (`DOMAIN_NAME="qcircuit_exact"`, `n3_I_B9+C2`, random-walk goals only for now, goals `data/n3_goals_exact.pkl`, see [exact](exact.md)).

`DOMAIN_NAME` (default `qcircuit`) selects the domain class passed as `--domain $DOMAIN_NAME.$DOMAIN` in both scripts.

deepxube arg strings are `<value><name>` tokens joined by `_`; see `deepxube updater_info`, `trainer_info`, `nnet_info` for the names.

Output goes to `tmp/<DOMAIN>/<HEUR>/`:
- `heur.pt`, `heur_targ.pt` — current and target network
- `heur_status.pkl`, `heur_train_summary.pkl` — training state (resumes if present)
- `output.txt` — stdout log (nothing is printed to the terminal)
- `heur_tboard/` — TensorBoard logs

## Solving (`scripts/solve.sh`)

`bash scripts/solve.sh [config]` loads `tmp/<DOMAIN>/<HEUR>/heur.pt`, searches `SOLVE_GOALS` with `SOLVE_PATHFIND` (e.g. `graph_v.100B_0.8W`) under `SOLVE_TIME_LIMIT` seconds per goal, and writes `results.pkl` and `output.txt` to `tmp/<DOMAIN>/<HEUR>/paths/<SOLVE_PATHFIND>/`.

## Goal Generation (`scripts/goal_gen.sh`)

```bash
deepxube problem_inst --domain qcircuit.n1_R --step_max 1 --num 1000 \
                      --file tmp/n1_goals_R_1K.pkl --redo
```
Generates 1000 random 1-qubit goals by random walks from identity.

## Converting Goals for the Exact Domain (`scripts/goals_to_exact.py`)

```bash
python scripts/goals_to_exact.py --input tmp/n3_goals.pkl --output data/n3_goals_exact.pkl
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
Writes one `<i>.qasm` file per solved goal in OpenQASM 3.0 format.
