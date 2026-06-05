# Training & Solving

## Setup

```bash
source setup.sh   # adds repo root to PYTHONPATH; run in every new shell
```

## Training (`scripts/train.sh`)

Calls `deepxube train` with:

| Variable | Example | Meaning |
|----------|---------|---------|
| `DOMAIN` | `n1_L15_P_H_e0.001` | Domain string (parsed by `QCircuitParser`) |
| `HEUR` | `resnet_fc.2000H_2B_bn` | Neural network architecture |
| `PATHFIND` | `bwqs.1_0.8_0.1` | Path-finding algorithm config |
| `BATCH_SIZE` | 200 | Training batch size |
| `MAX_ITRS` | 100000 | Max training iterations |
| `STEP_MAX` | 1000 | Max walk length for goal generation |
| `SEARCH_ITRS` | 1000 | Search iterations during training |
| `PROCS` | 4 | Parallel processes |

Output goes to `tmp/<DOMAIN>/<HEUR>/`:
- `heur.pt` — saved heuristic model
- `heur_targ.pt` — target network
- `status.pkl` — training state (resumes if present)
- `output.txt` — stdout log
- `events.out.tfevents.*` — TensorBoard logs

## Solving (`scripts/solve.sh`)

Calls `deepxube solve`. Loads a goal `.pkl`, runs search with a trained heuristic, writes results.

Key flags: `--heur_file`, `--file` (goals pkl), `--results` (output dir), `--time_limit`, `--redo`.

## Goal Generation (`scripts/goal_gen.sh`)

```bash
deepxube problem_inst --domain qcircuit.n1_R --step_max 1 --num 1000 \
                      --file tmp/n1_goals_R_1K.pkl --redo
```
Generates 1000 random 1-qubit goals by random walks from identity.

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

## heur_type: QFix

`--heur_type QFix` is used in all training/solving scripts. This selects the fixed-action-set heuristic variant, matching `HasFlatSGActsEnumFixedIn` implemented by `QCircuit`.
