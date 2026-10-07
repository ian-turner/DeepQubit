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

`configs/n3_exact` is the 3-qubit exact run (`n3_e0.000001_I`); `configs/n3_exact_ring` runs the integer-ring domain (`DOMAIN_NAME="qcircuit_exact"`, domain `n3_I_Z9` (compact int16 input; same features as `n3_I_B9` with 18× less data per state), network `resnet_fc_ring.1000H_4B_bn_2C` which expands the bits and computes the channel features on the GPU, solves `data/targets/3qubit` converted to ring goals, see [exact](exact.md)). Runs from before 2026-10-05 used `n3_I_B9` and live in `tmp/n3_I_B9/...`; their checkpoints load unchanged into the Z9 run directory. `configs/n2_exact_ring` is the same ring setup on 2 qubits (`n2_I_Z9`, solves `data/targets/2qubit` with a 100 s limit; all five targets are within 7 gates, and `crk_2` is the same unitary as `cs`).

`DOMAIN_NAME` (default `qcircuit`) selects the domain class passed as `--domain $DOMAIN_NAME.$DOMAIN` in both scripts. No test-set args (`--t_file`, `--t_search_itrs`, `--t_pathfinds`) are passed: deepxube 0.3.2's test-set code is commented out.

deepxube arg strings are `<value><name>` tokens joined by `_`; see `deepxube updater_info`, `trainer_info`, `nnet_info` for the names.

Output goes to `tmp/<DOMAIN>/<HEUR>/`:
- `heur.pt`, `heur_targ.pt` — current and target network
- `heur_status.pkl`, `heur_train_summary.pkl` — training state (resumes if present)
- `output.txt` — stdout log (nothing is printed to the terminal; appended on resume)
- `heur_tboard/` — TensorBoard logs (`train/loss` per iteration; `train/pathfind/*`, `train/ctgs/*` per update)
- `heur_chkpt_<itr>.pt` — weights every `CHECKPOINT` updates

`heur_train_summary.pkl` is a deepxube `TrainSummary`: `itr_to_steps_to_pathfindstats[itr][step]` holds `per_solved`, `path_costs`, `search_itrs`, `ctgs_backup`, `num_instances` for each update (keyed by the iteration it started at) and each random-walk length; `itr_to_in_out[itr]` holds the (target, prediction) arrays of that update's first training batch. Only searches that finished during the update (solved, or out of `SEARCH_ITRS`) are counted.

## Training Progress Graphs (`scripts/train_report.py`)

```bash
python scripts/train_report.py [--dir tmp] [--bins 100]
```
Graphs % solved per update, the same number as `output.txt`'s `Data - %solved` (mean over random-walk lengths of the % of that update's training searches solved; not a test set). Every run under `tmp/` (any directory with a `heur_train_summary.pkl`) gets `data/training/<DOMAIN>_<HEUR>.csv` (`update, itr, per_solved`), unless that CSV already has more updates (a CSV from the cluster is not overwritten by a short local run of the same name). The graphs read only these CSVs, so runs trained on the cluster need just their committed CSVs, not the model files. Each entry of `COMPARISONS` in the script — a list of (legend label, config) pairs, resolved to `tmp/$DOMAIN/$HEUR` by sourcing the config — gets `paper/images/<name>.png` (300 dpi): each run is averaged over bins of consecutive updates, about `--bins` points per run (same bin width for every run in a graph). Runs without a CSV are listed and left out, and a graph is drawn once two of its runs have one; a run's color is its position in the list, so it doesn't change as runs are added. Current comparisons: `n1_e0.01_encodings` (M, H, Q, H+M, Q+M, H+Q, H+M+Q) and `n1_e0.01_nerf_{M,H,H+M+Q}` (no NeRF vs. L5–L20, in a light-to-dark blue ramp). Unpickling needs deepxube installed.

## Solving (`scripts/solve.sh`)

`bash scripts/solve.sh [config] [goals]` loads `tmp/<DOMAIN>/<HEUR>/heur.pt`, searches the goals with `SOLVE_PATHFIND` (e.g. `graph_v.100B_0.8W`) under `SOLVE_TIME_LIMIT` seconds per goal, and writes to `tmp/<DOMAIN>/<HEUR>/paths/<SOLVE_PATHFIND>/`:
- `results.pkl`, `output.txt` — deepxube solve output
- `summary.txt` — per-goal table (name, solved, gates, T-count, time, nodes generated), also printed; from `scripts/solve_summary.py --results <dir> [--goals <pkl>]`
- `goals.pkl` — only when the goals are a directory (below)

The goals (`SOLVE_GOALS`, or the optional second argument) are either a goals `.pkl` or a directory of `.txt` targets. A directory is first converted with `goals_from_txt.py --domain $DOMAIN_NAME` (so `qcircuit_exact` gets ring goals) into `goals.pkl`, which records the file names so the summary can label each goal; `SOLVE_REACHABLE_ONLY=1` adds `--reachable_only` (drops targets the gate set cannot reach exactly, see [data](data.md#reachability)). A `.pkl` without names is summarized by index.

Configs point at the benchmark directories: `n1_*` → `data/targets/1qubit`, `n2_exact*` → `data/targets/2qubit`, `n3_exact*` (incl. `n3_exact_ring`) → `data/targets/3qubit`; the `n2`/`n3` configs set `SOLVE_REACHABLE_ONLY=1`. `configs/test` solves `data/targets/1qubit/random_1000.pkl`. Example: `bash scripts/solve.sh configs/n1_e0.01 data/targets/1qubit/random_1000.pkl` solves the random 1K set instead.

## Goals from `.txt` Targets (`scripts/goals_from_txt.py`)

```bash
python scripts/goals_from_txt.py --input data/targets/3qubit --output goals.pkl [--domain qcircuit_exact] [--reachable_only] [--tol 1e-6]
```
`--input` takes files and/or directories (directories expand to their `*.txt`, sorted by name; all targets must have the same qubit count). Writes `{'states', 'goals', 'names', 'skipped'}`; `names` are the file stems. With `--domain qcircuit_exact`, targets not in Z[ω,1/√2] (e.g. `rz4`–`rz7`) are skipped; `--tol` is the fitting tolerance (the `.txt` files carry ~8 digits, e.g. `ch`).

## Random Goals (`scripts/goal_gen.py`)

```bash
python scripts/goal_gen.py [--num_qubits 1] [--num 1000] [--seed 0] [--output <pkl>]
```
Samples `--num` Haar-random 2^n×2^n unitaries with qiskit's `random_unitary` (seeded, so reproducible) as goals from the identity, in deepxube's `{'states', 'goals'}` layout. Default output `data/targets/<n>qubit/random_<num>.pkl`. Replaces the old `R` domain flag and `goal_gen.sh` (`deepxube problem_inst` on `qcircuit.n1_R`).

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

Runs the [trasyn](https://github.com/haoty/trasyn) baseline synthesizer on a goals `.pkl` (default `data/targets/1qubit/random_1000.pkl`):
```bash
python scripts/trasyn_bench.py [goals.pkl] [--epsilon 0.01] [--t_budget 30] [--output <csv>]
```
Writes `data/baselines/trasyn_n<N>_<goals stem>_e<epsilon>_<t_budget>T.csv` with columns `goal, time, t_count, gate_count, error, solved` (`goal` is the pkl's name for it, else its index; `error` is `unitary_distance`, so `solved` = `error ≤ ε` matches `is_solved`), one row per goal written as it goes, then prints the % solved and mean time/T-count/gate count. The first goal is synthesized once untimed before the loop (printed as `Warm-up`): trasyn's first call in a process pays one-time startup costs (~4 s on the cluster GPU vs ~0.2 s per goal after); it keeps no search state between calls, so later goals are not sped up. One qubit uses `trasyn.synthesize`; more qubits go through `synthesize_qiskit_circuit` (transpiles to rotations + CNOTs, so even exact Clifford+T targets like `cs` come out as long approximate circuits). Without cupy, trasyn falls back to numpy on the CPU (~3 s per 1-qubit goal at ε = 0.01).

## Comparing with Trasyn (`scripts/trasyn_compare.py`)

```bash
python scripts/trasyn_compare.py [summary.txt ...] [--trasyn <csv>] [--labels <label> ...] [--fig <prefix>]
```
Pairs one or more solve summaries (`summary.txt` from `solve.sh`, or the copies in `data/results/`; default `data/results/1qubit/e0.01_L10_random_100B_0.0W.txt`) with a `trasyn_bench.py` CSV (default `data/baselines/trasyn_n1_random_1000_e0.01_30T.csv`) by goal name, on the goals in every file (a warning is printed if a file has others, e.g. an interrupted run). Prints per method the goals solved, mean/median T-count and gate count (over its solved goals) and time; then, per run on the goals both solved, how many goals ours has fewer/equal/more T gates and gates for, and the mean difference (ours − trasyn). Draws the goals at each T-count and gate-count difference as two figures, since their counts differ in scale: `paper/images/compare_<trasyn csv stem>_t_count.png` and `…_gate_count.png` (`--fig` sets the prefix; one color per run, legend for ≥ 2). trasyn's default gate set `tshxyz` is our `CliffT`, so gate counts compare directly; both sides must use the same ε. On random_1000 at ε = 0.01 (L10 model, `100B_0.0W`): both solve 1000/1000; T-count is equal on 897 goals and 1–3 higher on 103 (mean +0.11); gate count is lower on 691 and never higher (mean −1.21).

## Synthetiq Benchmark (`scripts/synthetiq_bench.py`)

Runs the [Synthetiq](https://github.com/eth-sri/synthetiq) baseline on `.txt` targets (files or directories, searched recursively; `.pkl` files are ignored; default `data/targets`). The binary is a parameter, since the checkout lives elsewhere on the cluster: `--bin` (default `$SYNTHETIQ_BIN`, else `~/research/synthetiq/bin/rust`), either the C++ `bin/main` or the Rust `bin/rust`; it is run from the checkout holding it (found by walking up to `data/gates`).
```bash
python scripts/synthetiq_bench.py data/targets/1qubit --epsilon 0.01 --time 10 --bin <synthetiq>/bin/main        # approximate
python scripts/synthetiq_bench.py data/targets/3qubit --exact --reachable_only --time 10 --bin <synthetiq>/bin/main # exact
```
- **Approximate** (default): a circuit counts if `unitary_distance ≤ ε` (default 0.01), as in `QCircuit.is_solved`. **`--exact`**: it counts if it equals the target over Z[ω, 1/√2] up to ω^k (canonical `ring.from_complex` forms), as in `QCircuitExact.is_solved` (ε defaults to 1e-6); targets not in the ring (rz4–rz7) are skipped, and `--reachable_only` also skips cct/csqrtswap (same goals as the n2/n3 benchmarks).
- Synthetiq's own distance is `unitary_distance / √2`, so it is passed `ε/√2` (its paper's `-eps 0.01` allowed `unitary_distance` up to 0.0141).
- Each target is rewritten as a fully specified spec (the 1-qubit `.txt` targets have no cover lines, which Synthetiq would read as all-unspecified). Gate set `CliffordT` = {h, s, sdg, t, tdg, cx} = our `CliffT_inv` (`I`). A goal stops after `--circuits` (default 10, Synthetiq's default) or `--time` s (default 100); `--threads` default 1 (only single-threaded runs are seeded).
- Writes `data/baselines/synthetiq_<inputs>_<e<ε>|exact>_<time>s_<circuits>c.csv` with columns `goal, num_qubits, solved, time, gate_count, t_count, t_depth, error, num_circuits, best_gate_count, best_t_count, total_time`: `time` and the gate/T stats are for the first counted circuit (time from Synthetiq's own clock; the wall time if none), `best_*` are minimums over all counted circuits (the first is often far from the best: e.g. 12 gates for rz3 = T), `total_time` is the process wall time. The circuits (OpenQASM 2.0, qiskit qubit order) stay in `data/baselines/<csv stem>/<goal>/`.

## Converting Results to QASM (`scripts/paths_to_qasm.py`)

```bash
python scripts/paths_to_qasm.py --input <results.pkl> --output <dir>
```
Writes one `<i>.qasm` file per solved goal in OpenQASM 3.0 format (`i` = goal index; names are in the `goals.pkl` the solve used).
