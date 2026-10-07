"""Runs the Synthetiq baseline synthesizer on .txt targets and writes a per-goal CSV.

Two modes, matching the two DeepQubit domains:
  approximate (default): a circuit counts if its error <= --epsilon (default 0.01), as in QCircuit.is_solved.
  --exact:               a circuit counts if it equals the target over Z[omega, 1/sqrt2] up to a power of omega, as in
                         QCircuitExact.is_solved (synthetiq runs at --epsilon 1e-6). Targets not in the ring (rz4-rz7)
                         are skipped; --reachable_only also skips those whose determinant rules out exact synthesis
                         (cct, csqrtswap; see check_goals.py), leaving the n2/n3 benchmark goals.
error is unitary_distance. Synthetiq's own distance is unitary_distance / sqrt2 (ExactEqualityComputer in its cost.cpp,
PartialMatrix::exact_cost in its partial.rs), so it is passed epsilon / sqrt2 and accepts exactly the circuits counted
here (its paper's -eps 0.01 runs allowed unitary_distance up to 0.0141).

Targets are .txt files and/or directories, searched recursively for *.txt (.pkl goal files are ignored). Each target is
rewritten as a fully specified Synthetiq spec (the 1-qubit targets have no cover lines, which Synthetiq would read as
all-unspecified) and run in a fresh synthetiq process, from the Synthetiq checkout that holds --bin (found by walking up
to data/gates), with its CliffordT gate set {h, s, sdg, t, tdg, cx} = DeepQubit's CliffT_inv. --bin is the C++ bin/main
or the Rust bin/rust: both take the same flags and print the same output. A goal stops after --circuits circuits or
--time seconds. The circuits (OpenQASM 2.0, same qubit order as qiskit) are kept in <output stem>/<goal>/.

CSV columns: goal, num_qubits, solved, time (s to the first counted circuit, Synthetiq's own clock; the process wall
time if none), gate_count, t_count, t_depth (of that first circuit; T-depth as qiskit counts it), error, num_circuits
(counted), best_gate_count, best_t_count (minimums over the counted circuits), total_time (process wall time).
Default output: data/baselines/synthetiq_<inputs>_<e<epsilon>|exact>_<time>s_<circuits>c.csv

Usage: python scripts/synthetiq_bench.py [targets ...] [--bin <synthetiq binary>] [--exact] [--reachable_only]
                                         [--epsilon 0.01] [--time 100] [--circuits 10] [--threads 1] [--output <csv>]
       (default targets: data/targets; default --bin: $SYNTHETIQ_BIN, else ~/research/synthetiq/bin/rust)
"""
import os
import re
import csv
import glob
import shutil
import subprocess
import tempfile
from time import time
from typing import List, Dict, Tuple, Optional
from argparse import ArgumentParser

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from utils.matrix_utils import load_matrix_from_file, unitary_distance
from utils import ring
from check_goals import det_omega_power, is_reachable

RING_TOL: float = 1e-6  # .txt targets carry ~8 digits (same default as goals_from_txt.py --tol)


def expand_inputs(inputs: List[str]) -> List[str]:
    """directories expand to every *.txt file below them (sorted); files are kept in the given order"""
    files: List[str] = []
    for x in inputs:
        if os.path.isdir(x):
            files.extend(sorted(glob.glob(os.path.join(x, '**', '*.txt'), recursive=True)))
        elif x.endswith('.txt'):
            files.append(x)
    return files


def find_root(binary: str) -> str:
    """the Synthetiq checkout holding the binary: synthetiq resolves its gate sets relative to the working directory"""
    d = os.path.dirname(os.path.realpath(binary))
    while not os.path.isdir(os.path.join(d, 'data', 'gates')):
        if os.path.dirname(d) == d:
            raise SystemExit(f"no Synthetiq checkout (a data/gates folder) above {binary}")
        d = os.path.dirname(d)
    return d


def write_spec(U: NDArray, name: str, filename: str) -> None:
    """Synthetiq's .txt spec: name, qubit count, matrix, then an all-ones cover (fully specified target)"""
    n = int(np.log2(U.shape[0]))
    with open(filename, 'w') as f:
        f.write(f"{name}\n{n}\n")
        for row in U:
            f.write(' '.join('(%r,%r)' % (float(z.real), float(z.imag)) for z in row) + '\n')
        for _ in range(U.shape[0]):
            f.write(' '.join(['1'] * U.shape[0]) + '\n')


def ring_form(U: NDArray) -> Optional[Tuple[NDArray, int]]:
    """canonical Z[omega, 1/sqrt2] form (unique up to the global phase), or None if U is not in the ring"""
    try:
        return ring.from_complex(U, RING_TOL)
    except ValueError:
        return None


def run_synthetiq(binary: str, root: str, U: NDArray, name: str, out_dir: str, epsilon: float, time_limit: float,
                  circuits: int, threads: int) -> Tuple[List[Tuple[float, str]], float]:
    """Runs synthetiq on U; returns ([(seconds to find it, qasm file)] in the order found, process wall time)"""
    shutil.rmtree(out_dir, ignore_errors=True)  # stale circuits from an earlier run would be read as found
    os.makedirs(out_dir)
    with tempfile.TemporaryDirectory() as tmp:
        spec = os.path.join(tmp, f"{name}.txt")
        write_spec(U, name, spec)
        # the output folder is given relative to root: the C++ build fails to create an absolute one (it tries "")
        cmd = [binary, spec, '--absolute-input', '--absolute-output', '--output',
               os.path.relpath(os.path.realpath(out_dir), root),
               '--epsilon', repr(epsilon / 2 ** 0.5), '--time', str(time_limit), '--circuits', str(circuits),
               '--threads', str(threads)]
        start_time = time()
        try:
            proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=2 * time_limit + 60)
        except subprocess.TimeoutExpired as e:
            # synthetiq only checks its clock between annealing runs; keep whatever it printed and saved
            out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or '')
            proc = subprocess.CompletedProcess(cmd, 0, out, '')
            print(f"synthetiq overran its time limit on {name}; killed")
        wall_time = time() - start_time
    if proc.returncode != 0:
        raise SystemExit(f"synthetiq failed on {name} ({proc.returncode}):\n{proc.stderr}{proc.stdout[-2000:]}")

    # each found circuit prints "<output folder>/ <seconds since the previous one> <index>" and is saved as
    # <cost>-<count>-<depth>-<thread>-<index>.qasm
    gaps: Dict[int, float] = {int(m.group(2)): float(m.group(1))
                              for m in re.finditer(r'^.*/ (\S+) (\d+)$', proc.stdout, re.MULTILINE)}
    found: List[Tuple[float, str]] = []
    for f in glob.glob(os.path.join(out_dir, '*.qasm')):
        idx = int(os.path.splitext(f)[0].rsplit('-', 1)[1])
        found.append((sum(v for i, v in gaps.items() if i <= idx), f))
    return sorted(found), wall_time


def circuit_stats(filename: str) -> Tuple[NDArray, int, int, int]:
    """(unitary, gate count, T-count, T-depth) of a synthetiq .qasm circuit; its qubit order is qiskit's"""
    qc = QuantumCircuit.from_qasm_file(filename)
    is_t = lambda x: x.operation.name in ('t', 'tdg')
    return Operator(qc).data, len(qc.data), sum(1 for x in qc.data if is_t(x)), qc.depth(is_t)


def main():
    parser = ArgumentParser()
    parser.add_argument('targets', type=str, nargs='*', default=['data/targets'],
                        help='.txt targets and/or directories of them (searched recursively; .pkl files are ignored)')
    parser.add_argument('--bin', type=str,
                        default=os.environ.get('SYNTHETIQ_BIN', os.path.expanduser('~/research/synthetiq/bin/rust')),
                        help='synthetiq binary (bin/main or bin/rust inside a Synthetiq checkout); default $SYNTHETIQ_BIN')
    parser.add_argument('--exact', action='store_true',
                        help='exact synthesis: count circuits equal to the target over the Clifford+T ring; skips '
                             'targets not in the ring')
    parser.add_argument('--reachable_only', action='store_true',
                        help='skip targets whose determinant rules out exact Clifford+T synthesis (see check_goals.py)')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='unitary_distance tolerance; default 0.01, or 1e-6 with --exact')
    parser.add_argument('--time', type=float, default=100, help='time limit per goal (s)')
    parser.add_argument('--circuits', type=int, default=10, help='stop a goal after this many circuits')
    parser.add_argument('--threads', type=int, default=1, help='synthetiq threads per goal (seeded runs need 1)')
    parser.add_argument('--output', type=str, default=None,
                        help='default: data/baselines/synthetiq_<inputs>_<e<epsilon>|exact>_<time>s_<circuits>c.csv')
    args = parser.parse_args()
    epsilon = args.epsilon if args.epsilon is not None else (1e-6 if args.exact else 0.01)

    binary = os.path.abspath(os.path.expanduser(args.bin))
    if not os.access(binary, os.X_OK):
        raise SystemExit(f"synthetiq binary not found: {binary} (pass --bin or set SYNTHETIQ_BIN)")
    root = find_root(binary)
    files = expand_inputs(args.targets)
    if len(files) == 0:
        raise SystemExit(f"no .txt targets found in {args.targets}")

    stem = '+'.join(os.path.splitext(os.path.basename(os.path.normpath(x)))[0] for x in args.targets)
    mode = 'exact' if args.exact else f"e{epsilon}"
    output = args.output or f"data/baselines/synthetiq_{stem}_{mode}_{args.time:g}s_{args.circuits}c.csv"
    circuits_dir = os.path.splitext(output)[0]
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    print('Running Synthetiq (%s) %s benchmark, epsilon=%.2e, %gs / %d circuits per goal, on %d targets' %
          (binary, 'exact' if args.exact else 'approximate', epsilon, args.time, args.circuits, len(files)))
    rows = []
    with open(output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['goal', 'num_qubits', 'solved', 'time', 'gate_count', 't_count', 't_depth', 'error',
                         'num_circuits', 'best_gate_count', 'best_t_count', 'total_time'])
        for filename in files:
            name = os.path.splitext(os.path.basename(filename))[0]
            num_qubits, U = load_matrix_from_file(filename)
            target_ring = ring_form(U) if args.exact else None
            if args.exact and target_ring is None:
                print(f"skipped {name}: not in the Clifford+T ring")
                continue
            if args.reachable_only and not is_reachable(U):
                print(f"skipped {name}: not exactly reachable (det = omega^{det_omega_power(U)})")
                continue

            found, wall_time = run_synthetiq(binary, root, U, name, os.path.join(circuits_dir, name), epsilon,
                                             args.time, args.circuits, args.threads)
            # (time, gate count, T-count, T-depth, error) of each circuit that passes the check
            counted, rejected = [], []
            for synth_time, qasm in found:
                V, gate_count, t_count, t_depth = circuit_stats(qasm)
                err = unitary_distance(U, V)
                if args.exact:
                    V_ring = ring_form(V)
                    ok = V_ring is not None and V_ring[1] == target_ring[1] and np.array_equal(V_ring[0], target_ring[0])
                else:
                    ok = err <= epsilon
                (counted if ok else rejected).append((synth_time, gate_count, t_count, t_depth, err))
            if len(rejected) > 0:
                print('Goal %s: %d synthetiq circuit(s) fail the check (errors %s)' %
                      (name, len(rejected), ', '.join('%.3e' % x[-1] for x in rejected)))

            if len(counted) == 0:
                writer.writerow([name, num_qubits, 0, '%.4f' % wall_time] + [''] * 4 + [0, '', '', '%.4f' % wall_time])
                f.flush()
                rows.append((wall_time, False))
                print('Goal %s | not solved | time: %.3f' % (name, wall_time))
                continue
            synth_time, gate_count, t_count, t_depth, err = counted[0]
            best_gate_count, best_t_count = min(x[1] for x in counted), min(x[2] for x in counted)
            rows.append((synth_time, True, best_gate_count, best_t_count))
            # written as it goes, so an interrupted run keeps its finished goals
            writer.writerow([name, num_qubits, 1, '%.4f' % synth_time, gate_count, t_count, t_depth, '%.6e' % err,
                             len(counted), best_gate_count, best_t_count, '%.4f' % wall_time])
            f.flush()
            print('Goal %s | time: %.3f | gate count: %i | T-count: %i | T-depth: %i | error: %.3e | '
                  '%i circuits in %.1fs, best gate count %i, best T-count %i' %
                  (name, synth_time, gate_count, t_count, t_depth, err, len(counted), wall_time, best_gate_count,
                   best_t_count))

    solved_rows = [x for x in rows if x[1]]
    print('Solved %.1f%% (%d/%d)' % (100 * len(solved_rows) / max(len(rows), 1), len(solved_rows), len(rows)), end='')
    if len(solved_rows) > 0:
        times, _, gate_counts, t_counts = [np.array(x) for x in zip(*solved_rows)]
        print(' | mean time to first circuit: %.3f | mean best gate count: %.2f | mean best T-count: %.2f' %
              (times.mean(), gate_counts.mean(), t_counts.mean()), end='')
    print('\nWrote %s (circuits in %s/)' % (output, circuits_dir))


if __name__ == '__main__':
    main()
