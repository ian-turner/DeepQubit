"""Writes data/targets/reference_values.csv: published and baseline reference values for the exact benchmark targets.

Columns: target, num_qubits, source, metric, value, tool, citation, optimality, notes
- target: the .txt target as <dir>/<stem> under data/targets (3qubit/toffoli, 3qubit_perms/7, 4qubit/U1, ...). Gates
  without a target file (not reachable exactly without an ancilla, ancilla constructions, undefined in their source)
  keep a bare name (CT, AND, 'Toffoli (1 ancilla)', ...) and say why in notes.
- metric: t_count, t_depth, gate_count, depth (gate_count/depth over {h, s, sdg, t, tdg, cx} = CliffT_inv, depth as
  qiskit counts it), or time_s (the source's own run time on its own hardware). An empty value = the tool failed.
- optimality: proven_optimal (the value meets a proven lower bound, given in notes), best_known (the smallest value in
  this table for the target and metric), reported (larger, a failure, a time, or a value below a proven lower bound,
  which the notes flag as inconsistent).
Values are qubit-order independent, so a circuit for the qubit-reversed operator (Synthetiq's specs are little-endian)
counts for our big-endian target.

Lower bounds: rows marked proven (Gosset et al.'s T-count 7 for Toffoli/Fredkin, carried to every gate that is
Clifford-equivalent to Toffoli, each checked here against its target file; Amy et al.'s depth-optimal circuits;
Clifford targets: T-count 0), T-count/T-depth >= 1 for a non-Clifford target, and T-depth >= ceil(T-count / qubits)
(a T layer holds at most one T per qubit), which makes T-depth 3 optimal for the 3-qubit Toffoli class.

Sources: Rietsch et al. 2024 Table II (v4), Amy et al. 2013 (figures of Sec. 6), Gosset et al. 2014,
Mosca & Mukhopadhyay 2021 Table 1 (MIN-T-SYNTH), all transcribed below; from the Synthetiq checkout (--synthetiq):
circuits/62/table4.csv, circuits/64/table6.csv, the circuits in circuits/62 and circuits/64 (counted with qiskit and
checked against our target files up to global phase and qubit order), baselines/mosca (Mosca-Mukhopadhyay T-count:
the length of the printed path), baselines/gheorgiu (Gheorghiu et al. T-depth: the 'T-depth: k' line) and the
per-class T-count/T-depth Synthetiq reports for the 30 permutations (its notebooks/post_processing/64.ipynb);
qiskit's RCCXGate/RC3XGate definitions for rccx/rcccx.

Usage: python scripts/make_reference_values.py [--synthetiq ~/research/synthetiq] [--output data/targets/reference_values.csv]
"""
import csv
import glob
import math
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Operator

from utils.matrix_utils import load_matrix_from_file

COLUMNS = ['target', 'num_qubits', 'source', 'metric', 'value', 'tool', 'citation', 'optimality', 'notes']
METRICS = ['t_count', 't_depth', 'gate_count', 'depth', 'time_s']
BASIS = ['h', 's', 'sdg', 't', 'tdg', 'cx']

RIETSCH = 'Rietsch et al. 2024 (IEEE QCE 2024; arXiv:2404.14865v4)'
AMY = 'Amy Maslov Mosca Roetteler 2013 (IEEE TCAD 32(6):818-830; arXiv:1206.0758)'
GOSSET = 'Gosset Kliuchnikov Mosca Russo 2014 (QIC 14(15&16):1261-1276; arXiv:1308.4134)'
MOSCA = 'Mosca & Mukhopadhyay 2021 (Quantum Sci. Technol. 7(1); doi:10.1088/2058-9565/ac2d3a; arXiv:2006.12440)'
GHEORGHIU = 'Gheorghiu Mosca Mukhopadhyay 2022 (npj Quantum Inf. 8:110; arXiv:2101.03142)'
SYNTHETIQ = 'Paradis Dekoninck Bichsel Vechev 2024 (OOPSLA; Synthetiq)'
MASLOV = 'Maslov 2016 (Phys. Rev. A 93:022311; arXiv:1508.03273)'

# Rietsch et al. 2024 Table II (v4): (gate as named in the paper, target, qubits, (T-count, time s) per tool; None = '-')
RIETSCH_TOOLS = ['Gumbel AlphaZero', 'MIN-T-SYNTH', 'QCOpt', 'Synthetiq']
RIETSCH_TABLE = [
    ('CV', '2qubit/csx', 2, [(3, 9.82), None, (3, 4.16), (3, 0.089)]),
    ('CY', '2qubit/cy', 2, [(0, 9.50), (0, 0.224), (0, 16.84), (0, 0.084)]),
    ('CZ', '2qubit/cz', 2, [(0, 9.41), (0, 0.330), (0, 0.1), (0, 0.08)]),
    ('SWAP', '2qubit/swap', 2, [(0, 9.27), (0, 0.353), (0, 0.01), (0, 0.08)]),
    ('W', '2qubit/w', 2, [(2, 9.62), None, (0, 3221.21), (2, 0.087)]),
    ('CP', '2qubit/cs', 2, [(3, 10.69), (3, 0.007), (3, 6.15), (3, 0.082)]),
    ('CH', '2qubit/ch', 2, [(2, 16.23), None, (2, 147.12), (2, 0.086)]),
    ('CT', 'CT', 2, [None, None, None, None]),
    ('Toffoli', '3qubit/toffoli', 3, [(7, 25.92), (7, 4.99), (7, 10800), (7, 0.238)]),
    ('Single Negated Toffoli', '3qubit/toffoli_neg1', 3, [(7, 16.83), (7, 5.70), None, (7, 0.315)]),
    ('Double Negated Toffoli', '3qubit/toffoli_neg2', 3, [(7, 16.78), (7, 3.39), None, (7, 0.252)]),
    ('Fredkin', '3qubit/fredkin', 3, [(7, 16.96), (7, 4.85), None, (7, 0.779)]),
    ('Peres', '3qubit/peres', 3, [(7, 16.87), (7, 5.00), None, (7, 0.321)]),
    ('QOR', '3qubit/qor', 3, [(7, 16.56), (7, 5.09), None, (7, 0.250)]),
    ('AND', 'AND', 3, [(7, 16.83), (7, 3.44), None, (7, 0.498)]),
    ('TR', '3qubit/tr', 3, [(7, 16.19), (7, 3.47), None, (7, 0.342)]),
    ('3 Toffoli (U2)', '4qubit/U2', 4, [(7, 18.54), (7, 309.84), None, (7, 12.058)]),
    ('1-bit adder', '4qubit/adder', 4, [(7, 18.53), (7, 312.55), None, (8, 24.365)]),
    ('2 Peres', '2 Peres', 4, [(7, 18.85), (7, 312.55), None, (7, 13.495)]),
    ('2 Toffoli (U1)', '4qubit/U1', 4, [(12, 19.27), (11, 6642.16), None, None]),
    ('Toffoli (1 ancilla)', 'Toffoli (1 ancilla)', 4, [(7, 31.50), (7, 233.54), None, (7, 4.132)]),
    ('x-Mod 5', 'x-Mod 5', 5, [(12, 60.773), None, None, None]),
]
RIETSCH_TOOL_NOTES = {'Synthetiq': '800 s limit on 48 cores', 'QCOpt': '3 h limit'}
RIETSCH_FAILED = {'MIN-T-SYNTH': 'not run: would need code changes'}  # others: time limit or no convergence

# how a target relates to the gate a source names, when that needs saying (applied to the literature rows)
TARGET_NOTES = {
    'CT': 'no target file: not exactly reachable without an ancilla (det omega)',
    'AND': 'no target file: no definition found in Rietsch et al. or its references (TODO)',
    '2 Peres': "no target file: not defined by Rietsch et al.; Amy et al.'s 1-bit adder (Fig. 11) = 4qubit/adder is "
               "itself two Peres gates",
    'Toffoli (1 ancilla)': 'no target file: uses a clean ancilla',
    'x-Mod 5': 'no target file: 5 qubits; not defined by Rietsch et al.',
    'CT (1 ancilla)': 'no target file: uses a clean ancilla',
    '3qubit/toffoli_neg2': 'source gives no definition; target assumes both controls negated',
    '3qubit/tr': 'source gives no definition; target is the Thapliyal-Ranganathan gate (same name and T-count)',
    '4qubit/U1': "source's U1 = (TOF x I)(I x TOF) (Mosca & Mukhopadhyay); our file (Synthetiq's) is its inverse: same "
                 "T-count and T-depth",
    '4qubit/U2': "source's U2 = (TOF x I)(I x TOF)(TOF x I); our file (Synthetiq's) is CX(0;1) TOF(2,3;0) CX(0;1): both "
                 "are CNOT conjugates of one Toffoli, so T-count and T-depth agree but other counts may not",
    '4qubit/adder': "assumed to be Amy et al.'s 1-bit adder (Fig. 11), which our file is up to qubit order",
}

# Amy et al. 2013: (target, figure, t_count, t_depth, depth); Figs. 4-7 are total-depth optimal over
# {H, P, P^dag, CNOT, T, T^dag} (Sec. 6.1), and Sec. 6.2 shows CP, CV and W T-depth optimal
AMY_TABLE = [
    ('2qubit/cy', '4b', None, None, 3), ('2qubit/cz', '4c', None, None, 3),
    ('2qubit/ch', '5a', 2, 2, 7), ('2qubit/cs', '5b', 3, 2, 4), ('2qubit/csx', '5c', 3, 2, 5), ('2qubit/w', '6', 2, 1, 9),
    ('3qubit/toffoli', '7a', 7, 4, 8), ('3qubit/toffoli_neg1', '7b', 7, 4, 8), ('3qubit/qor', '7c', 7, 4, 8),
    ('3qubit/peres', '7d', 7, 4, 8), ('3qubit/fredkin', '7e', 7, 4, 10),
    ('4qubit/adder', '11', 8, 2, None), ('2qubit/ch', '12', 2, 1, 9), ('3qubit/toffoli', '13', 7, 3, 9),
    ('CT (1 ancilla)', '14', 9, 3, None), ('Toffoli (1 ancilla)', '15', 7, 2, None),
]
AMY_DEPTH_OPTIMAL = {'4b', '4c', '5a', '5b', '5c', '6', '7a', '7b', '7c', '7d', '7e'}
AMY_TDEPTH_OPTIMAL = {'5b', '5c', '6'}
AMY_FIG_NOTES = {'11': 'peephole-optimized', '13': 'T-depth 3 conjectured minimal by the authors',
                 '14': '1 ancilla', '15': '1 ancilla', '12': 'minimum-T-depth CH'}

# Mosca & Mukhopadhyay 2021 Table 1 (their MIN-T-SYNTH): (target, T-count, time s)
MOSCA_TABLE = [('3qubit/toffoli', 7, 5.75), ('3qubit/fredkin', 7, 5.9), ('3qubit/peres', 7, 5.74),
               ('3qubit/qor', 7, 5.74), ('3qubit/toffoli_neg1', 7, 5.75), ('4qubit/adder', 7, 429.17),
               ('4qubit/U1', 11, 2.17 * 3600), ('4qubit/U2', 7, 391.27)]

# gates equal to C1 Toffoli C2 for Clifford C1, C2 (circuits on qubits 0-2, big-endian; checked against the files up to
# qubit order): Gosset et al.'s T-count 7 for Toffoli carries over
TOFFOLI_CLASS = {
    '3qubit/toffoli': [('ccx', 0, 1, 2)],
    '3qubit/ccz': [('h', 2), ('ccx', 0, 1, 2), ('h', 2)],
    '3qubit/fredkin': [('cx', 2, 1), ('ccx', 0, 1, 2), ('cx', 2, 1)],
    '3qubit/peres': [('ccx', 0, 1, 2), ('cx', 0, 1)],
    '3qubit/toffoli_neg1': [('x', 0), ('ccx', 0, 1, 2), ('x', 0)],
    '3qubit/toffoli_neg2': [('x', 0), ('x', 1), ('ccx', 0, 1, 2), ('x', 0), ('x', 1)],
    '3qubit/qor': [('x', 0), ('x', 1), ('ccx', 0, 1, 2), ('x', 0), ('x', 1), ('x', 2)],
    '3qubit/tr': [('x', 1), ('ccx', 0, 1, 2), ('cx', 0, 1), ('x', 1)],
    '3qubit/maj': [('cx', 2, 1), ('cx', 2, 0), ('ccx', 0, 1, 2)],
    '3qubit/uma': [('ccx', 0, 1, 2), ('cx', 2, 0), ('cx', 0, 1)],
    '3qubit_perms/1': [('ccx', 0, 1, 2)],
    '3qubit_perms/2': [('cx', 2, 1), ('ccx', 0, 1, 2), ('cx', 2, 1)],
}
# the 4-qubit embeddings (qubit 3 idle) inherit the 3-qubit circuits as upper bounds only
EMBEDDED = {'4qubit/toffoli': [('ccx', 0, 1, 2)], '4qubit/ccz': [('h', 2), ('ccx', 0, 1, 2), ('h', 2)]}

# Synthetiq's 30 permutation classes in four groups with the T-count/T-depth Synthetiq found for each
# (groups_same / normal_data in its notebooks/post_processing/64.ipynb)
PERM_GROUPS = {0: [0], 1: [1, 2, 5, 6, 13, 22, 29], 2: [3, 4, 7, 8, 11, 12, 15, 16, 19, 20, 23, 24, 27, 28],
               3: [9, 10, 14, 17, 18, 21, 25, 26]}
PERM_GROUP_VALUES = {0: (0, 0), 1: (7, 3), 2: (8, 3), 3: (15, 7)}


def row(target, n, source, metric, value, tool, citation, notes='', proven=False) -> Dict:
    return {'target': target, 'num_qubits': n, 'source': source, 'metric': metric, 'value': value, 'tool': tool,
            'citation': citation, 'optimality': 'proven_optimal' if proven else '', 'notes': notes}


def join(*notes: str) -> str:
    return '; '.join(x for x in notes if x)


def circuit(n: int, ops) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for op, *args in ops:
        getattr(qc, op)(*args)
    return qc


def same_operator(U: np.ndarray, V: np.ndarray) -> Optional[str]:
    """'equal' if V = U up to a global phase, 'qubit-reversed' if V is U with its qubits in reverse order, else None"""
    def phase_eq(A, B):
        k = np.argmax(np.abs(B))
        p = A.ravel()[k] / B.ravel()[k]
        return abs(abs(p) - 1) < 1e-6 and np.allclose(A, p * B, atol=1e-6)
    n = int(np.log2(U.shape[0]))
    if U.shape != V.shape:
        return None
    if phase_eq(U, V):
        return 'equal'
    perm = [n - 1 - q for q in range(n)]
    W = V.reshape([2] * 2 * n).transpose(perm + [n + q for q in perm]).reshape(V.shape)
    return 'qubit-reversed' if phase_eq(U, W) else None


def target_matrix(target: str) -> np.ndarray:
    return load_matrix_from_file(os.path.join('data/targets', f'{target}.txt'))[1]


def check_implements(qc: QuantumCircuit, target: str, what: str) -> str:
    match = same_operator(target_matrix(target), Operator(qc).data)
    if match is None:
        raise SystemExit(f"{what} does not implement {target}")
    return match


def circuit_stats(qc: QuantumCircuit) -> Dict[str, int]:
    is_t = lambda x: x.operation.name in ('t', 'tdg')
    return {'t_count': sum(1 for x in qc.data if is_t(x)), 't_depth': qc.depth(is_t), 'gate_count': len(qc.data),
            'depth': qc.depth()}


def qasm_rows(filename: str, target: str, metrics: List[str], source: str, notes: str = '',
              check: bool = True) -> List[Dict]:
    """counts of a Synthetiq .qasm circuit, after checking that it implements the target"""
    qc = QuantumCircuit.from_qasm_file(filename)
    if check:
        notes = join(notes, f"circuit checked against the target ({check_implements(qc, target, filename)})")
    stats = circuit_stats(qc)
    return [row(target, qc.num_qubits, source, m, stats[m], 'Synthetiq', SYNTHETIQ, notes) for m in metrics]


def literature_rows() -> List[Dict]:
    rows: List[Dict] = []
    for name, target, n, cells in RIETSCH_TABLE:
        for tool, cell in zip(RIETSCH_TOOLS, cells):
            notes = join(f"Table II row '{name}'", TARGET_NOTES.get(target, ''), RIETSCH_TOOL_NOTES.get(tool, ''))
            source = 'Rietsch et al. 2024 Table II'
            if cell is None:
                rows.append(row(target, n, source, 't_count', '', tool, RIETSCH,
                                join(notes, RIETSCH_FAILED.get(tool, 'failed: time limit or no convergence'))))
                continue
            t, s = cell
            rows.append(row(target, n, source, 't_count', t, tool, RIETSCH, notes))
            rows.append(row(target, n, source, 'time_s', s, tool, RIETSCH,
                            join(notes, 'time limit reached' if (tool, s) == ('QCOpt', 10800) else '')))
    for target, fig, t_count, t_depth, depth in AMY_TABLE:
        n = {'CT (1 ancilla)': 3, 'Toffoli (1 ancilla)': 4}.get(target) or int(target[0])
        source = f'Amy et al. 2013 Fig. {fig}'
        notes = join(TARGET_NOTES.get(target, ''), AMY_FIG_NOTES.get(fig, ''))
        for metric, v in [('t_count', t_count), ('t_depth', t_depth), ('depth', depth)]:
            if v is None:
                continue
            proven = (metric == 'depth' and fig in AMY_DEPTH_OPTIMAL) or (metric == 't_depth' and fig in AMY_TDEPTH_OPTIMAL)
            why = {'depth': 'total-depth optimal (exhaustive search, Sec. 6.1)',
                   't_depth': 'T-depth optimal (Sec. 6.2)'}.get(metric, '') if proven else ''
            rows.append(row(target, n, source, metric, v, 'meet-in-the-middle search', AMY, join(notes, why), proven))
    for target in ['3qubit/toffoli', '3qubit/fredkin']:
        rows.append(row(target, 3, 'Gosset et al. 2014 Sec. 5', 't_count', 7, 'T-count algorithm', GOSSET,
                        'proven minimal: no circuit with 6 or fewer T gates', proven=True))
    for target, t, s in MOSCA_TABLE:
        n = int(target[0])
        notes = join(TARGET_NOTES.get(target, ''), "Table 1 row 'Negated Toffoli'" if target.endswith('neg1') else '')
        rows.append(row(target, n, 'Mosca & Mukhopadhyay 2021 Table 1', 't_count', t, 'MIN-T-SYNTH', MOSCA, notes))
        rows.append(row(target, n, 'Mosca & Mukhopadhyay 2021 Table 1', 'time_s', s, 'MIN-T-SYNTH', MOSCA, notes))
    return rows


def derived_rows() -> List[Dict]:
    """Toffoli-class lower bounds and upper bounds, and qiskit's relative-phase Toffoli definitions"""
    rows: List[Dict] = []
    for target, ops in TOFFOLI_CLASS.items():
        qc = circuit(3, ops)
        match = check_implements(qc.reverse_bits(), target, f'{target} Toffoli-class circuit')
        word = ' '.join(f"{op}({','.join(map(str, args))})" for op, *args in ops)
        notes = f'= {word} ({match}), Clifford-equivalent to Toffoli'
        rows.append(row(target, 3, 'derived: Clifford equivalence to Toffoli', 't_count', 7,
                        'Gosset et al. lower bound + Toffoli circuit', GOSSET, notes, proven=True))
        rows.append(row(target, 3, 'derived: Clifford equivalence to Toffoli', 't_depth', 3,
                        'Amy et al. Fig. 13 Toffoli + Clifford gates', AMY, notes))
    for target, ops in EMBEDDED.items():
        check_implements(circuit(4, ops).reverse_bits(), target, f'{target} circuit')
        notes = 'the 3-qubit circuit with qubit 3 idle; no lower bound proven for the 4-qubit operator'
        for metric, v in [('t_count', 7), ('t_depth', 3)]:
            rows.append(row(target, 4, 'derived: 3-qubit Toffoli circuit', metric, v,
                            'Amy et al. Fig. 13 Toffoli + Clifford gates', AMY, notes))
    for target, n, gate, tool, citation in [('3qubit/rccx', 3, 'rccx', 'qiskit RCCXGate definition (Margolus)', ''),
                                            ('4qubit/rcccx', 4, 'rcccx', 'qiskit RC3XGate definition', MASLOV)]:
        qc = transpile(circuit(n, [(gate, *range(n))]), basis_gates=BASIS, optimization_level=0)
        notes = f"the definition's gates as written ({check_implements(qc, target, gate)})"
        stats = circuit_stats(qc)
        rows += [row(target, n, f'qiskit {gate} definition', m, stats[m], tool, citation, notes)
                 for m in ['t_count', 't_depth', 'gate_count', 'depth']]
    return rows


def mosca_rows(filename: str, target: str, n: int, source: str, notes: str = '') -> List[Dict]:
    """T-count (the length of the printed path) and run time (s) of a Mosca-Mukhopadhyay output file"""
    text = open(filename).read()
    m = re.search(r'^Path\s*[:=]\s*\[(.*)\]', text, re.MULTILINE)
    t = None if m is None else len([x for x in re.split(r'[,:]', m.group(1)) if x.strip()])
    rows = [row(target, n, source, 't_count', '' if t is None else t, 'Mosca-Mukhopadhyay', MOSCA,
                join(notes, 'no circuit found' if t is None else ''))]
    m = re.search(r'^Execution time\s*[:=]\s*([\d.]+)', text, re.MULTILINE)
    if m is not None:
        rows.append(row(target, n, source, 'time_s', float(m.group(1)), 'Mosca-Mukhopadhyay', MOSCA,
                        join('t_count search', 'no circuit found' if t is None else '')))
    return rows


def parse_gheorghiu(text: str) -> Optional[int]:
    m = re.search(r'T-depth: (\d+)', text)
    return None if m is None else int(m.group(1))


def synthetiq_rows(root: str) -> List[Dict]:
    rows: List[Dict] = []
    base = os.path.join(root, 'data')
    all4 = ['t_count', 't_depth', 'gate_count', 'depth']

    # Table 4 (circuits/62): Synthetiq's best T-depth circuits and the paper's baseline T-depths
    table4 = {r['operator']: r for r in csv.DictReader(open(os.path.join(base, 'circuits/62/table4.csv')))}
    t4 = 'synthetiq data/circuits/62/table4.csv'
    for op, target in [('cciswap', '4qubit/cciswap'), ('csqrtiswap', '3qubit/csqrtiswap')]:
        rows += qasm_rows(os.path.join(base, f'circuits/62/{op}.qasm'), target, all4,
                          f'synthetiq data/circuits/62/{op}.qasm', 'T-depth-optimized run')
    rows.append(row('3qubit/csqrtiswap', 3, t4, 't_depth', int(table4['csqrtiswap']['baseline_t_depth']),
                    'Table 4 baseline', SYNTHETIQ, 'gate-by-gate controlled sqrt(iSWAP) circuit after resynthesis'))
    rows.append(row('CCiSWAP (1 ancilla)', 5, t4, 't_depth', int(table4['cciswap']['baseline_t_depth']),
                    'Table 4 baseline', SYNTHETIQ, 'no target file: RCCX-iSWAP-RCCX construction on 5 qubits'))
    # rcccx: Synthetiq solved its relative-phase spec (any phases on the C3X support), a weaker target than our
    # rcccx (qiskit RC3XGate phases); its baseline is Maslov's RC3X = our rcccx
    qasm = os.path.join(base, 'circuits/62/rcccx.qasm')
    V = Operator(QuantumCircuit.from_qasm_file(qasm)).data
    support = np.eye(16)[:, [15 if i == 7 else 7 if i == 15 else i for i in range(16)]]  # little-endian C3X
    assert np.allclose(np.abs(V), support, atol=1e-6), qasm
    rows += qasm_rows(qasm, 'rcccx (relative-phase spec)', all4, 'synthetiq data/circuits/62/rcccx.qasm',
                      'no target file: any phases on the C3X support (input/62/rcccx.txt); circuit checked to be C3X '
                      'up to relative phases', check=False)
    rows.append(row('4qubit/rcccx', 4, t4, 't_depth', int(table4['rcccx']['baseline_t_depth']), 'Table 4 baseline',
                    SYNTHETIQ, 'Maslov 2016 RC3X (= qiskit RC3XGate) as written'))
    for op, name in [('cct', 'CCT'), ('csqrtswap', 'C-sqrtSWAP')]:
        rows += qasm_rows(os.path.join(base, f'circuits/62/{op}.qasm'), f'{name} (1 ancilla)', all4,
                          f'synthetiq data/circuits/62/{op}.qasm',
                          'no target file: not exactly reachable without an ancilla; circuit on 4 qubits', check=False)
        rows.append(row(f'{name} (1 ancilla)', 4, t4, 't_depth', int(table4[op]['baseline_t_depth']),
                        'Table 4 baseline', SYNTHETIQ, 'no target file'))

    # Table 6 (circuits/64/comparison): Synthetiq vs Mosca-Mukhopadhyay (T-count) and Gheorghiu et al. (T-depth)
    table6 = {r['name']: r for r in csv.DictReader(open(os.path.join(base, 'circuits/64/table6.csv')))}
    for op, target in [('U1', '4qubit/U1'), ('U1_var', '4qubit/U1_var'), ('U2', '4qubit/U2'), ('adder', '4qubit/adder'),
                       ('cch', '3qubit/cch'), ('ccx', '3qubit/toffoli')]:
        n = int(target[0])
        rows += qasm_rows(os.path.join(base, f'circuits/64/{op}_tcount.qasm'), target, ['t_count', 'gate_count', 'depth'],
                          f'synthetiq data/circuits/64/{op}_tcount.qasm', 'T-count-optimized run')
        rows += qasm_rows(os.path.join(base, f'circuits/64/{op}_tdepth.qasm'), target, ['t_depth'],
                          f'synthetiq data/circuits/64/{op}_tdepth.qasm', 'T-depth-optimized run')
        for metric, col in [('t_count', 'tcount_time'), ('t_depth', 'tdepth_time')]:
            rows.append(row(target, n, 'synthetiq data/circuits/64/table6.csv', 'time_s', float(table6[op][col]),
                            'Synthetiq', SYNTHETIQ, f'{metric} run: mean time to an optimal circuit'))
        rows += mosca_rows(os.path.join(base, f'baselines/mosca/comparison/{op}.txt'), target, n,
                           f'synthetiq data/baselines/mosca/comparison/{op}.txt')
        text = open(os.path.join(base, f'baselines/gheorgiu/comparison/{op}.txt')).read()
        d = parse_gheorghiu(text)
        source = f'synthetiq data/baselines/gheorgiu/comparison/{op}.txt'
        rows.append(row(target, n, source, 't_depth', '' if d is None else d, 'Gheorghiu-Mosca-Mukhopadhyay',
                        GHEORGHIU, 'run did not finish' if d is None else ''))
        if d is not None:
            ms = float(re.findall(r'Took: (\d+) ms', text)[-1])
            rows.append(row(target, n, source, 'time_s', ms / 1000, 'Gheorghiu-Mosca-Mukhopadhyay', GHEORGHIU,
                            't_depth search'))

    # the 30 permutation classes (input/64/permutations)
    gheorghiu: Dict[int, Tuple[Optional[int], float]] = {}
    lines = [x.strip() for x in open(os.path.join(base, 'baselines/gheorgiu/permutations.txt')) if x.strip()]
    for i in range(0, len(lines) - 3, 4):  # blocks of: number, 'OUT:', 'T-depth: k' or 'none', 'Took: <ms> ms'
        assert lines[i + 1] == 'OUT:' and lines[i + 3].startswith('Took:'), lines[i:i + 4]
        gheorghiu[int(lines[i])] = (parse_gheorghiu(lines[i + 2]), float(lines[i + 3].split()[1]))
    group_of = {p: g for g, ps in PERM_GROUPS.items() for p in ps}
    for p in range(30):
        target = f'3qubit_perms/{p}'
        for metric, v in zip(['t_count', 't_depth'], PERM_GROUP_VALUES[group_of[p]]):
            rows.append(row(target, 3, 'synthetiq notebooks/post_processing/64.ipynb', metric, v, 'Synthetiq',
                            SYNTHETIQ, f'permutation group {group_of[p]}'))
        rows += mosca_rows(os.path.join(base, f'baselines/mosca/permutations/{p}.txt'), target, 3,
                           f'synthetiq data/baselines/mosca/permutations/{p}.txt',
                           'identity: the search starts at T-count 2, so it returns a T-count-4 Clifford circuit'
                           if p == 0 else '')
        if p in gheorghiu:
            d, ms = gheorghiu[p]
            source = 'synthetiq data/baselines/gheorgiu/permutations.txt'
            rows.append(row(target, 3, source, 't_depth', '' if d is None else d, 'Gheorghiu-Mosca-Mukhopadhyay',
                            GHEORGHIU, 'no circuit found' if d is None else ''))
            rows.append(row(target, 3, source, 'time_s', ms / 1000, 'Gheorghiu-Mosca-Mukhopadhyay', GHEORGHIU,
                            join('t_depth search', 'no circuit found' if d is None else '')))
    return rows


def clifford_targets() -> Dict[str, bool]:
    """target -> whether it is a Clifford, for every .txt target"""
    out = {}
    for f in sorted(glob.glob('data/targets/*/*.txt')):
        try:
            Clifford.from_matrix(load_matrix_from_file(f)[1])
            out[os.path.relpath(os.path.splitext(f)[0], 'data/targets')] = True
        except QiskitError:
            out[os.path.relpath(os.path.splitext(f)[0], 'data/targets')] = False
    return out


def set_optimality(rows: List[Dict], is_clifford: Dict[str, bool]) -> None:
    """proven_optimal / best_known / reported per (target, metric), against the lower bounds in the module docstring"""
    lower: Dict[Tuple[str, str], Tuple[int, str]] = {}

    def raise_bound(key, v, why):
        if key not in lower or v > lower[key][0]:
            lower[key] = (v, why)
    for r in rows:
        if r['optimality'] == 'proven_optimal':
            raise_bound((r['target'], r['metric']), r['value'], f"{r['tool']} ({r['source']})")
    for target, cliff in is_clifford.items():
        if not cliff:
            for metric in ['t_count', 't_depth']:
                raise_bound((target, metric), 1, 'not a Clifford')
    n_of = {r['target']: r['num_qubits'] for r in rows}
    for (target, metric), (v, why) in list(lower.items()):
        if metric == 't_count' and v > 1:
            raise_bound((target, 't_depth'), math.ceil(v / n_of[target]),
                        f"T-depth >= ceil(T-count {v} / {n_of[target]} qubits)")

    groups = defaultdict(list)
    for r in rows:
        r['optimality'] = 'reported'
        if r['value'] != '' and r['metric'] != 'time_s':
            groups[(r['target'], r['metric'])].append(r)
    for key, rs in groups.items():
        bound, why = lower.get(key, (None, ''))
        valid = []
        for r in rs:
            if bound is not None and r['value'] < bound:
                r['notes'] = join(r['notes'], f"INCONSISTENT: below the proven lower bound {bound} ({why})")
                print(f"warning: {key[0]} {key[1]} = {r['value']} from {r['tool']} is below the lower bound {bound}")
            else:
                valid.append(r)
        best = min(r['value'] for r in valid)
        for r in valid:
            if r['value'] == best:
                r['optimality'] = 'proven_optimal' if best == bound else 'best_known'
                if best == bound:
                    r['notes'] = join(r['notes'], f'lower bound: {why}' if why not in r['notes'] else '')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--synthetiq', type=str, default=os.path.expanduser('~/research/synthetiq'))
    parser.add_argument('--output', type=str, default='data/targets/reference_values.csv')
    args = parser.parse_args()

    is_clifford = clifford_targets()
    rows = literature_rows() + derived_rows() + synthetiq_rows(args.synthetiq)
    rows += [row(t, int(t[0]), 'Clifford check (qiskit Clifford.from_matrix)', m, 0, 'trivial', '', 'Clifford operator',
                 proven=True) for t, c in is_clifford.items() if c for m in ['t_count', 't_depth']]
    for r in rows:
        assert r['metric'] in METRICS, r
        if '/' in r['target']:
            assert r['target'] in is_clifford, f"no target file for {r['target']}"
    set_optimality(rows, is_clifford)

    # an optimal T-depth is at most the optimal T-count: flag T-depths above the best T-count for the same target
    best_t = {r['target']: r['value'] for r in rows if r['metric'] == 't_count' and r['optimality'] != 'reported'}
    for r in rows:
        if r['metric'] == 't_depth' and r['value'] != '' and r['value'] > best_t.get(r['target'], r['value']):
            r['notes'] = join(r['notes'], f"exceeds the best T-count {best_t[r['target']]}")

    rows.sort(key=lambda r: (r['num_qubits'], '/' not in r['target'], r['target'], METRICS.index(r['metric']),
                             r['value'] == '', r['value'] if r['value'] != '' else 0, r['tool']))
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows for {len({r['target'] for r in rows})} targets to {args.output}")
