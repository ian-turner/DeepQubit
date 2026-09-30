"""Tests for the exact ring domain (utils/ring.py, domains/qcircuit_exact.py).

Run:  python tests/test_exact.py        (or pytest tests/test_exact.py)
"""
import os
import re
import sys
import time
import pickle
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import deepxube  # noqa: F401,E402  (imports the local domains/ and nnets/ packages in the right order and registers them)
from domains.qcircuit import *  # noqa: F401,F403,E402
from domains.qcircuit_exact import QCircuitExact, QStateExact, QGoalExact
from deepxube.factories.domain_factory import get_domain_from_arg
from utils import ring
from utils.matrix_utils import unitary_distance

FLOAT = {gs: get_domain_from_arg(f'qcircuit.n3_e0.000001{gs}')[0] for gs in ('', '_I')}
EXACT = {gs: get_domain_from_arg(f'qcircuit_exact.n3{gs}_G_B9+C2+M')[0] for gs in ('', '_I')}


def walk_both(gs, word):
    """Apply a list of action indices in both domains; returns (float unitary, exact state)"""
    fd, ed = FLOAT[gs], EXACT[gs]
    fs, es = fd.sample_start_states(1)[0], ed.sample_start_states(1)[0]
    for idx in word:
        fs = fd.next_state([fs], [fd.actions[idx]])[0][0]
        es = ed.next_state([es], [ed.actions[idx]])[0][0]
    return fs.unitary, es


def test_ring_element_ops():
    w = np.exp(1j * np.pi / 4)
    c = np.array([[[[1, 2, 3, 4]]]], dtype=np.int64)
    z = ring.to_complex(c, np.array([0]))[0, 0, 0]
    assert np.isclose(z, 1 + 2 * w + 3 * w ** 2 + 4 * w ** 3)
    assert np.isclose(ring.to_complex(ring.rotate(c, 5), np.array([0]))[0, 0, 0], z * w ** 5)
    assert np.isclose(ring.to_complex(ring.conj(c), np.array([0]))[0, 0, 0], np.conj(z))
    assert np.isclose(ring.to_complex(ring.mul_sqrt2(c), np.array([0]))[0, 0, 0], z * np.sqrt(2))
    c2 = np.array([[[-2, 0, 5, 1]]], dtype=np.int64)
    z2 = ring.to_complex(c2, np.array(0))[0, 0]
    assert np.isclose(ring.to_complex(ring.matmul(c[0], c2), np.array(0))[0, 0], z * z2)
    assert np.isclose(ring.to_complex(ring.matmul_dagger(c[0], c2), np.array(0))[0, 0], z * np.conj(z2))


def test_matches_float_domain_on_random_words():
    rng = np.random.RandomState(0)
    for gs in ('', '_I'):
        n_act = len(FLOAT[gs].actions)
        for length in [1, 2, 5, 10, 20, 40]:
            for _ in range(8):
                word = rng.randint(n_act, size=length).tolist()
                U, es = walk_both(gs, word)
                assert unitary_distance(U, es.unitary) < 1e-9, (gs, word)
                assert es.k >= 0 and np.abs(es.coeffs).max() < 2 ** 26


def test_exact_identities():
    ed = EXACT['_I']
    ident = ed.sample_start_states(1)[0]
    names = {repr(a): a for a in ed.actions}
    h0, t0, s0, sdg0, tdg0 = names['h qs[0]'], names['t qs[0]'], names['s qs[0]'], ed.actions[6], ed.actions[12]
    cx01 = names['cx qs[0], qs[1]']
    cx10 = names['cx qs[1], qs[0]']

    def run(acts):
        s = ident
        for a in acts:
            s = ed.next_state([s], [a])[0][0]
        return s
    assert run([h0, h0]) == ident
    assert run([t0] * 8) == ident
    assert run([s0] * 4) == ident
    assert run([cx01, cx01]) == ident
    assert run([s0, sdg0]) == ident and run([t0, tdg0]) == ident
    assert hash(run([h0, t0, h0, h0, tdg0, h0])) == hash(ident) and run([h0, t0, h0, h0, tdg0, h0]) == ident
    # SWAP by two different CNOT words -> identical canonical state
    assert run([cx01, cx10, cx01]) == run([cx10, cx01, cx10])
    # global phase: X Z X Z = -I must canonicalize to the identity (X = H Z H, Z = S S)
    x0 = [h0, s0, s0, h0]
    z0 = [s0, s0]
    assert run(x0 + z0 + x0 + z0) == ident
    assert run([t0, t0, t0, t0]) == run(z0)


def test_from_complex_roundtrip_and_rejection():
    rng = np.random.RandomState(1)
    ed = EXACT['_I']
    for length in [0, 3, 12, 30]:
        _, es = walk_both('_I', rng.randint(21, size=length).tolist())
        back = QStateExact.from_complex(es.unitary * np.exp(0.3j))  # arbitrary phase must not matter
        assert back == es
    from qiskit.quantum_info import random_unitary
    try:
        ring.from_complex(random_unitary(8).data, k_max=6)
        assert False, "random unitary should be rejected"
    except ValueError:
        pass


def _qasm_word(path, domain, qmap):
    lines = [l.strip() for l in open(path) if re.match(r'^(h|s|sdg|t|tdg|cx) ', l.strip())]
    acts = []
    for l in lines:
        g, qs = l.rstrip(';').split(' ', 1)
        qs = [qmap(int(x)) for x in re.findall(r'\[(\d+)\]', qs)]
        cls = {'h': HGate, 's': SGate, 'sdg': SdgGate, 't': TGate, 'tdg': TdgGate, 'cx': CNOTGate}[g]
        acts.append(domain._lookup_action(cls, tuple(qs)))
    return acts


def test_benchmark_goals_solved_by_reference_circuits():
    ed = EXACT['_I']
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pickle.load(open(os.path.join(root, 'tmp/n3_goals.pkl'), 'rb'))
    names = ['cch', 'ccrz_2', 'ccz', 'csqrtiswap', 'fredkin', 'toffoli']
    for name in ['ccz', 'toffoli', 'fredkin', 'cch']:
        goal = QGoalExact.from_complex(data['goals'][names.index(name)].unitary)
        solved = False
        for qmap in (lambda q: q, lambda q: 2 - q):
            s = ed.sample_start_states(1)[0]
            for a in _qasm_word(os.path.join(root, f'data/circuits/3qubit/{name}.qasm'), ed, qmap):
                s = ed.next_state([s], [a])[0][0]
            solved |= ed.is_solved([s], [goal])[0]
        assert solved, name


def test_macro_goals_and_problem_instances():
    ed = EXACT['_I']
    np.random.seed(0)
    states, goals = ed.sample_problem_instances([0, 1, 5, 15, 16, 30] * 5)
    assert len(goals) == 30
    ident = ed.sample_start_states(1)[0]
    for k, g in zip([0, 1, 5, 15, 16, 30] * 5, goals):
        if k == 0:
            assert g.k == 0 and np.array_equal(g.coeffs, ident.coeffs)
        assert np.isclose(abs(np.linalg.det(g.unitary)), 1)
    # a full Toffoli macro word reaches the benchmark Toffoli
    tof = ed._expand_macro(MACROS['toffoli'][0](0, 1, 2))
    s = ident
    for a in tof:
        s = ed.next_state([s], [a])[0][0]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pickle.load(open(os.path.join(root, 'tmp/n3_goals.pkl'), 'rb'))
    assert ed.is_solved([s], [QGoalExact.from_complex(data['goals'][5].unitary)])[0]


def test_encoding_dims_and_bits_lossless():
    rng = np.random.RandomState(2)
    for enc in ['B9', 'C2', 'M', 'B9+C2', 'B4+C1+M']:
        ed = get_domain_from_arg(f'qcircuit_exact.n3_I_{enc}')[0]
        states = [walk_both('_I', rng.randint(21, size=L).tolist())[1] for L in [0, 5, 20, 30]]
        goals = [QGoalExact(s.coeffs, s.k) for s in states[::-1]]
        x = ed.to_np_flat_sg(states, goals)[0]
        assert x.shape == (4, ed.get_input_info_flat_sg()[0][0]), enc
        assert x.dtype == np.float32 and set(np.unique(x[:, :64 * 4 * 9])) <= {0.0, 1.0} if enc.startswith('B9') else True
    # B9 bits reconstruct the coefficients exactly while k <= 16
    ed = get_domain_from_arg('qcircuit_exact.n3_I_B9')[0]
    for L in [1, 10, 30]:
        s = walk_both('_I', rng.randint(21, size=L).tolist())[1]
        g = QGoalExact(*ring.identity(8))
        c, k = ed._relative([s], [g])  # = S^dagger canonical
        bits = ed.to_np_flat_sg([s], [g])[0][0, :8 * 8 * 4 * 9].reshape(8, 8, 4, 9).astype(np.int64)
        res = (bits * (1 << np.arange(9))).sum(-1)
        signed = np.where(res >= 256, res - 512, res)
        assert k[0] <= 16 and np.array_equal(signed, c[0]), L


def test_channel_rows_clifford_vs_toffoli():
    ed = EXACT['_I']
    ident = ed.sample_start_states(1)[0]
    # identity: every generator maps to itself, exponent 0
    coef, ex = ed._channel(ident.coeffs[None], np.array([0]))
    assert (ex == 0).all()
    support = (coef != 0).any(-1).sum(-1)  # (1, 6)
    assert (support == 1).all()
    gens = ring.pauli_generator_indices(3)
    for r, p in enumerate(gens):
        assert coef[0, r, p, 0] == 1 and coef[0, r, p, 1] == 0
    # random Clifford words: single +-1 per row, exponent 0
    rng = np.random.RandomState(4)
    cliff = [i for i, a in enumerate(ed.actions) if not isinstance(a, (TGate, TdgGate))]
    for L in [3, 10, 25]:
        s = walk_both('_I', [cliff[i] for i in rng.randint(len(cliff), size=L)])[1]
        coef, ex = ed._channel(s.coeffs[None], np.array([s.k]))
        assert (ex == 0).all() and ((coef != 0).any(-1).sum(-1) == 1).all()
        assert set(np.unique(coef[..., 0])) <= {-1, 0, 1} and (coef[..., 1] == 0).all()
    # Toffoli(0,1,2): X0, X1, Z2 rows spread over 4 Paulis with coefficient 1/2 (exponent 2); X2, Z0, Z1 stay single
    s = ident
    for a in ed._expand_macro(MACROS['toffoli'][0](0, 1, 2)):
        s = ed.next_state([s], [a])[0][0]
    coef, ex = ed._channel(s.coeffs[None], np.array([s.k]))
    support = (coef != 0).any(-1).sum(-1)[0]
    assert support.tolist() == [4, 4, 1, 1, 1, 4], support.tolist()   # rows: X0 X1 X2 Z0 Z1 Z2
    assert ex[0].tolist() == [2, 2, 0, 0, 0, 2], ex[0].tolist()
    # T on qubit 0: X0 -> (X0 + Y0)/sqrt2, exponent 1
    t0 = ed.next_state([ident], [ed.actions[9]])[0][0]
    coef, ex = ed._channel(t0.coeffs[None], np.array([t0.k]))
    assert ex[0].tolist() == [1, 0, 0, 0, 0, 0] and (coef != 0).any(-1).sum(-1)[0].tolist() == [2, 1, 1, 1, 1, 1]


def test_parser():
    d = get_domain_from_arg('qcircuit_exact.n3_I_G0.3_B7+C3_K12')[0]
    assert d.num_qubits == 3 and d.gateset == 'CliffT_inv' and d.macro_frac == 0.3 and d.k_cap == 12
    assert d._parts == [('B', 7), ('C', 3)]
    assert get_domain_from_arg('qcircuit_exact.n2')[0]._parts == [('B', 9), ('C', 2)]
    try:
        get_domain_from_arg('qcircuit_exact.n3_e0.01')
        assert False
    except ValueError:
        pass


def test_next_state_batched_mixed_actions():
    ed, fd = EXACT['_I'], FLOAT['_I']
    rng = np.random.RandomState(5)
    starts = ed.sample_start_states(50)
    fstarts = fd.sample_start_states(50)
    for _ in range(6):
        acts = [ed.actions[i] for i in rng.randint(21, size=50)]
        starts = ed.next_state(starts, acts)[0]
        fstarts = fd.next_state(fstarts, acts)[0]
    for s, f in zip(starts, fstarts):
        assert unitary_distance(s.unitary, f.unitary) < 1e-9


def test_resnet_fc_ring_layer_matches_numpy_encodings():
    import torch
    from nnets.resnet_fc_ring import RingFeatures
    dom_b = get_domain_from_arg('qcircuit_exact.n3_I_B9')[0]
    dom_ref = get_domain_from_arg('qcircuit_exact.n3_I_B9+C2+M')[0]
    rng = np.random.RandomState(7)
    states = [walk_both('_I', rng.randint(21, size=L).tolist())[1] for L in [0, 1, 4, 9, 15, 22, 30, 30]]
    goals = [QGoalExact(s.coeffs, s.k) for s in states[::-1]]
    x = dom_b.to_np_flat_sg(states, goals)[0]
    ref = dom_ref.to_np_flat_sg(states, goals)[0]
    layer = RingFeatures(dom_b, chan_bits=2, float_view=True)
    out = layer(torch.tensor(x)).numpy()
    assert out.shape == (8, x.shape[1] + layer.extra_dim) and out.shape[1] == ref.shape[1]
    assert np.array_equal(out[:, :x.shape[1]], x)
    n_c = dom_ref._part_dim('C', 2)
    assert np.array_equal(out[:, x.shape[1]:x.shape[1] + n_c], ref[:, x.shape[1]:x.shape[1] + n_c])   # channel bits + one-hots
    assert np.allclose(out[:, x.shape[1] + n_c:], ref[:, x.shape[1] + n_c:], atol=1e-6)               # float view
    # exponent beyond the lossless range (alternating H, T on one qubit raises k by one every two rounds): blocks zeroed
    ed = EXACT['_I']
    s = ed.sample_start_states(1)[0]
    for i in range(36):
        s = ed.next_state([s], [ed.actions[0]])[0][0]
        s = ed.next_state([s], [ed.actions[9]])[0][0]
    assert layer.k_lossless < s.k <= dom_b.k_cap, s.k
    x_deep = dom_b.to_np_flat_sg([s], [QGoalExact(*ring.identity(8))])[0]
    out_deep = layer(torch.tensor(x_deep)).numpy()
    assert np.array_equal(out_deep[:, :x_deep.shape[1]], x_deep) and not out_deep[:, x_deep.shape[1]:].any()
    # chan_bits=0 gives only the float view
    layer0 = RingFeatures(dom_b, chan_bits=0, float_view=True)
    assert layer0.extra_dim == 128 and layer0(torch.tensor(x)).shape[1] == x.shape[1] + 128


def test_resnet_fc_ring_builds_through_factory():
    import torch
    from deepxube.factories.pathfind_fns_factory import get_path_fns_nnet_par_dict
    dom, name = get_domain_from_arg('qcircuit_exact.n3_I_B9')
    pf, npd = get_path_fns_nnet_par_dict(dom, name, ['heurv,resnet_fc_ring.50H_1B_bn_2C_fv'], torch.device('cpu'))
    nnet = npd['heurv'].get_nnet()
    assert nnet.ring_features.chan_bits == 2 and nnet.ring_features.float_view
    assert nnet.heur[0].in_features == 2325 + 1782 + 128
    rng = np.random.RandomState(8)
    states = [walk_both('_I', rng.randint(21, size=L).tolist())[1] for L in [0, 3, 12]]
    goals = [QGoalExact(*ring.identity(8)) for _ in states]
    x = torch.tensor(dom.to_np_flat_sg(states, goals)[0])
    nnet.eval()
    out = nnet([x])
    assert out[0].shape == (3, 1)
    # the default parser values: no C flag -> 2 bits, no fv
    pf2, npd2 = get_path_fns_nnet_par_dict(dom, name, ['heurv,resnet_fc_ring.50H_1B_bn'], torch.device('cpu'))
    assert npd2['heurv'].get_nnet().heur[0].in_features == 2325 + 1782
    try:
        get_path_fns_nnet_par_dict(*get_domain_from_arg('qcircuit_exact.n3_I_B9+C2'), ['heurv,resnet_fc_ring.50H_1B_bn'], torch.device('cpu'))
        assert False, "domain with a C part must be rejected"
    except (AssertionError, ValueError) as e:
        assert 'resnet_fc_ring' in str(e)


def timing_info():
    import time as _t
    import torch
    from nnets.resnet_fc_ring import RingFeatures
    dom_b = get_domain_from_arg('qcircuit_exact.n3_I_B9')[0]
    rng0 = np.random.RandomState(9)
    st = dom_b.sample_start_states(2000)
    for _ in range(12):
        st = dom_b.next_state(st, [dom_b.actions[i] for i in rng0.randint(21, size=2000)])[0]
    gl = [QGoalExact(s.coeffs, s.k) for s in st[::-1]]
    t = _t.time(); xb = dom_b.to_np_flat_sg(st, gl)[0]; t_b = _t.time() - t
    layer = RingFeatures(dom_b, 2, False); xt = torch.tensor(xb); layer(xt)
    t = _t.time(); layer(xt); t_l = _t.time() - t
    print(f"  timing (2000 states): domain B9 encode {t_b * 1e3:.0f} ms, torch RingFeatures (CPU) {t_l * 1e3:.0f} ms")
    ed = get_domain_from_arg('qcircuit_exact.n3_I_B9+C2')[0]
    rng = np.random.RandomState(6)
    states = ed.sample_start_states(2000)
    for _ in range(12):
        states = ed.next_state(states, [ed.actions[i] for i in rng.randint(21, size=2000)])[0]
    goals = [QGoalExact(s.coeffs, s.k) for s in states[::-1]]
    t = time.time(); ed.next_state(states, [ed.actions[i] for i in rng.randint(21, size=2000)]); t_ns = time.time() - t
    t = time.time(); ed.to_np_flat_sg(states, goals); t_enc = time.time() - t
    print(f"  timing (2000 states): next_state {t_ns * 1e3:.0f} ms, encode B9+C2 {t_enc * 1e3:.0f} ms, "
          f"max k = {max(s.k for s in states)}, input dim = {ed.get_input_info_flat_sg()[0][0]}")


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print('ok', t.__name__)
    timing_info()
    print('all passed')
