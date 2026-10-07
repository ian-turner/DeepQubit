"""Tests for the exact ring domain (utils/ring.py, domains/qcircuit_exact.py).

Run:  python tests/test_exact.py        (or pytest tests/test_exact.py)
"""
import os
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
from utils.matrix_utils import unitary_distance, load_matrix_from_file

FLOAT = {gs: get_domain_from_arg(f'qcircuit.n3_e0.000001{gs}')[0] for gs in ('', '_I')}
EXACT = {gs: get_domain_from_arg(f'qcircuit_exact.n3{gs}_B9+C2+M')[0] for gs in ('', '_I')}


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


TARGETS = 'data/targets/3qubit'


def _target(name):
    """Benchmark target from data/targets/3qubit as an exact goal (the .txt entries carry ~8 digits, hence tol)"""
    return QGoalExact.from_complex(load_matrix_from_file(os.path.join(TARGETS, f'{name}.txt'))[1], tol=1e-6)


def _run(domain, word):
    """State reached from the identity by a circuit-order list of (gate class, qubits)"""
    s = domain.sample_start_states(1)[0]
    for cls, qs in word:
        s = domain.next_state([s], [domain._lookup_action(cls, qs)])[0][0]
    return s


def _toffoli_word(a=0, b=1, t=2):
    """15-gate Toffoli (controls a, b; target t), T-count 7 (Nielsen & Chuang Fig. 4.9)"""
    return [(HGate, (t,)), (CNOTGate, (b, t)), (TdgGate, (t,)), (CNOTGate, (a, t)), (TGate, (t,)),
            (CNOTGate, (b, t)), (TdgGate, (t,)), (CNOTGate, (a, t)), (TGate, (b,)), (TGate, (t,)),
            (HGate, (t,)), (CNOTGate, (a, b)), (TGate, (a,)), (TdgGate, (b,)), (CNOTGate, (a, b))]


def test_benchmark_targets_solved_by_known_circuits():
    """Qubit 0 is the most significant bit: toffoli flips qubit 2 when 0 and 1 are set, fredkin swaps 1 and 2 when 0 is"""
    ed = EXACT['_I']
    words = {
        'toffoli': _toffoli_word(),
        'ccz': [g for g in _toffoli_word() if g != (HGate, (2,))],  # the Toffoli word without the target H gates
        'fredkin': [(CNOTGate, (2, 1))] + _toffoli_word() + [(CNOTGate, (2, 1))],
    }
    for name, word in words.items():
        assert ed.is_solved([_run(ed, word)], [_target(name)])[0], name


def test_problem_instances():
    ed = EXACT['_I']
    np.random.seed(0)
    states, goals = ed.sample_problem_instances([0, 1, 5, 15, 16, 30] * 5)
    assert len(goals) == 30
    ident = ed.sample_start_states(1)[0]
    for k, g in zip([0, 1, 5, 15, 16, 30] * 5, goals):
        if k == 0:
            assert g.k == 0 and np.array_equal(g.coeffs, ident.coeffs)
        assert np.isclose(abs(np.linalg.det(g.unitary)), 1)


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
    s = _run(ed, _toffoli_word())
    coef, ex = ed._channel(s.coeffs[None], np.array([s.k]))
    support = (coef != 0).any(-1).sum(-1)[0]
    assert support.tolist() == [4, 4, 1, 1, 1, 4], support.tolist()   # rows: X0 X1 X2 Z0 Z1 Z2
    assert ex[0].tolist() == [2, 2, 0, 0, 0, 2], ex[0].tolist()
    # T on qubit 0: X0 -> (X0 + Y0)/sqrt2, exponent 1
    t0 = ed.next_state([ident], [ed.actions[9]])[0][0]
    coef, ex = ed._channel(t0.coeffs[None], np.array([t0.k]))
    assert ex[0].tolist() == [1, 0, 0, 0, 0, 0] and (coef != 0).any(-1).sum(-1)[0].tolist() == [2, 1, 1, 1, 1, 1]


def test_parser():
    d = get_domain_from_arg('qcircuit_exact.n3_I_B7+C3_K12')[0]
    assert d.num_qubits == 3 and d.gateset == 'CliffT_inv' and d.k_cap == 12
    assert d._parts == [('B', 7), ('C', 3)]
    assert get_domain_from_arg('qcircuit_exact.n2')[0]._parts == [('Z', 9)]
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


def test_next_state_output_is_canonical():
    """next_state skips reduction/canonicalization for gates that cannot change them; check against a full normalize"""
    rng = np.random.RandomState(10)
    for gs in ('', '_I'):
        ed = EXACT[gs]
        states = ed.sample_start_states(40)
        for _ in range(15):
            states = ed.next_state(states, [ed.actions[i] for i in rng.randint(len(ed.actions), size=40)])[0]
        for a in ed.actions:
            raw_c, raw_k = ed._apply_group(np.stack([s.coeffs for s in states]).astype(np.int64),
                                           np.array([s.k for s in states]), a)
            ref_c, ref_k = ring.normalize_batch(raw_c, raw_k)
            nxt = ed.next_state(states, [a] * len(states))[0]
            assert all(s.k == kk and np.array_equal(s.coeffs, cc) for s, cc, kk in zip(nxt, ref_c, ref_k)), (gs, a)


def test_incremental_relative_matches_scratch():
    """G S^dagger derived from the parent's cached relative by column ops equals the full product, along a simulated
    search (roots evaluated, popped nodes expanded, children evaluated), for both gate sets, and after goal relabeling"""
    rng = np.random.RandomState(11)
    for gs in ('', '_I'):
        ed = EXACT[gs]
        np.random.seed(11)
        roots, goals = ed.sample_problem_instances([0, 3, 8, 20, 40, 60])
        frontier = list(zip(roots, goals))
        ed.to_np_flat_sg(roots, goals)  # caches the roots' relatives
        scratch_calls = []
        scratch = ed._relative_scratch
        for _ in range(4):
            parents = [frontier[i] for i in rng.choice(len(frontier), size=6, replace=False)]
            st = [p for p, _ in parents for _ in ed.actions]
            gl = [g for _, g in parents for _ in ed.actions]
            children = ed.next_state(st, [a for _ in parents for a in ed.actions])[0]
            ed._relative_scratch = lambda *a: scratch_calls.append(len(a[0])) or scratch(*a)
            c, k = ed._relative(children, gl)
            ed._relative_scratch = scratch
            assert not scratch_calls, scratch_calls  # every relative came from a parent's by column ops
            c_ref, k_ref = ed._relative_scratch(children, gl)
            assert np.array_equal(c, c_ref) and np.array_equal(k, k_ref), gs
            frontier = list(zip(children, gl))
        # popped states re-encoded (cache hits) and relabeled to other goals (scratch) still agree
        sts = [s for s, _ in frontier[:30]]
        for gl in ([g for _, g in frontier[:30]], [goals[1]] * 30):
            c, k = ed._relative(sts, gl)
            c_ref, k_ref = ed._relative_scratch(sts, gl)
            assert np.array_equal(c, c_ref) and np.array_equal(k, k_ref), gs


def test_compact_storage_hash_and_pickle():
    ed = EXACT['_I']
    rng = np.random.RandomState(12)
    s = walk_both('_I', rng.randint(21, size=25).tolist())[1]
    assert s.coeffs.dtype == np.int16
    s64 = QStateExact(s.coeffs.astype(np.int64), s.k)  # e.g. built by from_complex
    assert s64.coeffs.dtype == np.int16 and s64 == s and hash(s64) == hash(s)
    child = ed.next_state([s], [ed.actions[0]])[0][0]
    assert child._parent is s
    back = pickle.loads(pickle.dumps(child))
    assert back == child and hash(back) == hash(child) and back._parent is None and back._rel is None
    # goals pickled before compact storage (their state is the plain __dict__: int64 coefficients and k, no caches)
    # still load and compare
    g = _target('cch')
    old = QGoalExact.__new__(QGoalExact)
    old.__setstate__({'coeffs': g.coeffs.astype(np.int64), 'k': g.k})  # what unpickling such a file does
    assert old.coeffs.dtype == np.int16 and old == g and hash(old) == hash(g)
    # coefficients beyond int16 stay int64 (and equal states hash alike)
    big = np.zeros((8, 8, 4), dtype=np.int64)
    big[0, 0, 0] = 1 << 20
    assert QStateExact(big, 40).coeffs.dtype == np.int64


def test_z_encoding_matches_b_encoding_through_ring_layer():
    import torch
    from nnets.resnet_fc_ring import RingFeatures
    dom_z = get_domain_from_arg('qcircuit_exact.n3_I_Z9')[0]
    dom_b = get_domain_from_arg('qcircuit_exact.n3_I_B9')[0]
    dom_ref = get_domain_from_arg('qcircuit_exact.n3_I_B9+C2+M')[0]
    rng = np.random.RandomState(13)
    states = [walk_both('_I', rng.randint(21, size=L).tolist())[1] for L in [0, 1, 4, 9, 15, 22, 30, 30, 50, 80]]
    goals = [QGoalExact(s.coeffs, s.k) for s in states[::-1]]
    xz = dom_z.to_np_flat_sg(states, goals)[0]
    assert xz.dtype == np.int16 and xz.shape == (10, dom_z.get_input_info_flat_sg()[0][0]) == (10, 257)
    ref = dom_ref.to_np_flat_sg(states, goals)[0]
    out_z = RingFeatures(dom_z, chan_bits=2, float_view=True)(torch.tensor(xz)).numpy()
    out_b = RingFeatures(dom_b, chan_bits=2, float_view=True)(torch.tensor(dom_b.to_np_flat_sg(states, goals)[0])).numpy()
    assert out_z.shape == out_b.shape == ref.shape
    n_exact = ref.shape[1] - 128
    assert np.array_equal(out_z[:, :n_exact], ref[:, :n_exact]) and np.array_equal(out_z[:, :n_exact], out_b[:, :n_exact])
    assert np.allclose(out_z[:, n_exact:], ref[:, n_exact:], atol=1e-6)
    # beyond the B9 range Z still has the exact integers (k <= 29): channel rows equal the numpy reference
    ed = EXACT['_I']
    s = ed.sample_start_states(1)[0]
    for i in range(36):
        s = ed.next_state([s], [ed.actions[0]])[0][0]
        s = ed.next_state([s], [ed.actions[9]])[0][0]
    g = [QGoalExact(*ring.identity(8))]
    assert 16 < s.k <= 29
    out_deep = RingFeatures(dom_z, chan_bits=2)(torch.tensor(dom_z.to_np_flat_sg([s], g)[0])).numpy()
    ref_deep = get_domain_from_arg('qcircuit_exact.n3_I_B9+C2')[0].to_np_flat_sg([s], g)[0]
    assert np.array_equal(out_deep[:, 2304:], ref_deep[:, 2304:])  # k one-hot, channel bits and exponents
    try:
        get_domain_from_arg('qcircuit_exact.n3_I_Z9+C2')
        assert False, "Z must be the only part"
    except ValueError:
        pass


def test_resnet_fc_ring_checkpoint_works_with_z_input():
    import torch
    from deepxube.factories.pathfind_fns_factory import get_path_fns_nnet_par_dict
    dom_b, name = get_domain_from_arg('qcircuit_exact.n3_I_B9')
    dom_z, _ = get_domain_from_arg('qcircuit_exact.n3_I_Z9')
    net_b = get_path_fns_nnet_par_dict(dom_b, name, ['heurv,resnet_fc_ring.50H_1B_bn_2C'], torch.device('cpu'))[1]['heurv'].get_nnet()
    net_z = get_path_fns_nnet_par_dict(dom_z, name, ['heurv,resnet_fc_ring.50H_1B_bn_2C'], torch.device('cpu'))[1]['heurv'].get_nnet()
    assert net_z.heur[0].in_features == net_b.heur[0].in_features == 2325 + 1782
    net_z.load_state_dict(net_b.state_dict())
    net_b.eval()
    net_z.eval()
    rng = np.random.RandomState(14)
    states = [walk_both('_I', rng.randint(21, size=L).tolist())[1] for L in [0, 3, 12, 40]]
    goals = [QGoalExact(*ring.identity(8)) for _ in states]
    out_b = net_b([torch.tensor(dom_b.to_np_flat_sg(states, goals)[0])])[0]
    out_z = net_z([torch.tensor(dom_z.to_np_flat_sg(states, goals)[0])])[0]
    assert torch.allclose(out_b, out_z, atol=1e-5)
    # the layer's tables are not saved; checkpoints from before (which saved them) still load strictly
    sd = net_b.state_dict()
    assert not any('ring_features' in key for key in sd)
    for name in ('pow2', 'rot4f', 'gens', 'chan_idx', 'chan_sign', 'cbit_shifts'):
        sd['ring_features.' + name] = torch.zeros(1)
    net_z.load_state_dict(sd)


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
    dom_z = get_domain_from_arg('qcircuit_exact.n3_I_Z9')[0]
    gl = [QGoalExact(g.coeffs, g.k) for g in gl]  # new goal objects: no cached relatives
    t = _t.time(); xz = dom_z.to_np_flat_sg(st, gl)[0]; t_z = _t.time() - t
    print(f"  timing (2000 states): domain Z9 encode from scratch {t_z * 1e3:.0f} ms ({xz.nbytes / 2000:.0f} bytes/state vs "
          f"{xb.nbytes / 2000:.0f} for B9 floats)")
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
