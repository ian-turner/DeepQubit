"""resnet_fc_ring: the fully-connected resnet heuristic with a fixed, parameter-free front layer that does the
ring algebra on the device.

The qcircuit_exact domain sends the canonical relative unitary R = G S^dagger either as its B<m> residue bits plus the
exponent one-hot (domain encoding B<m>), or -- preferred, 18x fewer bytes through the worker/runner/training buffers --
as int16 coefficients plus k (domain encoding Z<m>), in which case this layer expands the same B<m> bits on the device.
`RingFeatures` then computes the channel-representation features -- the same numbers the domain's `C<m>` encoding
produces in numpy -- and optionally the float view (`M`). The resnet sees [B bits | channel bits | (floats)], with the
same layout for B<m> and Z<m> domains, so a checkpoint works with either.

The channel uses the two Galois embeddings omega -> omega, omega -> omega^3 of Z[omega]: R P R^dagger for all 2n Pauli
generators is one complex (N x N)(N x 2nN) product per embedding (a Pauli is a phased permutation), the 4^n Pauli traces
are one GEMM with a fixed signed selection matrix, and the Z[sqrt2] value a + b sqrt2 of each trace is recovered from
its two real embeddings a +- b sqrt2 by rounding: 16x fewer float64 FLOPs than the 4N x 4N companion form the numpy
reference uses. The layer's tables are rebuilt from the domain and not saved in checkpoints.

Network string: resnet_fc_ring.<H>H_<B>B[_bn][_<m>C][_fv]   e.g. resnet_fc_ring.1000H_4B_bn_2C
  <m>C  channel residue bits per numerator (default 2; 0 disables the channel block)
  fv    append the float view of R
Rows whose integers cannot be reconstructed (k > 2m - 2 for B<m> input, k > 29 for Z<m>) get a zeroed channel/float
block.
"""
from typing import List, Type

import numpy as np
import torch
from torch import nn, Tensor

from deepxube.base.nnet_input import FlatIn
from deepxube.base.nnet import HeurNNet
from deepxube.factories.nnet_factory import deepxube_nnet_factory
from deepxube.nnets.resnet_fc import ResnetFCParser
from deepxube.pytorch.pytorch_models import FullyConnectedModel, ResnetModel

from domains.qcircuit_exact import QCircuitExact
from utils import ring


def _val2(x: Tensor) -> Tensor:
    """2-adic valuation of int64 entries (large for 0)"""
    low = (x & -x).to(torch.float64)
    return torch.where(x == 0, torch.full_like(x, 10 ** 6), torch.log2(torch.clamp(low, min=1.0)).round().to(torch.int64))


class RingFeatures(nn.Module):
    # tables saved in checkpoints by earlier versions of this layer (now rebuilt from the domain instead)
    _LEGACY_BUFFERS = ('pow2', 'rot4f', 'gens', 'chan_idx', 'chan_sign', 'cbit_shifts')

    def __init__(self, domain: QCircuitExact, chan_bits: int = 2, float_view: bool = False):
        super().__init__()
        parts = domain._parts
        self.z_input: bool = parts[0][0] == 'Z'
        assert (self.z_input or parts[0][0] == 'B') and all(kind != 'C' for kind, _ in parts), \
            "resnet_fc_ring expects the domain encoding to be Z<m>, or to start with B<m> and not include C (the network computes it)"
        self.n: int = domain.num_qubits
        self.N: int = domain.N
        self.m: int = parts[0][1]
        self.k_cap: int = domain.k_cap
        # coefficients are exact while |c| < 2^(m-1) (B bits) or < 2^15 (int16), and |c| <= sqrt2^k
        self.k_lossless: int = 29 if self.z_input else 2 * self.m - 2
        self.chan_bits: int = chan_bits
        self.chan_cap: int = 2 * self.k_cap
        self.float_view: bool = float_view
        self.n_bits: int = self.N * self.N * 4 * self.m
        self.n_k: int = self.k_cap + 1
        self.num_paulis: int = 4 ** self.n

        self.register_buffer('pow2', torch.tensor([2.0 ** i for i in range(self.m)], dtype=torch.float64), persistent=False)
        self.register_buffer('bit_shifts', torch.arange(self.m, dtype=torch.int64), persistent=False)
        # Galois embeddings j = 1, 3: z = sum_t c_t omega^t -> sum_t c_t omega^(j t)
        w = np.exp(1j * np.pi / 4)
        js = np.array([1, 3])
        self.register_buffer('emb', torch.tensor(w ** (np.arange(4)[:, None] * js[None, :]), dtype=torch.complex128), persistent=False)  # (4, 2)
        sigma, phase = ring.pauli_tables(self.n)  # P[i, sigma[p, i]] = omega^phase[p, i]; every phase is even
        gens = ring.pauli_generator_indices(self.n)
        self.register_buffer('gen_sigma', torch.tensor(sigma[gens], dtype=torch.int64), persistent=False)  # (2n, N)
        self.register_buffer('gen_ph', torch.tensor(w ** (js[:, None, None] * phase[gens][None]), dtype=torch.complex128), persistent=False)  # (2, 2n, N)
        # Re tr(Q M) = sum_i Re(omega^(j phase[Q, i]) M[sigma[Q, i], i]) is a signed selection of Re M / Im M (the phase
        # is a power of i): proj[e] maps view_as_real(M) flattened (a, b, re/im) to the 4^n traces
        proj = np.zeros((2, 2 * self.N * self.N, self.num_paulis))
        part_sign = {0: (0, 1.0), 2: (1, -1.0), 4: (0, -1.0), 6: (1, 1.0)}  # Re(i^p m) = Re m, -Im m, -Re m, Im m
        for e, j in enumerate(js):
            for q in range(self.num_paulis):
                for i in range(self.N):
                    part, sign = part_sign[int(j * phase[q, i]) % 8]
                    proj[e, (sigma[q, i] * self.N + i) * 2 + part, q] += sign
        self.register_buffer('proj', torch.tensor(proj, dtype=torch.float64), persistent=False)  # (2, 2N^2, 4^n)
        self.register_buffer('cbit_shifts', torch.arange(max(chan_bits, 1), dtype=torch.int64), persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):  # type: ignore[no-untyped-def]
        for name in self._LEGACY_BUFFERS:
            state_dict.pop(prefix + name, None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def extra_dim(self) -> int:
        dim = 0
        if self.chan_bits > 0:
            dim += 2 * self.n * self.num_paulis * 2 * self.chan_bits + 2 * self.n * (self.chan_cap + 1)
        if self.float_view:
            dim += 2 * self.N * self.N
        return dim

    def _decode(self, x: Tensor):
        """B input: bits + exponent one-hot -> signed integer coefficients (B, N, N, 4) float64 and exponents (B,) int64"""
        B = x.shape[0]
        bits = x[:, :self.n_bits].to(torch.float64).reshape(B, self.N, self.N, 4, self.m)
        r = bits @ self.pow2
        c = r - (2.0 ** self.m) * (r >= 2.0 ** (self.m - 1)).to(torch.float64)
        k = x[:, self.n_bits:self.n_bits + self.n_k].argmax(dim=1)
        return c, k

    def _expand_z(self, x: Tensor):
        """Z input (int16 coefficients, k) -> [B<m> bits | k one-hot] exactly as the domain's B<m> encoding, the
        coefficients (B, N, N, 4) float64 and exponents (B,) int64"""
        B = x.shape[0]
        ci = x[:, :-1].to(torch.int64)
        k = x[:, -1].to(torch.int64)
        res = (ci & ((1 << self.m) - 1)).unsqueeze(-1)  # two's complement: residue mod 2^m
        bits = ((res >> self.bit_shifts) & 1).reshape(B, -1).to(torch.float32)
        onehot = nn.functional.one_hot(torch.clamp(k, max=self.k_cap), self.n_k).to(torch.float32)
        return torch.cat([bits, onehot], dim=1), ci.to(torch.float64).reshape(B, self.N, self.N, 4), k

    def _channel(self, c: Tensor, k: Tensor):
        B, N, n = c.shape[0], self.N, self.n
        E = torch.einsum('bxyt,te->ebxy', c.to(torch.complex128), self.emb)  # (2, B, N, N): both embeddings of R
        # R P_g R^dagger = E (ph_g(i) E^dagger[sigma_g(i), :])_i -- one (N x N)(N x 2nN) product covers every generator
        G = E.conj().transpose(-1, -2)[:, :, self.gen_sigma, :] * self.gen_ph.view(2, 1, 2 * n, N, 1)  # (2, B, 2n, N, N)
        G = G.permute(0, 1, 3, 2, 4).reshape(2, B, N, 2 * n * N)
        M = (E @ G).reshape(2, B, N, 2 * n, N).permute(0, 1, 3, 2, 4).contiguous()  # (2, B, 2n, N, N)
        T = torch.view_as_real(M).reshape(2, B * 2 * n, 2 * N * N) @ self.proj  # traces a + b sqrt2, a - b sqrt2
        T = T.reshape(2, B, 2 * n, self.num_paulis)
        a = torch.round((T[0] + T[1]) / 2.0)
        b = torch.round((T[0] - T[1]) / (2.0 * np.sqrt(2.0)))
        coef = torch.stack([a, b], dim=-1).to(torch.int64)  # (B, 2n, 4^n, 2): numerators a + b*sqrt2
        ex = (2 * k + 2 * n).view(B, 1).expand(B, 2 * n)
        a, b = coef[..., 0], coef[..., 1]
        v = torch.minimum(2 * _val2(a), 2 * _val2(b) + 1).min(dim=-1).values
        r = torch.minimum(v, ex)
        half = (r // 2).unsqueeze(-1)
        a2, b2 = a >> half, b >> half
        odd = (r % 2 == 1).unsqueeze(-1)
        coef = torch.stack([torch.where(odd, b2, a2), torch.where(odd, a2 >> 1, b2)], dim=-1)
        return coef, ex - r

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        if self.z_input:
            base, c, k = self._expand_z(x)
        else:
            base = x.to(torch.float32)
            c, k = self._decode(base)
        valid = (k <= self.k_lossless).to(base.dtype).view(B, 1)
        feats: List[Tensor] = [base]
        if self.chan_bits > 0:
            coef, ex = self._channel(c, k)
            res = torch.remainder(coef, 1 << self.chan_bits).reshape(B, -1, 1)
            bits = ((res >> self.cbit_shifts) & 1).reshape(B, -1).to(base.dtype)
            onehot = nn.functional.one_hot(torch.clamp(ex, max=self.chan_cap), self.chan_cap + 1).reshape(B, -1).to(base.dtype)
            feats += [bits * valid, onehot * valid]
        if self.float_view:
            a, b, cc, d = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
            scale = (2.0 ** (k.to(torch.float64) / 2)).view(B, 1, 1)
            re = (a + (b - d) / np.sqrt(2.0)) / scale
            im = (cc + (b + d) / np.sqrt(2.0)) / scale
            feats.append(torch.cat([re.reshape(B, -1), im.reshape(B, -1)], dim=1).to(base.dtype) * valid)
        return torch.cat(feats, dim=1)


@deepxube_nnet_factory.register_class("resnet_fc_ring")
class ResnetFCRing(HeurNNet[FlatIn]):
    @staticmethod
    def nnet_input_type() -> Type[FlatIn]:
        return FlatIn

    def __init__(self, nnet_input: FlatIn, out_dim: int, q_fix: bool, res_dim: int = 1000, num_blocks: int = 4,
                 batch_norm: bool = False, weight_norm: bool = False, layer_norm: bool = False, act_fn: str = "RELU",
                 chan_bits: int = 2, float_view: bool = False):
        super().__init__(nnet_input, out_dim, q_fix)
        domain = nnet_input.domain
        assert isinstance(domain, QCircuitExact), "resnet_fc_ring needs the qcircuit_exact domain"
        input_dims, one_hot_depths = nnet_input.get_input_info()
        assert len(input_dims) == 1 and one_hot_depths[0] == 1
        self.ring_features = RingFeatures(domain, chan_bits=chan_bits, float_view=float_view)
        base_dim: int = self.ring_features.n_bits + self.ring_features.n_k if self.ring_features.z_input else input_dims[0]
        input_dim_tot: int = base_dim + self.ring_features.extra_dim

        self.res_dim: int = res_dim
        group_norm: int = 1 if layer_norm else -1

        def res_block_init() -> nn.Module:
            return FullyConnectedModel(res_dim, [res_dim] * 2, [act_fn, "LINEAR"],
                                       batch_norms=[batch_norm] * 2, weight_norms=[weight_norm] * 2,
                                       group_norms=[group_norm] * 2)

        self.heur = nn.Sequential(
            nn.Linear(input_dim_tot, res_dim),
            ResnetModel(res_block_init, num_blocks, act_fn),
            nn.Linear(res_dim, self.out_dim)
        )

    def _forward(self, inputs: List[Tensor]) -> Tensor:
        with torch.no_grad():
            feats: Tensor = self.ring_features(inputs[0])
        return self.heur(feats)


@deepxube_nnet_factory.register_parser("resnet_fc_ring")
class ResnetFCRingParser(ResnetFCParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("C", "chan_bits", int, "channel-representation residue bits computed on the device (0 disables)", default=2)
        self.add_argument("fv", "float_view", None, "append the float view of the relative unitary")
