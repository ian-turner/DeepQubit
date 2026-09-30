"""resnet_fc_ring: the fully-connected resnet heuristic with a fixed, parameter-free front layer that does the
ring algebra on the device.

The qcircuit_exact domain sends only the cheap B<m> residue bits of the canonical relative unitary R = G S^dagger
(plus the exponent one-hot). `RingFeatures` reconstructs the integer coefficients from the bits (exact while the
exponent k <= 2m - 2), builds the companion form, and computes the channel-representation features -- the same
numbers the domain's `C<m>` encoding produces in numpy, but as batched matmuls and gathers on whatever device the
network lives on -- and optionally the float view (`M`). The resnet then sees [B bits | channel bits | (floats)].

Network string: resnet_fc_ring.<H>H_<B>B[_bn][_<m>C][_fv]   e.g. resnet_fc_ring.1000H_4B_bn_2C
  <m>C  channel residue bits per numerator (default 2; 0 disables the channel block)
  fv    append the float view of R
Rows whose exponent exceeds the lossless range of the B block get a zeroed channel/float block.
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
    def __init__(self, domain: QCircuitExact, chan_bits: int = 2, float_view: bool = False):
        super().__init__()
        parts = domain._parts
        assert parts[0][0] == 'B' and all(kind != 'C' for kind, _ in parts), \
            "resnet_fc_ring expects the domain encoding to start with B<m> and not to include C (the network computes it)"
        self.n: int = domain.num_qubits
        self.N: int = domain.N
        self.m: int = parts[0][1]
        self.k_cap: int = domain.k_cap
        self.k_lossless: int = 2 * self.m - 2
        self.chan_bits: int = chan_bits
        self.chan_cap: int = 2 * self.k_cap
        self.float_view: bool = float_view
        self.n_bits: int = self.N * self.N * 4 * self.m
        self.n_k: int = self.k_cap + 1
        self.num_paulis: int = 4 ** self.n

        self.register_buffer('pow2', torch.tensor([2.0 ** i for i in range(self.m)], dtype=torch.float64))
        self.register_buffer('rot4f', torch.tensor(ring._ROT4F, dtype=torch.float64))
        gens = np.stack([ring.companion(ring.pauli_coeffs(self.n, p)) for p in ring.pauli_generator_indices(self.n)])
        self.register_buffer('gens', torch.tensor(gens, dtype=torch.float64))  # (2n, 4N, 4N)
        self.register_buffer('chan_idx', torch.tensor(domain._chan_idx.reshape(-1), dtype=torch.int64))
        self.register_buffer('chan_sign', torch.tensor(domain._chan_sign, dtype=torch.float64))  # (1, 4^n, N, 1)
        self.register_buffer('cbit_shifts', torch.arange(max(chan_bits, 1), dtype=torch.int64))

    @property
    def extra_dim(self) -> int:
        dim = 0
        if self.chan_bits > 0:
            dim += 2 * self.n * self.num_paulis * 2 * self.chan_bits + 2 * self.n * (self.chan_cap + 1)
        if self.float_view:
            dim += 2 * self.N * self.N
        return dim

    def _decode(self, x: Tensor):
        """bits + exponent one-hot -> signed integer coefficients (B, N, N, 4) float64 and exponents (B,) int64"""
        B = x.shape[0]
        bits = x[:, :self.n_bits].to(torch.float64).reshape(B, self.N, self.N, 4, self.m)
        r = bits @ self.pow2
        c = r - (2.0 ** self.m) * (r >= 2.0 ** (self.m - 1)).to(torch.float64)
        k = x[:, self.n_bits:self.n_bits + self.n_k].argmax(dim=1)
        return c, k

    def _channel(self, c: Tensor, k: Tensor):
        B, N, n = c.shape[0], self.N, self.n
        comp = (c.reshape(-1, 4) @ self.rot4f).reshape(B, N, N, 4, 4)
        R = comp.permute(0, 1, 3, 2, 4).reshape(B, 4 * N, 4 * N)
        Rt = R.transpose(1, 2)
        rows: List[Tensor] = []
        for g in range(2 * n):
            M = (R @ self.gens[g] @ Rt).reshape(B, -1)
            sel = M[:, self.chan_idx].reshape(B, self.num_paulis, N, 2) * self.chan_sign
            rows.append(sel.sum(dim=2))
        coef = torch.round(torch.stack(rows, dim=1)).to(torch.int64)  # (B, 2n, 4^n, 2): a + b*sqrt2
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
        c, k = self._decode(x)
        valid = (k <= self.k_lossless).to(x.dtype).view(B, 1)
        feats: List[Tensor] = [x]
        if self.chan_bits > 0:
            coef, ex = self._channel(c, k)
            res = torch.remainder(coef, 1 << self.chan_bits).reshape(B, -1, 1)
            bits = ((res >> self.cbit_shifts) & 1).reshape(B, -1).to(x.dtype)
            onehot = nn.functional.one_hot(torch.clamp(ex, max=self.chan_cap), self.chan_cap + 1).reshape(B, -1).to(x.dtype)
            feats += [bits * valid, onehot * valid]
        if self.float_view:
            a, b, cc, d = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
            scale = (2.0 ** (k.to(torch.float64) / 2)).view(B, 1, 1)
            re = (a + (b - d) / np.sqrt(2.0)) / scale
            im = (cc + (b + d) / np.sqrt(2.0)) / scale
            feats.append(torch.cat([re.reshape(B, -1), im.reshape(B, -1)], dim=1).to(x.dtype) * valid)
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
        input_dim_tot: int = input_dims[0] + self.ring_features.extra_dim

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
            feats: Tensor = self.ring_features(inputs[0].float())
        return self.heur(feats)


@deepxube_nnet_factory.register_parser("resnet_fc_ring")
class ResnetFCRingParser(ResnetFCParser):
    def __init__(self) -> None:
        super().__init__()
        self.add_argument("C", "chan_bits", int, "channel-representation residue bits computed on the device (0 disables)", default=2)
        self.add_argument("fv", "float_view", None, "append the float view of the relative unitary")
