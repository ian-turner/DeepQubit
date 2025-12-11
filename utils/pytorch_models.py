import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from typing import List
from deepxube.base.heuristic import HeurNNetModule, HeurNNetV, HeurNNetQFixOut, HeurNNetQIn
from deepxube.nnet.pytorch_models import ResnetModel


class NeRFEmbedding(nn.Module):
    """Does NeRF embeddings for input Tensors."""
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        self._emb_vec = torch.tensor(
            [2 ** i for i in range(L)], requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d = x.shape
        if self._emb_vec.device != x.device:
            self._emb_vec = self._emb_vec.to(x.device)
        l = self.L
        x = repeat(x, 'b d-> b d l', l=l)
        _sin = (x * self._emb_vec).sin()
        _cos = (x * self._emb_vec).cos()
        interleaved = torch.stack([_sin, _cos], dim=-1).view(b, d * l * 2)
        return interleaved


class ResnetModel(HeurNNetModule):
    def __init__(self, input_size: int, L: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super(ResnetModel, self).__init__()
        self.L = L
        self.num_resnet_blocks = num_resnet_blocks
        self.batch_norm = batch_norm
        self.blocks = nn.ModuleList()

        # NeRF for input encoding
        if L > 0:
            self.nerf = NeRFEmbedding(L=L)
            self.fc1 = nn.Linear(input_size * L * 2, h1_dim)
        else:
            self.fc1 = nn.Linear(input_size, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for _ in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_goals_l: List[torch.Tensor]):
        # processing input
        x = states_goals_l[0].float()
        if self.L > 0:
            x = self.nerf(x).float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x
