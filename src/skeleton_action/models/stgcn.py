"""Model C: Spatial-Temporal Graph Convolutional Network (Yan et al., AAAI 2018).

Notation: N batch, C channels, T frames, V = 15 joints, K adjacency subsets.

* Input ``[N, T, V, C]`` -> ``[N, C, T, V]``; data BatchNorm over the V*C channels.
* Spatial graph convolution (per block)::

      Y[n, c', t, w] = sum_k sum_v (W_k X)[n, c', t, v] * (A_k ⊙ M_k)[v, w]

  ``W_k`` is a 1x1 convolution (C -> K*C'), ``A_k`` the normalised partitioned
  adjacency (see :mod:`skeleton_action.models.graph`) and ``M_k`` an optional learnable
  edge-importance mask (initialised to 1).
* Temporal convolution: ``Conv2d(C', C', kernel=(k_t, 1), stride=(s, 1))`` along T,
  shared across joints.
* Block: GCN -> BN -> ReLU -> TCN -> BN -> Dropout, + residual, -> ReLU.
* Head: global average pooling over (T, V) -> dropout -> linear -> logits ``[N, K_cls]``.
"""

from __future__ import annotations

import torch
from torch import nn

from skeleton_action.models.graph import build_adjacency


class GraphConv(nn.Module):
    """Partitioned spatial graph convolution."""

    def __init__(self, in_channels: int, out_channels: int, num_subsets: int) -> None:
        super().__init__()
        self.k = num_subsets
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        if a.shape != (self.k, v, v):
            raise ValueError(f"adjacency shape {tuple(a.shape)} != {(self.k, v, v)}")
        y = self.conv(x).view(n, self.k, self.out_channels, t, v)
        return torch.einsum("nkctv,kvw->nctw", y, a).contiguous()


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_subsets: int,
        temporal_kernel: int = 9,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 != 1:
            raise ValueError("temporal_kernel must be odd")
        pad = (temporal_kernel - 1) // 2
        self.gcn = GraphConv(in_channels, out_channels, num_subsets)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (temporal_kernel, 1), (stride, 1), (pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout),
        )
        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        res = 0 if self.residual is None else self.residual(x)
        return self.relu(self.tcn(self.gcn(x, a)) + res)


class STGCN(nn.Module):
    """Compact ST-GCN for 15-joint skeletons.

    Args:
        channels: output channels of each block (default 6 blocks, ~0.5 M parameters).
        strides: temporal stride of each block (same length as ``channels``).
        temporal_kernel: odd temporal kernel size.
        graph_strategy: ``spatial`` (K=3), ``distance`` (K=2) or ``uniform`` (K=1).
        edge_importance: learn a multiplicative mask per block on the adjacency.
    """

    def __init__(
        self,
        num_classes: int,
        num_joints: int = 15,
        in_channels: int = 3,
        channels: tuple[int, ...] | list[int] = (32, 32, 64, 64, 128, 128),
        strides: tuple[int, ...] | list[int] = (1, 1, 2, 1, 2, 1),
        temporal_kernel: int = 9,
        dropout: float = 0.2,
        head_dropout: float = 0.3,
        graph_strategy: str = "spatial",
        edge_importance: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError("channels and strides must have equal length")
        a = torch.from_numpy(build_adjacency(graph_strategy, num_nodes=num_joints))
        if not torch.isfinite(a).all():
            raise ValueError("non-finite adjacency")
        self.register_buffer("A", a)
        k = a.shape[0]
        self.data_bn = nn.BatchNorm1d(num_joints * in_channels)
        blocks = []
        c_in = in_channels
        for i, (c_out, s) in enumerate(zip(channels, strides, strict=True)):
            blocks.append(
                STGCNBlock(
                    c_in,
                    c_out,
                    k,
                    temporal_kernel,
                    s,
                    dropout=0.0 if i == 0 else dropout,
                    residual=i > 0,
                )
            )
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        if edge_importance:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones_like(a)) for _ in blocks]
            )
        else:
            self.edge_importance = None
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(c_in, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, t, v, c = x.shape
        x = x.permute(0, 2, 3, 1).reshape(n, v * c, t)  # [N, V*C, T]
        x = self.data_bn(x)
        x = x.reshape(n, v, c, t).permute(0, 2, 3, 1).contiguous()  # [N, C, T, V]
        for i, block in enumerate(self.blocks):
            a = self.A if self.edge_importance is None else self.A * self.edge_importance[i]
            x = block(x, a)
        x = x.mean(dim=(2, 3))  # global spatio-temporal average pooling -> [N, C]
        return self.head(x)
