"""Model A: per-frame MLP encoder + order-preserving temporal aggregation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

AGGREGATIONS = ("mean", "meanmax", "segments", "flatten")


class MLPClassifier(nn.Module):
    """Frame-wise MLP baseline.

    Shapes::

        x [B, T, J, C] -> reshape [B, T, J*C] -> frame MLP -> [B, T, D]
          -> aggregation -> [B, A] -> head -> logits [B, num_classes]

    Aggregation (``D`` = last hidden dim):

    * ``mean`` / ``meanmax``: order-invariant pooling, A = D / 2D. Cannot separate
      time-reversed actions (Sit down vs Stand up) unless velocity channels are used.
    * ``segments`` (default): mean over K equal temporal segments, concatenated
      (A = K*D). Keeps coarse temporal order at a small parameter cost.
    * ``flatten``: concatenate all frames (A = T*D); exact order, most parameters,
      tied to a fixed T.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames: int,
        num_joints: int = 15,
        in_channels: int = 3,
        hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        aggregation: str = "segments",
        num_segments: int = 4,
        head_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
        if aggregation == "segments" and num_segments > num_frames:
            raise ValueError("num_segments cannot exceed num_frames")
        self.aggregation = aggregation
        self.num_segments = num_segments
        self.num_frames = num_frames
        layers: list[nn.Module] = []
        d_in = num_joints * in_channels
        for h in hidden_dims:
            layers += [nn.Linear(d_in, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d_in = h
        self.frame_mlp = nn.Sequential(*layers)
        agg_dim = {
            "mean": d_in,
            "meanmax": 2 * d_in,
            "segments": num_segments * d_in,
            "flatten": num_frames * d_in,
        }[aggregation]
        self.head = nn.Sequential(
            nn.Linear(agg_dim, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, j, c = x.shape
        h = self.frame_mlp(x.reshape(b, t, j * c))  # [B, T, D]
        if self.aggregation == "mean":
            z = h.mean(dim=1)
        elif self.aggregation == "meanmax":
            z = torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=-1)
        elif self.aggregation == "segments":
            z = F.adaptive_avg_pool1d(h.transpose(1, 2), self.num_segments).flatten(1)
        else:
            if t != self.num_frames:
                raise ValueError(f"flatten aggregation needs T={self.num_frames}, got {t}")
            z = h.flatten(1)
        return self.head(z)
