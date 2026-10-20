"""Model B: per-frame joint encoder + (Bi)LSTM sequence classifier."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):
    """Frame encoder -> (Bi)LSTM -> temporal pooling -> linear head.

    Shapes::

        x [B, T, J, C] -> [B, T, J*C] -> encoder -> [B, T, E]
          -> LSTM -> [B, T, H * dirs] -> pool -> [B, H * dirs] -> logits [B, K]

    ``lengths`` (optional, ``[B]``) marks valid prefixes for padded batches; sequences
    are then packed so that padding never enters the recurrence, and mean pooling is
    masked. With the default fixed-length resampling all lengths equal T.

    Pooling: ``mean`` over valid time steps or ``last`` (final hidden state of the
    forward direction concatenated with the backward direction's state at t=0).
    """

    def __init__(
        self,
        num_classes: int,
        num_joints: int = 15,
        in_channels: int = 3,
        embed_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in ("mean", "last"):
            raise ValueError("pooling must be 'mean' or 'last'")
        self.pooling = pooling
        self.bidirectional = bidirectional
        self.encoder = nn.Sequential(
            nn.Linear(num_joints * in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim), nn.Dropout(dropout), nn.Linear(out_dim, num_classes)
        )
        self._init_lstm()

    def _init_lstm(self) -> None:
        h = self.lstm.hidden_size
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                for k in range(4):
                    nn.init.orthogonal_(p.data[k * h : (k + 1) * h])
            elif "bias" in name:
                nn.init.zeros_(p)
                if "bias_ih" in name:
                    p.data[h : 2 * h] = 1.0  # forget gate (gate order i, f, g, o)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        b, t, j, c = x.shape
        e = self.encoder(x.reshape(b, t, j * c))
        if lengths is not None:
            packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, (h_n, _) = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=t)
            mask = torch.arange(t, device=x.device)[None, :] < lengths.to(x.device)[:, None]
        else:
            out, (h_n, _) = self.lstm(e)
            mask = None
        if self.pooling == "mean":
            if mask is None:
                z = out.mean(dim=1)
            else:
                m = mask.unsqueeze(-1).to(out.dtype)
                z = (out * m).sum(1) / m.sum(1).clamp_min(1.0)
        else:
            z = torch.cat([h_n[-2], h_n[-1]], dim=-1) if self.bidirectional else h_n[-1]
        return self.head(z)
