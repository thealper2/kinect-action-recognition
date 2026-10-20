from __future__ import annotations

import numpy as np
import pytest
import torch

from skeleton_action.constants import KARD_EDGES, NUM_JOINTS, J
from skeleton_action.models import build_model, count_parameters
from skeleton_action.models.graph import (
    adjacency_matrix,
    build_adjacency,
    graph_distance_to_root,
)
from skeleton_action.models.lstm import LSTMClassifier


def test_graph_definition() -> None:
    assert len(KARD_EDGES) == 14  # tree over 15 joints
    a = adjacency_matrix(NUM_JOINTS, KARD_EDGES)
    assert a.shape == (15, 15)
    np.testing.assert_array_equal(a, a.T)
    assert np.all(np.diag(a) == 0)
    d = graph_distance_to_root(NUM_JOINTS, KARD_EDGES, J.TORSO)
    assert d[J.TORSO] == 0 and d[J.NECK] == 1 and d[J.HEAD] == 2 and d[J.R_HAND] == 4


@pytest.mark.parametrize("strategy,k", [("spatial", 3), ("distance", 2), ("uniform", 1)])
def test_adjacency_partition(strategy: str, k: int) -> None:
    a = build_adjacency(strategy)
    assert a.shape == (k, 15, 15) and np.isfinite(a).all()
    total = a.sum(0)
    support = (adjacency_matrix(15, KARD_EDGES) + np.eye(15)) > 0
    np.testing.assert_array_equal(total > 0, support)  # subsets partition A + I
    np.testing.assert_array_equal(support, support.T)
    np.testing.assert_allclose(total.sum(axis=0), 1.0, atol=1e-6)  # column-normalised
    if strategy == "spatial":
        root, close, far = a
        np.testing.assert_array_equal(root > 0, np.eye(15) > 0)  # tree: root = self only
        np.testing.assert_array_equal((close > 0), (far > 0).T)  # parent/child duality
        assert close[J.NECK, J.HEAD] > 0 and far[J.HEAD, J.NECK] > 0


MODEL_PARAMS = {
    "mlp": {"hidden_dims": [32, 16], "num_segments": 4},
    "lstm": {"embed_dim": 16, "hidden_size": 16, "num_layers": 2},
    "stgcn": {"channels": [8, 16, 16], "strides": [1, 2, 1], "temporal_kernel": 5},
}


@pytest.mark.parametrize("name", ["mlp", "lstm", "stgcn"])
@pytest.mark.parametrize("channels", [3, 6])
def test_forward_backward(name: str, channels: int) -> None:
    torch.manual_seed(0)
    m = build_model(name, 18, 24, 15, channels, **MODEL_PARAMS[name])
    x = torch.randn(5, 24, 15, channels)
    y = torch.randint(0, 18, (5,))
    out = m(x)
    assert out.shape == (5, 18)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert count_parameters(m) > 0
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(x)).all()


@pytest.mark.parametrize("agg", ["mean", "meanmax", "segments", "flatten"])
def test_mlp_aggregations(agg: str) -> None:
    m = build_model("mlp", 18, 16, 15, 3, hidden_dims=[16], aggregation=agg)
    assert m(torch.randn(2, 16, 15, 3)).shape == (2, 18)


def test_mlp_segments_are_order_sensitive() -> None:
    torch.manual_seed(0)
    m = build_model("mlp", 18, 16, 15, 3, hidden_dims=[16], aggregation="segments").eval()
    x = torch.randn(1, 16, 15, 3)
    assert not torch.allclose(m(x), m(x.flip(1)))
    mm = build_model("mlp", 18, 16, 15, 3, hidden_dims=[16], aggregation="mean").eval()
    assert torch.allclose(mm(x), mm(x.flip(1)), atol=1e-6)


def test_lstm_lengths_ignore_padding() -> None:
    torch.manual_seed(0)
    m = LSTMClassifier(18, 15, 3, embed_dim=8, hidden_size=8, num_layers=1).eval()
    x = torch.randn(2, 12, 15, 3)
    lengths = torch.tensor([12, 7])
    x2 = x.clone()
    x2[1, 7:] = 100.0  # garbage in the padded tail
    with torch.no_grad():
        assert torch.allclose(m(x, lengths), m(x2, lengths), atol=1e-5)
        for pooling in ("last",):
            ml = LSTMClassifier(18, 15, 3, embed_dim=8, hidden_size=8, pooling=pooling).eval()
            assert torch.allclose(ml(x, lengths), ml(x2, lengths), atol=1e-5)


def test_reproducible_init() -> None:
    torch.manual_seed(3)
    a = build_model("stgcn", 18, 16, 15, 3, **MODEL_PARAMS["stgcn"])
    torch.manual_seed(3)
    b = build_model("stgcn", 18, 16, 15, 3, **MODEL_PARAMS["stgcn"])
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert ka == kb and torch.equal(va, vb)


def test_stgcn_default_size_and_stability() -> None:
    m = build_model("stgcn", 18, 64, 15, 3)
    assert 2e5 < count_parameters(m) < 2e6  # compact by design
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 64, 15, 3) * 50)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        build_model("unknown", 18, 64, 15, 3)
