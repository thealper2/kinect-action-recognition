"""Skeleton graph and ST-GCN adjacency partitioning (Yan et al., AAAI 2018).

For the 15-joint KARD skeleton with the torso as root, the *spatial configuration*
partition splits the 1-hop neighbourhood of every joint i (including i) into
three subsets, according to the hop distance d(.) to the root:

* k = 0, root:        {j : d(j) == d(i)}   (i itself; no same-level neighbours in a tree)
* k = 1, centripetal: {j : d(j) <  d(i)}   (the parent, closer to the torso)
* k = 2, centrifugal: {j : d(j) >  d(i)}   (the children)

Every subset is normalised with the in-degree of the full neighbourhood,
``A_k <- A_k D^{-1}`` with ``D_ii = sum_j (A + I)_{ji}``, so that, summed over k, each output
joint takes the mean over its neighbourhood (columns of ``sum_k A_k`` sum to 1).
"""

from __future__ import annotations

import numpy as np

from skeleton_action.constants import KARD_EDGES, NUM_JOINTS, J


def adjacency_matrix(num_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Binary symmetric adjacency without self loops."""
    a = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for i, j in edges:
        if i == j or not (0 <= i < num_nodes and 0 <= j < num_nodes):
            raise ValueError(f"invalid edge {(i, j)}")
        a[i, j] = a[j, i] = 1.0
    return a


def hop_distance(num_nodes: int, edges: list[tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    """``[V, V]`` hop distances up to ``max_hop`` (``inf`` beyond)."""
    a = adjacency_matrix(num_nodes, edges) + np.eye(num_nodes)
    dist = np.full((num_nodes, num_nodes), np.inf)
    reach = [np.linalg.matrix_power(a, d) > 0 for d in range(max_hop + 1)]
    for d in range(max_hop, -1, -1):
        dist[reach[d]] = d
    return dist


def normalize_digraph(a: np.ndarray) -> np.ndarray:
    """Column normalisation ``A D^{-1}`` with ``D_ii = sum_j A_ji``."""
    deg = a.sum(axis=0)
    inv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-12), 0.0)
    return a @ np.diag(inv)


def graph_distance_to_root(num_nodes: int, edges: list[tuple[int, int]], root: int) -> np.ndarray:
    """BFS distance of every node to ``root`` over the (unbounded) skeleton graph."""
    adj = adjacency_matrix(num_nodes, edges)
    dist = np.full(num_nodes, np.inf)
    dist[root] = 0
    frontier = [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(adj[u]):
                if np.isinf(dist[v]):
                    dist[v] = dist[u] + 1
                    nxt.append(int(v))
        frontier = nxt
    if np.isinf(dist).any():
        raise ValueError("skeleton graph is not connected")
    return dist


def build_adjacency(
    strategy: str = "spatial",
    num_nodes: int = NUM_JOINTS,
    edges: list[tuple[int, int]] | None = None,
    center: int = J.TORSO,
) -> np.ndarray:
    """Return the stacked, normalised adjacency ``[K, V, V]``.

    Strategies: ``uniform`` (K=1), ``distance`` (K=2: self / 1-hop neighbours) and
    ``spatial`` (K=3, default, see module docstring).
    """
    edges = KARD_EDGES if edges is None else edges
    hop = hop_distance(num_nodes, edges, max_hop=1)
    a_full = (hop <= 1).astype(np.float64)  # A + I
    a_norm = normalize_digraph(a_full)
    if strategy == "uniform":
        return a_norm[None].astype(np.float32)
    if strategy == "distance":
        return np.stack([a_norm * (hop == d) for d in (0, 1)]).astype(np.float32)
    if strategy != "spatial":
        raise ValueError(f"unknown strategy {strategy!r}")
    d_root = graph_distance_to_root(num_nodes, edges, center)
    root = np.zeros_like(a_norm)
    close = np.zeros_like(a_norm)
    far = np.zeros_like(a_norm)
    # a[j, i]: contribution of source joint j to target joint i.
    for i in range(num_nodes):
        for j in range(num_nodes):
            if hop[j, i] > 1:
                continue
            if d_root[j] == d_root[i]:
                root[j, i] = a_norm[j, i]
            elif d_root[j] < d_root[i]:
                close[j, i] = a_norm[j, i]
            else:
                far[j, i] = a_norm[j, i]
    return np.stack([root, close, far]).astype(np.float32)
