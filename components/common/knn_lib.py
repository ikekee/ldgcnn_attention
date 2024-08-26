"""This module contains a function for applying knn algorithm."""
import torch
from torch_cluster import knn_graph


def apply_knn(x: torch.Tensor, batch: torch.Tensor, k=30) -> torch.Tensor:
    """Applies knn for making a graph.

    Args:
        x: Node feature matrix.
        batch: Batch vector, which assigns each node to a specific example.
        k: The number of neighbors.

    Returns:
        Tensor with graph edges for EdgeConv.
    """
    idx = knn_graph(x, k=k, batch=batch)
    return idx
