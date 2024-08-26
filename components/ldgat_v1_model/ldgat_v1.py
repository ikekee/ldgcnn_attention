"""This module contains a class with an implementation of the LDGATv1 model."""
import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.nn import MLP
from torch_geometric.nn.dense.linear import Linear

from components.common.knn_lib import apply_knn


class LDGATv1(nn.Module):
    """Class with an implementation of a model based on the LDGCNN architecture with a graph attention mechanism.

    This model uses a first version of the graph attention mechanism (check https://arxiv.org/abs/1710.10903
    and https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html for details).

    Attributes:
        k: Number of nearest neighbors for creating graph using KNN.
        attention_conv1: First attention convolution layer.
        multihead_l1: First linear layer for rolling up the attention heads' results.
        attention_conv2: Second attention convolution layer.
        multihead_l2: Second linear layer for rolling up the attention heads' results.
        attention_conv3: Third attention convolution layer.
        multihead_l3: Third linear layer for rolling up the attention heads' results.
        attention_conv4: Fourth attention convolution layer.
        multihead_l4: Fourth linear layer for rolling up the attention heads' results.
        fe_mlp: MLP that uses extracted local features for creating global features vector.
        mlp: MLP that uses concatenated local and global features vectors
         for predicting segmentation scores.
    """

    def __init__(self, in_channels: int, out_channels: int, k=30, heads: int = 3):
        """Creates an instance of the class.

        Args:
            in_channels: Number of the input channels.
            out_channels: Number of output segmentation classes.
            k: Number of nearest neighbors for creating graph using KNN.
            heads: Number of the attention heads.
        """
        super().__init__()

        self.k = k

        # Extracting global features
        self.attention_conv1 = GATConv(
            in_channels=in_channels, out_channels=64, heads=heads)
        self.multihead_l1 = Linear(64 * heads, 64)

        self.attention_conv2 = GATConv(
            in_channels=(64 + in_channels), out_channels=64, heads=heads)
        self.multihead_l2 = Linear(64 * heads, 64)

        self.attention_conv3 = GATConv(
            in_channels=(64 + 64 + in_channels), out_channels=64, heads=heads)
        self.multihead_l3 = Linear(64 * heads, 64)

        self.attention_conv4 = GATConv(
            in_channels=(64 + 64 + 64 + in_channels), out_channels=128, heads=heads)
        self.multihead_l4 = Linear(128 * heads, 128)

        self.fe_mlp = MLP(
            [in_channels + 64 + 64 + 64 + 128, 1024, 1024],
            norm=None,
            dropout=0.5
        )

        # MLP for prediction segmentation scores
        self.mlp = MLP(
            [in_channels + 64 + 64 + 64 + 128 + 1024, 256, 256, 128, out_channels],
            dropout=0.5,
            norm=None
        )

    def forward(self, data) -> torch.Tensor:
        """Performs forward propagation.

        Args:
            data: DataBatch.

        Returns:
            Model output for desired input as a torch.Tensor.
        """
        if data.pos is not None and data.x is not None:
            x, pos, batch = data.x, data.pos, data.batch
            # x0 is (num_points, in_channels)
            x0 = torch.cat([x, pos], dim=-1)
        elif data.pos is None:
            x, batch = data.x, data.batch
            x0 = x
        elif data.x is None:
            pos, batch = data.pos, data.batch
            x0 = pos
        num_points = batch.size(0)
        edge_index = apply_knn(x0, batch, k=self.k)
        # (num_points, in_channels) -> (num_points, 64)
        x1 = self.attention_conv1(x0, edge_index)
        x1 = self.multihead_l1(x1)

        edge_index = apply_knn(x1, batch, k=self.k)
        link_1 = torch.cat([x0, x1], dim=1)
        # (num_points, in_channels + 64) -> (num_points, 64)
        x2 = self.attention_conv2(link_1, edge_index)
        x2 = self.multihead_l2(x2)

        edge_index = apply_knn(x2, batch, k=self.k)
        link_2 = torch.cat([x0, x1, x2], dim=1)
        # (num_points, in_channels + 64 + 64) -> (num_points, 64)
        x3 = self.attention_conv3(link_2, edge_index)
        x3 = self.multihead_l3(x3)

        edge_index = apply_knn(x2, batch, k=self.k)
        link_3 = torch.cat([x0, x1, x2, x3], dim=1)
        # (num_points, in_channels + 64 + 64 + 64) -> (num_points, 128)
        x4 = self.attention_conv4(link_3, edge_index)
        x4 = self.multihead_l4(x4)

        link_4 = torch.cat([x0, x1, x2, x3, x4], dim=-1)
        # (num_points, in_channels + 64 + 64 + 64 + 128) -> (num_points, 1024)
        x5 = self.fe_mlp(link_4)

        # x6 is a global feature tensor
        # (num_points, 1024) -> (1, 1024)
        global_features, _ = torch.max(x5, dim=0, keepdim=True)
        # (1, 1024) -> (num_points, 1024)
        global_features_repeated = global_features.repeat(num_points, 1)
        # (num_points, in_channels + 64 + 64 + 64 + 128) + (num_points, 1024)
        # -> (num_points, in_channels + 64 + 64 + 64 + 128 + 1024)
        local_global_features = torch.cat([link_4, global_features_repeated], axis=1)
        # (num_points, in_channels + 64 + 64 + 64 + 128 + 1024) -> (num_points, out_channels)
        out = self.mlp(local_global_features)
        return nn.functional.log_softmax(out, dim=1)
