"""
Heterogeneous GNN Encoder
=========================
Encodes the heterogeneous graph created by `hetero_graph_builder`.
Returns embeddings for employee/current_job/post_type nodes that can be consumed
by downstream multi-task heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


@dataclass
class HeteroGNNConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    use_layernorm: bool = True
    add_self_loop: bool = True
    return_dict: bool = False


class HeteroGNN(nn.Module):
    """
    HGT-based encoder for the heterogeneous graph.
    """

    def __init__(
        self,
        metadata: Tuple[list, list],
        input_dims: Dict[str, int],
        config: Optional[HeteroGNNConfig] = None,
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.config = config or HeteroGNNConfig()

        self.proj = nn.ModuleDict(
            {node_type: nn.Linear(input_dims[node_type], self.config.hidden_dim) for node_type in metadata[0]}
        )

        self.gnn_layers = nn.ModuleList(
            [
                HGTConv(
                    in_channels=self.config.hidden_dim,
                    out_channels=self.config.hidden_dim,
                    metadata=self.metadata,
                    heads=self.config.heads,
                    group="sum",
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(self.config.hidden_dim)
                        if self.config.use_layernorm
                        else nn.Identity()
                        for node_type in metadata[0]
                    }
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        missing_ntypes = [ntype for ntype in self.metadata[0] if ntype not in data.node_types]
        if missing_ntypes:
            raise KeyError(f"Graph 缺少模型需要的节点类型: {missing_ntypes}")

        x_dict = {
            ntype: self.proj[ntype](data[ntype].x.float())
            for ntype in self.metadata[0]
        }
        edge_index_dict = data.edge_index_dict

        for layer, conv in enumerate(self.gnn_layers):
            h_dict = conv(x_dict, edge_index_dict)

            out_dict = {}
            for ntype, h in h_dict.items():
                h = self.activation(h)
                h = self.dropout(h)
                h = self.norms[layer][ntype](h)
                out_dict[ntype] = h
            x_dict = out_dict

        return x_dict

    def encode(self, data: HeteroData) -> Tuple[Tensor, Tensor, Tensor]:
        embeddings = self.forward(data)
        return embeddings["employee"], embeddings["current_job"], embeddings["post_type"]
