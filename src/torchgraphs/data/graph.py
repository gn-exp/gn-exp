from __future__ import annotations
from typing import Optional
import dataclasses

import torch
import networkx as nx

from .base import _BaseGraph


@dataclasses.dataclass
class Graph(_BaseGraph):
    global_features: Optional[torch.Tensor] = None

    _feature_fields = _BaseGraph._feature_fields + ('global_features',)

    @property
    def global_features_shape(self):
        return self.global_features.shape if self.global_features is not None else None

    @property
    def global_features_as_edges(self) -> torch.Tensor:
        """Broadcast `global_features` along the the first dimension to match the first dimension of `edge_features`,
        therefore the shape of the returned tensor is `(num_edges,) + self.global_features.shape`
        """
        return self.global_features.expand(self.num_edges, *self.global_features.shape)

    @property
    def global_features_as_nodes(self):
        """Broadcast `global_features` along the the first dimension to match the first dimension of `node_features`,
        therefore the shape of the returned tensor is `(num_nodes,) + self.global_features.shape`
        """
        return self.global_features.expand(self.num_nodes, *self.global_features.shape)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"n={self.num_nodes}, "
                f"e={self.num_edges}, "
                f"n_shape={self.node_features_shape}, "
                f"e_shape={self.edge_features_shape}, "
                f"g_shape={self.global_features_shape})")

    def to_networkx(self, cls=nx.MultiDiGraph):
        g = cls()
        if self.node_features is not None:
            g.add_nodes_from([(i, {'features': f}) for i, f in enumerate(self.node_features)])
        else:
            g.add_nodes_from(range(self.num_nodes))

        if self.edge_features is None:
            g.add_edges_from([(s.item(), r.item()) for s, r in zip(self.senders, self.receivers)])
        else:
            g.add_edges_from([(s.item(), r.item(), {'features': f})
                              for s, r, f in zip(self.senders, self.receivers, self.edge_features)])
        if self.global_features is not None:
            g.graph['features'] = self.global_features
        return g

    @classmethod
    def from_networkx(cls, graph_nx: nx.Graph) -> Graph:
        # Handle node features
        if graph_nx.number_of_nodes() > 0 and 'features' in graph_nx.nodes[0]:
            node_features = torch.stack([features for node_id, features in graph_nx.nodes(data='features')])
        else:
            node_features = None

        # Handle edge features
        if graph_nx.number_of_edges() > 0:
            senders, receivers, edge_features = zip(*graph_nx.edges(data='features'))
            senders = torch.tensor(senders, dtype=torch.long)
            receivers = torch.tensor(receivers, dtype=torch.long)
            if edge_features[0] is not None:
                edge_features = torch.stack(edge_features)
            else:
                edge_features = None
        else:
            senders = torch.tensor([], dtype=torch.long)
            receivers = torch.tensor([], dtype=torch.long)
            edge_features = None

        # Handle global features
        global_features = graph_nx.graph.get('features', None)

        return cls(
            num_nodes=graph_nx.number_of_nodes(),
            num_edges=graph_nx.number_of_edges(),
            node_features=node_features,
            edge_features=edge_features,
            senders=senders,
            receivers=receivers,
            global_features=global_features,
        )
