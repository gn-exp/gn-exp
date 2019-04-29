import warnings

import torch.nn as nn

from ..data import GraphBatch


class GraphNetwork(nn.Module):
    def __init__(self, node_fn, edge_fn, global_fn,
                 edges_to_sender, edges_to_receiver, nodes_to_global, edges_to_global):
        super(GraphNetwork, self).__init__()
        self.node_fn = node_fn
        self.edge_fn = edge_fn
        self.global_fn = global_fn
        self.edges_to_sender = edges_to_sender
        self.edges_to_receiver = edges_to_receiver
        self.nodes_to_global = nodes_to_global
        self.edges_to_global = edges_to_global

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        edge_features = self.edge_fn(graphs)

        edges_to_sender = self.edges_to_sender(graphs, edge_features)
        edges_to_receiver = self.edges_to_receiver(graphs, edge_features)
        node_features = self.node_fn(graphs, edges_to_sender, edges_to_receiver)

        edge_to_global = self.edges_to_global(graphs, edge_features)
        node_to_global = self.nodes_to_global(graphs, node_features)
        global_features = self.global_fn(graphs, node_to_global, edge_to_global)

        return graphs.evolve(
            node_features=node_features,
            edge_features=edge_features,
            global_features=global_features
        )


class GraphLayer(nn.Module):
    def __init__(self, edge_fn=None, node_fn=None, global_fn=None):
        super(GraphLayer, self).__init__()
        self.node_fn = node_fn
        self.edge_fn = edge_fn
        self.global_fn = global_fn

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        if self.edge_fn is not None:
            new_edge_features = self.edge_fn(graphs)
            graphs.evolve(edge_features=new_edge_features)

        if self.node_fn is not None:
            new_node_features = self.node_fn(graphs)
            graphs.evolve(node_features=new_node_features)

        if self.global_fn is not None:
            new_global_features = self.global_fn(graphs)
            graphs.evolve(global_features=new_global_features)

        return graphs
