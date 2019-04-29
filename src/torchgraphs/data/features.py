from typing import overload

import torch
import networkx as nx

from .graph import Graph
from .graphbatch import GraphBatch


@overload
def add_random_features(graph: nx.Graph, *, node_features_shape=None, edge_features_shape=None,
                        global_features_shape=None) -> nx.Graph: ...


@overload
def add_random_features(graph: Graph, *, node_features_shape=None, edge_features_shape=None,
                        global_features_shape=None) -> Graph: ...


@overload
def add_random_features(graph: GraphBatch, *, node_features_shape=None, edge_features_shape=None,
                        global_features_shape=None) -> GraphBatch: ...


def add_random_features(graph, *, node_features_shape=None, edge_features_shape=None, global_features_shape=None):
    if isinstance(node_features_shape, int):
        node_features_shape = (node_features_shape,)
    if isinstance(edge_features_shape, int):
        edge_features_shape = (edge_features_shape,)
    if isinstance(global_features_shape, int):
        global_features_shape = (global_features_shape,)

    if isinstance(graph, nx.Graph):
        if node_features_shape is not None:
            for node, data in graph.nodes(data=True):
                data['features'] = torch.rand(*node_features_shape)
        if edge_features_shape is not None:
            for start, end, data in graph.edges(data=True):
                data['features'] = torch.rand(*edge_features_shape)
        if global_features_shape is not None:
            graph.graph['features'] = torch.rand(*global_features_shape)
        return graph
    elif isinstance(graph, Graph):
        return graph.evolve(
            node_features=None if node_features_shape is None else torch.rand(graph.num_nodes, *node_features_shape),
            edge_features=None if edge_features_shape is None else torch.rand(graph.num_edges, *edge_features_shape),
            global_features=None if global_features_shape is None else torch.rand(*global_features_shape)
        )
    elif isinstance(graph, GraphBatch):
        return graph.evolve(
            node_features=None if node_features_shape is None else torch.rand(graph.num_nodes, *node_features_shape),
            edge_features=None if edge_features_shape is None else torch.rand(graph.num_edges, *edge_features_shape),
            global_features=None if global_features_shape is None else torch.rand(
                graph.num_graphs, *global_features_shape)
        )

    raise ValueError(
        f'`graph` must be instance of `networkx.Graph`, `torchgraphs.data.Graph` or '
        f'`torchgraphs.data.GraphBatch`, found {type(graph)}')


@overload
def add_dummy_features(graph: nx.Graph) -> nx.Graph: ...


@overload
def add_dummy_features(graph: Graph) -> Graph: ...


def add_dummy_features(graph):
    if isinstance(graph, nx.Graph):
        for node, data in graph.nodes(data=True):
            data['features'] = torch.empty(3).fill_(node)
        for start, end, data in graph.edges(data=True):
            data['features'] = torch.tensor([start, start, end, end]).float()
        graph.graph['features'] = torch.ones(5)
        return graph
    elif isinstance(graph, Graph):
        return graph.evolve(
            node_features=torch.arange(graph.num_nodes).expand(3, -1).t().float(),
            edge_features=torch.tensor([[s, s, r, r] for s, r in zip(graph.senders, graph.receivers)]).float(),
            global_features=torch.ones(5)
        )
    raise ValueError(f'`graph` must be instance of `networkx.Graph` or `torchgraphs.data.Graph`, found {type(graph)}')
