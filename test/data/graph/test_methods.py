import pytest

import networkx as nx

from torchgraphs import Graph
from torchgraphs.data.features import add_dummy_features
from graphs_for_test import graphs_for_test


@pytest.fixture(params=[g for g in graphs_for_test().values() if g.number_of_edges() > 0],
                ids=[n for n, g in graphs_for_test().items() if g.number_of_edges() > 0])
def graph_nx(request) -> nx.Graph:
    return request.param


def test_graph_properties(graph_nx):
    graph_nx = add_dummy_features(graph_nx)
    graph = Graph.from_networkx(graph_nx)

    assert list(graph.degree) == [d for _, d in graph_nx.degree]
    assert list(graph.in_degree) == [d for _, d in graph_nx.in_degree]
    assert list(graph.out_degree) == [d for _, d in graph_nx.out_degree]


def test_edge_functions(graph_nx):
    graph_nx = add_dummy_features(graph_nx)
    graph = Graph.from_networkx(graph_nx)

    # Edge features
    # By edge index
    for edge_index in range(graph.num_edges):
        assert graph.edge_features[edge_index].shape == graph.edge_features_shape
    # Iterator
    for edge_features in iter(graph.edge_features):
        assert edge_features.shape == graph.edge_features_shape
    # As tensor
    assert graph.edge_features.shape == (graph.num_edges, *graph.edge_features_shape)

    # Features of the sender nodes
    # By edge index
    for edge_index in range(graph.num_edges):
        assert graph.sender_features[edge_index].shape == graph.node_features_shape
    # Iterator
    for edge_features in graph.sender_features:
        assert edge_features.shape == graph.node_features_shape
    # As tensor
    assert graph.sender_features().shape == (graph.num_edges, *graph.node_features_shape)

    # Features of the receiver nodes
    # By edge index
    for edge_index in range(graph.num_edges):
        assert graph.receiver_features[edge_index].shape == graph.node_features_shape
    # Iterator
    for edge_features in graph.receiver_features:
        assert edge_features.shape == graph.node_features_shape
    # As tensor
    assert graph.receiver_features().shape == (graph.num_edges, *graph.node_features_shape)


def test_node_functions(graph_nx):
    graph_nx = add_dummy_features(graph_nx)
    graph = Graph.from_networkx(graph_nx)

    # Features of the outgoing edges
    # By node index
    for node_index in range(graph.num_nodes):
        assert graph.out_edge_features[node_index].shape[1:] == graph.edge_features_shape
    # Iterator
    for out_edges in iter(graph.out_edge_features):
        assert out_edges.shape[1:] == graph.edge_features_shape
    # As tensor
    assert graph.out_edge_features(aggregation='sum').shape == (graph.num_nodes, *graph.edge_features_shape)

    # Features of the incoming edges
    # By node index
    for node_index in range(graph.num_nodes):
        assert graph.in_edge_features[node_index].shape[1:] == graph.edge_features_shape
    # Iterator
    for in_edges in iter(graph.in_edge_features):
        assert in_edges.shape[1:] == graph.edge_features_shape
    # As tensor
    assert graph.in_edge_features(aggregation='sum').shape == (graph.num_nodes, *graph.edge_features_shape)

    # Features of the successor nodes
    # By node index
    for node_index in range(graph.num_nodes):
        assert graph.successor_features[node_index].shape[1:] == graph.node_features_shape
    # Iterator
    for in_edges in iter(graph.successor_features):
        assert in_edges.shape[1:] == graph.node_features_shape
    # As tensor
    assert graph.successor_features(aggregation='sum').shape == (graph.num_nodes, *graph.node_features_shape)

    # Features of the predecessor nodes
    # By node index
    for node_index in range(graph.num_nodes):
        assert graph.predecessor_features[node_index].shape[1:] == graph.node_features_shape
    # Iterator
    for in_edges in iter(graph.predecessor_features):
        assert in_edges.shape[1:] == graph.node_features_shape
    # As tensor
    assert graph.predecessor_features(aggregation='sum').shape == (graph.num_nodes, *graph.node_features_shape)


def test_global_functions(graph_nx):
    graph_nx = add_dummy_features(graph_nx)
    graph = Graph.from_networkx(graph_nx)

    assert graph.global_features.shape == graph.global_features_shape
    assert graph.global_features_as_nodes.shape == (graph.num_nodes, *graph.global_features_shape)
    assert graph.global_features_as_edges.shape == (graph.num_edges, *graph.global_features_shape)
