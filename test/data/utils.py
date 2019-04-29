from typing import Union

import torch
import networkx as nx

import torchgraphs as tg


def assert_graphs_equal(graph1: Union[tg.Graph, nx.Graph], graph2: Union[tg.Graph, nx.Graph]):
    if isinstance(graph1, tg.Graph) and isinstance(graph2, tg.Graph):
        _assert_graphs_equals(graph1, graph2)
    elif isinstance(graph1, nx.Graph) and isinstance(graph2, nx.Graph):
        _assert_graphs_nx_equal(graph1, graph2)
    elif isinstance(graph1, nx.Graph) and isinstance(graph2, tg.Graph):
        _assert_graph_and_graph_nx_equals(graph_nx=graph1, graph=graph2)
    elif isinstance(graph1, tg.Graph) and isinstance(graph2, nx.Graph):
        _assert_graph_and_graph_nx_equals(graph_nx=graph2, graph=graph1)


def has_node_features(graph: Union[tg.Graph, nx.Graph]):
    if isinstance(graph, tg.Graph):
        return graph.node_features is not None and graph.node_features.shape[0] != 0
    if isinstance(graph, nx.Graph):
        return graph.number_of_nodes() > 0 and graph.nodes(data='features')[0] is not None
    raise ValueError(f'Wrong type: {type(graph)}')


def has_edge_features(graph: Union[tg.Graph, nx.Graph]):
    if isinstance(graph, tg.Graph):
        return graph.edge_features is not None and graph.edge_features.shape[0] != 0
    if isinstance(graph, nx.Graph):
        return graph.number_of_edges() > 0 and list(graph.edges(data='features'))[0][-1] is not None
    raise ValueError(f'Wrong type: {type(graph)}')


def has_global_features(graph: Union[tg.Graph, nx.Graph]):
    if isinstance(graph, tg.Graph):
        return graph.global_features is not None
    if isinstance(graph, nx.Graph):
        return 'features' in graph.graph
    raise ValueError(f'Wrong type: {type(graph)}')


def _assert_graphs_nx_equal(g1: nx.Graph, g2: nx.Graph):
    # Check number of nodes and edges
    assert g1.number_of_nodes() == g2.number_of_nodes()
    assert g1.number_of_edges() == g2.number_of_edges()

    # Check node features
    for (node_id_1, node_features_1), (node_id_2, node_features_2) in \
            zip(g1.nodes(data='features'), g2.nodes(data='features')):
        assert node_id_1 == node_id_2
        assert (node_features_1 is not None) == (node_features_2 is not None)
        if node_features_1 is not None and node_features_2 is not None:
            torch.testing.assert_allclose(node_features_1, node_features_2)

    # Check edge features
    for (sender_id_1, receiver_id_1, edge_features_1), (sender_id_2, receiver_id_2, edge_features_2) in \
            zip(g1.edges(data='features'), g2.edges(data='features')):
        assert sender_id_1 == sender_id_2
        assert receiver_id_1 == receiver_id_2
        assert (edge_features_1 is not None) == (edge_features_2 is not None)
        if edge_features_1 is not None and edge_features_2 is not None:
            torch.testing.assert_allclose(edge_features_1, edge_features_2)

    # Check graph features
    assert has_global_features(g1) == has_global_features(g2)
    if has_global_features(g1) and has_global_features(g2):
        torch.testing.assert_allclose(g1.graph['features'], g2.graph['features'])


def _assert_graphs_equals(g1: tg.Graph, g2: tg.Graph):
    # Check number of nodes and edges
    assert g1.num_nodes == g2.num_nodes
    assert g1.num_edges == g2.num_edges

    # Check node features
    assert has_node_features(g1) == has_node_features(g2)
    if has_node_features(g1) and has_node_features(g2):
        torch.testing.assert_allclose(g1.node_features, g2.node_features)

    # Check edge indexes
    for sender_id_1, receiver_id_1, sender_id_2, receiver_id_2 in \
            zip(g1.senders, g1.receivers, g2.senders, g2.receivers):
        assert sender_id_1 == sender_id_2
        assert receiver_id_1 == receiver_id_2

    # Check edge features
    assert has_edge_features(g1) == has_edge_features(g2)
    if has_edge_features(g1) and has_edge_features(g2):
        torch.testing.assert_allclose(g1.edge_features, g2.edge_features)

    # Check graph features
    assert has_global_features(g1) == has_global_features(g2)
    if has_global_features(g1) and has_global_features(g2):
        torch.testing.assert_allclose(g1.global_features, g2.global_features)


def _assert_graph_and_graph_nx_equals(graph: tg.Graph, graph_nx: nx.Graph):
    # Check number of nodes and edges
    assert graph_nx.number_of_nodes() == graph.num_nodes
    assert graph_nx.number_of_edges() == graph.num_edges

    # Check node features
    assert has_node_features(graph) == has_node_features(graph_nx)
    if has_node_features(graph) and has_node_features(graph_nx):
        for node_features_nx, node_features in zip(graph_nx.nodes(data='features'), graph.node_features):
            torch.testing.assert_allclose(node_features_nx[1], node_features)

    # Check edge indexes
    for (sender_id_nx, receiver_id_nx, *_), sender_id, receiver_id in \
            zip(graph_nx.edges, graph.senders, graph.receivers):
        assert sender_id_nx == sender_id
        assert receiver_id_nx == receiver_id

    assert has_edge_features(graph) == has_edge_features(graph_nx)
    if has_edge_features(graph) and has_edge_features(graph_nx):
        for (*_, edge_features_nx), edge_features in zip(graph_nx.edges(data='features'), graph.edge_features):
            torch.testing.assert_allclose(edge_features_nx, edge_features)

    # Check graph features
    assert has_global_features(graph) == has_global_features(graph_nx)
    if has_global_features(graph) and has_global_features(graph_nx):
        torch.testing.assert_allclose(graph_nx.graph['features'], graph.global_features)
