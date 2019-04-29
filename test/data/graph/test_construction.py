import pytest
import torch

from torchgraphs import Graph

def test_empty():
    graph = Graph()
    validate_graph(graph)

    assert graph.num_nodes == 0
    assert graph.node_features is None
    assert graph.node_features_shape is None

    assert graph.num_edges == len(graph.senders) == len(graph.receivers) == 0
    assert graph.edge_features is None
    assert graph.edge_features_shape is None

    assert graph.global_features is None
    assert graph.global_features_shape is None


def test_nodes():
    graph = Graph(num_nodes=0)
    validate_graph(graph)
    assert graph.num_nodes == 0

    graph = Graph(num_nodes=10)
    validate_graph(graph)
    assert graph.num_nodes == 10

    graph = Graph(node_features=torch.rand(15, 2))
    validate_graph(graph)
    assert graph.num_nodes == 15
    assert graph.node_features_shape == (2,)

    with pytest.raises(ValueError):
        Graph(num_nodes=-1)

    with pytest.raises(ValueError):
        Graph(num_nodes=0, node_features=torch.rand(15, 2))


def test_edges():
    graph = Graph(num_nodes=6, senders=torch.tensor([0, 1, 2, 5, 5]), receivers=torch.tensor([3, 4, 5, 5, 5]))
    validate_graph(graph)
    assert graph.num_edges == len(graph.senders) == len(graph.receivers) == 5

    graph = Graph(num_nodes=6, edge_features=torch.rand(5, 2),
                  senders=torch.tensor([0, 1, 2, 5, 5]), receivers=torch.tensor([3, 4, 5, 5, 5]))
    validate_graph(graph)
    assert graph.num_edges == len(graph.senders) == len(graph.receivers) == len(graph.edge_features) == 5
    assert graph.edge_features_shape == (2,)

    # Negative number of edges
    with pytest.raises(ValueError):
        Graph(num_edges=-1)

    # Senders and receivers not given
    with pytest.raises(ValueError):
        Graph(num_edges=3)

    # Senders not given
    with pytest.raises(ValueError):
        Graph(num_edges=3, receivers=torch.arange(10))

    # Receivers not given
    with pytest.raises(ValueError):
        Graph(num_edges=3, senders=torch.arange(10))

    # Senders and receivers given, but not matching number of edges
    with pytest.raises(ValueError):
        Graph(num_edges=3, senders=torch.arange(10), receivers=torch.arange(10))

    # Edges on a graph with no nodes
    with pytest.raises(ValueError):
        Graph(senders=torch.tensor([0, 1, 2]), receivers=torch.tensor([3, 4, 5]))

    # Different number of senders and receivers
    with pytest.raises(ValueError):
        Graph(num_nodes=6, senders=torch.tensor([0]), receivers=torch.tensor([3, 4, 5]))

    # Indexes out-of-bounds
    with pytest.raises(ValueError):
        Graph(num_nodes=6, senders=torch.tensor([0, 1, 1000]), receivers=torch.tensor([3, 4, 5]))

    # Indexes out-of-bounds
    with pytest.raises(ValueError):
        Graph(num_nodes=6, senders=torch.tensor([0, 1, 2]), receivers=torch.tensor([3, 4, 1000]))

    # Indexes out-of-bounds
    with pytest.raises(ValueError):
        Graph(num_nodes=6, senders=torch.tensor([-1000, 1, 2]), receivers=torch.tensor([3, 4, 5]))

    # Indexes out-of-bounds
    with pytest.raises(ValueError):
        Graph(num_nodes=6, senders=torch.tensor([0, 1, 2]), receivers=torch.tensor([-1000, 4, 5]))

    # Senders, receivers and number of edges given, but not matching features
    with pytest.raises(ValueError):
        Graph(num_nodes=6, senders=torch.tensor([0, 1]), receivers=torch.tensor([3, 4]), edge_features=torch.rand(9, 2))


def test_globals():
    graph = Graph(global_features=torch.rand(3))
    validate_graph(graph)

    graph = Graph(node_features=torch.rand(6, 2), edge_features=torch.rand(5, 2), global_features=torch.rand(3),
                  senders=torch.tensor([0, 0, 1, 1, 2]), receivers=torch.tensor([0, 0, 3, 4, 5]))
    validate_graph(graph)


def validate_graph(graph: Graph):
    assert graph.num_nodes >= 0
    assert graph.num_edges >= 0
    assert graph.node_features is None or graph.num_nodes == len(graph.node_features)

    assert graph.num_edges == len(graph.senders) == len(graph.receivers)
    assert (graph.senders < graph.num_nodes).all() and (graph.senders >= 0).all()
    assert (graph.receivers < graph.num_nodes).all() and (graph.receivers >= 0).all()
    assert graph.edge_features is None or graph.num_edges == len(graph.edge_features)

    assert graph.global_features is None or graph.global_features.shape == graph.global_features_shape
