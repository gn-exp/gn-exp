from torchgraphs import Graph
from torchgraphs.data.features import add_random_features

from data.utils import assert_graphs_equal


def test_from_networkx(graph_nx, features_shapes):
    graph_nx = add_random_features(graph_nx, **features_shapes)
    graph = Graph.from_networkx(graph_nx)
    assert_graphs_equal(graph_nx, graph)


def test_to_networkx(graph, features_shapes):
    graph = add_random_features(graph, **features_shapes)
    graph_nx = graph.to_networkx()
    assert_graphs_equal(graph, graph_nx)


def test_device(graph, features_shapes, device):
    graph = add_random_features(graph, **features_shapes)
    other_graph = graph.to(device)

    for k in other_graph._feature_fields:
        assert (getattr(other_graph, k) is None) or (getattr(other_graph, k).device == device)

    assert_graphs_equal(graph, other_graph.cpu())
