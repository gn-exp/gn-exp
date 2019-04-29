import pytest
import torch

from torchgraphs import Graph, GraphBatch
from torchgraphs.data.features import add_random_features
from data.utils import assert_graphs_equal


def test_from_to_networkxs(graphs_nx, features_shapes, device):
    graphs_nx = [add_random_features(g, **features_shapes) for g in graphs_nx]
    graphbatch = GraphBatch.from_networkxs(graphs_nx).to(device)

    validate_batch(graphbatch)

    assert len(graphs_nx) == len(graphbatch) == graphbatch.num_graphs
    assert [g.number_of_nodes() for g in graphs_nx] == graphbatch.num_nodes_by_graph.tolist()
    assert [g.number_of_edges() for g in graphs_nx] == graphbatch.num_edges_by_graph.tolist()

    # Test sequential access (__iter__)
    for g_nx, g in zip(graphs_nx, graphbatch):
        assert_graphs_equal(g_nx, g.cpu())

    # Test random access (__getitem__)
    for i in range(len(graphbatch)):
        assert_graphs_equal(graphs_nx[i], graphbatch[i].cpu())

    # Test back conversion
    graphs_nx_back = graphbatch.cpu().to_networkxs()
    for g1, g2 in zip(graphs_nx, graphs_nx_back):
        assert_graphs_equal(g1, g2)


def test_corner_cases(features_shapes, device):
    # Only some graphs have node/edge features, global features are either present on all of them or absent from all
    gfs = features_shapes['global_features_shape']
    graphs = [
        add_random_features(Graph(num_nodes=0, num_edges=0), global_features_shape=gfs),
        add_random_features(Graph(num_nodes=0, num_edges=0), global_features_shape=gfs),
        add_random_features(Graph(num_nodes=3, num_edges=0), **features_shapes),
        add_random_features(Graph(num_nodes=0, num_edges=0), **features_shapes),
        add_random_features(
            Graph(num_nodes=2, senders=torch.tensor([0, 1]), receivers=torch.tensor([1, 0])), **features_shapes)
    ]
    graphbatch = GraphBatch.from_graphs(graphs).to(device)
    validate_batch(graphbatch)

    for g_orig, g_batch in zip(graphs, graphbatch):
        assert_graphs_equal(g_orig, g_batch.cpu())

    # Global features should be either present on all graphs or absent from all graphs
    with pytest.raises(ValueError):
        GraphBatch.from_graphs([
            Graph(num_nodes=0, num_edges=0),
            add_random_features(Graph(num_nodes=0, num_edges=0), global_features_shape=10)
        ])
    with pytest.raises(ValueError):
        GraphBatch.from_graphs([
            add_random_features(Graph(num_nodes=0, num_edges=0), global_features_shape=10),
            Graph(num_nodes=0, num_edges=0)
        ])


def test_from_graphs(graphs, features_shapes, device):
    graphs = [add_random_features(g, **features_shapes).to(device) for g in graphs]
    graphbatch = GraphBatch.from_graphs(graphs)

    validate_batch(graphbatch)

    assert len(graphs) == len(graphbatch) == graphbatch.num_graphs
    assert [g.num_nodes for g in graphs] == graphbatch.num_nodes_by_graph.tolist()
    assert [g.num_edges for g in graphs] == graphbatch.num_edges_by_graph.tolist()

    # Test sequential access (__iter__)
    for g, gb in zip(graphs, graphbatch):
        assert_graphs_equal(g, gb)

    # Test random access (__getitem__)
    for i in range(len(graphbatch)):
        assert_graphs_equal(graphs[i], graphbatch[i])


def validate_batch(graphbatch):
    assert len(graphbatch) == graphbatch.num_graphs
    assert (graphbatch.senders < graphbatch.num_nodes).all()
    assert (graphbatch.receivers < graphbatch.num_nodes).all()
    assert (graphbatch.degree == graphbatch.in_degree + graphbatch.out_degree).all()
