import torch

from torchgraphs import Graph, GraphBatch
from torchgraphs.data.features import add_random_features
from data.utils import assert_graphs_equal


def test_collate_tuples(graphs_nx, features_shapes, device):
    graphs_in = [add_random_features(Graph.from_networkx(g), **features_shapes).to(device) for g in graphs_nx]
    graphs_out = list(reversed(graphs_in))
    xs = torch.rand(len(graphs_in), 10, 32)
    ys = torch.rand(len(graphs_in), 7)

    samples = list(zip(graphs_in, xs, ys, graphs_out))
    batch = GraphBatch.collate(samples)

    for g1, g2 in zip(graphs_in, batch[0]):
        assert_graphs_equal(g1, g2)

    torch.testing.assert_allclose(xs, batch[1])

    torch.testing.assert_allclose(ys, batch[2])

    for g1, g2 in zip(graphs_out, batch[3]):
        assert_graphs_equal(g1, g2)


def test_collate_dicts(graphs_nx, features_shapes, device):
    graphs_in = [add_random_features(Graph.from_networkx(g), **features_shapes).to(device) for g in graphs_nx]
    graphs_out = list(reversed(graphs_in))
    xs = torch.rand(len(graphs_in), 10, 32)
    ys = torch.rand(len(graphs_in), 7)

    samples = [{'in': gi, 'x': x, 'y': y, 'out': go} for gi, x, y, go in zip(graphs_in, xs, ys, graphs_out)]
    batch = GraphBatch.collate(samples)

    for g1, g2 in zip(graphs_in, batch['in']):
        assert_graphs_equal(g1, g2)

    torch.testing.assert_allclose(xs, batch['x'])

    torch.testing.assert_allclose(ys, batch['y'])

    for g1, g2 in zip(graphs_out, batch['out']):
        assert_graphs_equal(g1, g2)
