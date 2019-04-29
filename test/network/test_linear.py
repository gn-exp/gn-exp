from collections import OrderedDict

import torch

from torchgraphs import GraphBatch
from torchgraphs.network import NodeLinear, EdgeLinear, GlobalLinear, EdgeReLU, NodeReLU, GlobalReLU

from features_shapes import linear_features
from torchgraphs.data.features import add_random_features


def test_linear_graph_network(graphbatch: GraphBatch, device):
    graphbatch = add_random_features(graphbatch, **linear_features).to(device)

    node_linear = NodeLinear(
        out_features=linear_features['node_features_shape'],
        incoming_features=linear_features['edge_features_shape'],
        node_features=linear_features['node_features_shape'],
        global_features=linear_features['global_features_shape'],
        aggregation='mean'
    )
    edge_linear = EdgeLinear(
        out_features=linear_features['edge_features_shape'],
        edge_features=linear_features['edge_features_shape'],
        sender_features=linear_features['node_features_shape'],
        receiver_features=linear_features['node_features_shape'],
        global_features=linear_features['global_features_shape']
    )
    global_linear = GlobalLinear(
        out_features=linear_features['global_features_shape'],
        edge_features=linear_features['edge_features_shape'],
        node_features=linear_features['node_features_shape'],
        global_features=linear_features['global_features_shape'],
        aggregation='mean'
    )

    net = torch.nn.Sequential(OrderedDict([
        ('edge', edge_linear),
        ('edge_relu', EdgeReLU()),
        ('node', node_linear),
        ('node_relu', NodeReLU()),
        ('global', global_linear),
        ('global_relu', GlobalReLU()),
        
    ]))
    net.to(device)

    result = net.forward(graphbatch)

    assert graphbatch.num_graphs == result.num_graphs
    assert graphbatch.num_nodes == result.num_nodes
    assert graphbatch.num_edges == result.num_edges
    assert (graphbatch.num_nodes_by_graph == result.num_nodes_by_graph).all()
    assert (graphbatch.num_edges_by_graph == result.num_edges_by_graph).all()
    assert (graphbatch.senders == result.senders).all()
    assert (graphbatch.receivers == result.receivers).all()
