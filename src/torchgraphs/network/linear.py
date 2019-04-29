import math

import torch
import torch.nn as nn

from .aggregation import get_aggregation
from ..data import GraphBatch
from ..utils import repeat_tensor, segment_lengths_to_ids


class EdgeLinear(nn.Module):
    def __init__(self, out_features, edge_features=None, sender_features=None, receiver_features=None,
                 global_features=None, bias=True):
        super(EdgeLinear, self).__init__()
        self.out_features = out_features

        self.W_edge = nn.Parameter(torch.Tensor(out_features, edge_features)) \
            if edge_features is not None else None
        self.W_sender = nn.Parameter(torch.Tensor(out_features, sender_features)) \
            if sender_features is not None else None
        self.W_receiver = nn.Parameter(torch.Tensor(out_features, receiver_features)) \
            if receiver_features is not None else None
        self.W_global = nn.Parameter(torch.Tensor(out_features, global_features)) \
            if global_features is not None else None
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        _reset_parameters(self)

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        new_edges = 0

        if self.W_edge is not None:
            new_edges += graphs.edge_features @ self.W_edge.t()
        if self.W_sender is not None:
            new_edges += torch.index_select(
                graphs.node_features @ self.W_sender.t(), dim=0, index=graphs.senders)
        if self.W_receiver is not None:
            new_edges += torch.index_select(
                graphs.node_features @ self.W_receiver.t(), dim=0, index=graphs.receivers)
        if self.W_global is not None:
            new_edges += repeat_tensor(
                graphs.global_features @ self.W_global.t(), dim=0, repeats=graphs.num_edges_by_graph)
        if self.bias is not None:
            new_edges += self.bias

        return graphs.evolve(edge_features=new_edges)


class NodeLinear(nn.Module):
    def __init__(self, out_features, node_features=None, incoming_features=None, outgoing_features=None,
                 global_features=None, aggregation=None, bias=True):
        super(NodeLinear, self).__init__()
        self.out_features = out_features
        if isinstance(aggregation, str):
            aggregation = get_aggregation(aggregation)
        self.aggregation = aggregation

        self.W_node = nn.Parameter(torch.Tensor(out_features, node_features)) \
            if node_features is not None else None
        self.W_incoming = nn.Parameter(torch.Tensor(out_features, incoming_features)) \
            if incoming_features is not None else None
        self.W_outgoing = nn.Parameter(torch.Tensor(out_features, outgoing_features)) \
            if outgoing_features is not None else None
        self.W_global = nn.Parameter(torch.Tensor(out_features, global_features)) \
            if global_features is not None else None
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        if incoming_features is not None and aggregation is None:
            raise ValueError('An aggregation function is needed to process incoming edges')
        if outgoing_features is not None and aggregation is None:
            raise ValueError('An aggregation function is needed to process outgoing edges')

        _reset_parameters(self)

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        new_nodes = 0

        if self.W_node is not None:
            new_nodes += graphs.node_features @ self.W_node.t()
        if self.W_incoming is not None:
            new_nodes += self.aggregation(
                graphs.edge_features, dim=0, index=graphs.receivers, dim_size=graphs.num_nodes) @ self.W_incoming.t()
        if self.W_outgoing is not None:
            new_nodes += self.aggregation(
                graphs.edge_features, dim=0, index=graphs.senders, dim_size=graphs.num_nodes) @ self.W_outgoing.t()
        if self.W_global is not None:
            new_nodes += repeat_tensor(
                graphs.global_features @ self.W_global.t(), dim=0, repeats=graphs.num_nodes_by_graph)
        if self.bias is not None:
            new_nodes += self.bias

        return graphs.evolve(node_features=new_nodes)


class GlobalLinear(nn.Module):

    def __init__(self, out_features, node_features=None, edge_features=None, global_features=None,
                 aggregation=None, bias=True):
        super(GlobalLinear, self).__init__()
        self.W_node = nn.Parameter(torch.Tensor(out_features, node_features)) \
            if node_features is not None else None
        self.W_edges = nn.Parameter(torch.Tensor(out_features, edge_features)) \
            if edge_features is not None else None
        self.W_global = nn.Parameter(torch.Tensor(out_features, global_features)) \
            if global_features is not None else None
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None

        if isinstance(aggregation, str):
            aggregation = get_aggregation(aggregation)
        self.aggregation = aggregation

        if node_features is not None and aggregation is None:
            raise ValueError('An aggregation function is needed to process node features')

        if edge_features is not None and aggregation is None:
            raise ValueError('An aggregation function is needed to process edge features')

        _reset_parameters(self)

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        new_globals = 0

        if self.W_node is not None:
            index = segment_lengths_to_ids(graphs.num_nodes_by_graph)
            new_globals = new_globals + self.aggregation(
                graphs.node_features, dim=0, index=index, dim_size=graphs.num_graphs) @ self.W_node.t()
        if self.W_edges is not None:
            index = segment_lengths_to_ids(graphs.num_edges_by_graph)
            new_globals = new_globals + self.aggregation(
                graphs.edge_features, dim=0, index=index, dim_size=graphs.num_graphs) @ self.W_edges.t()
        if self.W_global is not None:
            new_globals = new_globals + graphs.global_features @ self.W_global.t()
        if self.bias is not None:
            new_globals += self.bias

        return graphs.evolve(global_features=new_globals)


def _reset_parameters(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            bound = 1 / math.sqrt(param.numel())
            nn.init.uniform_(param, -bound, bound)
        else:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
