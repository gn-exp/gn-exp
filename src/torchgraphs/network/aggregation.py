import torch_scatter
import torch.nn as nn

from ..data import GraphBatch
from ..utils import segment_lengths_to_ids


def get_aggregation(name):
    if name in ('add', 'sum'):
        return torch_scatter.scatter_add
    elif name in ('mean', 'avg'):
        return torch_scatter.scatter_mean
    elif name == 'max':
        from functools import wraps

        @wraps(torch_scatter.scatter_max)
        def wrapper(*args, **kwargs):
            return torch_scatter.scatter_max(*args, **kwargs)[0]

        return wrapper


class _BatchAggregator(nn.Module):
    def __init__(self, aggregation):
        super().__init__()
        if isinstance(aggregation, str):
            aggregation = get_aggregation(aggregation)
        self.aggregation = aggregation

    def forward(self, graphs: GraphBatch):
        raise NotImplementedError


class EdgesToSender(_BatchAggregator):
    def forward(self, graphs: GraphBatch):
        # It's necessary to specify the shape of the output dimension, otherwise when max(receivers) != num_nodes
        # the pooling operation would output a minimal tensor with shape (max(receivers), *edge_features_shape)
        # instead of (num_nodes, *edge_features_shape), same would happen for senders
        return self.aggregation(graphs.edge_features, graphs.senders, dim_size=graphs.num_nodes)


class EdgesToReceiver(_BatchAggregator):
    def forward(self, graphs: GraphBatch):
        return self.aggregation(graphs.edge_features, graphs.receivers, dim_size=graphs.num_nodes)


class EdgesToGlobal(_BatchAggregator):
    def forward(self, graphs: GraphBatch):
        index = segment_lengths_to_ids(graphs.num_edges_by_graph)
        return self.aggregation(graphs.edge_features, index, dim_size=graphs.num_graphs)


class NodesToGlobal(_BatchAggregator):
    def forward(self, graphs: GraphBatch):
        index = segment_lengths_to_ids(graphs.num_nodes_by_graph)
        return self.aggregation(graphs.node_features, index, dim_size=graphs.num_graphs)
