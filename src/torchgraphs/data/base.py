import dataclasses
from typing import Optional, Iterator

import torch
import torch_scatter


@dataclasses.dataclass
class _BaseGraph(object):
    num_nodes: int = None
    num_edges: int = None
    node_features: Optional[torch.Tensor] = None
    edge_features: Optional[torch.Tensor] = None
    senders: torch.LongTensor = None
    receivers: torch.LongTensor = None

    _feature_fields = ('node_features', 'edge_features')
    _index_fields = ('senders', 'receivers')

    def __post_init__(self):
        # Try filling in missing info
        if self.num_nodes is None:
            self.num_nodes = len(self.node_features) if self.node_features is not None else 0
        if self.num_edges is None or self.num_edges == 0:
            if self.senders is None:
                self.senders = torch.LongTensor()
            if self.receivers is None:
                self.receivers = torch.LongTensor()
            self.num_edges = len(self.senders)
        self._validate()

    def _validate(self):
        # Check nodes
        if self.num_nodes is None or self.num_nodes < 0:
            raise ValueError(f"`num_nodes` cannot be None or negative, got {self.num_nodes}")
        if self.node_features is not None and len(self.node_features) != self.num_nodes:
            raise ValueError(f"`num_nodes`, `len(node_features)` must match, "
                             f"got {self.num_nodes}, {len(self.node_features)}")

        # Check edges
        if self.num_edges is None or self.num_nodes < 0:
            raise ValueError(f"`num_edges` cannot be None or negative, got {self.num_edges}")
        if self.senders is None or self.receivers is None:
            raise ValueError(f"`senders`, `receivers` cannot be None")
        if not (self.num_edges == len(self.senders) == len(self.receivers)):
            raise ValueError(f"`num_edges`, `len(senders)`, `len(receivers)` must match, "
                             f"got {self.num_edges}, {len(self.senders)}, {len(self.receivers)}")
        if self.edge_features is not None and len(self.edge_features) != self.num_edges:
            raise ValueError(f"`num_edges`, `len(edge_features)` must match, "
                             f"got {self.num_edges}, {len(self.edge_features)}")

        # Check out-of-bounds edge indexes
        send_oob = (self.senders < 0) | (self.senders >= self.num_nodes)
        recv_oob = (self.receivers < 0) | (self.receivers >= self.num_nodes)
        if send_oob.any():
            wrongs = [f'{s.item()} -> {r.item()}' for s, r in zip(self.senders[send_oob], self.receivers[send_oob])]
            raise ValueError(f"Edge sender out of bounds for: {wrongs}")
        if recv_oob.any():
            wrongs = [f'{s.item()} -> {r.item()}' for s, r in zip(self.senders[recv_oob], self.receivers[recv_oob])]
            raise ValueError(f"Edge receiver out of bounds for: {wrongs}")

    @property
    def sender_features(self):
        """For every edge, the features of the sender node.
        
        Examples:
            * Access the sender's features of a single edge

              >>> graph.sender_features[node_index]

            * Iterate over the the sender's features of every edge

              >>> iter(graph.sender_features)

            * Get a tensor of sender features with shape (num_edges, *node_features_shape)

              >>> graph.sender_features()
        """
        return _SenderNodeView(self)

    @property
    def receiver_features(self):
        """For every edge, the features of the receiver node.

        Examples:
            * Access the receiver's features of a single edge

              >>> graph.receiver_features[node_index]

            * Iterate over the the receiver's features of every edge

              >>> iter(graph.receiver_features)

            * Get a tensor of receivers' features with shape (num_edges, *node_features_shape)

              >>> graph.receiver_features()
        """
        return _ReceiverNodeView(self)

    @property
    def out_edge_features(self):
        """For every node, the features of the outgoing edges
        i.e. the features of the edges that have that node as sender

        Examples:
            * Access the features of the outgoing edges of a single node

              >>> graph.out_edge_features[node_index]

            * Iterate node by node over the features of the outgoing edges

              >>> iter(graph.out_edge_features)

            * Get a tensor of aggregated edge features with shape (num_nodes, *edge_features_shape)

              >>> graph.out_edge_features(aggregation='sum')
        """
        return _OutEdgeView(self)

    edge_features_by_sender = out_edge_features

    @property
    def in_edge_features(self):
        """For every node, the features of the incoming edges,
        i.e. the features of the edges that have that node as receiver.

        Examples:
            * Access the features of the incoming edges of a single node

              >>> graph.in_edge_features[node_index]

            * Iterate node by node over the features of the incoming edges

              >>> iter(graph.in_edge_features)

            * Get a tensor of aggregated edge features with shape (num_nodes, *edge_features_shape)

              >>> graph.in_edge_features(aggregation='sum')
        """
        return _InEdgeView(self)

    edge_features_by_receiver = in_edge_features

    @property
    def successor_features(self):
        """For every node, the features of the successor nodes.

        Examples:
            * Access the features of the successor nodes of a single node
              as a tensor of shape (num_successors, *node_features_shape)

              >>> graph.successor_features[node_index]

            * Iterate over the successors of every node

              >>> iter(graph.successor_features)

            * Get a tensor of aggregated successor features with shape (num_nodes, *node_features_shape)

              >>> graph.successor_features(aggregation='sum')
        """
        return _SuccessorView(self)

    @property
    def predecessor_features(self):
        """For every node, the features of the predecessor nodes.

        Examples:
            * Access the features of the predecessor nodes of a single node
              as a tensor of shape (num_predecessors, *node_features_shape)

              >>> graph.predecessor_features[node_index]

            * Iterate over the predecessors of every node

              >>> iter(graph.predecessor_features)

            * Get a tensor of aggregated predecessor features with shape (num_nodes, *node_features_shape)

              >>> graph.predecessor_features(aggregation='sum')
        """
        return _PredecessorView(self)

    def neighbors(self, node_index):
        """The indexes of the nodes that are directly reachable from the node `node_index`.
        """
        return self.receivers[self.senders == node_index]

    def neighbors_features(self, node_index):
        """The features of the nodes that are directly reachable from the node `node_index`.
        """
        return self.node_features.index_select(index=self.neighbors(node_index), dim=0)

    @property
    def degree(self) -> torch.LongTensor:
        """For every node, the number of edges adjacent to that node.

        If an edge is a self connection it is counted twice, both as outgoing and as incoming.
        """
        return self.in_degree + self.out_degree

    @property
    def out_degree(self) -> torch.LongTensor:
        """For every node, the number edges pointing out from that node.

        I.e. the number of edges that have that node as a sender.
        """
        # TODO still buggy
        return self.senders.new_zeros(self.num_nodes).index_add_(
            dim=0, index=self.senders, source=self.senders.new_ones(self.num_edges))

    @property
    def in_degree(self) -> torch.LongTensor:
        """For every node, the number edges pointing in to that node.

        I.e. the number of edges that have that node as a receiver.
        """
        return self.receivers.new_zeros(self.num_nodes).index_add_(
            dim=0, index=self.receivers, source=self.receivers.new_ones(self.num_edges))

    @property
    def node_features_shape(self):
        return self.node_features.shape[1:] if self.node_features is not None else None

    @property
    def edge_features_shape(self):
        return self.edge_features.shape[1:] if self.edge_features is not None else None

    def cpu(self):
        return self.to('cpu')

    def cuda(self, device=None, non_blocking=False):
        if device is None:
            device = torch.cuda.current_device()
        return self.to(device, non_blocking)

    def to(self, device, non_blocking=False):
        feature_fields = {
            field_name: getattr(self, field_name).to(device=device, non_blocking=non_blocking)
            for field_name in self._feature_fields if getattr(self, field_name) is not None
        }
        index_fields = {
            field_name: getattr(self, field_name).to(device=device, non_blocking=non_blocking)
            for field_name in self._index_fields
        }
        return self.evolve(**index_fields, **feature_fields)

    def requires_grad_(self, requires_grad=True):
        for field_name in self._feature_fields:
            if getattr(self, field_name) is not None:
                getattr(self, field_name).requires_grad_(requires_grad)
        return self

    def zero_grad_(self):
        for field_name in self._feature_fields:
            if getattr(self, field_name) is not None:
                getattr(self, field_name).grad = None
        return self

    def evolve(self, **updates):
        return dataclasses.replace(self, **updates)


class _InOutEdgeView(object):
    def __init__(self, graph: _BaseGraph):
        self._graph = graph
        # TODO move these to the class definition or somewhere else
        self._pooling_functions = {
            'mean': lambda src, idx: torch_scatter.scatter_mean(src, idx, dim=0, dim_size=graph.num_nodes),
            'sum': lambda src, idx: torch_scatter.scatter_add(src, idx, dim=0, dim_size=graph.num_nodes),
            'max': lambda src, idx: torch_scatter.scatter_max(src, idx, dim=0, dim_size=graph.num_nodes)[0],
        }

    def __len__(self):
        return self._graph.num_nodes

    def __getitem__(self, node_index) -> torch.Tensor:
        raise NotImplemented

    def __iter__(self) -> Iterator[torch.Tensor]:
        for node_index in range(self._graph.num_nodes):
            yield self[node_index]


class _InEdgeView(_InOutEdgeView):
    def __call__(self, aggregation, *args, **kwargs) -> torch.Tensor:
        if isinstance(aggregation, str):
            aggregation = self._pooling_functions[aggregation]
        return aggregation(self._graph.edge_features, self._graph.receivers, *args, **kwargs)

    def __getitem__(self, node_index) -> torch.Tensor:
        return self._graph.edge_features[self._graph.receivers == node_index]


class _OutEdgeView(_InOutEdgeView):
    def __call__(self, aggregation, *args, **kwargs) -> torch.Tensor:
        if isinstance(aggregation, str):
            aggregation = self._pooling_functions[aggregation]
        return aggregation(self._graph.edge_features, self._graph.senders, *args, **kwargs)

    def __getitem__(self, node_index) -> torch.Tensor:
        return self._graph.edge_features[self._graph.senders == node_index]


class _NodeView(object):
    def __init__(self, graph: _BaseGraph):
        self._graph = graph

    def __len__(self):
        return self._graph.num_edges

    def __getitem__(self, edge_index) -> torch.Tensor:
        raise NotImplemented

    def __iter__(self) -> Iterator[torch.Tensor]:
        for edge_index in range(self._graph.num_edges):
            yield self[edge_index]


class _ReceiverNodeView(_NodeView):
    def __call__(self) -> torch.Tensor:
        return self._graph.node_features.index_select(index=self._graph.receivers, dim=0)

    def __getitem__(self, edge_index) -> torch.Tensor:
        return self._graph.node_features[self._graph.receivers[edge_index]]


class _SenderNodeView(_NodeView):
    def __call__(self) -> torch.Tensor:
        return self._graph.node_features.index_select(index=self._graph.senders, dim=0)

    def __getitem__(self, edge_index) -> torch.Tensor:
        return self._graph.node_features[self._graph.senders[edge_index]]


class _SuccessorPredecessorView(object):
    def __init__(self, graph: _BaseGraph):
        self._graph = graph
        # TODO move these to the class definition or somewhere else
        self._pooling_functions = {
            'mean': lambda src, idx: torch_scatter.scatter_mean(src, idx, dim=0, dim_size=graph.num_nodes),
            'sum': lambda src, idx: torch_scatter.scatter_add(src, idx, dim=0, dim_size=graph.num_nodes),
            'max': lambda src, idx: torch_scatter.scatter_max(src, idx, dim=0, dim_size=graph.num_nodes)[0],
        }

    def __len__(self):
        return self._graph.num_nodes

    def __getitem__(self, edge_index) -> torch.Tensor:
        raise NotImplemented

    def __iter__(self) -> Iterator[torch.Tensor]:
        for node_index in range(self._graph.num_nodes):
            yield self[node_index]


class _SuccessorView(_SuccessorPredecessorView):
    def __call__(self, aggregation, *args, **kwargs) -> torch.Tensor:
        # For every edge get the features of the receiving node
        successors = self._graph.node_features.index_select(index=self._graph.receivers, dim=0)
        # Aggregate the features of the receiving nodes according to the sender
        fn = self._pooling_functions.get(aggregation, aggregation)
        return fn(successors, self._graph.senders)

    def __getitem__(self, node_index) -> torch.Tensor:
        successors = self._graph.receivers[self._graph.senders == node_index]
        return self._graph.node_features.index_select(index=successors, dim=0)


class _PredecessorView(_SuccessorPredecessorView):
    def __call__(self, aggregation, *args, **kwargs) -> torch.Tensor:
        # For every edge get the features of the sender node
        predecessors = self._graph.node_features.index_select(index=self._graph.senders, dim=0)
        # Aggregate the features of the sender nodes according to the receiver
        fn = self._pooling_functions.get(aggregation, aggregation)
        return fn(predecessors, self._graph.receivers)

    def __getitem__(self, node_index) -> torch.Tensor:
        predecessors = self._graph.receivers[self._graph.senders == node_index]
        return self._graph.node_features.index_select(index=predecessors, dim=0)
