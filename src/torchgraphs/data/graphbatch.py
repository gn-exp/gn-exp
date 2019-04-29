from __future__ import annotations

import dataclasses
import collections.abc
from typing import Iterator, Sequence, Iterable, Optional, Tuple

import networkx as nx
import torch
import torch_scatter
from torch.utils.data.dataloader import default_collate

from .base import _BaseGraph
from .graph import Graph
from ..utils import segment_lengths_to_slices, segment_lengths_to_ids, repeat_tensor


@dataclasses.dataclass
class GraphBatch(_BaseGraph):
    num_graphs: int = None
    global_features: Optional[torch.Tensor] = None
    num_nodes_by_graph: torch.LongTensor = None
    num_edges_by_graph: torch.LongTensor = None

    _feature_fields = _BaseGraph._feature_fields + ('global_features',)
    _index_fields = _BaseGraph._index_fields + ('num_nodes_by_graph', 'num_edges_by_graph')

    def __post_init__(self):
        # super().__post_init__() will also validate the instance using the _validate methods,
        # so we first fill in missing values that are will be used in self._validate()
        if self.num_graphs is None:
            if self.num_nodes_by_graph is not None:
                self.num_graphs = len(self.num_nodes_by_graph)
            elif self.num_nodes_by_graph is not None:
                self.num_graphs = len(self.num_nodes_by_graph)
            elif self.global_features is not None:
                self.num_graphs = len(self.global_features)
            else:
                raise ValueError('Could not infer number of graphs from batch fields')

        if self.num_nodes_by_graph is None and self.num_nodes == 0:
            self.num_nodes_by_graph = torch.zeros(self.num_graphs, dtype=torch.long)
        if self.num_edges_by_graph is None and self.num_edges == 0:
            self.num_edges_by_graph = torch.zeros(self.num_graphs, dtype=torch.long)
        super(GraphBatch, self).__post_init__()

    def _validate(self):
        super(GraphBatch, self)._validate()

        if self.global_features is not None and self.num_graphs != len(self.global_features):
            raise ValueError(f'Total number of graphs and length of global features must correspond: '
                             f'`num_graphs`={self.num_graphs} '
                             f'`len(self.global_features)`={len(self.global_features)}')
        if self.num_graphs != len(self.num_nodes_by_graph):
            raise ValueError(f'Total number of graphs and length of nodes by graph must correspond: '
                             f'`num_graphs`={self.num_graphs} '
                             f'`len(self.num_nodes_by_graph)`={len(self.num_nodes_by_graph)}')
        if self.num_graphs != len(self.num_edges_by_graph):
            raise ValueError(f'Total number of graphs and length of edges by graph must correspond: '
                             f'`num_graphs`={self.num_graphs} '
                             f'`len(self.num_edges_by_graph)`={len(self.num_edges_by_graph)}')

        if self.num_nodes != self.num_nodes_by_graph.sum():
            raise ValueError(f'Total number of nodes and number of nodes by graph must correspond: '
                             f'`num_nodes`={self.num_nodes} '
                             f'`sum(self.num_nodes_by_graph)`={self.num_nodes_by_graph.sum().item()}')
        if self.num_edges != self.num_edges_by_graph.sum():
            raise ValueError(f'Total number of edges and number of edges by graph must correspond: '
                             f'`num_edges`={self.num_edges} '
                             f'`sum(self.num_edges_by_graph)`={self.num_edges_by_graph.sum().item()}')

    def __len__(self):
        return self.num_graphs

    @property
    def node_features_by_graph(self):
        """For every graph in the batch, the features of their nodes

        Examples:

            * Access the node features of a single graph

              >>> batch.node_features_by_graph[graph_index]

            * Iterate over the node features of every graph in the batch

              >>> iter(batch.node_features_by_graph)

            * Get a tuple of tensors containing the node features of every graph

              >>> batch.node_features_by_graph.astuple()

            * Get a tensor of aggregated node features with shape (num_graphs, *node_features_shape)

              >>> batch.node_features_by_graph(aggregation='sum')
        """
        return _BatchNodeView(self)

    @property
    def edge_features_by_graph(self):
        """For every graph in the batch, the features of their edges

        Examples:

            * Access the edge features of a single graph

              >>> batch.edge_features_by_graph[graph_index]

            * Iterate over the edge features of every graph in the batch

              >>> iter(batch.edge_features_by_graph)

            * Get a tuple of tensors containing the edge features of every graph

              >>> batch.edge_features_by_graph.astuple()

            * Get a tensor of aggregated edge features with shape (num_graphs, *edge_features_shape)

              >>> batch.edge_features_by_graph(aggregation='sum')
        """
        return _BatchEdgeView(self)

    @property
    def global_features_shape(self):
        return self.global_features.shape[1:] if self.global_features is not None else None

    def global_features_as_edges(self) -> torch.Tensor:
        """Broadcast `global_features` along the the first dimension to match `edge_features`,
        respecting the edge-to-graph assignment

        Returns:
            a tensor of shape `(num_edges, *global_features_shape)`
        """
        return repeat_tensor(self.global_features, self.num_edges_by_graph)

    def global_features_as_nodes(self) -> torch.Tensor:
        """Broadcast `global_features` along the the first dimension to match `node_features`,
        respecting the node-to-graph assignment

        Returns:
            a tensor of shape `(num_nodes, *global_features_shape)`
        """
        return repeat_tensor(self.global_features, self.num_nodes_by_graph)

    def __getitem__(self, graph_index):
        """Use for random access, as in `batch[i]`. For sequential access use `iter(batch)` or `for g in batch`
        """
        node_offset = self.num_nodes_by_graph[:graph_index].sum()
        edge_offset = self.num_edges_by_graph[:graph_index].sum()
        n_nodes = self.num_nodes_by_graph[graph_index]
        n_edges = self.num_edges_by_graph[graph_index]
        return Graph(
            num_nodes=n_nodes.item(),
            num_edges=n_edges.item(),
            node_features=None if self.node_features is None else self.node_features[node_offset:node_offset + n_nodes],
            edge_features=None if self.edge_features is None else self.edge_features[edge_offset:edge_offset + n_edges],
            global_features=self.global_features[graph_index] if self.global_features is not None else None,
            senders=self.senders[edge_offset:edge_offset + n_edges] - node_offset,
            receivers=self.receivers[edge_offset:edge_offset + n_edges] - node_offset
        )

    def __iter__(self):
        """Use for sequential access, as in `iter(batch)` or `for g in batch`. For random access use `batch[i].`
        """
        node_slices = segment_lengths_to_slices(self.num_nodes_by_graph)
        edge_slices = segment_lengths_to_slices(self.num_edges_by_graph)
        for graph_index, node_slice, edge_slice in zip(range(self.num_graphs), node_slices, edge_slices):
            yield Graph(
                num_nodes=self.num_nodes_by_graph[graph_index].item(),
                num_edges=self.num_edges_by_graph[graph_index].item(),
                node_features=self.node_features[node_slice] if self.node_features is not None else None,
                edge_features=self.edge_features[edge_slice] if self.edge_features is not None else None,
                global_features=self.global_features[graph_index] if self.global_features is not None else None,
                senders=self.senders[edge_slice] - node_slice.start,
                receivers=self.receivers[edge_slice] - node_slice.start
            )
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"#{self.num_graphs}, "
                f"n={self.num_nodes_by_graph}, "
                f"e={self.num_edges_by_graph}, "
                f"n_shape={self.node_features_shape}, "
                f"e_shape={self.edge_features_shape}, "
                f"g_shape={self.global_features_shape})")

    def to_networkxs(self):
        return [g.to_networkx() for g in self]

    def to_graphs(self):
        return list(self)

    @classmethod
    def from_graphs(cls, graphs: Sequence[Graph]) -> GraphBatch:
        """Merges multiple graphs in a batch. All node, edge and graph features must have the same shape if present.

        If some graph of the sequence have values for `node_features`, `edge_features`, but some of the others
        don't (maybe they were created with `num_nodes = 0` or `num_edges = 0` and None as node/edge features),
        this method will still try to correctly batch the graphs together. It is however advised to replace the None
        values on those graphs with empty tensors of shape `(0, *node_features_shape)` and `(0, *edge_features_shape)`.
          
        The field `global_features` is instead required to be either present on all graphs or absent from all graphs.
        """
        # TODO if the graphs in `graphs` require grad the resulting batch should require grad too
        if len(graphs) == 0:
            raise ValueError('Graphs list can not be empty')

        node_features = []
        edge_features = []
        global_features = []
        num_nodes_by_graph = []
        num_edges_by_graph = []
        senders = []
        receivers = []
        node_offset = 0
        for i, g in enumerate(graphs):
            if g.node_features is not None:
                node_features.append(g.node_features)
            if g.edge_features is not None:
                edge_features.append(g.edge_features)
            if g.global_features is not None:
                global_features.append(g.global_features)
            num_nodes_by_graph.append(g.num_nodes)
            num_edges_by_graph.append(g.num_edges)
            senders.append(g.senders + node_offset)
            receivers.append(g.receivers + node_offset)
            node_offset += g.num_nodes

        from torch.utils.data.dataloader import _use_shared_memory

        if len(node_features) > 0:
            out = None
            if _use_shared_memory:
                numel = sum([x.numel() for x in node_features])
                storage = node_features[0].storage()._new_shared(numel)
                out = node_features[0].new(storage)
            node_features = torch.cat(node_features, out=out)
        else:
            node_features = None

        if len(edge_features) > 0:
            out = None
            if _use_shared_memory:
                numel = sum([x.numel() for x in edge_features])
                storage = edge_features[0].storage()._new_shared(numel)
                out = edge_features[0].new(storage)
            edge_features = torch.cat(edge_features, out=out)
        else:
            edge_features = None

        if len(global_features) == len(graphs):
            out = None
            if _use_shared_memory:
                numel = sum([x.numel() for x in global_features])
                storage = global_features[0].storage()._new_shared(numel)
                out = global_features[0].new(storage)
            global_features = torch.stack(global_features, out=out)
        elif len(global_features) == 0:
            global_features = None
        else:
            raise ValueError('The field `global_features` must either be None on all graphs or present on all graphs')

        out = None
        if _use_shared_memory:
            numel = sum([x.numel() for x in senders])
            storage = graphs[0].senders.storage()._new_shared(numel)
            out = senders[0].new(storage)
        senders = torch.cat(senders, out=out)

        out = None
        if _use_shared_memory:
            numel = sum([x.numel() for x in receivers])
            storage = graphs[0].receivers.storage()._new_shared(numel)
            out = receivers[0].new(storage)
        receivers = torch.cat(receivers, out=out)

        return cls(
            num_nodes=node_offset,
            num_edges=len(senders),
            num_nodes_by_graph=senders.new_tensor(num_nodes_by_graph),
            num_edges_by_graph=senders.new_tensor(num_edges_by_graph),
            node_features=node_features,
            edge_features=edge_features,
            global_features=global_features,
            senders=senders,
            receivers=receivers
        )

    @classmethod
    def from_networkxs(cls, networkxs: Iterable[nx.Graph]) -> GraphBatch:
        return cls.from_graphs([Graph.from_networkx(graph_nx) for graph_nx in networkxs])

    @classmethod
    def collate(cls, samples):
        """Collates a sequence of samples containing graphs into a batch

        The samples in the sequence can contain multiple types of inputs, such as:

        >>> [
        >>>   (input_graph, tensor, other_tensor, output_graph),
        >>>   (input_graph, tensor, other_tensor, output_graph),
        >>>   ...
        >>> ]

        """
        if isinstance(samples[0], Graph):
            return cls.from_graphs(samples)
        elif isinstance(samples[0], collections.abc.Mapping):
            return {key: cls.collate([d[key] for d in samples]) for key in samples[0]}
        elif isinstance(samples[0], collections.abc.Sequence):
            transposed = zip(*samples)
            return [cls.collate(samples) for samples in transposed]
        else:
            return default_collate(samples)


class _BatchView(object):
    def __init__(self, batch: GraphBatch):
        self._batch = batch
        self._pooling_functions = {
            'mean': lambda src, idx: torch_scatter.scatter_mean(src, idx, dim=0, dim_size=batch.num_graphs),
            'sum': lambda src, idx: torch_scatter.scatter_add(src, idx, dim=0, dim_size=batch.num_graphs),
            'max': lambda src, idx: torch_scatter.scatter_max(src, idx, dim=0, dim_size=batch.num_graphs)[0],
        }

    def __len__(self):
        return self._batch.num_graphs


class _BatchNodeView(_BatchView):
    def __getitem__(self, graph_index) -> torch.Tensor:
        node_offset = self._batch.num_nodes_by_graph[:graph_index].sum()
        num_nodes = self._batch.num_nodes_by_graph[graph_index]
        return self._batch.node_features[node_offset:node_offset + num_nodes]

    def __iter__(self) -> Iterator[torch.Tensor]:
        for slice_ in segment_lengths_to_slices(self._batch.num_nodes_by_graph):
            yield self._batch.node_features[slice_]

    def as_tuple(self) -> Tuple[torch.Tensor]:
        """Convenience method to get a tuple of non-aggregated node features.

        Better than building a tuple from the iterator: `tuple(batch.node_features_by_graph)`"""
        return torch.split_with_sizes(self._batch.node_features, self._batch.num_nodes_by_graph.tolist(), dim=0)

    def __call__(self, aggregation) -> torch.Tensor:
        aggregation = self._pooling_functions[aggregation]
        return aggregation(self._batch.node_features, segment_lengths_to_ids(self._batch.num_nodes_by_graph))


class _BatchEdgeView(_BatchView):
    def __getitem__(self, graph_index) -> torch.Tensor:
        edge_offset = self._batch.num_edges_by_graph[:graph_index].sum()
        num_edges = self._batch.num_edges_by_graph[graph_index]
        return self._batch.edge_features[edge_offset:edge_offset + num_edges]

    def __iter__(self) -> Iterator[torch.Tensor]:
        for slice_ in segment_lengths_to_slices(self._batch.num_edges_by_graph):
            yield self._batch.edge_features[slice_]

    def as_tuple(self) -> Tuple[torch.Tensor]:
        """Convenience method to get a tuple of non-aggregated edge features.

        Better than building a tuple from the iterator: `tuple(batch.edge_features_by_graph)`"""
        return torch.split_with_sizes(self._batch.edge_features, self._batch.num_edges_by_graph.tolist(), dim=0)

    def __call__(self, aggregation) -> torch.Tensor:
        aggregation = self._pooling_functions[aggregation]
        return aggregation(self._batch.edge_features, segment_lengths_to_ids(self._batch.num_edges_by_graph))
