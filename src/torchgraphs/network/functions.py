import torch

from ..data import GraphBatch


class _FeatureFunction(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function


class EdgeFunction(_FeatureFunction):
    def forward(self, graphs: GraphBatch) -> GraphBatch:
        return graphs.evolve(edge_features=self.function(graphs.edge_features))


class NodeFunction(_FeatureFunction):

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        return graphs.evolve(node_features=self.function(graphs.node_features))


class GlobalFunction(_FeatureFunction):

    def forward(self, graphs: GraphBatch) -> GraphBatch:
        return graphs.evolve(global_features=self.function(graphs.global_features))


class NodeReLU(NodeFunction):
    def __init__(self):
        super(NodeReLU, self).__init__(torch.nn.functional.relu)


class EdgeReLU(EdgeFunction):
    def __init__(self):
        super(EdgeReLU, self).__init__(torch.nn.functional.relu)


class GlobalReLU(GlobalFunction):
    def __init__(self):
        super(GlobalReLU, self).__init__(torch.nn.functional.relu)

class EdgeDroput(torch.nn.Dropout):
    def forward(self, graphs):
        return graphs.evolve(
            edge_features=super(EdgeDroput, self).forward(graphs.edge_features)
        )

class NodeDroput(torch.nn.Dropout):
    def forward(self, graphs):
        return graphs.evolve(
            node_features=super(NodeDroput, self).forward(graphs.node_features)
        )

class GlobalDroput(torch.nn.Dropout):
    def forward(self, graphs):
        return graphs.evolve(
            global_features=super(GlobalDroput, self).forward(graphs.global_features)
        )
