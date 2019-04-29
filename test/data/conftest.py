from typing import Sequence

import pytest
import networkx as nx

from torchgraphs import Graph
from graphs_for_test import graphs_for_test
from features_shapes import all_features


@pytest.fixture(params=all_features.values(), ids=all_features.keys())
def features_shapes(request):
    return request.param


@pytest.fixture(params=graphs_for_test().values(), ids=graphs_for_test().keys())
def graph_nx(request) -> nx.Graph:
    return request.param


@pytest.fixture(params=graphs_for_test().values(), ids=graphs_for_test().keys())
def graph(request) -> Graph:
    return Graph.from_networkx(request.param)


@pytest.fixture
def graphs_nx() -> Sequence[nx.Graph]:
    return list(graphs_for_test().values())


@pytest.fixture
def graphs() -> Sequence[Graph]:
    return [Graph.from_networkx(g) for g in graphs_for_test().values()]
