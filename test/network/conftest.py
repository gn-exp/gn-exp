import pytest

from torchgraphs import GraphBatch
from graphs_for_test import graphs_for_test


@pytest.fixture
def graphbatch() -> GraphBatch:
    return GraphBatch.from_networkxs(graphs_for_test().values())
