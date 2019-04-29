import networkx as nx


def noedges():
    noedges = nx.DiGraph()
    noedges.add_nodes_from(range(3))
    return noedges


def directed():
    directed = nx.DiGraph()
    directed.add_nodes_from(range(4))
    directed.add_edges_from([
        (0, 3),
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0)
    ])
    return directed


def selfloops():
    selfloops = nx.MultiDiGraph()
    selfloops.add_nodes_from(range(3))
    selfloops.add_edges_from([
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 2),
        (1, 0)
    ])
    return selfloops


def complete():
    return nx.complete_graph(3, create_using=nx.DiGraph)


def nonconnected():
    """Non connected + nodes with no edges at all"""
    nonconnected = nx.MultiDiGraph()
    nonconnected.add_nodes_from(range(6))
    nonconnected.add_edges_from([
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (1, 0),
        (1, 0),
        (3, 2),
        (3, 3),
        (4, 4),
        (4, 4)
    ])
    return nonconnected


def karate():
    karate = nx.Graph()
    karate.add_nodes_from(range(34))
    karate.add_edges_from([
        (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)
    ])
    karate = karate.to_directed()
    return karate


def graphs_for_test():
    return {
        'empty': nx.MultiDiGraph(),
        'noedges': noedges(),
        'directed': directed(),
        'selfloops': selfloops(),
        'complete': complete(),
        'nonconnected': nonconnected(),
        'karate': karate()
    }


if __name__ == '__main__':
    for name, graph in graphs_for_test().items():
        nx.nx_pydot.to_pydot(graph).write(f'{name}.png', format='png')
