import networkx as nx

from qweave.routing.shortest_path import deterministic_shortest_path


def test_shortest_path_tie_breaking():
    graph = nx.Graph([(0, 1), (1, 3), (0, 2), (2, 3)])
    assert deterministic_shortest_path(graph, 0, 3) == [0, 1, 3]

