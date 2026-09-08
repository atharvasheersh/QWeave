import networkx as nx

from qweave.oracle.cp_sat import solve_initial_mapping


def test_tiny_oracle_finds_adjacent_assignment():
    interactions = nx.Graph()
    interactions.add_edge(0, 1, weight=1)
    result = solve_initial_mapping(interactions, nx.path_graph(3))
    assert result.mapping[0] != result.mapping[1]
    assert abs(result.objective - 1.0) < 1e-9
    assert result.status in {"OPTIMAL", "FEASIBLE"}

