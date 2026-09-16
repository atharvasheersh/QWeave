import networkx as nx
import pytest

from qweave.oracle.cp_sat import solve_initial_mapping


def test_tiny_oracle_finds_adjacent_assignment():
    interactions = nx.Graph()
    interactions.add_edge(0, 1, weight=1)
    result = solve_initial_mapping(interactions, nx.path_graph(3))
    assert result.mapping[0] != result.mapping[1]
    assert abs(result.objective - 1.0) < 1e-9
    assert result.status in {"OPTIMAL", "FEASIBLE"}


def test_mapping_oracle_rejects_disconnected_hardware_before_solver() -> None:
    interactions = nx.Graph([(0, 1)])
    hardware = nx.Graph([(0, 1), (2, 3)])
    with pytest.raises(ValueError, match="connected hardware"):
        solve_initial_mapping(interactions, hardware)

