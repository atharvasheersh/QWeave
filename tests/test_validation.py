from qiskit import QuantumCircuit
import networkx as nx
import pytest

from qweave.core.validation import validate_mapping, validate_mapping_inverse, validate_two_qubit_legality


def test_invalid_mapping_rejected():
    with pytest.raises(ValueError, match="injective"):
        validate_mapping({0: 0, 1: 0}, 2, nx.path_graph(2))


def test_inverse_and_illegal_operation_validation():
    assert validate_mapping_inverse({0: 1, 1: 0}) == {1: 0, 0: 1}
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    with pytest.raises(ValueError, match="illegal"):
        validate_two_qubit_legality(circuit, nx.path_graph(3))

