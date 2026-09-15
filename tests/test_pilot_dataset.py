"""The pilot source instances and hardware graphs must be reproducible."""

from hashlib import sha256
import networkx as nx
from qiskit import qasm2

from qweave.experiments.pilot_dataset import pilot_manifest


def test_pilot_fixtures_are_stable_distinct_and_connected() -> None:
    first = pilot_manifest()
    assert first == pilot_manifest()
    assert first["pilot_version"] == "0.1"
    assert len(first["cases"]) == 27
    assert len({case["case_id"] for case in first["cases"]}) == 27
    for case in first["cases"]:
        width = case["logical_qubits"]
        assert qasm2.loads(case["source_qasm2"]).num_qubits == width
        assert sha256(case["source_qasm2"].encode("utf-8")).hexdigest() == case["source_sha256"]
        graph = nx.Graph()
        graph.add_nodes_from(range(width))
        graph.add_edges_from(case["hardware_edges"])
        assert nx.is_connected(graph)
