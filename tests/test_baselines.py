from qiskit import QuantumCircuit
import networkx as nx

from qweave.baselines.basic import run_basic_baseline
from qweave.baselines.sabre import run_sabre_baseline


def test_basic_and_sabre_smoke():
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)
    graph = nx.path_graph(3)
    basic = run_basic_baseline(circuit, graph)
    sabre = run_sabre_baseline(circuit, graph, seed=11)
    assert basic.swap_count >= 0
    assert sabre["depth"] >= 1
    assert sabre["seed"] == 11

