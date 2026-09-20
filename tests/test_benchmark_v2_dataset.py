"""Benchmark v2 stays deterministic, disjoint, and source-hash complete."""

from hashlib import sha256

import networkx as nx

from qweave.experiments.benchmark_v2_dataset import (
    HELD_OUT_FAMILIES,
    benchmark_v2_manifest,
    directed_v2_graph,
    load_v2_case,
)


def test_v2_manifest_counts_hashes_and_splits_are_frozen() -> None:
    first = benchmark_v2_manifest()
    assert first == benchmark_v2_manifest()
    assert len(first["cases"]) == 180
    counts = {split: sum(case["split"] == split for case in first["cases"])
              for split in ("train", "validation", "test")}
    assert counts == {"train": 60, "validation": 84, "test": 36}
    identifiers = [case["case_id"] for case in first["cases"]]
    assert len(identifiers) == len(set(identifiers))
    for case in first["cases"]:
        assert sha256(case["source_qasm2"].encode()).hexdigest() == case["source_sha256"]
        circuit, graph = load_v2_case(case)
        assert circuit.num_qubits == case["logical_qubits"]
        assert graph.number_of_nodes() == case["physical_qubits"]
        assert nx.is_connected(graph)
        assert set(directed_v2_graph(case).nodes) == set(graph.nodes)


def test_v2_test_split_is_held_out_and_includes_idle_hardware() -> None:
    test = [case for case in benchmark_v2_manifest()["cases"] if case["split"] == "test"]
    assert {case["family"] for case in test} == set(HELD_OUT_FAMILIES)
    assert {case["topology"] for case in test} == {
        "grid2", "chorded_ring", "grid2_idle"}
    assert any(case["physical_qubits"] > case["logical_qubits"] for case in test)
