"""Tiny exact CP-SAT oracle for the initial mapping objective."""

from dataclasses import dataclass
import math
import time
import networkx as nx
from ortools.sat.python import cp_model

from qweave.core.types import Mapping
from qweave.core.qiskit_adapter import validate_hardware_graph
from qweave.core.validation import validate_mapping


@dataclass
class OracleResult:
    mapping: Mapping
    objective: float
    status: str
    runtime_seconds: float
    gap: float | None


def solve_initial_mapping(interaction_graph: nx.Graph, coupling_graph: nx.Graph, time_limit: float = 10.0) -> OracleResult:
    """Solve weighted shortest-path initial mapping exactly for at most 8 qubits."""

    logicals = sorted(interaction_graph.nodes)
    physicals = sorted(coupling_graph.nodes)
    if not logicals:
        raise ValueError("oracle interaction graph must contain logical qubits")
    if logicals != list(range(len(logicals))):
        raise ValueError("oracle logical nodes must be contiguous indices 0..n-1")
    if len(logicals) > 8:
        raise ValueError("CP-SAT oracle is intentionally limited to <= 8 logical qubits")
    validate_hardware_graph(coupling_graph, len(logicals))
    if not nx.is_connected(coupling_graph):
        raise ValueError("CP-SAT mapping oracle requires connected hardware")
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("oracle time limit must be finite and positive")
    distances = dict(nx.all_pairs_shortest_path_length(coupling_graph))
    if any(not math.isfinite(data.get("weight", 1)) or data.get("weight", 1) < 0
           for _, _, data in interaction_graph.edges(data=True)):
        raise ValueError("oracle interaction weights must be finite and non-negative")
    scale = 1_000_000
    model = cp_model.CpModel()
    assignment = {(logical, physical): model.NewBoolVar(f"x_{logical}_{physical}") for logical in logicals for physical in physicals}
    for logical in logicals:
        model.Add(sum(assignment[logical, physical] for physical in physicals) == 1)
    for physical in physicals:
        model.Add(sum(assignment[logical, physical] for logical in logicals) <= 1)
    # Linearize pair products so CP-SAT can optimize the quadratic mapping cost.
    products = []
    for first, second, data in interaction_graph.edges(data=True):
        weight = int(round(data.get("weight", 1) * scale))
        for physical_first in physicals:
            for physical_second in physicals:
                product = model.NewBoolVar(f"p_{first}_{second}_{physical_first}_{physical_second}")
                model.AddMultiplicationEquality(product, [assignment[first, physical_first], assignment[second, physical_second]])
                products.append(weight * distances[physical_first][physical_second] * product)
    model.Minimize(sum(products))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 1
    started = time.perf_counter()
    status_code = solver.Solve(model)
    runtime = time.perf_counter() - started
    status = solver.StatusName(status_code)
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT could not find a mapping: {status}")
    mapping = {logical: next(physical for physical in physicals if solver.Value(assignment[logical, physical])) for logical in logicals}
    objective = solver.ObjectiveValue() / scale
    bound = solver.BestObjectiveBound() / scale
    gap = abs(objective - bound) / abs(objective) if objective else 0.0
    validate_mapping(mapping, len(logicals), coupling_graph)
    return OracleResult(mapping, objective, status, runtime, gap)
