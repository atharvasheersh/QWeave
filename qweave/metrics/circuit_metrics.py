"""Reusable circuit and routing metrics."""

import time


def _two_qubit_count(circuit: object) -> int:
    return sum(len(qargs) == 2 for _, qargs, _ in circuit.data)


def circuit_metrics(circuit: object, original: object | None = None, runtime: float | None = None) -> dict:
    """Return depth, operation counts, and overheads with zero-safe ratios."""

    original_depth = original.depth() if original is not None else circuit.depth()
    depth = circuit.depth()
    overhead = depth - original_depth
    return {
        "depth": depth,
        "operation_count": len(circuit.data),
        "two_qubit_count": _two_qubit_count(circuit),
        "swap_count": sum(operation.name == "swap" for operation, _, _ in circuit.data),
        "depth_overhead": overhead,
        "relative_depth_overhead": overhead / original_depth if original_depth else 0.0,
        "runtime_seconds": runtime,
    }


def timed(function: object, *args: object, **kwargs: object) -> tuple[object, float]:
    """Run a callable and return its result with wall-clock runtime."""

    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started

