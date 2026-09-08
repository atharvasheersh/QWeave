"""Shared type aliases and result records."""

from dataclasses import dataclass, field
from typing import Any, Hashable

LogicalQubit = int
PhysicalQubit = int
Mapping = dict[LogicalQubit, PhysicalQubit]


@dataclass
class MappingResult:
    mapping: Mapping
    trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RoutingResult:
    circuit: Any
    final_mapping: Mapping
    trace: list[dict[str, Any]]
    swap_count: int


def stable_node_key(node: Hashable) -> tuple[str, str]:
    """Return a deterministic key for graph nodes, including non-integers."""

    return (type(node).__name__, repr(node))

