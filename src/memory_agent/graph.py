from __future__ import annotations

from collections.abc import Callable
from typing import Any


Node = Callable[[dict[str, Any]], dict[str, Any]]


class SimpleStateGraph:
    """Small LangGraph-like skeleton for deterministic node flow."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, str] = {}

    def add_node(self, name: str, node: Node) -> None:
        self.nodes[name] = node

    def add_edge(self, source: str, target: str) -> None:
        self.edges[source] = target

    def run(self, start: str, state: dict[str, Any]) -> dict[str, Any]:
        current = start
        while current in self.nodes:
            state = self.nodes[current](state)
            if current not in self.edges:
                return state
            current = self.edges[current]
        return state
