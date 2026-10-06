"""Read-only questions about a CodeGraph (IMPLEMENTATION_PLAN B6).

GraphIndex pre-computes adjacency lists once; the module-level functions are
convenience wrappers for one-off calls. Results are sets/dicts: callers must
sort before producing output (DECISIONS #8).
"""

from collections import deque

import networkx as nx

from tokencut.contracts import CodeGraph, EdgeKind

MODULE_SUFFIX = "::<module>"
_DEPENDENCY_KINDS = (EdgeKind.CALLS, EdgeKind.INHERITS)


def file_of(chunk_id: str) -> str:
    """'shop/cart.py::Cart.total' -> 'shop/cart.py' (file paths never contain '::')."""
    return chunk_id.split("::", 1)[0]


class GraphIndex:
    def __init__(self, graph: CodeGraph):
        self.chunk_ids = set(graph.chunk_ids)
        self.depends_on: dict[str, list[str]] = {cid: [] for cid in graph.chunk_ids}
        self.undirected: dict[str, list[str]] = {cid: [] for cid in graph.chunk_ids}
        self.parent: dict[str, str] = {}  # method id -> class id (from CONTAINS edges)
        for edge in graph.edges:  # graph.edges is sorted, so neighbour lists are deterministic
            if edge.source not in self.chunk_ids or edge.target not in self.chunk_ids:
                continue
            if edge.kind in _DEPENDENCY_KINDS:
                self.depends_on[edge.source].append(edge.target)
            if edge.kind == EdgeKind.CONTAINS:
                self.parent[edge.target] = edge.source
            self.undirected[edge.source].append(edge.target)
            self.undirected[edge.target].append(edge.source)

    def module_of(self, chunk_id: str) -> str | None:
        module_id = file_of(chunk_id) + MODULE_SUFFIX
        return module_id if module_id in self.chunk_ids else None

    def closure(self, chunk_ids: set[str], max_hops: int = 1) -> set[str]:
        """Start set + outgoing CALLS/INHERITS within max_hops, + parent classes, + module chunks."""
        result = {cid for cid in chunk_ids if cid in self.chunk_ids}
        frontier = sorted(result)
        for _ in range(max_hops):
            next_frontier = []
            for cid in frontier:
                for target in self.depends_on[cid]:
                    if target not in result:
                        result.add(target)
                        next_frontier.append(target)
            frontier = next_frontier
        for cid in sorted(result):
            if cid in self.parent:
                result.add(self.parent[cid])
        for cid in sorted(result):
            module_id = self.module_of(cid)
            if module_id is not None:
                result.add(module_id)
        return result

    def distances(
        self, sources: list[str] | set[str], max_hops: int | None = None
    ) -> dict[str, int]:
        """Undirected multi-source BFS: chunk id -> hops to the nearest source."""
        dist: dict[str, int] = {}
        queue: deque[str] = deque()
        for cid in sorted(sources):
            if cid in self.chunk_ids and cid not in dist:
                dist[cid] = 0
                queue.append(cid)
        while queue:
            cid = queue.popleft()
            if max_hops is not None and dist[cid] >= max_hops:
                continue
            for neighbour in self.undirected[cid]:
                if neighbour not in dist:
                    dist[neighbour] = dist[cid] + 1
                    queue.append(neighbour)
        return dist

    def neighbours(self, chunk_id: str) -> list[str]:
        """Direct neighbours along any edge, either direction."""
        return self.undirected.get(chunk_id, [])


def dependency_closure(graph: CodeGraph, chunk_ids: set[str], max_hops: int = 1) -> set[str]:
    return GraphIndex(graph).closure(chunk_ids, max_hops)


def neighbours_within(graph: CodeGraph, chunk_id: str, k: int) -> dict[str, int]:
    return GraphIndex(graph).distances([chunk_id], max_hops=k)


def multi_source_distances(graph: CodeGraph, sources: list[str] | set[str]) -> dict[str, int]:
    return GraphIndex(graph).distances(sources)


def to_networkx(graph: CodeGraph) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_nodes_from(graph.chunk_ids)
    for edge in graph.edges:
        g.add_edge(edge.source, edge.target, kind=edge.kind.value, resolved=edge.resolved)
    return g
