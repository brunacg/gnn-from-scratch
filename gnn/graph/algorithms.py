"""
Classic graph algorithms implemented from scratch.

Data structure: adjacency list using a dict of dicts  {u: {v: weight}}.
All algorithms work on undirected graphs by default; directed graphs
are supported by omitting the reverse edge in add_edge().

Algorithms
----------
- Dijkstra's shortest paths  O((V + E) log V)  via a binary min-heap
- BFS (breadth-first search) O(V + E)
- DFS (depth-first search)   O(V + E)

Graph statistics
----------------
- degree(u)
- clustering_coefficient(u)
- is_connected()
- num_edges()
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, List, Optional, Set, Tuple


class Graph:
    """
    Undirected weighted graph stored as an adjacency list.

    Nodes are labelled 0 .. N-1.
    """

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes
        # _adj[u][v] = weight of edge (u, v)
        self._adj: Dict[int, Dict[int, float]] = {i: {} for i in range(n_nodes)}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def add_edge(self, u: int, v: int, weight: float = 1.0, directed: bool = False) -> None:
        if weight < 0:
            raise ValueError(
                f"Negative edge weight {weight} is not supported by Dijkstra. "
                "Use Bellman-Ford for negative weights."
            )
        self._adj[u][v] = weight
        if not directed:
            self._adj[v][u] = weight

    def weight(self, u: int, v: int) -> float:
        return self._adj[u][v]

    def neighbours(self, u: int) -> List[int]:
        return list(self._adj[u])

    # ------------------------------------------------------------------
    # Dijkstra's algorithm
    # ------------------------------------------------------------------

    def dijkstra(self, source: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        """
        Single-source shortest paths from `source` using Dijkstra's algorithm.

        Implementation uses a binary min-heap (Python's heapq) for the
        priority queue.  Time complexity: O((V + E) log V).

        Returns
        -------
        dist : dict mapping node -> shortest distance from source
               (float('inf') for unreachable nodes)
        prev : dict mapping node -> previous node on shortest path
               (None for source and unreachable nodes)
        """
        dist: Dict[int, float] = {u: float("inf") for u in range(self.n_nodes)}
        prev: Dict[int, Optional[int]] = {u: None for u in range(self.n_nodes)}
        dist[source] = 0.0

        # heap entries: (distance, node)
        heap = [(0.0, source)]

        while heap:
            d_u, u = heapq.heappop(heap)

            # Stale entry -- a shorter path was already found
            if d_u > dist[u]:
                continue

            for v, w in self._adj[u].items():
                alt = dist[u] + w
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(heap, (alt, v))

        return dist, prev

    def reconstruct_path(
        self, prev: Dict[int, Optional[int]], source: int, target: int
    ) -> List[int]:
        """
        Reconstruct the shortest path from source to target using the
        `prev` dict returned by dijkstra().

        Returns an empty list if target is unreachable.
        """
        if prev[target] is None and target != source:
            return []
        path = []
        node: Optional[int] = target
        while node is not None:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path if path[0] == source else []

    # ------------------------------------------------------------------
    # BFS
    # ------------------------------------------------------------------

    def bfs(self, source: int) -> Tuple[List[int], Dict[int, int]]:
        """
        Breadth-first search from `source`.

        Returns
        -------
        visited : nodes in BFS discovery order
        dist    : hop-distance from source (unweighted)
        """
        visited_set: Set[int] = {source}
        visited: List[int] = [source]
        dist: Dict[int, int] = {source: 0}
        queue = deque([source])

        while queue:
            u = queue.popleft()
            for v in self._adj[u]:
                if v not in visited_set:
                    visited_set.add(v)
                    visited.append(v)
                    dist[v] = dist[u] + 1
                    queue.append(v)

        return visited, dist

    # ------------------------------------------------------------------
    # DFS
    # ------------------------------------------------------------------

    def dfs(self, source: int) -> List[int]:
        """
        Depth-first search from `source` (iterative, using an explicit stack).

        Returns nodes in DFS discovery order.
        """
        visited: List[int] = []
        visited_set: Set[int] = set()
        stack = [source]

        while stack:
            u = stack.pop()
            if u in visited_set:
                continue
            visited_set.add(u)
            visited.append(u)
            # push neighbours in reverse order so smaller-index nodes are
            # explored first (consistent ordering for tests)
            for v in sorted(self._adj[u], reverse=True):
                if v not in visited_set:
                    stack.append(v)

        return visited

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------

    def degree(self, u: int) -> int:
        return len(self._adj[u])

    def num_edges(self) -> int:
        """Number of undirected edges."""
        return sum(len(nbrs) for nbrs in self._adj.values()) // 2

    def is_connected(self) -> bool:
        if self.n_nodes == 0:
            return True
        visited, _ = self.bfs(0)
        return len(visited) == self.n_nodes

    def clustering_coefficient(self, u: int) -> float:
        """
        Local clustering coefficient of node u.

        Fraction of pairs among u's neighbours that are themselves connected:

            cc(u) = |{(v,w): v,w in N(u), (v,w) in E}| / (k*(k-1)/2)

        where k = degree(u).  Returns 0.0 if degree < 2.
        """
        nbrs = list(self._adj[u])
        k = len(nbrs)
        if k < 2:
            return 0.0
        triangles = sum(
            1
            for i, v in enumerate(nbrs)
            for w in nbrs[i + 1 :]
            if w in self._adj[v]
        )
        return triangles / (k * (k - 1) / 2)

    def average_clustering(self) -> float:
        return sum(self.clustering_coefficient(u) for u in range(self.n_nodes)) / self.n_nodes

    def diameter(self) -> float:
        """
        Graph diameter: maximum shortest-path distance over all node pairs.
        Returns float('inf') if the graph is disconnected.
        """
        diam = 0.0
        for u in range(self.n_nodes):
            dist, _ = self.dijkstra(u)
            max_d = max(dist.values())
            if max_d == float("inf"):
                return float("inf")
            diam = max(diam, max_d)
        return diam

    def __repr__(self) -> str:
        return f"Graph(nodes={self.n_nodes}, edges={self.num_edges()})"
