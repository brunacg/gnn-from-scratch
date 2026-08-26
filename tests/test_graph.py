import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gnn.graph.algorithms import Graph


class TestDijkstra:
    def setup_method(self):
        # Weighted graph:
        #   0 --1-- 1
        #   |       |
        #   4       2
        #   |       |
        #   3 --1-- 2
        #   (node 3 connects 0 and 2 as well)
        self.g = Graph(5)
        self.g.add_edge(0, 1, 1.0)
        self.g.add_edge(1, 2, 2.0)
        self.g.add_edge(0, 3, 4.0)
        self.g.add_edge(3, 2, 1.0)
        self.g.add_edge(2, 4, 3.0)

    def test_direct_edge(self):
        dist, _ = self.g.dijkstra(0)
        assert dist[1] == 1.0

    def test_shortest_path(self):
        dist, _ = self.g.dijkstra(0)
        # 0->1->2 = 3,  0->3->2 = 5
        assert dist[2] == 3.0

    def test_longer_path_avoided(self):
        dist, _ = self.g.dijkstra(0)
        assert dist[3] == 4.0

    def test_source_zero(self):
        dist, _ = self.g.dijkstra(0)
        assert dist[0] == 0.0

    def test_path_reconstruction(self):
        dist, prev = self.g.dijkstra(0)
        path = self.g.reconstruct_path(prev, 0, 4)
        # 0 -> 1 -> 2 -> 4
        assert path == [0, 1, 2, 4]
        assert sum(
            self.g.weight(path[i], path[i+1]) for i in range(len(path)-1)
        ) == dist[4]

    def test_disconnected_node(self):
        g = Graph(3)
        g.add_edge(0, 1, 1.0)
        dist, _ = g.dijkstra(0)
        assert dist[2] == float("inf")

    def test_single_node(self):
        g = Graph(1)
        dist, _ = g.dijkstra(0)
        assert dist[0] == 0.0

    def test_negative_weight_raises(self):
        g = Graph(2)
        with pytest.raises(ValueError):
            g.add_edge(0, 1, -1.0)


class TestBFS:
    def setup_method(self):
        self.g = Graph(6)
        for u, v in [(0,1),(0,2),(1,3),(1,4),(2,5)]:
            self.g.add_edge(u, v)

    def test_visited_all_reachable(self):
        visited, _ = self.g.bfs(0)
        assert sorted(visited) == [0, 1, 2, 3, 4, 5]

    def test_bfs_levels(self):
        _, dist = self.g.bfs(0)
        assert dist[0] == 0
        assert dist[1] == 1 and dist[2] == 1
        assert dist[3] == 2 and dist[4] == 2 and dist[5] == 2

    def test_disconnected_component(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        visited, _ = g.bfs(0)
        assert 2 not in visited
        assert 3 not in visited


class TestDFS:
    def setup_method(self):
        self.g = Graph(5)
        for u, v in [(0,1),(0,2),(1,3),(2,4)]:
            self.g.add_edge(u, v)

    def test_visits_all_reachable(self):
        visited = self.g.dfs(0)
        assert sorted(visited) == [0, 1, 2, 3, 4]

    def test_disconnected(self):
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(2, 3)
        assert sorted(g.dfs(0)) == [0, 1]
        assert sorted(g.dfs(2)) == [2, 3]


class TestGraphStats:
    def setup_method(self):
        # Complete graph K4
        self.g = Graph(4)
        for u in range(4):
            for v in range(u+1, 4):
                self.g.add_edge(u, v)

    def test_degree(self):
        for u in range(4):
            assert self.g.degree(u) == 3

    def test_is_connected(self):
        assert self.g.is_connected()

    def test_not_connected(self):
        g = Graph(3)
        g.add_edge(0, 1)
        assert not g.is_connected()

    def test_num_edges(self):
        assert self.g.num_edges() == 6

    def test_clustering_coefficient(self):
        # In K4, every triangle is present -> cc = 1.0
        cc = self.g.clustering_coefficient(0)
        assert abs(cc - 1.0) < 1e-9

    def test_clustering_coefficient_path(self):
        # Path graph 0-1-2: node 1 has 2 neighbours (0,2) but no edge between them
        g = Graph(3)
        g.add_edge(0, 1); g.add_edge(1, 2)
        assert g.clustering_coefficient(1) == 0.0
