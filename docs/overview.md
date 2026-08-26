# GNN from Scratch - Project Overview

This guide explains what this project does, how the pieces fit together, and how the graph algorithms are implemented.
It is written for people with a basic understanding of Python, arrays, and neural networks.

## What this project is about

The repo builds a **Graph Neural Network (GNN)** system from scratch using only NumPy + custom code.  
It does not depend on PyTorch or TensorFlow for automatic differentiation or graph layers.

- `gnn/autograd/tensor.py` defines a tiny autograd engine (custom `Tensor`, reverse-mode autodiff).
- `gnn/layers` and `gnn/models` define GCN, GAT, and MLP models.
- `gnn/data/cora.py` loads and prepares the Cora citation graph + labels.
- `gnn/graph/algorithms.py` adds general graph utilities: graph construction + BFS/DFS/Dijkstra and stats.
- `scripts/train_cora.py` is the command-line entry point to run training runs.

The project name is `gnn-from-scratch` and the graph is treated both as:

1. A **data structure** (nodes, edges, weights, neighbors), and  
2. A **message-passing operator** for deep learning.

## High-level pipeline

1. Load Cora graph + node features + labels.
2. Represent the graph as an adjacency structure.
3. Train one of:
   - `mlp`: uses features only (baseline),
   - `gcn`: uses graph neighborhood aggregation,
   - `gat`: uses graph attention.
4. Evaluate predictions on test nodes.

Why this is useful: if you only use node features (`mlp`), you ignore graph structure.  
If you use `gcn`/`gat`, you also learn from connected neighbors.

## How a graph is represented in this project (theory + code)

### Theory

In machine learning, many problems are naturally graph-structured:

- node = entity (paper, user, product),
- edge = relation (citation, follow, purchase, similarity),
- optional edge weight = strength/cost of the relation.

For a graph with `N` nodes, the usual representation is an adjacency matrix `A (N x N)`:

- `A[i][j] = 1` (or weight) if edge `i -> j` exists,
- `A[i][j] = 0` if no edge.

For training pipelines, many GNNs use a normalized adjacency:

- add self-loops `A_tilde = A + I`,
- normalize with `D^{-1/2} A_tilde D^{-1/2}`,
- then propagate as `A_hat @ H @ W`.

### In this code

The class in `gnn/graph/algorithms.py` stores graph data as an adjacency list:

- `self._adj[u]` maps neighbors `v` and edge weight `w`,
- `add_edge(u, v, weight, directed=False)` inserts edges,
- default is undirected (adds both `u->v` and `v->u`).

This is efficient for sparse graphs and simple neighbor iteration.

## Dijkstra's algorithm (single-source shortest path)

### What it solves

Given one source node, Dijkstra finds the minimum-cost distance to all other reachable nodes using non-negative edge weights.

### Core idea

1. Start with known distance of source = `0`, others = `inf`.
2. Repeatedly pick the currently unknown node with the smallest tentative distance.
3. Relax edges: try to improve neighboring nodes via that node.
4. When a node is popped from the priority queue with outdated distance, skip it.

### In the project

- Implemented in `Graph.dijkstra(source)`.
- Uses `heapq` (binary min-heap) as priority queue.
- Returns:
  - `dist`: shortest distance map,
  - `prev`: predecessor map (for path reconstruction).
- Path reconstruction helper:
  - `Graph.reconstruct_path(prev, source, target)`.

Why heap + stale-check:
- Better performance than scanning all nodes for each step.
- If a better path is found after a node was already pushed, older heap entries become stale; they are ignored by checking if `d_u > dist[u]`.

## BFS (unweighted shortest-hop traversal)

### What it solves

BFS finds shortest paths in an **unweighted** graph by hop count and gives traversal order by distance layers.

### Core idea

1. Start with a queue containing `source`.
2. Pop front, visit unseen neighbors, enqueue them.
3. Record discovery distance as `dist[parent] + 1`.

### In the project

- Implemented in `Graph.bfs(source)`.
- Returns:
  - `visited`: visit order,
  - `dist`: hop distance from source.
- Uses `collections.deque`.

## DFS (depth-first exploration)

### What it solves

DFS explores as far as possible before backtracking. It is great for ordering, connectivity, and component checks.

### Core idea

1. Use stack (or recursion) starting from source.
2. Pop node, mark visited, push neighbors.
3. Continue until all reachable nodes are processed.

### In the project

- Implemented as iterative DFS to avoid recursion depth issues:
  - `Graph.dfs(source)`.
- Neighbors are pushed in reverse-sorted order to keep deterministic output for tests.

## Why these algorithms exist in a GNN repo

Even though `gcn.py` and `gat.py` are central to training, the graph utilities serve several roles:

- sanity-checking graph connectivity (`is_connected`),
- measuring graph structure (`clustering_coefficient`, `average_clustering`, `diameter`),
- validating algorithm behavior through `tests/test_graph.py`,
- explaining graph intuition in notebooks/docs.

## One-page mental model

Think of this project in two layers:

1. **Graph math layer**  
   Build/inspect a graph, compute paths and properties.

2. **Learning layer**  
   Use graph structure to aggregate neighbor features and learn node representations.

The first layer ensures data is correct and understandable; the second layer learns from that structure.

## Quick commands you can tell people

- Train default GCN: `python scripts/train_cora.py`
- Train GAT: `python scripts/train_cora.py --model gat`
- Train baseline MLP: `python scripts/train_cora.py --model mlp`
- Run tests: `python -m pytest tests/ -v`

## Suggested walkthrough for onboarding

1. Open `gnn/graph/algorithms.py` and read `Graph` API signatures first.
2. Open `gnn/data/cora.py` to see how adjacency is built from data files.
3. Open `gnn/layers/gcn_layer.py` for `A_hat @ H` aggregation.
4. Open `tests/test_graph.py` and trace expectations for BFS/DFS/Dijkstra.
5. Re-run a short training command and compare `mlp` vs `gcn`.

## Where to dig next

- `README.md`: benchmark numbers and end-to-end architecture diagram.
- `notebooks/01_autograd.ipynb`: intuition for custom gradients.
- `notebooks/02_graph_theory.ipynb`: deeper graph concepts.
- `tests/` for compact executable truths about what each class should do.

This file is meant to be a short primer so team members can move from "I know graphs" to "I can read this repo without getting lost."
