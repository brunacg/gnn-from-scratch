"""
Cora citation-network dataset loader.

Cora contains 2708 scientific publications (nodes) connected by 5429
citation links (edges).  Each paper is described by a 1433-dimensional
binary bag-of-words feature vector and belongs to one of 7 classes.

Reference
---------
McCallum, A. et al. (2000). Automating the Construction of Internet
    Portals with Machine Learning. Information Retrieval, 3(2), 127-163.

Download
--------
The raw files are fetched once from the official mirror and cached in
~/.cache/gnn_scratch/cora/.  Set GNN_DATA_DIR to override the location.
"""

from __future__ import annotations

import os
import urllib.request
import tarfile
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, Tuple, Optional

CoraData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]

_URL          = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
_CONTENT_FILE = "cora/cora.content"
_CITES_FILE   = "cora/cora.cites"

# Splits from Kipf & Welling (2017): 20 per class train, 500 val, 1000 test
_N_TRAIN_PER_CLASS = 20
_N_VAL  = 500
_N_TEST = 1000

CLASS_NAMES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]


def load_cora(data_dir: Optional[str] = None, verbose: bool = True) -> CoraData:
    """
    Load the Cora dataset, downloading it on first use.

    Returns
    -------
    A_hat      : dense normalised adjacency  (N x N)
    X          : node feature matrix  (N x F)
    y          : integer class labels (N,)
    train_mask : boolean mask for 140 training nodes
    val_mask   : boolean mask for 500 validation nodes
    test_mask  : boolean mask for 1000 test nodes
    """
    cache = _get_cache_dir(data_dir)
    if not (cache / _CONTENT_FILE).exists():
        _download_cora(cache, verbose)

    if verbose:
        print("Parsing Cora ...")

    node_ids, X, y = _parse_content(cache / _CONTENT_FILE)
    X = _row_normalise_features(X)
    adj = _parse_cites(cache / _CITES_FILE, node_ids)
    A_hat = _normalise_adjacency(adj)
    train_mask, val_mask, test_mask = _build_masks(y, len(node_ids))

    if verbose:
        print(
            f"  Nodes: {X.shape[0]}  |  Edges: {adj.nnz // 2}  |  "
            f"Features: {X.shape[1]}  |  Classes: {int(y.max()) + 1}"
        )
        print(
            f"  Train: {train_mask.sum()}  |  "
            f"Val: {val_mask.sum()}  |  "
            f"Test: {test_mask.sum()}"
        )

    return A_hat, X, y, train_mask, val_mask, test_mask


def _get_cache_dir(data_dir: Optional[str]) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("GNN_DATA_DIR")
    return Path(env) if env else Path.home() / ".cache" / "gnn_scratch"


def _download_cora(cache: Path, verbose: bool) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    tgz_path = cache / "cora.tgz"
    if verbose:
        print(f"Downloading Cora -> {tgz_path} ...")

    try:
        urllib.request.urlretrieve(_URL, tgz_path, reporthook=_progress(verbose))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Cora from {_URL}.\n"
            "Please download manually from:\n"
            "  https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz\n"
            f"and place the .tgz in {cache}"
        ) from exc

    if verbose:
        print("\nExtracting ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(cache)
    if verbose:
        print("Done.")


def _progress(verbose: bool):
    def hook(count, block_size, total_size):
        if verbose and total_size > 0:
            pct = min(100, int(count * block_size * 100 / total_size))
            print(f"\r  {pct}%", end="", flush=True)
    return hook


def _parse_content(
    path: Path,
) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    """Parse cora.content: <paper_id> <feat_0> ... <feat_1432> <class_label>"""
    class_map: Dict[str, int] = {}
    node_ids:  Dict[str, int] = {}
    feat_rows, label_list = [], []

    with open(path, "r") as f:
        for line in f:
            parts     = line.strip().split()
            paper_id  = parts[0]
            label_str = parts[-1]

            if label_str not in class_map:
                class_map[label_str] = len(class_map)

            node_ids[paper_id] = len(node_ids)
            feat_rows.append(list(map(float, parts[1:-1])))
            label_list.append(class_map[label_str])

    return node_ids, np.array(feat_rows, dtype=np.float32), np.array(label_list, dtype=np.int64)


def _parse_cites(path: Path, node_ids: Dict[str, int]) -> sp.csr_matrix:
    """Parse cora.cites into a symmetric sparse adjacency matrix."""
    N = len(node_ids)
    rows, cols = [], []

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src, dst = parts[0], parts[1]
            if src in node_ids and dst in node_ids:
                i, j = node_ids[src], node_ids[dst]
                rows += [i, j]
                cols += [j, i]

    adj = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, N))
    return (adj > 0).astype(np.float32)


def _row_normalise_features(X: np.ndarray) -> np.ndarray:
    """
    Scale each node's feature vector to sum to 1.

    Standard preprocessing for Cora (Kipf & Welling, 2017): bag-of-words rows
    have wildly different L1 norms based on document length, which biases the
    early-layer activations towards long documents and slows convergence.
    Rows that are entirely zero are left as zero.
    """
    rowsum = X.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0.0] = 1.0
    return (X / rowsum).astype(X.dtype, copy=False)


def _normalise_adjacency(adj: sp.csr_matrix) -> np.ndarray:
    """
    Compute A_hat = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
    where A_tilde = A + I.  Eigenvalues of A_hat lie in (0, 1].
    """
    N = adj.shape[0]
    A_tilde = adj + sp.eye(N, format="csr", dtype=np.float32)
    deg = np.asarray(A_tilde.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(np.where(deg > 0, deg ** -0.5, 0.0), format="csr")
    return (D_inv_sqrt @ A_tilde @ D_inv_sqrt).toarray().astype(np.float64)


def _build_masks(
    labels: np.ndarray, N: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the Kipf & Welling (2017) fixed dataset splits."""
    n_classes = int(labels.max()) + 1
    train_mask = np.zeros(N, dtype=bool)
    for c in range(n_classes):
        train_mask[np.where(labels == c)[0][:_N_TRAIN_PER_CLASS]] = True

    remaining = np.where(~train_mask)[0]
    val_mask  = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)
    val_mask[remaining[:_N_VAL]] = True
    test_mask[remaining[_N_VAL : _N_VAL + _N_TEST]] = True

    return train_mask, val_mask, test_mask
