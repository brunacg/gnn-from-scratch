# Benchmark snapshot

This repository includes a small reproducibility benchmark on the Cora citation network.
The numbers below come from three random seeds (`42`, `43`, `44`) with a maximum of
200 epochs, early stopping (`patience=20`), and gradient clipping at `5.0`.

| Model | Test accuracy (mean ± population SD) | Mean best epoch | Mean trial time* |
|---|---:|---:|---:|
| MLP | 52.00 ± 0.54% | 28.0 | 4.6 s |
| GCN | 78.30 ± 0.24% | 46.3 | 10.5 s |
| GAT | 80.93 ± 1.86% | 26.7 | 20.7 s |

\*Timing is hardware- and environment-dependent and should be treated as illustrative,
not as a portable performance claim.

## Reproduce

```bash
python scripts/benchmark.py \
  --trials 3 \
  --epochs 200 \
  --patience 20 \
  --grad-clip 5.0
```

The benchmark is intentionally simple: its purpose is to verify that the custom
autograd engine, graph layers, sparse aggregation paths, training loop, and baselines
behave coherently across multiple seeds.

## Interpretation

The MLP uses node features only. GCN adds normalized neighborhood aggregation, while
GAT learns edge-level attention coefficients. On this snapshot, graph structure gives
a large gain over the feature-only baseline, and GAT provides a smaller additional
improvement over GCN.

See the implementation postmortems for the debugging and optimization history:

- [GCN sparse aggregation](postmortems/gcn.md)
- [GAT gradient-flow and sparse-attention rewrite](postmortems/gat.md)
- [MLP baseline cleanup](postmortems/mlp.md)
