# dsc180-quantization

Compare per-channel and per-token KV-cache quantization using attention tensors
from Qwen3-1.7B. The experiment measures reconstruction and attention errors and
plots the results.

## Setup

Use Python 3.13 and [uv](https://docs.astral.sh/uv/). From the repo root, install
the dependencies with:

```bash
uv sync --locked
```

This creates a `.venv` with the versions recorded in `uv.lock`. The project uses
PyTorch, Transformers, Accelerate, NumPy, and Matplotlib, plus pytest for tests.

## Get the data

No separate dataset is needed: the input passage is included in
`src/inference/pull_cache.py`. Run:

```bash
uv run python src/inference/pull_cache.py
```

On the first run, this downloads `Qwen/Qwen3-1.7B` from Hugging Face, so you need
internet access and enough disk space for the model weights. Downloads are kept
in Hugging Face's local cache configured on the amchine.

The script runs the passage through the model and saves query, key, and value
tensors to `kv_snapshots/layer.pt`. This generated file is ignored by Git.

## Reproduce the results

After generating the tensors, run from the repo root:

```bash
uv run python src/quantization/main.py
```

This compares both methods using 2-bit quantization, a group size of 32, and a
residual-length setting of 128. It prints average errors across query heads and
writes two figures:

- `figures/kv_magnitudes.png`: key/value magnitudes for the first layer and KV head.
- `figures/error_comparison.png`: reconstruction and attention errors by method.

To check the quantization, metrics, and plotting code:

```bash
uv run pytest
```

## AI Usage

We utilized Claude Code and Codex for understanding PyTorch syntax, though the code itself was written by hand.
