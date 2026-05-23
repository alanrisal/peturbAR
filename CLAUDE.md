# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**aivc-dcm** implements discrete diffusion models for single-cell gene expression modeling, specifically for predicting cell state changes under genetic perturbations. Published at MLGenX @ ICLR 2026.

## Commands

This project uses **uv** for dependency management (not pip/conda directly).

```bash
# Install dependencies
uv sync
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# Train unconditional model (Dentate Gyrus dataset)
uv run scripts/train_perturbseq.py \
    CONFIG=configs/perturb_seq_small.yaml \
    TRAIN_DATA_PATH=datasets/dentate_gyrus.h5ad

# Train conditional model (with perturbation labels)
uv run scripts/train_perturbseq.py \
    CONFIG=configs/perturb_seq_small.yaml \
    TRAIN_DATA_PATH=datasets/replogle.h5ad \
    COND_LABELS_PT_PATH=datasets/protein_embeddings.pt

# Run inference (conditional generation)
uv run scripts/inference_conditional.py \
    EXPERIMENT_DIR=experiments/dcm \
    CELL_TYPE=hepg2 \
    NUM_SAMPLES_PER_PERT=1000 \
    NUM_STEPS=100

# SLURM cluster training
sbatch bash/train_pseq.sbatch
sbatch bash/inf_cond.sbatch
```

There is no test suite or linter configured.

## Architecture

### Core Library (`src/sedd/`)

**`model.py`** — All model architectures:
- `SEDDTransformer`: Base transformer for unconditional generation. Uses token/gene/time embeddings, multi-head self-attention with RoPE (rotary positional embeddings), and AdaLN (Adaptive Layer Norm) conditioned on diffusion timestep σ.
- `SEDDPerturbationTransformer`: Extends the base with perturbation label embeddings and optional cell-type conditioning. Concatenates time + perturbation + cell_type embeddings for AdaLN input.
- `SEDDPerturbationTransformerSeparateFiLM`: Advanced variant applying FiLM (Feature-wise Linear Modulation) sequentially per condition type (time → perturbation → cell_type) for better compositional generalization.

Model sizes: Small (128d, 4 layers), Medium (256d, 6 layers), Large (512d, 8 layers).

**`graph.py`** — `AbsorbingGraph`: Discrete diffusion where tokens transition to a mask token (index = `num_bins`). Transition probabilities: `p_stay = exp(-σ)`, `p_to_mask = 1 - p_stay`.

**`noise.py`** — Noise schedules:
- `LogLinearNoise` (default): σ(t) = -log(1 - (1-ε)·t)
- `GeometricNoise`: σ(t) = σ_min^(1-t) · σ_max^t

**`data.py`** — Dataset classes:
- `RNASeqDataset`: Wraps h5ad expression matrices (AnnData via scanpy) as PyTorch Dataset with gene binning.
- `PerturbSeqDataset`: Handles control-perturbed paired data with perturbation label lookup.

**`trainer.py`** — Training loops:
- `SEDDTrainer`: Mixed precision (bfloat16), gradient clipping (default 1.0), loss computed only at masked positions.
- `PerturbationTrainer`: Handles perturbation label lookup tables and held-out validation cell types.

**`sampling.py`** — Inference samplers:
- `EulerSampler`: Numerical reverse diffusion (unconditional).
- `PerturbationEulerSampler`: Conditional variant; final step uses argmax denoising.

### Data Flow

```
h5ad file → PerturbSeqDataset (bin genes 0..num_bins-1) → DataLoader
→ Sample t∈[0,1], compute σ(t), apply absorbing diffusion x_t
→ Forward: model(x_t, σ, pert_label) → logits
→ Cross-entropy loss at masked positions → checkpoint
→ Inference: reverse diffusion → predicted perturbed expression
```

### Configuration

YAML config files in `configs/`. Key top-level sections: `model`, `data`, `training`, `checkpointing`, `diffusion`, `inference`. CLI overrides use `KEY=value` syntax (parsed in `scripts/utils.py`). Experiments save to `experiments/dcm/<run_name>/`.

### External Embeddings

Pre-computed protein embeddings (e.g., ESM2) can be passed via `COND_LABELS_PT_PATH`. The model projects these to hidden_dim via a linear layer when `use_pretrained_embeddings: true` in config.
