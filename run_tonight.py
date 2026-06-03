import gc
import json as _json
import scanpy as sc
import numpy as np
import torch
from pathlib import Path
from scipy import sparse
from torch.utils.data import Subset

from sedd.model import SEDDPerturbationTransformerSeparateFiLMMedium
from sedd.trainer import PerturbationTrainer
from sedd.data import PerturbSeqDataset
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = '/workspace/data/k562_essential_processed.h5ad'
CHECKPOINT_DIR = '/workspace/checkpoints/k562_crossattn'
NUM_GENES      = 2000
NUM_BINS       = 51       # 0..50 expression levels + mask token at 51
BATCH_SIZE     = 64
NUM_EPOCHS     = 30
DEVICE         = torch.device('cuda')
CONTROL_NAME   = 'non-targeting'

# Rows per chunk when we densify for binning. 20k × 2000 float32 ≈ 160 MB
# per chunk; tune down if needed.
BIN_CHUNK_CELLS = 20000

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
adata = sc.read_h5ad(DATA_PATH)
print(f"Loaded shape: {adata.shape}, X type: {type(adata.X).__name__}, "
      f"X dtype: {adata.X.dtype}")

# If the h5ad happened to be stored with a dense X, force CSR right away.
# Replogle counts are >95% zeros, so sparse storage cuts RAM ~20-30x and lets
# normalize_total / log1p run in place without intermediate dense copies.
if not sparse.issparse(adata.X):
    print("X is dense — converting to CSR to save RAM...")
    adata.X = sparse.csr_matrix(adata.X)
    gc.collect()

# HVG on raw counts. subset=True prunes columns in place — no boolean-indexed
# view that subsequent ops may densify behind your back.
sc.pp.highly_variable_genes(
    adata, n_top_genes=NUM_GENES, flavor='seurat_v3', subset=True
)
gc.collect()
print(f"After HVG filter: {adata.shape}")

# Save HVG gene names so inference and eval can align gene columns
GENE_NAMES = list(adata.var_names)
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
with open(Path(CHECKPOINT_DIR) / "genes.json", "w") as _f:
    _json.dump(GENE_NAMES, _f)
print(f"Saved {len(GENE_NAMES)} gene names to {CHECKPOINT_DIR}/genes.json")

# Standard scRNA-seq normalization. Both run in place on CSR.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
print(f"After normalize_total + log1p: max={adata.X.max():.2f}")

# ── Chunked, int8 binning ─────────────────────────────────────────────────────
# Original code:
#     X = adata.X.toarray()
#     X_binned = np.clip(np.round(X).astype(np.int32), 0, NUM_BINS - 1)
# holds up to four simultaneous dense (n_cells, n_genes) arrays in flight
# (X + np.round + astype int32 + clip output) → ~10 GB peak at 310k × 2000.
#
# Rewrite: pre-allocate the int8 output, densify ONE chunk at a time, round
# and clip in place on that chunk's float buffer, then cast to int8. int8
# fits [0, 51] exactly and is 4x smaller than int32.
n_cells, n_genes = adata.shape
X_binned = np.empty((n_cells, n_genes), dtype=np.int8)
for start in range(0, n_cells, BIN_CHUNK_CELLS):
    end = min(start + BIN_CHUNK_CELLS, n_cells)
    chunk = adata.X[start:end]
    if sparse.issparse(chunk):
        chunk = chunk.toarray()  # float32 (n_chunk, n_genes)
    np.round(chunk, out=chunk)                       # in place
    np.clip(chunk, 0, NUM_BINS - 1, out=chunk)       # in place
    X_binned[start:end] = chunk.astype(np.int8, copy=False)
    del chunk
print(f"Binned: dtype={X_binned.dtype}, "
      f"size={X_binned.nbytes / 1e9:.2f} GB")

# Keep what we still need from obs, then drop adata to free the sparse buffer.
pert_series = adata.obs['gene'].copy()
del adata
gc.collect()

# ── Build dataset ──────────────────────────────────────────────────────────────
print("Building dataset...")
dataset = PerturbSeqDataset(
    expression=X_binned,
    pert_labels=pert_series.values,
    num_bins=NUM_BINS,
    control_pert_name=CONTROL_NAME,
)
print(f"num_perturbations: {dataset.num_perturbations}")

# Split by perturbation (not by cell)
all_perts = [p for p in list(dataset.pert_to_idx.keys()) if p != CONTROL_NAME]
np.random.seed(42)
np.random.shuffle(all_perts)
val_perts  = set(all_perts[:int(0.1 * len(all_perts))])
train_perts = set(all_perts) - val_perts

train_mask = [i for i in range(len(dataset))
              if dataset.idx_to_pert[dataset.pert_labels[i].item()] in train_perts]
val_mask   = [i for i in range(len(dataset))
              if dataset.idx_to_pert[dataset.pert_labels[i].item()] in val_perts]

train_dataset = Subset(dataset, train_mask)
val_dataset   = Subset(dataset, val_mask)
print(f"Train cells: {len(train_dataset)}, Val cells: {len(val_dataset)}")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=True
)

# ── GRN attention bias ─────────────────────────────────────────────────────────
# np.corrcoef upcasts internally to float64, which for (n_genes, n_ctrl_cells)
# briefly allocates ~2x the gene×cell matrix. Do the correlation in float32 by
# hand: column-mean center, unit-normalize columns, gram matrix.
print("Computing GRN correlation bias from control cells...")
ctrl_mask = (pert_series == CONTROL_NAME).values
ctrl_X = X_binned[ctrl_mask].astype(np.float32, copy=True)   # (n_ctrl, n_genes)
ctrl_X -= ctrl_X.mean(axis=0, keepdims=True)
ctrl_norms = np.linalg.norm(ctrl_X, axis=0, keepdims=True)
ctrl_norms[ctrl_norms == 0] = 1.0
ctrl_X /= ctrl_norms
A = (ctrl_X.T @ ctrl_X).astype(np.float32)                   # (n_genes, n_genes)
np.nan_to_num(A, copy=False, nan=0.0)
del ctrl_X, ctrl_norms
gc.collect()
attn_bias = torch.tensor(0.05 * A, device=DEVICE)
del A, X_binned, pert_series  # both arrays are now fully consumed
gc.collect()
print(f"GRN bias shape: {attn_bias.shape}")

# ── Build model ────────────────────────────────────────────────────────────────
print("Building model...")
model = SEDDPerturbationTransformerSeparateFiLMMedium(
    num_genes=NUM_GENES,
    num_bins=NUM_BINS,
    num_perturbations=dataset.num_perturbations,
    num_cell_types=None,
    max_seq_len=NUM_GENES,   # pin to actual gene count; default is 4096 which wastes memory
)
model.register_buffer('attn_bias', attn_bias)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Build trainer ──────────────────────────────────────────────────────────────
graph   = AbsorbingGraph(num_states=NUM_BINS + 1)
noise   = LogLinearNoise()

trainer = PerturbationTrainer(
    model=model,
    graph=graph,
    noise=noise,
    device=DEVICE,
    use_amp=True,
    amp_dtype=torch.bfloat16,
    gradient_clip=1.0,
)

# ── Train ──────────────────────────────────────────────────────────────────────
print("Starting training...")
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    mask_ratio=0.15,
    log_interval=50,
    val_interval=1,
    checkpoint_dir=CHECKPOINT_DIR,
    save_interval=5,
)
print("Done. Checkpoints at:", CHECKPOINT_DIR)