import scanpy as sc
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Subset

from sedd.model import SEDDPerturbationTransformerSeparateFiLMMedium
from sedd.trainer import PerturbationTrainer
from sedd.data import PerturbSeqDataset
from sedd.graph import AbsorbingGraph
from sedd.noise import LogLinearNoise  # check noise.py for exact class name

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = '/workspace/data/replogle_k562_essential/perturb_processed.h5ad'
CHECKPOINT_DIR = '/workspace/checkpoints/k562_crossattn'
NUM_GENES      = 2000
NUM_BINS       = 51       # 0..50 expression levels + mask token at 51
BATCH_SIZE     = 64
NUM_EPOCHS     = 30
DEVICE         = torch.device('cuda')
CONTROL_NAME   = 'non-targeting'  # update if your print above showed different

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
adata = sc.read_h5ad(DATA_PATH)

# Filter to highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=NUM_GENES, flavor='seurat_v3')
adata = adata[:, adata.var.highly_variable]
print(f"After HVG filter: {adata.shape}")

# Bin to discrete tokens
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
X_binned = np.clip(np.round(X).astype(np.int32), 0, NUM_BINS - 1)

# ── Build dataset ──────────────────────────────────────────────────────────────
print("Building dataset...")
dataset = PerturbSeqDataset(
    expression=X_binned,
    pert_labels=adata.obs['gene'].values,
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
print("Computing GRN correlation bias from control cells...")
ctrl_mask  = adata.obs['gene'] == CONTROL_NAME
ctrl_X     = X_binned[ctrl_mask.values]
A          = np.corrcoef(ctrl_X.T).astype(np.float32)
A          = np.nan_to_num(A, nan=0.0)   # handle any NaN from zero-variance genes
attn_bias  = torch.tensor(0.05 * A, device=DEVICE)
print(f"GRN bias shape: {attn_bias.shape}")

# ── Build model ────────────────────────────────────────────────────────────────
print("Building model...")
model = SEDDPerturbationTransformerSeparateFiLMMedium(
    num_genes=NUM_GENES,
    num_bins=NUM_BINS,
    num_perturbations=dataset.num_perturbations,
    num_cell_types=None,
)
model.register_buffer('attn_bias', attn_bias)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# ── Build trainer ──────────────────────────────────────────────────────────────
graph   = AbsorbingGraph(num_states=NUM_BINS + 1)
noise   = LogLinearNoise()   # check noise.py — may be named differently

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