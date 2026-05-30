import scanpy as sc
import numpy as np

adata = sc.read_h5ad('/workspace/data/k562_essential_processed.h5ad')

print("=== Shape ===")
print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

print("\n=== obs columns ===")
print(list(adata.obs.columns))

print("\n=== Perturbation label column ===")
# run_tonight.py expects adata.obs['gene']
if 'gene' in adata.obs.columns:
    uniq = adata.obs['gene'].unique()
    print(f"  'gene' column found: {len(uniq)} unique values")
    print(f"  Sample: {list(uniq[:10])}")
    ctrl_count = (adata.obs['gene'] == 'non-targeting').sum()
    print(f"  Cells with 'non-targeting' label: {ctrl_count}")
else:
    print("  WARNING: 'gene' column NOT found. Available columns above.")

print("\n=== Expression matrix ===")
X = adata.X
sample = X[:500, :500]
if hasattr(sample, 'toarray'):
    sample = sample.toarray()
print(f"  dtype: {sample.dtype}")
print(f"  min: {sample.min():.4f}, max: {sample.max():.4f}, mean:{sample.mean():.4f}")
print(f"  Fraction of zeros: {(sample == 0).mean():.2%}")
print(f"  Are values integers? {np.all(sample == np.round(sample))}")