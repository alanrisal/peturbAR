import numpy as np
import scanpy as sc
import torch
from scipy.spatial.distance import cdist
import ot  # pip install POT
from sklearn.metrics.pairwise import rbf_kernel


# Must match NUM_BINS in run_tonight.py
NUM_BINS = 51


def normalize_and_bin(X, num_bins=NUM_BINS):
    """Apply the same preprocessing used during training.

    Training pipeline (run_tonight.py):
      1. normalize_total(target_sum=1e4)  — equalize sequencing depth per cell
      2. log1p                            — compress dynamic range; values → [0, ~8]
      3. np.round + clip to [0, num_bins-1] — discretize to tokens

    Generated cells are already token indices (steps 1-3 already applied).
    Test h5ad contains raw counts (max ~210), so we must apply all three steps
    before comparing — otherwise W2 is computed across incompatible scales.
    """
    if hasattr(X, 'toarray'):
        X = X.toarray().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    # Detect whether normalization is still needed (generated cells are already
    # small integers [0, num_bins-1]; raw counts have much higher max values).
    if X.max() > num_bins:
        # normalize_total: scale each cell to target_sum=1e4
        cell_sums = X.sum(axis=1, keepdims=True)
        cell_sums = np.where(cell_sums == 0, 1, cell_sums)  # avoid div-by-zero
        X = X / cell_sums * 1e4
        # log1p
        X = np.log1p(X)

    return np.clip(np.round(X).astype(np.int32), 0, num_bins - 1)


def compute_mmd(X, Y, gamma=None):
    """Maximum Mean Discrepancy with RBF kernel."""
    if gamma is None:
        # median heuristic
        combined = np.vstack([X, Y])
        dists = cdist(combined, combined, 'sqeuclidean')
        gamma = 1.0 / np.median(dists[dists > 0])

    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def compute_w2(X, Y):
    """Wasserstein-2 distance via POT (exact for small N, entropy-regularized
    for large)."""
    n, m = len(X), len(Y)
    a = np.ones(n) / n
    b = np.ones(m) / m
    M = ot.dist(X, Y, metric='sqeuclidean')  # cost matrix
    w2_sq = ot.emd2(a, b, M)  # returns squared W2
    return np.sqrt(w2_sq)


def evaluate(generated_h5ad, test_h5ad, pert_col='perturbation',
             test_pert_col=None, n_subsample=1000, n_top_genes=50, num_bins=NUM_BINS):
    """
    Args:
        pert_col:      obs column name in the GENERATED h5ad  (default: 'perturbation')
        test_pert_col: obs column name in the TEST h5ad.
                       Defaults to pert_col when not set, but the raw Replogle
                       file uses 'gene' while generated cells use 'perturbation',
                       so pass test_pert_col='gene' when evaluating against it.
    """
    if test_pert_col is None:
        test_pert_col = pert_col

    gen = sc.read_h5ad(generated_h5ad)
    test = sc.read_h5ad(test_h5ad)

    # --- Normalize test BEFORE gene alignment ---
    # Critical ordering: training did normalize_total(all 8563 genes) → log1p → HVG subset.
    # If we subset genes first and then normalize, we normalize over ~2000 genes instead of
    # 8563, inflating values by ~4x (HVGs carry only ~25% of total counts).
    # Generated cells are already binned tokens — normalize_and_bin is a no-op for them.
    gen_X_raw = gen.X.toarray() if hasattr(gen.X, 'toarray') else np.asarray(gen.X, dtype=np.float32)
    test_X_raw = test.X.toarray() if hasattr(test.X, 'toarray') else np.asarray(test.X, dtype=np.float32)

    print(f"[pre-bin] Generated X range: {gen_X_raw.min():.3f} .. {gen_X_raw.max():.3f}")
    print(f"[pre-bin] Test X range:      {test_X_raw.min():.3f} .. {test_X_raw.max():.3f}")

    gen_binned = normalize_and_bin(gen_X_raw, num_bins=num_bins)
    # Normalize test over the FULL gene set before subsetting
    if test_X_raw.max() > num_bins:
        cell_sums = test_X_raw.sum(axis=1, keepdims=True)
        cell_sums = np.where(cell_sums == 0, 1, cell_sums)
        test_X_norm = np.log1p(test_X_raw / cell_sums * 1e4)
        test = sc.AnnData(X=test_X_norm, obs=test.obs.copy(), var=test.var.copy())
        test_X_raw = test_X_norm  # update for post-bin print below

    # --- Gene alignment (after normalization) ---
    if gen.var is not None and len(gen.var) > 0 and gen.n_vars != test.n_vars:
        shared_genes = [g for g in gen.var_names if g in test.var_names]
        if len(shared_genes) == 0:
            raise ValueError(
                f"No gene names overlap between generated ({gen.n_vars} genes) "
                f"and test ({test.n_vars} genes). Check that var_names are set on "
                f"generated_cells.h5ad and that the same HVGs were used."
            )
        print(f"Gene alignment: generated={gen.n_vars}, test={test.n_vars} -> "
              f"using {len(shared_genes)} shared genes")
        gen  = gen[:, shared_genes]
        test = test[:, shared_genes]
    elif gen.n_vars != test.n_vars:
        raise ValueError(
            f"Generated has {gen.n_vars} genes, test has {test.n_vars} genes, "
            f"and generated h5ad has no var_names to align by."
        )

    # Bin the (now-aligned) test expression
    test_X_aligned = test.X.toarray() if hasattr(test.X, 'toarray') else np.asarray(test.X)
    test_binned = np.clip(np.round(test_X_aligned).astype(np.int32), 0, num_bins - 1)

    print(f"[post-bin] Generated X range: {gen_binned.min()} .. {gen_binned.max()}")
    print(f"[post-bin] Test X range:      {test_binned.min()} .. {test_binned.max()}")

    # Rebuild AnnData with binned values
    gen = sc.AnnData(X=gen_binned, obs=gen.obs.copy(), var=gen.var.copy())
    test = sc.AnnData(X=test_binned, obs=test.obs.copy(), var=test.var.copy())

    perturbations = gen.obs[pert_col].unique()
    results = {}

    for pert in perturbations:
        gen_cells = gen[gen.obs[pert_col] == pert].X
        test_cells = test[test.obs[test_pert_col] == pert].X

        # X is now a dense ndarray after the rebuild, but guard anyway.
        if hasattr(gen_cells, 'toarray'):
            gen_cells = gen_cells.toarray()
        if hasattr(test_cells, 'toarray'):
            test_cells = test_cells.toarray()

        if len(gen_cells) == 0 or len(test_cells) == 0:
            print(f"{pert}: skipped (gen={len(gen_cells)}, test={len(test_cells)})")
            continue

        # Subsample for tractability
        gen_cells = gen_cells[np.random.choice(len(gen_cells),
                              min(n_subsample, len(gen_cells)),
                              replace=False)]
        test_cells = test_cells[np.random.choice(len(test_cells),
                                min(n_subsample, len(test_cells)),
                                replace=False)]

        # Top variable genes within the shared HVG set — both sides have the same
        # columns at this point, so top_idx is valid for both.
        gene_var = np.var(test_cells, axis=0)
        top_idx = np.argsort(gene_var)[-n_top_genes:]
        gen_sub = gen_cells[:, top_idx].astype(float)
        test_sub = test_cells[:, top_idx].astype(float)

        mmd = compute_mmd(gen_sub, test_sub)
        w2 = compute_w2(gen_sub, test_sub)

        results[pert] = {'mmd': mmd, 'w2': w2}
        print(f"{pert}: MMD={mmd:.4f}, W2={w2:.4f}")

    return results


if __name__ == "__main__":
    import sys
    results = evaluate(
        generated_h5ad=sys.argv[1],   # e.g. /workspace/checkpoints/k562_crossattn/inference_results/generated_cells.h5ad
        test_h5ad=sys.argv[2],        # e.g. /workspace/data/k562_essential_processed.h5ad
        pert_col='perturbation',      # column name in generated h5ad
        test_pert_col='gene',         # column name in raw Replogle h5ad
        n_subsample=500,
        n_top_genes=50,
        num_bins=NUM_BINS,
    )
    # Summary
    mmds = [v['mmd'] for v in results.values()]
    w2s  = [v['w2']  for v in results.values()]
    print(f"\nMean MMD: {np.mean(mmds):.4f} ± {np.std(mmds):.4f}")
    print(f"Mean W2:  {np.mean(w2s):.4f} ± {np.std(w2s):.4f}")