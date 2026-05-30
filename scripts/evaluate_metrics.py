import numpy as np
import scanpy as sc
import torch
from scipy.spatial.distance import cdist
import ot  # pip install POT
from sklearn.metrics.pairwise import rbf_kernel


# Must match NUM_BINS in run_tonight.py
NUM_BINS = 51


def bin_expression(X, num_bins=NUM_BINS):
    """Apply the same discretization used during training.

    Generated cells are saved as integer bin indices in [0, num_bins - 1].
    The test h5ad contains raw expression counts, so we must apply the
    identical binning before any distance comparison — otherwise the two
    distributions live on completely different scales and metrics like W2
    become meaningless (e.g. W2 ~ 4000 in a 50-dim space where the
    theoretical max with both sides in [0, 50] is ~354).
    """
    if hasattr(X, 'toarray'):
        X = X.toarray()
    else:
        X = np.asarray(X)
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
             n_subsample=1000, n_top_genes=50, num_bins=NUM_BINS):
    gen = sc.read_h5ad(generated_h5ad)
    test = sc.read_h5ad(test_h5ad)

    # --- Gene alignment: generated cells only have the HVGs used at training time.
    # If the test h5ad has more genes, subset it to the same gene set in the same order.
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
            f"and generated h5ad has no var_names to align by. "
            f"Re-run inference and make sure gene names are saved into the h5ad."
        )

    # --- Sanity check: confirm scale mismatch is resolved ---
    gen_X_raw = gen.X.toarray() if hasattr(gen.X, 'toarray') else np.asarray(gen.X)
    test_X_raw = test.X.toarray() if hasattr(test.X, 'toarray') else np.asarray(test.X)
    print(f"[pre-bin] Generated X range: {gen_X_raw.min():.3f} .. {gen_X_raw.max():.3f}")
    print(f"[pre-bin] Test X range:      {test_X_raw.min():.3f} .. {test_X_raw.max():.3f}")

    # Apply identical binning to BOTH sides. Generated is already binned
    # (idempotent for ints already in range); test is raw and must be
    # discretized to match.
    gen_binned = bin_expression(gen_X_raw, num_bins=num_bins)
    test_binned = bin_expression(test_X_raw, num_bins=num_bins)

    print(f"[post-bin] Generated X range: {gen_binned.min()} .. {gen_binned.max()}")
    print(f"[post-bin] Test X range:      {test_binned.min()} .. {test_binned.max()}")

    # Rebuild AnnData objects so the obs/var alignment stays correct when
    # we index by perturbation below.
    gen = sc.AnnData(X=gen_binned, obs=gen.obs.copy(), var=gen.var.copy())
    test = sc.AnnData(X=test_binned, obs=test.obs.copy(), var=test.var.copy())

    perturbations = gen.obs[pert_col].unique()
    results = {}

    for pert in perturbations:
        gen_cells = gen[gen.obs[pert_col] == pert].X
        test_cells = test[test.obs[pert_col] == pert].X

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
        generated_h5ad=sys.argv[1],         # inference_results/generated_cells.h5ad
        test_h5ad=sys.argv[2],              # datasets/replogle_k562_test.h5ad
        pert_col='perturbation',
        n_subsample=500,
        n_top_genes=50,
        num_bins=NUM_BINS,
    )
    # Summary
    mmds = [v['mmd'] for v in results.values()]
    w2s  = [v['w2']  for v in results.values()]
    print(f"\nMean MMD: {np.mean(mmds):.4f} ± {np.std(mmds):.4f}")
    print(f"Mean W2:  {np.mean(w2s):.4f} ± {np.std(w2s):.4f}")