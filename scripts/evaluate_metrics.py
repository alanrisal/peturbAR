import numpy as np
import scanpy as sc
import torch
from scipy.spatial.distance import cdist
import ot  # pip install POT
from sklearn.metrics.pairwise import rbf_kernel


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
            n_subsample=1000, n_top_genes=50):
    gen = sc.read_h5ad(generated_h5ad)
    test = sc.read_h5ad(test_h5ad)

    perturbations = gen.obs[pert_col].unique()
    results = {}

    for pert in perturbations:
        gen_cells = gen[gen.obs[pert_col] == pert].X
        test_cells = test[test.obs[pert_col] == pert].X

        if hasattr(gen_cells, 'toarray'):
            gen_cells = gen_cells.toarray()
        if hasattr(test_cells, 'toarray'):
            test_cells = test_cells.toarray()

        # Subsample for tractability
        gen_cells = gen_cells[np.random.choice(len(gen_cells),
                            min(n_subsample, len(gen_cells)),
replace=False)]
        test_cells = test_cells[np.random.choice(len(test_cells),
                                min(n_subsample, len(test_cells)),
replace=False)]

        # Use top variable genes to reduce dimensionality before W2
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
        generated_h5ad=sys.argv[1],         #inference_results/generated_cells.h5ad
        test_h5ad=sys.argv[2],              #datasets/replogle_k562_test.h5ad
        pert_col='perturbation',
        n_subsample=500,
        n_top_genes=50
    )
    # Summary
    mmds = [v['mmd'] for v in results.values()]
    w2s  = [v['w2']  for v in results.values()]
    print(f"\nMean MMD: {np.mean(mmds):.4f} ± {np.std(mmds):.4f}")
    print(f"Mean W2:  {np.mean(w2s):.4f} ± {np.std(w2s):.4f}")