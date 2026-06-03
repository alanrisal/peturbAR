#!/usr/bin/env python3
"""
Visualization script for DCM perturbation prediction results.
Run on RunPod after inference + evaluation.

Usage:
    python scripts/visualize_results.py \
        --results     results.txt \
        --generated   /workspace/checkpoints/k562_crossattn/inference_results/generated_cells.h5ad \
        --test        /workspace/data/k562_essential_processed.h5ad \
        --outdir      /workspace/peturbAR/visuals
"""
import argparse
import os
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import sparse
from scipy.stats import gaussian_kde

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_results(path):
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            gene = parts[0].rstrip(':')
            mmd = float(parts[1].split('=')[1].rstrip(','))
            w2  = float(parts[2].split('=')[1])
            results[gene] = {'mmd': mmd, 'w2': w2}
    return results


def load_and_preprocess(gen_path, test_path, num_bins=51):
    gen  = sc.read_h5ad(gen_path)
    test = sc.read_h5ad(test_path)

    # Normalize test over full gene set BEFORE subsetting
    test_X = test.X.toarray() if sparse.issparse(test.X) else np.asarray(test.X, dtype=np.float32)
    cell_sums = test_X.sum(axis=1, keepdims=True)
    cell_sums = np.where(cell_sums == 0, 1, cell_sums)
    test_X = np.log1p(test_X / cell_sums * 1e4)
    test = sc.AnnData(X=test_X, obs=test.obs.copy(), var=test.var.copy())

    # Gene alignment
    if gen.n_vars != test.n_vars:
        shared = [g for g in gen.var_names if g in test.var_names]
        gen  = gen[:, shared]
        test = test[:, shared]

    # Bin test
    test_X_sub = test.X.toarray() if sparse.issparse(test.X) else np.asarray(test.X)
    test_binned = np.clip(np.round(test_X_sub).astype(np.int32), 0, num_bins - 1)
    test = sc.AnnData(X=test_binned, obs=test.obs.copy(), var=test.var.copy())

    gen_X = gen.X.toarray() if sparse.issparse(gen.X) else np.asarray(gen.X)
    gen = sc.AnnData(X=gen_X.astype(np.int32), obs=gen.obs.copy(), var=gen.var.copy())

    return gen, test


def get_cells(gen, test, pert, num_bins=51):
    g = gen[gen.obs['perturbation'] == pert].X
    t = test[test.obs['gene'] == pert].X
    if sparse.issparse(g): g = g.toarray()
    if sparse.issparse(t): t = t.toarray()
    return g.astype(float), t.astype(float)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_metric_distributions(results, outdir):
    """Histogram + violin of W2 and MMD across all perturbations."""
    w2s  = [v['w2']  for v in results.values()]
    mmds = [v['mmd'] for v in results.values()]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Metric Distributions Across 205 Held-Out Perturbations', fontsize=14, fontweight='bold')

    # W2 histogram
    axes[0,0].hist(w2s, bins=30, color='steelblue', alpha=0.8, edgecolor='white')
    axes[0,0].axvline(np.mean(w2s), color='red', lw=2, linestyle='--', label=f'Mean={np.mean(w2s):.2f}')
    axes[0,0].axvline(7.0, color='orange', lw=2, linestyle=':', label='Paper baseline=7.0')
    axes[0,0].set_xlabel('W2 Distance')
    axes[0,0].set_ylabel('Count')
    axes[0,0].set_title('W2 Distribution')
    axes[0,0].legend(fontsize=9)

    # MMD histogram
    axes[0,1].hist(mmds, bins=30, color='salmon', alpha=0.8, edgecolor='white')
    axes[0,1].axvline(np.mean(mmds), color='red', lw=2, linestyle='--', label=f'Mean={np.mean(mmds):.3f}')
    axes[0,1].axvline(0.6, color='orange', lw=2, linestyle=':', label='Paper baseline=0.60')
    axes[0,1].set_xlabel('MMD')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('MMD Distribution')
    axes[0,1].legend(fontsize=9)

    # W2 violin
    axes[1,0].violinplot(w2s, showmedians=True)
    axes[1,0].axhline(7.0, color='orange', lw=2, linestyle=':', label='Paper baseline')
    axes[1,0].set_xticks([])
    axes[1,0].set_ylabel('W2 Distance')
    axes[1,0].set_title('W2 Spread')
    axes[1,0].legend(fontsize=9)

    # MMD violin
    axes[1,1].violinplot(mmds, showmedians=True)
    axes[1,1].axhline(0.6, color='orange', lw=2, linestyle=':', label='Paper baseline')
    axes[1,1].set_xticks([])
    axes[1,1].set_ylabel('MMD')
    axes[1,1].set_title('MMD Spread')
    axes[1,1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / '1_metric_distributions.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: 1_metric_distributions.png")


def plot_w2_mmd_scatter(results, outdir):
    """W2 vs MMD scatter — shows correlation and flags outliers."""
    names = list(results.keys())
    w2s   = [results[n]['w2']  for n in names]
    mmds  = [results[n]['mmd'] for n in names]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(w2s, mmds, alpha=0.6, c=np.array(w2s), cmap='coolwarm', s=40, zorder=2)
    plt.colorbar(scatter, ax=ax, label='W2 Distance')

    # Label top 5 worst and top 5 best by W2
    sorted_by_w2 = sorted(zip(w2s, mmds, names), key=lambda x: x[0])
    for w, m, n in sorted_by_w2[:5]:    # best
        ax.annotate(n, (w, m), fontsize=7, color='green',
                    xytext=(4, 4), textcoords='offset points')
    for w, m, n in sorted_by_w2[-5:]:   # worst
        ax.annotate(n, (w, m), fontsize=7, color='red',
                    xytext=(4, 4), textcoords='offset points')

    # Reference lines for paper baseline
    ax.axvline(7.0,  color='orange', lw=1.5, linestyle='--', label='Paper W2=7.0')
    ax.axhline(0.60, color='purple', lw=1.5, linestyle='--', label='Paper MMD=0.60')
    ax.set_xlabel('W2 Distance (lower = better)')
    ax.set_ylabel('MMD (lower = better)')
    ax.set_title('W2 vs MMD per Held-Out Perturbation\n(green=best, red=worst by W2)')
    ax.legend(fontsize=9)

    corr = np.corrcoef(w2s, mmds)[0, 1]
    ax.text(0.03, 0.97, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(outdir / '2_w2_vs_mmd_scatter.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: 2_w2_vs_mmd_scatter.png")


def plot_ranked_perturbations(results, outdir):
    """Ranked bar charts — which perturbations are best/worst predicted."""
    sorted_by_w2 = sorted(results.items(), key=lambda x: x[1]['w2'])
    names = [n for n, _ in sorted_by_w2]
    w2s   = [v['w2']  for _, v in sorted_by_w2]
    mmds  = [v['mmd'] for _, v in sorted_by_w2]

    n = len(names)
    colors_w2  = ['#2ecc71' if w < np.percentile(w2s, 25) else
                  '#e74c3c' if w > np.percentile(w2s, 75) else
                  '#3498db' for w in w2s]
    colors_mmd = ['#2ecc71' if m < np.percentile(mmds, 25) else
                  '#e74c3c' if m > np.percentile(mmds, 75) else
                  '#3498db' for m in mmds]

    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    fig.suptitle('Per-Perturbation Performance (All 205 Held-Out Genes, Ranked by W2)',
                 fontsize=13, fontweight='bold')

    axes[0].bar(range(n), w2s, color=colors_w2, alpha=0.85)
    axes[0].axhline(7.0, color='orange', lw=2, linestyle='--', label='Paper baseline W2=7.0')
    axes[0].axhline(np.mean(w2s), color='red', lw=1.5, linestyle=':', label=f'Your mean={np.mean(w2s):.2f}')
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(names, rotation=90, fontsize=5.5)
    axes[0].set_ylabel('W2 Distance')
    axes[0].set_title('W2 per Perturbation (green=top quartile, red=bottom quartile)')
    axes[0].legend(fontsize=9)

    sorted_by_mmd = sorted(zip(mmds, names, colors_mmd))
    mmds_s = [m for m, _, _ in sorted_by_mmd]
    names_s = [n for _, n, _ in sorted_by_mmd]
    cols_s  = [c for _, _, c in sorted_by_mmd]

    axes[1].bar(range(n), mmds_s, color=cols_s, alpha=0.85)
    axes[1].axhline(0.6, color='orange', lw=2, linestyle='--', label='Paper baseline MMD=0.60')
    axes[1].axhline(np.mean(mmds), color='red', lw=1.5, linestyle=':', label=f'Your mean={np.mean(mmds):.3f}')
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(names_s, rotation=90, fontsize=5.5)
    axes[1].set_ylabel('MMD')
    axes[1].set_title('MMD per Perturbation (ranked separately)')
    axes[1].legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(outdir / '3_ranked_perturbations.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: 3_ranked_perturbations.png")


def plot_overlaid_distributions(results, gen, test, outdir, num_bins=51):
    """Overlaid expression histograms for best, median, and worst perturbations."""
    sorted_by_w2 = sorted(results.items(), key=lambda x: x[1]['w2'])
    n = len(sorted_by_w2)
    picks = {
        'Best (lowest W2)':    sorted_by_w2[0][0],
        '25th percentile':     sorted_by_w2[n // 4][0],
        'Median':              sorted_by_w2[n // 2][0],
        '75th percentile':     sorted_by_w2[3 * n // 4][0],
        'Worst (highest W2)':  sorted_by_w2[-1][0],
    }

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    fig.suptitle('Overlaid Expression Distributions: Generated vs Real\n'
                 '(All held-out perturbations — model never saw these during training)',
                 fontsize=12, fontweight='bold')

    bins = np.arange(0, num_bins + 1) - 0.5
    for ax, (label, pert) in zip(axes, picks.items()):
        g_cells, t_cells = get_cells(gen, test, pert, num_bins)
        if len(g_cells) == 0 or len(t_cells) == 0:
            ax.set_title(f'{label}\n{pert}\n(no data)')
            continue

        g_flat = g_cells.flatten()
        t_flat = t_cells.flatten()

        ax.hist(t_flat, bins=bins, density=True, alpha=0.55, color='steelblue',
                label='Real', histtype='stepfilled')
        ax.hist(g_flat, bins=bins, density=True, alpha=0.55, color='salmon',
                label='Generated', histtype='stepfilled')
        ax.hist(t_flat, bins=bins, density=True, color='steelblue',
                histtype='step', lw=1.5)
        ax.hist(g_flat, bins=bins, density=True, color='red',
                histtype='step', lw=1.5)

        w2  = results[pert]['w2']
        mmd = results[pert]['mmd']
        ax.set_title(f'{label}\n{pert}\nW2={w2:.2f}, MMD={mmd:.3f}', fontsize=9)
        ax.set_xlabel('Expression Bin')
        ax.set_xlim(-0.5, 12)  # zoom in on real data range [0-9]
        if ax == axes[0]:
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / '4_overlaid_distributions.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: 4_overlaid_distributions.png")


def plot_mean_expression_profile(results, gen, test, outdir):
    """Mean expression per gene: generated vs real, for best and worst perturbations."""
    sorted_by_w2 = sorted(results.items(), key=lambda x: x[1]['w2'])
    picks = [
        ('Best predicted', sorted_by_w2[0][0]),
        ('2nd best',       sorted_by_w2[1][0]),
        ('Worst predicted', sorted_by_w2[-1][0]),
        ('2nd worst',      sorted_by_w2[-2][0]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mean Gene Expression Profile: Generated vs Real\n'
                 '(each dot = one of 2000 genes; diagonal = perfect prediction)',
                 fontsize=12, fontweight='bold')

    for ax, (label, pert) in zip(axes.flatten(), picks):
        g_cells, t_cells = get_cells(gen, test, pert)
        if len(g_cells) == 0 or len(t_cells) == 0:
            ax.set_title(f'{label}: {pert} (no data)')
            continue

        gen_mean  = g_cells.mean(axis=0)
        test_mean = t_cells.mean(axis=0)

        corr = np.corrcoef(gen_mean, test_mean)[0, 1]
        ax.scatter(test_mean, gen_mean, alpha=0.15, s=5, color='steelblue')

        lim = max(test_mean.max(), gen_mean.max()) * 1.05
        ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='y=x (perfect)')
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel('Real mean expression bin')
        ax.set_ylabel('Generated mean expression bin')
        ax.set_title(f'{label}: {pert}\n'
                     f'W2={results[pert]["w2"]:.2f}, Pearson r={corr:.3f}')
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / '5_mean_expression_scatter.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: 5_mean_expression_scatter.png")


def plot_sparsity_comparison(results, gen, test, outdir):
    """Fraction of zero-expression genes: generated vs real per perturbation."""
    perts_in_both = list(results.keys())
    gen_sparsity  = []
    test_sparsity = []
    names = []

    for pert in perts_in_both:
        g, t = get_cells(gen, test, pert)
        if len(g) == 0 or len(t) == 0:
            continue
        gen_sparsity.append((g == 0).mean())
        test_sparsity.append((t == 0).mean())
        names.append(pert)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(test_sparsity, gen_sparsity, alpha=0.5, s=30, color='steelblue')

    lim = max(max(test_sparsity), max(gen_sparsity)) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='y=x (perfect match)')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Real sparsity (fraction of genes at bin 0)')
    ax.set_ylabel('Generated sparsity')
    ax.set_title('Sparsity Comparison: Generated vs Real\n'
                 '(each dot = one held-out perturbation)')
    ax.legend(fontsize=9)

    corr = np.corrcoef(test_sparsity, gen_sparsity)[0, 1]
    ax.text(0.03, 0.97, f'Pearson r = {corr:.3f}', transform=ax.transAxes,
            fontsize=10, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(outdir / '6_sparsity_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved: 6_sparsity_comparison.png")


def plot_umap_embedding(results, gen, test, outdir, n_cells_per_pert=75, num_bins=51):
    """
    3-panel UMAP figure:
      Panel 1 — All cells, colored by source (generated=red, real=blue).
                 Good MMD → point clouds overlap.
      Panel 2 — Same embedding, cells colored by how well their perturbation
                 was predicted (W2 score). Shows whether hard/easy perturbations
                 cluster in latent space.
      Panel 3 — 2D kernel density contour overlay: generated (red) vs real (blue)
                 contours on the same UMAP coordinates. Overlapping rings = good fit.
    """
    import pandas as pd

    print("\nBuilding UMAP embedding...")
    print(f"  Subsampling to {n_cells_per_pert} cells per perturbation per source...")

    # ── Gather cells ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    gen_Xs, real_Xs = [], []
    meta_source, meta_pert, meta_w2 = [], [], []

    for pert, metrics in results.items():
        g_mask = gen.obs['perturbation'] == pert
        t_mask = test.obs['gene'] == pert

        g_cells = gen[g_mask].X
        t_cells = test[t_mask].X
        if sparse.issparse(g_cells): g_cells = g_cells.toarray()
        if sparse.issparse(t_cells): t_cells = t_cells.toarray()

        if len(g_cells) == 0 or len(t_cells) == 0:
            continue

        n_g = min(n_cells_per_pert, len(g_cells))
        n_t = min(n_cells_per_pert, len(t_cells))
        g_idx = rng.choice(len(g_cells), n_g, replace=False)
        t_idx = rng.choice(len(t_cells), n_t, replace=False)

        gen_Xs.append(g_cells[g_idx])
        real_Xs.append(t_cells[t_idx])
        meta_source.extend(['Generated'] * n_g + ['Real'] * n_t)
        meta_pert.extend([pert] * (n_g + n_t))
        meta_w2.extend([metrics['w2']] * (n_g + n_t))

    X_all = np.vstack(gen_Xs + real_Xs).astype(np.float32)
    meta  = pd.DataFrame({
        'source': meta_source,
        'perturbation': meta_pert,
        'w2': meta_w2,
    })
    print(f"  Total cells in UMAP: {len(X_all):,} "
          f"({(meta.source=='Generated').sum():,} gen, {(meta.source=='Real').sum():,} real)")

    # ── PCA → neighbours → UMAP via scanpy ───────────────────────────────────
    adata_umap = sc.AnnData(X=X_all, obs=meta)
    sc.pp.pca(adata_umap, n_comps=50, svd_solver='randomized')
    sc.pp.neighbors(adata_umap, n_neighbors=15, n_pcs=50)
    sc.tl.umap(adata_umap, min_dist=0.3)
    coords = adata_umap.obsm['X_umap']
    print("  UMAP done.")

    is_gen  = meta['source'].values == 'Generated'
    is_real = ~is_gen
    w2_vals = meta['w2'].values

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(21, 7))
    gs  = gridspec.GridSpec(1, 3, wspace=0.08)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    fig.suptitle(
        'UMAP of Generated vs Real Cells Across 205 Held-Out Perturbations\n'
        'PCA(50) → UMAP | model never saw any of these perturbations during training',
        fontsize=13, fontweight='bold', y=1.01
    )

    # ── Panel 1: source coloring ──────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(coords[is_real,  0], coords[is_real,  1],
               c='steelblue', s=2, alpha=0.25, rasterized=True, label='Real')
    ax.scatter(coords[is_gen,   0], coords[is_gen,   1],
               c='salmon',    s=2, alpha=0.25, rasterized=True, label='Generated')
    legend_els = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Real'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='salmon',    markersize=8, label='Generated'),
    ]
    ax.legend(handles=legend_els, fontsize=10, loc='upper right')
    ax.set_title('Panel 1 — Source\nOverlap = low MMD = good fit', fontsize=11)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 2: W2-score coloring ────────────────────────────────────────────
    ax = axes[1]
    sc2 = ax.scatter(coords[:, 0], coords[:, 1],
                     c=w2_vals, cmap='RdYlGn_r',
                     s=2, alpha=0.3, rasterized=True,
                     vmin=np.percentile(w2_vals, 5),
                     vmax=np.percentile(w2_vals, 95))
    cb = plt.colorbar(sc2, ax=ax, pad=0.02, shrink=0.85)
    cb.set_label('W2 of this perturbation\n(green=well predicted, red=hard)', fontsize=9)
    ax.set_title('Panel 2 — Prediction Difficulty\nDo hard perturbations cluster?', fontsize=11)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 3: KDE density contours ────────────────────────────────────────
    ax = axes[2]
    ax.scatter(coords[is_real, 0], coords[is_real, 1],
               c='steelblue', s=1, alpha=0.08, rasterized=True)
    ax.scatter(coords[is_gen, 0], coords[is_gen, 1],
               c='salmon', s=1, alpha=0.08, rasterized=True)

    xmin, xmax = coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5
    ymin, ymax = coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5
    xx, yy = np.mgrid[xmin:xmax:150j, ymin:ymax:150j]
    grid_pts = np.vstack([xx.ravel(), yy.ravel()])

    for subset, color, label in [
        (is_real, 'steelblue', 'Real'),
        (is_gen,  'red',       'Generated'),
    ]:
        pts = coords[subset].T
        if pts.shape[1] < 3:
            continue
        kde = gaussian_kde(pts, bw_method=0.25)
        z   = kde(grid_pts).reshape(xx.shape)
        # filled low-alpha background + crisp contour lines
        ax.contourf(xx, yy, z, levels=5, colors=[color], alpha=0.12)
        ax.contour( xx, yy, z, levels=5, colors=[color], linewidths=1.4, alpha=0.85)

    legend_els = [
        Line2D([0],[0], color='steelblue', lw=2, label='Real density'),
        Line2D([0],[0], color='red',       lw=2, label='Generated density'),
    ]
    ax.legend(handles=legend_els, fontsize=10, loc='upper right')
    ax.set_title('Panel 3 — Density Contours\nOverlapping rings = distributions match', fontsize=11)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    ax.set_xticks([]); ax.set_yticks([])

    fig.savefig(outdir / '7_umap_embedding.png', bbox_inches='tight', dpi=180)
    plt.close(fig)
    print("Saved: 7_umap_embedding.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results',   default='results.txt')
    parser.add_argument('--generated', default='/workspace/checkpoints/k562_crossattn/inference_results/generated_cells.h5ad')
    parser.add_argument('--test',      default='/workspace/data/k562_essential_processed.h5ad')
    parser.add_argument('--outdir',    default='/workspace/peturbAR/visuals')
    parser.add_argument('--num_bins',  type=int, default=51)
    parser.add_argument('--umap-only', action='store_true',
                        help='Skip the other plots and only generate the UMAP panel.')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving visuals to: {outdir}\n")

    print("Parsing results...")
    results = parse_results(args.results)
    print(f"  {len(results)} perturbations loaded\n")

    print("Loading h5ad files (test normalization takes ~30s)...")
    gen, test = load_and_preprocess(args.generated, args.test, args.num_bins)
    print(f"  Generated: {gen.shape}, Test: {test.shape}\n")

    if not vars(args).get('umap_only', False):
        print("Generating metric plots...")
        plot_metric_distributions(results, outdir)
        plot_w2_mmd_scatter(results, outdir)
        plot_ranked_perturbations(results, outdir)
        plot_overlaid_distributions(results, gen, test, outdir, args.num_bins)
        plot_mean_expression_profile(results, gen, test, outdir)
        plot_sparsity_comparison(results, gen, test, outdir)

    # UMAP is the most expensive — run last so other plots are already saved
    print("\nGenerating UMAP (PCA + UMAP fitting, ~2-5 min)...")
    plot_umap_embedding(results, gen, test, outdir, num_bins=args.num_bins)

    print(f"\nAll done. {len(list(outdir.glob('*.png')))} PNGs saved to {outdir}")


if __name__ == '__main__':
    main()
