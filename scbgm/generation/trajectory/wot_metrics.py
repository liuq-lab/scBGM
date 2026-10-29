#!/usr/bin/env python3
"""Shared WOT evaluation helpers: training-only representation, distribution metrics, DPT and markers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


TARGET_TO_TIME = {"D3.5": 1.0 / 3.0, "D4": 2.0 / 3.0}
MARKERS = ("Shisa8", "Prrx1", "Fut9")


def dense_float(matrix: Any) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().astype(np.float32)
    return np.asarray(matrix, dtype=np.float32)


def normalized_log_hvg(
    data: ad.AnnData,
    hvg_names: np.ndarray,
    already_normalized_log: bool,
) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(data.var_names.astype(str))}
    missing = [name for name in hvg_names if name not in lookup]
    if missing:
        raise RuntimeError(f"missing evaluation genes, first entries: {missing[:5]}")
    indices = np.asarray([lookup[name] for name in hvg_names], dtype=np.int64)
    selected = dense_float(data.X[:, indices])
    if already_normalized_log:
        return selected
    totals = np.asarray(data.X.sum(axis=1)).reshape(-1).astype(np.float64)
    if np.any(totals <= 0):
        raise RuntimeError("zero-library real cell in evaluation")
    return np.log1p(selected * (10_000.0 / totals)[:, None]).astype(np.float32)


def fit_train_only_representation(
    train_path: Path,
    genes: np.ndarray,
    max_cells: int,
    n_hvg: int,
    seed: int,
) -> tuple[np.ndarray, StandardScaler, PCA]:
    train = ad.read_h5ad(train_path)
    if "period" not in train.obs:
        raise RuntimeError("training H5AD lacks period")
    relevant = {"D1.5", "D2", "D2.5", "D3", "D4.5", "D5", "D5.5", "D6"}
    keep = np.flatnonzero(train.obs["period"].astype(str).isin(relevant).to_numpy())
    rng = np.random.default_rng(seed)
    if len(keep) > max_cells:
        keep = np.sort(rng.choice(keep, max_cells, replace=False))
    train = train[keep].copy()
    if not np.array_equal(train.var_names.astype(str), genes):
        train = train[:, genes].copy()
    sc.pp.normalize_total(train, target_sum=10_000.0)
    sc.pp.log1p(train)
    sc.pp.highly_variable_genes(train, n_top_genes=min(n_hvg, train.n_vars), flavor="seurat")
    hvg_names = train.var_names[train.var["highly_variable"].to_numpy()].astype(str).to_numpy()
    matrix = dense_float(train[:, hvg_names].X)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix).astype(np.float32)
    n_components = min(50, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(scaled)
    return hvg_names, scaler, pca


def transform_expression(
    data: ad.AnnData,
    hvg_names: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    already_normalized_log: bool,
) -> np.ndarray:
    values = normalized_log_hvg(data, hvg_names, already_normalized_log)
    return pca.transform(scaler.transform(values)).astype(np.float32)


def official_mmd_rbf(real: np.ndarray, generated: np.ndarray) -> float:
    if len(real) != len(generated):
        raise ValueError("MMD requires equal group sizes")
    total = np.concatenate([real, generated], axis=0).astype(np.float64)
    squared = (
        np.sum(total * total, axis=1)[:, None]
        + np.sum(total * total, axis=1)[None, :]
        - 2.0 * total @ total.T
    )
    np.maximum(squared, 0.0, out=squared)
    n_total = len(total)
    bandwidth = squared.sum() / max(n_total * n_total - n_total, 1)
    bandwidth = max(float(bandwidth), 1e-12) / (2.0 ** (5 // 2))
    kernel = sum(np.exp(-squared / (bandwidth * (2.0**i))) for i in range(5))
    n = len(real)
    return float(
        np.mean(
            kernel[:n, :n]
            + kernel[n:, n:]
            - kernel[:n, n:]
            - kernel[n:, :n]
        )
    )


def normalized_knn_ilisi(embedding: np.ndarray, batches: np.ndarray, k: int = 10) -> float:
    k = min(int(k), len(embedding) - 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(embedding)
    indices = neighbors.kneighbors(return_distance=False)[:, 1:]
    p_generated = batches[indices].mean(axis=1)
    inverse_simpson = 1.0 / (
        p_generated * p_generated + (1.0 - p_generated) ** 2
    )
    return float(np.mean(inverse_simpson - 1.0))


def pair_metrics(
    real: ad.AnnData,
    generated: ad.AnnData,
    hvg_names: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    n: int,
    seed: int,
) -> tuple[float, float, int]:
    count = min(int(n), real.n_obs, generated.n_obs)
    rng = np.random.default_rng(seed)
    real_idx = np.sort(rng.choice(real.n_obs, count, replace=False))
    generated_idx = np.sort(rng.choice(generated.n_obs, count, replace=False))
    real_pca = transform_expression(real[real_idx], hvg_names, scaler, pca, False)[:, :20]
    generated_pca = transform_expression(
        generated[generated_idx], hvg_names, scaler, pca, True
    )[:, :20]
    mmd = official_mmd_rbf(real_pca, generated_pca)
    joined = np.concatenate([real_pca, generated_pca], axis=0)
    batches = np.concatenate(
        [np.zeros(count, dtype=np.int8), np.ones(count, dtype=np.int8)]
    )
    return mmd, normalized_knn_ilisi(joined, batches, k=10), count


def make_generated_adata(
    expression: list[np.ndarray],
    counts: list[np.ndarray],
    times: list[float],
    genes: np.ndarray,
    method: str,
) -> ad.AnnData:
    n = int(expression[0].shape[0])
    obs_rows: list[dict[str, Any]] = []
    obs_names: list[str] = []
    for state_index, time in enumerate(times):
        for cell in range(n):
            obs_rows.append(
                {
                    "method": method,
                    "source": "generated",
                    "state_index": state_index,
                    "flow_time": float(time),
                    "state": f"t={time:.9f}",
                    "period": f"{method} t={time:.9f}",
                }
            )
            obs_names.append(f"{method}:t{time:.9f}:cell{cell}")
    result = ad.AnnData(
        X=sparse.csr_matrix(np.concatenate(expression, axis=0).astype(np.float32)),
        obs=pd.DataFrame(obs_rows, index=obs_names),
    )
    result.var_names = pd.Index(genes)
    result.layers["counts"] = sparse.csr_matrix(
        np.concatenate(counts, axis=0).astype(np.float32)
    )
    return result


def run_dpt(
    reference: ad.AnnData,
    generated: list[ad.AnnData],
    hvg_names: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    seed: int,
) -> tuple[ad.AnnData, pd.DataFrame]:
    periods = reference.obs["period"].astype(str)
    use_periods = {"D1.5", "D2", "D2.5", "D3", "D3.5", "D4", "D4.5", "D5", "D5.5", "D6"}
    reference = reference[periods.isin(use_periods).to_numpy()][::5].copy()
    real_pca = transform_expression(reference, hvg_names, scaler, pca, False)
    reference.obs["source"] = "real"
    reference.obs["method"] = "real"
    reference.obs["state"] = reference.obs["period"].astype(str).to_numpy()

    obs = [reference.obs.copy()]
    embeddings = [real_pca]
    for data in generated:
        obs.append(data.obs.copy())
        embeddings.append(transform_expression(data, hvg_names, scaler, pca, True))
    combined = ad.AnnData(
        X=np.zeros((sum(len(frame) for frame in obs), 1), dtype=np.float32),
        obs=pd.concat(obs, axis=0),
    )
    combined.obsm["X_pca"] = np.concatenate(embeddings, axis=0).astype(np.float32)
    sc.pp.neighbors(combined, n_neighbors=10, use_rep="X_pca", n_pcs=20, random_state=seed)
    sc.tl.umap(combined, random_state=seed)
    root = np.flatnonzero(
        (combined.obs["source"].astype(str).to_numpy() == "real")
        & (combined.obs["period"].astype(str).to_numpy() == "D1.5")
    )
    if len(root) == 0:
        raise RuntimeError("D1.5 root missing from DPT reference")
    combined.uns["iroot"] = int(root[0])
    sc.tl.diffmap(combined)
    sc.tl.dpt(combined)

    real_summary = (
        combined.obs.loc[combined.obs["source"].astype(str).eq("real")]
        .groupby("period", observed=True)["dpt_pseudotime"]
        .agg(["size", "mean", "median"])
        .reset_index()
        .rename(
            columns={
                "period": "state",
                "size": "n_cells",
                "mean": "mean_dpt",
                "median": "median_dpt",
            }
        )
    )
    real_summary.insert(0, "method", "real")
    real_summary["flow_time"] = np.nan
    generated_summary = (
        combined.obs.loc[combined.obs["source"].astype(str).eq("generated")]
        .groupby(["method", "flow_time"], observed=True)["dpt_pseudotime"]
        .agg(["size", "mean", "median"])
        .reset_index()
        .rename(
            columns={
                "size": "n_cells",
                "mean": "mean_dpt",
                "median": "median_dpt",
            }
        )
    )
    generated_summary["state"] = generated_summary["flow_time"].map(
        lambda value: f"t={float(value):.9f}"
    )
    return combined, pd.concat([real_summary, generated_summary], ignore_index=True)


def marker_table(reference: ad.AnnData, generated: list[ad.AnnData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    reference_names = {name.lower(): name for name in reference.var_names.astype(str)}
    for marker in MARKERS:
        key = marker.lower()
        if key not in reference_names:
            continue
        actual = reference_names[key]
        real_values = normalized_log_hvg(reference, np.asarray([actual]), False)[:, 0]
        periods = reference.obs["period"].astype(str).to_numpy()
        for period in ("D3", "D3.5", "D4", "D4.5"):
            mask = periods == period
            if np.any(mask):
                rows.append(
                    {
                        "method": "real",
                        "state": period,
                        "flow_time": TARGET_TO_TIME.get(period, 0.0 if period == "D3" else 1.0),
                        "gene": marker,
                        "mean_expression": float(np.mean(real_values[mask])),
                    }
                )
        for data in generated:
            generated_names = {name.lower(): name for name in data.var_names.astype(str)}
            if key not in generated_names:
                continue
            values = dense_float(data[:, [generated_names[key]]].X)[:, 0]
            for time in sorted(data.obs["flow_time"].unique()):
                mask = np.isclose(data.obs["flow_time"].to_numpy(dtype=float), float(time))
                rows.append(
                    {
                        "method": str(data.obs["method"].iloc[0]),
                        "state": f"t={float(time):.9f}",
                        "flow_time": float(time),
                        "gene": marker,
                        "mean_expression": float(np.mean(values[mask])),
                    }
                )
    return pd.DataFrame(rows)


def plot_markers(table: pd.DataFrame, destination: Path) -> None:
    if table.empty:
        return
    genes = list(table["gene"].drop_duplicates())
    fig, axes = plt.subplots(1, len(genes), figsize=(6 * len(genes), 5), squeeze=False)
    for ax, gene in zip(axes[0], genes):
        subset = table[(table["gene"] == gene) & (table["method"] != "real")]
        for method, group in subset.groupby("method", observed=True):
            group = group.sort_values("flow_time")
            ax.plot(group["flow_time"], group["mean_expression"], marker="o", label=method)
        real = table[(table["gene"] == gene) & (table["method"] == "real")]
        ax.scatter(real["flow_time"], real["mean_expression"], c="black", s=55, marker="x", label="real")
        ax.set_title(gene)
        ax.set_xlabel("Normalized flow time")
        ax.set_ylabel("Mean log-normalized expression")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)
