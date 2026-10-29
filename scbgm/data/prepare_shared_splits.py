#!/usr/bin/env python3
"""Create leakage-safe shared train/validation/test AnnData splits.

The split protocol is fixed to:

* outer test fraction: 20%;
* inner validation fraction: 10% of the remaining 80% (8% overall);
* gradient-training fraction: 72%;
* seed: 17;
* cell filtering: at least 10 detected genes;
* gene filtering: detected in at least 3 gradient-training cells.

All gene filtering is learned from gradient-training cells only. The source
H5AD is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


TRAIN_FRACTION = 0.72
VALIDATION_FRACTION = 0.08
TEST_FRACTION = 0.20
INNER_VALIDATION_FRACTION = (
    VALIDATION_FRACTION / (TRAIN_FRACTION + VALIDATION_FRACTION)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-genes", type=int, default=10)
    parser.add_argument("--min-cells", type=int, default=3)
    return parser.parse_args()


def load_dataset_row(path: Path, task_index: int) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not 0 <= task_index < len(rows):
        raise IndexError(
            f"task-index {task_index} is outside 0..{len(rows) - 1}"
        )
    required = {
        "dataset",
        "source_path",
        "matrix_source",
        "source_space",
        "stratify_columns",
    }
    missing = required - set(rows[task_index])
    if missing:
        raise ValueError(f"datasets.tsv is missing columns: {sorted(missing)}")
    return rows[task_index]


def sha256_lines(values) -> str:
    payload = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def as_csr_float32(matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        result = matrix.tocsr().astype(np.float32)
    else:
        result = sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))
    result.eliminate_zeros()
    if result.data.size:
        if not np.isfinite(result.data).all():
            raise ValueError("Expression matrix contains non-finite values")
        if float(result.data.min()) < 0:
            raise ValueError("Expression matrix contains negative values")
    return result


def choose_matrix(adata: ad.AnnData, matrix_source: str):
    if matrix_source == "X":
        matrix = adata.X
    elif matrix_source.startswith("layer:"):
        layer = matrix_source.split(":", 1)[1]
        if layer not in adata.layers:
            raise KeyError(f"Missing AnnData layer: {layer}")
        matrix = adata.layers[layer]
    else:
        raise ValueError(f"Unknown matrix_source: {matrix_source}")
    return as_csr_float32(matrix)


def validate_source_space(
    matrix: sparse.csr_matrix,
    source_space: str,
) -> dict[str, float]:
    sample = matrix[: min(1_000, matrix.shape[0])]
    values = sample.data
    cell_sums = np.asarray(sample.sum(axis=1)).ravel()
    nonzero = values[values != 0]
    integer_fraction = (
        float(np.mean(np.isclose(nonzero, np.round(nonzero))))
        if len(nonzero)
        else 1.0
    )
    stats = {
        "sample_nonzero_min": float(nonzero.min()) if len(nonzero) else 0.0,
        "sample_nonzero_max": float(nonzero.max()) if len(nonzero) else 0.0,
        "sample_integer_fraction": integer_fraction,
        "sample_cell_sum_min": float(cell_sums.min()),
        "sample_cell_sum_max": float(cell_sums.max()),
        "sample_cell_sum_mean": float(cell_sums.mean()),
    }
    if source_space == "raw_counts":
        if integer_fraction < 0.999:
            raise ValueError(
                "Configured raw_counts matrix is not integer-like: "
                f"{integer_fraction:.6f}"
            )
    elif source_space == "normalized_linear_1e4":
        if not np.allclose(cell_sums, 1e4, rtol=5e-4, atol=5.0):
            raise ValueError(
                "Configured normalized_linear_1e4 matrix does not have "
                "approximately 10,000 counts per sampled cell"
            )
    else:
        raise ValueError(f"Unknown source_space: {source_space}")
    return stats


def stable_cell_ids(dataset: str, original_names: np.ndarray) -> pd.Index:
    width = max(6, len(str(max(len(original_names) - 1, 0))))
    return pd.Index(
        [
            f"{dataset}__{index:0{width}d}__{name}"
            for index, name in enumerate(original_names)
        ],
        dtype=str,
    )


def make_strata(
    obs: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    missing = [column for column in columns if column not in obs.columns]
    if missing:
        raise KeyError(f"Missing stratification columns: {missing}")
    parts = [
        obs[column].astype("string").fillna("<NA>")
        for column in columns
    ]
    strata = parts[0]
    for part in parts[1:]:
        strata = strata.str.cat(part, sep="||")
    counts = strata.value_counts()
    too_small = counts[counts < 3]
    if len(too_small):
        raise ValueError(
            "Every stratum needs at least 3 cells for the two-stage split; "
            f"small strata={too_small.to_dict()}"
        )
    return strata.astype(str)


def split_indices(
    strata: pd.Series,
    seed: int,
) -> dict[str, np.ndarray]:
    all_indices = np.arange(len(strata))
    train_validation, test = train_test_split(
        all_indices,
        test_size=TEST_FRACTION,
        random_state=seed,
        shuffle=True,
        stratify=strata.to_numpy(),
    )
    train, validation = train_test_split(
        train_validation,
        test_size=INNER_VALIDATION_FRACTION,
        random_state=seed + 1,
        shuffle=True,
        stratify=strata.iloc[train_validation].to_numpy(),
    )
    result = {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }
    sets = {name: set(values.tolist()) for name, values in result.items()}
    if (
        sets["train"] & sets["validation"]
        or sets["train"] & sets["test"]
        or sets["validation"] & sets["test"]
    ):
        raise RuntimeError("Split overlap detected")
    if len(set.union(*sets.values())) != len(strata):
        raise RuntimeError("Split union does not cover every retained cell")
    return result


def train_gene_mask(
    matrix: sparse.csr_matrix,
    train_indices: np.ndarray,
    min_cells: int,
) -> np.ndarray:
    train_matrix = matrix[train_indices]
    detected = np.asarray((train_matrix > 0).sum(axis=0)).ravel()
    keep = detected >= min_cells
    if not keep.any():
        raise RuntimeError("Train-only gene filtering removed every gene")
    return keep


def expected_outputs(output_dir: Path, dataset: str) -> list[Path]:
    return [
        output_dir / f"{dataset}_train_source.h5ad",
        output_dir / f"{dataset}_validation_source.h5ad",
        output_dir / f"{dataset}_test_source.h5ad",
        output_dir / "genes.npy",
        output_dir / "cell_split.csv",
        output_dir / "split_manifest.json",
    ]


def main() -> None:
    args = parse_args()
    row = load_dataset_row(args.datasets, args.task_index)
    dataset = row["dataset"]
    source_path = Path(row["source_path"]).expanduser()
    output_dir = args.output_root / dataset / (
        f"outer80_inner10_seed{args.seed}"
    )
    outputs = expected_outputs(output_dir, dataset)

    if all(path.exists() for path in outputs):
        print("Completed split already exists; skipping:", output_dir)
        return
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset:", dataset)
    print("Source:", source_path)
    print("Matrix source:", row["matrix_source"])
    print("Source space:", row["source_space"])
    print("Output:", output_dir)

    adata = ad.read_h5ad(source_path)
    original_shape = tuple(map(int, adata.shape))
    original_names = adata.obs_names.astype(str).to_numpy()
    adata.obs["original_obs_name"] = original_names
    adata.obs_names = stable_cell_ids(dataset, original_names)
    adata.var_names_make_unique()

    matrix = choose_matrix(adata, row["matrix_source"])
    source_stats = validate_source_space(matrix, row["source_space"])
    adata.X = matrix
    for layer in list(adata.layers.keys()):
        del adata.layers[layer]
    adata.raw = None

    detected_genes = np.diff(matrix.indptr)
    keep_cells = detected_genes >= args.min_genes
    if not keep_cells.any():
        raise RuntimeError("Cell filtering removed every cell")
    removed_cells = int((~keep_cells).sum())
    if removed_cells:
        adata = adata[keep_cells].copy()
        matrix = adata.X.tocsr()

    stratify_columns = [
        item.strip()
        for item in row["stratify_columns"].split(",")
        if item.strip()
    ]
    strata = make_strata(adata.obs, stratify_columns)
    adata.obs["split_stratum"] = pd.Categorical(strata)
    splits = split_indices(strata, args.seed)

    keep_genes = train_gene_mask(
        matrix,
        splits["train"],
        args.min_cells,
    )
    removed_genes = int((~keep_genes).sum())
    adata = adata[:, keep_genes].copy()
    genes = np.asarray(adata.var_names.astype(str), dtype=str)
    np.save(output_dir / "genes.npy", genes, allow_pickle=False)

    split_labels = np.full(adata.n_obs, "", dtype=object)
    for split_name, indices in splits.items():
        split_labels[indices] = split_name
    adata.obs["split"] = pd.Categorical(
        split_labels,
        categories=["train", "validation", "test"],
    )

    split_table = pd.DataFrame(
        {
            "dataset": dataset,
            "barcode": adata.obs_names.astype(str),
            "original_obs_name": adata.obs[
                "original_obs_name"
            ].astype(str),
            "split": split_labels,
            "stratum": strata.to_numpy(),
        }
    )
    split_table.to_csv(output_dir / "cell_split.csv", index=False)

    for split_name, indices in splits.items():
        destination = (
            output_dir / f"{dataset}_{split_name}_source.h5ad"
        )
        subset = adata[indices].copy()
        subset.write_h5ad(destination, compression="gzip")
        print(split_name, subset.shape, "->", destination)
        del subset

    manifest = {
        "dataset": dataset,
        "protocol": "outer80_inner10",
        "seed": int(args.seed),
        "source_path": str(source_path),
        "matrix_source": row["matrix_source"],
        "source_space": row["source_space"],
        "stratify_columns": stratify_columns,
        "fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "filters": {
            "min_genes_per_cell": int(args.min_genes),
            "min_cells_per_gene_train_only": int(args.min_cells),
        },
        "original_shape": list(original_shape),
        "filtered_shape": [int(adata.n_obs), int(adata.n_vars)],
        "removed_cells": removed_cells,
        "removed_genes": removed_genes,
        "n_cells": {
            name: int(len(indices))
            for name, indices in splits.items()
        },
        "n_genes": int(len(genes)),
        "gene_order_sha256": sha256_lines(genes),
        "cell_order_sha256": sha256_lines(
            adata.obs_names.astype(str)
        ),
        "final_test_loaded_during_training": False,
        "gene_filter_fit_on": "gradient-training cells only",
        "source_validation": source_stats,
        "files": {
            name: f"{dataset}_{name}_source.h5ad"
            for name in ("train", "validation", "test")
        },
    }
    with (output_dir / "split_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    if not all(path.exists() for path in outputs):
        raise RuntimeError("One or more expected split outputs are missing")
    print("Shared split completed:", output_dir)
    print(json.dumps(manifest["n_cells"], indent=2))
    print("Genes:", manifest["n_genes"])
    print("Gene hash:", manifest["gene_order_sha256"])


if __name__ == "__main__":
    main()
