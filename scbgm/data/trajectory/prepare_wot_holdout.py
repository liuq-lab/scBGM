#!/usr/bin/env python3
"""Materialize a leakage-free WOT D3.5/D4 hold-out experiment.

TRAIN and VALIDATION have D3.5 and D4 removed.  The untouched cells from all
three original splits are combined only into evaluation files.  No evaluation
file is consumed by either the NB-AE or conditional scBGM trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


HOLDOUT = ("D3.5", "D4")
REFERENCE_PERIODS = (
    "D1.5",
    "D2",
    "D2.5",
    "D3",
    "D3.5",
    "D4",
    "D4.5",
    "D5",
    "D5.5",
    "D6",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-key", default="period")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_lines(values: pd.Index | np.ndarray) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def load(path: Path, label_key: str) -> ad.AnnData:
    data = ad.read_h5ad(path)
    if label_key not in data.obs:
        raise KeyError(f"{path} has no obs[{label_key!r}]")
    if data.obs[label_key].isna().any():
        raise ValueError(f"{path} contains missing {label_key!r}")
    return data


def tagged_subset(
    data: ad.AnnData,
    mask: np.ndarray,
    split: str,
) -> ad.AnnData:
    result = data[mask].copy()
    result.obs["_original_obs_name"] = result.obs_names.astype(str)
    result.obs["_source_split"] = split
    result.obs_names = pd.Index(
        [f"{split}:{name}" for name in result.obs_names.astype(str)]
    )
    return result


def atomic_h5ad(data: ad.AnnData, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    data.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source = args.source_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "PREPARE_COMPLETE.json"
    if complete.is_file() and not args.overwrite:
        print(f"Already complete: {complete}")
        return

    paths = {
        "train": source / "WOT_train_source.h5ad",
        "validation": source / "WOT_validation_source.h5ad",
        "test": source / "WOT_test_source.h5ad",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    data_by_split: dict[str, ad.AnnData] = {}
    genes: np.ndarray | None = None
    counts: dict[str, dict[str, int]] = {}
    for split, path in paths.items():
        print(f"Loading {split}: {path}", flush=True)
        data = load(path, args.label_key)
        current_genes = data.var_names.astype(str).to_numpy()
        if genes is None:
            genes = current_genes
        elif not np.array_equal(genes, current_genes):
            raise RuntimeError(f"Gene order differs in {split}")
        labels = data.obs[args.label_key].astype(str)
        counts[split] = labels.value_counts().sort_index().astype(int).to_dict()
        data_by_split[split] = data

    train_labels = data_by_split["train"].obs[args.label_key].astype(str).to_numpy()
    validation_labels = (
        data_by_split["validation"].obs[args.label_key].astype(str).to_numpy()
    )
    filtered = {
        "train": data_by_split["train"][~np.isin(train_labels, HOLDOUT)].copy(),
        "validation": data_by_split["validation"][~np.isin(validation_labels, HOLDOUT)].copy(),
    }
    for split, data in filtered.items():
        leaked = sorted(set(data.obs[args.label_key].astype(str)) & set(HOLDOUT))
        if leaked:
            raise RuntimeError(f"Hold-out leakage in {split}: {leaked}")
        path = output / f"WOT_{split}_no_D3.5_D4.h5ad"
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
        print(f"Writing {path} shape={data.shape}", flush=True)
        atomic_h5ad(data, path)

    holdout_parts: list[ad.AnnData] = []
    reference_parts: list[ad.AnnData] = []
    for split, data in data_by_split.items():
        labels = data.obs[args.label_key].astype(str).to_numpy()
        holdout_parts.append(tagged_subset(data, np.isin(labels, HOLDOUT), split))
        reference_parts.append(
            tagged_subset(data, np.isin(labels, REFERENCE_PERIODS), split)
        )
    # anndata.concat supports only "inner" and "outer" joins in the scBGM
    # environment.  Gene names and order were already asserted identical above,
    # so an inner join is equivalent to the intended exact join here.
    holdout = ad.concat(holdout_parts, join="inner", merge="same", index_unique=None)
    reference = ad.concat(
        reference_parts, join="inner", merge="same", index_unique=None
    )
    if holdout.n_obs == 0:
        raise RuntimeError("Hold-out truth contains no cells")
    if set(holdout.obs[args.label_key].astype(str)) != set(HOLDOUT):
        raise RuntimeError("Hold-out truth does not contain exactly D3.5 and D4")
    for name, data in (("holdout_truth", holdout), ("trajectory_reference", reference)):
        path = output / f"WOT_{name}.h5ad"
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
        print(f"Writing {path} shape={data.shape}", flush=True)
        atomic_h5ad(data, path)

    train_names = set(filtered["train"].obs_names.astype(str))
    validation_names = set(filtered["validation"].obs_names.astype(str))
    if train_names & validation_names:
        raise RuntimeError("Filtered TRAIN and VALIDATION overlap")

    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source_dir": str(source),
        "label_key": args.label_key,
        "holdout_labels": list(HOLDOUT),
        "reference_periods": list(REFERENCE_PERIODS),
        "source_counts": counts,
        "filtered_train_n": int(filtered["train"].n_obs),
        "filtered_validation_n": int(filtered["validation"].n_obs),
        "holdout_truth_n": int(holdout.n_obs),
        "trajectory_reference_n": int(reference.n_obs),
        "gene_order_sha256": sha256_lines(genes),
        "train_obs_sha256": sha256_lines(filtered["train"].obs_names),
        "validation_obs_sha256": sha256_lines(filtered["validation"].obs_names),
        "training_reads_evaluation_files": False,
    }
    temporary = complete.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, complete)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
