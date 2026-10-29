#!/usr/bin/env python3
"""Reconstruct one manifest-defined training H5AD in node-local storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train = ad.read_h5ad(manifest["source_train_h5ad"])
    labels = train.obs["celltype"].astype(str).to_numpy()
    target_names = np.asarray(
        train.obs_names[labels == manifest["target"]].astype(str)
    )
    if len(target_names) != manifest["n_target_original"]:
        raise RuntimeError("source target count differs from manifest")
    permutation = np.random.default_rng(int(manifest["seed"])).permutation(
        len(target_names)
    )
    selected = set(target_names[permutation[: int(manifest["n_target"])]])
    keep = np.asarray(
        [
            label != manifest["target"] or str(name) in selected
            for name, label in zip(train.obs_names, labels)
        ],
        dtype=bool,
    )
    subset = train[keep].copy()
    observed_target = subset.obs_names[
        subset.obs["celltype"].astype(str).to_numpy() == manifest["target"]
    ].astype(str)
    if set(observed_target) != selected or len(observed_target) != manifest["n_target"]:
        raise RuntimeError("reconstructed target subset differs from manifest")
    if subset.n_obs != manifest["n_train_arm"]:
        raise RuntimeError("reconstructed training size differs from manifest")
    subset.write_h5ad(output / "human_lung_train_source.h5ad", compression="gzip")
    # The conditional runner resolves validation/test filenames relative to
    # split_dir even when the manifest stores absolute source paths. Keep the
    # large immutable files in place and expose them through node-local links.
    for key in ("source_validation_h5ad", "source_test_h5ad"):
        source = Path(manifest[key])
        link = output / source.name
        if not link.exists():
            link.symlink_to(source)
    print(
        f"materialized {manifest['arm']}: {subset.n_obs} cells, "
        f"target={len(observed_target)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
