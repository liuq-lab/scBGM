"""Build the two training arms for the disease-status counterfactual (Human Lung, full gene).

condition = celltype||status.  Both arms drop combinations with < 20 TRAIN cells (the trainer
requires >= 20).  The held-out arm additionally removes three Disease combinations from TRAIN and
VALIDATION; TEST is never touched.
"""
import json
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path("/path/to/your/runs/counterfactual_status_20261018")
SRC = Path("/path/to/your/data/uncond_result/shared_splits/human_lung/outer80_inner10_seed17")
HELD_OUT = ["AT2||Disease", "Macrophages||Disease", "Endothelial Cells||Disease"]
MIN_TRAIN = 20


def condition(a):
    return (a.obs["celltype"].astype(str) + "||" + a.obs["status"].astype(str)).values


def main():
    train = ad.read_h5ad(SRC / "human_lung_train_source.h5ad")
    validation = ad.read_h5ad(SRC / "human_lung_validation_source.h5ad")
    test = ad.read_h5ad(SRC / "human_lung_test_source.h5ad", backed="r")
    for a in (train, validation):
        a.obs["condition"] = pd.Categorical(condition(a))
    counts = train.obs["condition"].value_counts()
    too_small = sorted(counts[counts < MIN_TRAIN].index)
    celltypes = sorted(train.obs["celltype"].astype(str).unique())
    spec = {"separator": "||", "factors": ["celltype", "status"],
            "levels": {"celltype": celltypes, "status": ["Control", "Disease"]}, "dims": [12, 4]}
    (ROOT / "label_factors.json").write_text(json.dumps(spec, indent=2) + "\n")
    test_counts = pd.Series(condition(test)).value_counts()
    summary = {"held_out": HELD_OUT, "dropped_lt20": too_small, "min_train_cells": MIN_TRAIN, "arms": {}}
    for arm, drop in (("heldout", too_small + HELD_OUT), ("oracle", too_small)):
        out = ROOT / arm / "splits"
        out.mkdir(parents=True, exist_ok=True)
        rec = {}
        for name, a in (("train", train), ("validation", validation)):
            keep = ~a.obs["condition"].astype(str).isin(drop).values
            sub = a[keep].copy()
            sub.obs["condition"] = sub.obs["condition"].astype(str).astype("category")
            sub.write_h5ad(out / f"human_lung_{name}_source.h5ad")
            rec[name] = {"n_cells": int(sub.n_obs), "n_conditions": int(sub.obs["condition"].nunique()),
                         "removed_cells": int((~keep).sum())}
        for c in HELD_OUT:
            rec[c] = {"train_cells_in_arm": int((train.obs["condition"].astype(str).values == c).sum()
                                                if arm == "oracle" else 0),
                      "test_cells": int(test_counts.get(c, 0))}
        summary["arms"][arm] = rec
    (ROOT / "ARMS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
