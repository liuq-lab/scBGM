"""Leave-one-combination-out arms (design B): each arm removes ONE Disease combination (plus the
< 20-cell combinations dropped in every arm) from TRAIN and VALIDATION.  TEST is never touched."""
import json
import os
from pathlib import Path

import anndata as ad
import pandas as pd

ROOT = Path(os.environ.get("CF_ROOT", "/path/to/your/project/counterfactual_status_20261018"))
SRC = Path("/path/to/your/data/uncond_result/shared_splits/human_lung/outer80_inner10_seed17")
ARMS = {"heldout_AT2": "AT2||Disease", "heldout_Macrophages": "Macrophages||Disease",
        "heldout_Endothelial": "Endothelial Cells||Disease"}
MIN_TRAIN = 20


def main():
    train = ad.read_h5ad(SRC / "human_lung_train_source.h5ad")
    validation = ad.read_h5ad(SRC / "human_lung_validation_source.h5ad")
    for a in (train, validation):
        a.obs["condition"] = pd.Categorical((a.obs["celltype"].astype(str) + "||" + a.obs["status"].astype(str)).values)
    counts = train.obs["condition"].value_counts()
    too_small = sorted(counts[counts < MIN_TRAIN].index)
    n_disease = int((train.obs["status"].astype(str) == "Disease").sum())
    summary = {"design": "leave one combination out", "dropped_lt20": too_small, "arms": {}}
    for arm, combo in ARMS.items():
        drop = too_small + [combo]
        out = ROOT / arm / "splits"
        out.mkdir(parents=True, exist_ok=True)
        rec = {"held_out": combo, "held_out_train_cells": int(counts[combo]),
               "share_of_disease_train_cells": round(int(counts[combo]) / n_disease, 4)}
        for name, a in (("train", train), ("validation", validation)):
            keep = ~a.obs["condition"].astype(str).isin(drop).values
            sub = a[keep].copy()
            sub.obs["condition"] = sub.obs["condition"].astype(str).astype("category")
            sub.write_h5ad(out / f"human_lung_{name}_source.h5ad")
            rec[name] = {"n_cells": int(sub.n_obs), "n_conditions": int(sub.obs["condition"].nunique())}
        summary["arms"][arm] = rec
    (ROOT / "ARMS_SINGLE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
