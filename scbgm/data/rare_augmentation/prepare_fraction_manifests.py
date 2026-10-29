#!/usr/bin/env python3
"""Create nested Ciliated fraction manifests without duplicating large H5ADs."""

from __future__ import annotations

import json
from pathlib import Path

SOURCE = Path(
    "/path/to/your/data/uncond_result/"
    "shared_splits/human_lung/outer80_inner10_seed17"
)
ROOT = Path(
    "/path/to/your/data/cond_result/downstream/"
    "rare_augmentation_human_lung_ciliated_fraction_sweep_v1"
)
TARGET = "Ciliated"
SEED = 101
FRACTIONS = (1.0, 0.50, 0.25, 0.10, 0.05, 0.03, 0.02, 0.01, 0.005)
N_TARGET_ORIGINAL = 10562
N_OTHER = 71802


def fraction_tag(fraction: float) -> str:
    return f"p{100.0 * fraction:g}".replace(".", "p")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    train_path = SOURCE / "human_lung_train_source.h5ad"

    summary = []
    for array_index, fraction in enumerate(FRACTIONS):
        n_keep = max(1, int(N_TARGET_ORIGINAL * fraction))
        tag = fraction_tag(fraction)
        name = f"human_lung_ciliated_{tag}_seed{SEED}_n{n_keep}"
        arm = ROOT / name
        arm.mkdir(parents=True, exist_ok=True)
        manifest = {
            "array_index": array_index,
            "arm": name,
            "arm_dir": str(arm),
            "dataset": "human_lung",
            "target": TARGET,
            "seed": SEED,
            "retained_fraction": fraction,
            "retained_percent": 100.0 * fraction,
            "n_target_original": N_TARGET_ORIGINAL,
            "n_target": int(n_keep),
            "n_other": N_OTHER,
            "n_train_arm": int(N_OTHER + n_keep),
            "n_added_to_original": int(N_TARGET_ORIGINAL - n_keep),
            "target_selection": "seeded_nested_permutation_reconstructed_at_runtime",
            "nested_target_subsets": True,
            "source_train_h5ad": str(train_path),
            "source_validation_h5ad": str(SOURCE / "human_lung_validation_source.h5ad"),
            "source_test_h5ad": str(SOURCE / "human_lung_test_source.h5ad"),
            "validation_and_test_untouched": True,
        }
        path = arm / "arm_manifest.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != manifest:
                raise RuntimeError(f"Existing manifest differs: {path}")
        else:
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        summary.append(manifest)

    (ROOT / "fraction_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    "array_index": item["array_index"],
                    "percent": item["retained_percent"],
                    "n_target": item["n_target"],
                    "n_added": item["n_added_to_original"],
                    "arm": item["arm"],
                }
                for item in summary
            ],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
