# Downstream analyses

Steps without a `scbgm` command are run directly with `python`.

## WOT trajectory interpolation

| Step | Command | Code |
| --- | --- | --- |
| Withhold D3.5/D4 and split | `data trajectory` | `scbgm/data/trajectory/prepare_wot_holdout.py` |
| Train the AE | `train ae` | `scbgm/training/unconditional/autoencoder.py` |
| Train the continuous-time model | `train trajectory` | `scbgm/training/trajectory/train_continuous_time_scbgm.py` |
| Generate and evaluate D3.5/D4 between real D3 and D4.5 cells | `generate trajectory` | `scbgm/generation/trajectory/evaluate_real_d3_anchored.py` |

```bash
scbgm generate trajectory --model-dir runs/wot-scbgm --ae-dir runs/wot-ae \
  --train-h5ad WOT_train_no_D3.5_D4.h5ad --holdout-h5ad WOT_holdout_truth.h5ad \
  --trajectory-reference-h5ad WOT_trajectory_reference.h5ad --output-dir results/wot
```

## Rare-cell augmentation (Human Lung)

| Step | Command | Code |
| --- | --- | --- |
| Define retained fractions | `data rare-manifests` | `scbgm/data/rare_augmentation/prepare_fraction_manifests.py` |
| Down-sample the rare class | `data rare-split` | `scbgm/data/rare_augmentation/materialize_fraction_split.py` |
| Train on the down-sampled split | `train conditional` | `scbgm/training/conditional/train_conditional.py` |

## Disease-state counterfactuals (Human Lung)

| Step | Command | Code |
| --- | --- | --- |
| Build held-out and oracle splits | `data counterfactual` | `scbgm/data/counterfactual/prepare_arms.py` |
| Train with cell-type and state embeddings | `train counterfactual` | `scbgm/training/counterfactual/train_conditional.py` |
| Generate withheld cell type × state combinations | `generate counterfactual` | `scbgm/generation/counterfactual/generate_conditional.py` |

```bash
scbgm generate counterfactual --model-dir runs/heldout-AT2 --ae-artifact-dir runs/heldout-AT2-ae \
  --extra-labels 'AT2||Disease' --labels 'AT2||Disease' --n-per-label 50 \
  --output results/AT2-disease.h5ad
```
