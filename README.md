# scBGM

scBGM applies the Bayesian generative modeling framework BayesGM to single-cell
RNA-seq data. It generates realistic cells, either unconditionally or for given
conditions such as cell type, disease state or time, including conditions that are
rare or absent in the training data.

This repository contains the code for data splitting, autoencoder (AE) and scBGM
training, generation and the downstream analyses in the paper
(trajectory interpolation, rare-cell augmentation and disease-state counterfactuals).
Datasets and trained models are not included; download the data from their public
sources and train the models with the commands below.

## How it works

1. **Autoencoder.** A negative-binomial (NB) autoencoder maps raw UMI counts to a
   128-dimensional latent space and decodes latents back to counts given a library size.
2. **Generator.** A generator maps a standard-normal prior draw *z* (plus a label or
   time embedding for conditional models) to a Gaussian, with a mean and a diagonal
   variance, over the standardized AE latent.
3. **Step 1, warm start.** The generator is initialized with adversarial training,
   together with an encoder and latent/data discriminators.
4. **Step 2, refinement.** Each training cell gets its own latent variable. Training
   alternates between updating the generator (Gaussian likelihood + MMD between
   generated and real latents + a variance regularizer) and updating the per-cell
   latents under a maximum a posteriori objective.
5. **Checkpoint selection.** Step 1 and Step 2 checkpoints are selected on the
   validation split by iLISI between generated and real cells (per-class macro iLISI
   for conditional models). The test split is never read during training.
6. **Generation.** Prior draws are passed through the generator, a library size is
   sampled from a model fitted on the training cells, and the AE decoder samples NB
   counts.


## Installation

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
scbgm setup --repair-scib
scbgm doctor
```

`scbgm setup` checks the Python, TensorFlow/TFP and PyTorch versions and runs the
native scIB iLISI once; `--repair-scib` first compiles that tool for this machine.
`scbgm doctor` checks that every command's script is present and parses, and lists missing dependencies.

## Input data

Each dataset is one H5AD file with cells as rows:

- **Raw integer UMI counts**, in `.X` or in a layer. Normalized or log-transformed
  values are rejected.
- Unique gene names in `.var_names`.
- The label used for conditioning (for example `celltype`) and the columns used to
  stratify the split, in `.obs`.

Describe the datasets in a tab-separated file, one row per dataset:

| Column | Meaning | Example |
| --- | --- | --- |
| `dataset` | Name used for output folders and files | `human_lung` |
| `source_path` | Path to the H5AD | `/path/to/your/data/human_lung.h5ad` |
| `matrix_source` | Where the counts are: `X` or `layer:<name>` | `X` |
| `source_space` | Must be `raw_counts` | `raw_counts` |
| `stratify_columns` | Comma-separated `.obs` columns for stratification | `celltype,status` |

## Workflow

Every step is a `scbgm` subcommand. `scbgm list` shows all commands, and
`scbgm <group> <command> --help` lists each command's options and defaults.
Trainers refuse to write into a non-empty output folder unless `--overwrite` is
given, so use a new folder for each run.

### 1. Split the data

```bash
scbgm data split --datasets datasets.tsv --task-index 0 --output-root data/splits
```

`--task-index` selects the row of the TSV (0-based). Cells with fewer than 10
detected genes are removed. The remaining cells are split 72% / 8% / 20% into
training, validation and test sets, stratified by `stratify_columns` (seed 17), and
only genes detected in at least 3 training cells are kept. Output:

```text
data/splits/<dataset>/outer80_inner10_seed17/
  <dataset>_train_source.h5ad
  <dataset>_validation_source.h5ad
  <dataset>_test_source.h5ad
  split_manifest.json
```

**Optional HVG selection.** `scbgm data hvg --dataset <name> --train <train.h5ad>
--output <train_hvg.h5ad> --n-hvg 3000` selects Seurat-v3 highly variable genes on the
training split only. Subset the validation and test files to the same ordered genes
before training on HVGs; the trainers require identical genes in every split.

The commands below use `S=data/splits/<dataset>/outer80_inner10_seed17` and `D=<dataset>`.

### 2. Train the autoencoder

```bash
scbgm train ae --split-dir $S --train-file ${D}_train_source.h5ad \
  --validation-file ${D}_validation_source.h5ad --output-dir runs/ae --device cpu
```

The AE is trained on the training split. The checkpoint is selected by
reconstruction iLISI on up to 2,000 validation cells. The output folder holds
`model.pt`, `metadata.json`, `genes.npy`, `history.csv` and the encoded training and
validation latents and library sizes used by the next step.

### 3. Train scBGM

Unconditional:

```bash
scbgm train unconditional --split-dir $S --train-file ${D}_train_source.h5ad \
  --validation-file ${D}_validation_source.h5ad \
  --ae-artifact-dir runs/ae --output-dir runs/unconditional
```

Conditional on an `.obs` label:

```bash
scbgm train conditional --split-dir $S --train-file ${D}_train_source.h5ad \
  --validation-file ${D}_validation_source.h5ad --label-key celltype \
  --ae-artifact-dir runs/ae --output-dir runs/conditional
```

Each model folder contains `config.json` (all resolved options), the selected
generator weights (`generator.weights.h5`), `step1.json` / `step2.json` with the
selected checkpoints, `step1_history.csv` / `step2_history.csv`, the intermediate
checkpoints and the latent standardization (`latent_scale.npz`). At the end of
training each trainer also writes a sample from the selected model
(2,000 cells for unconditional models; 200 cells per class for conditional models,
including `generated.h5ad`). Use `--decoder-device cuda` (unconditional) or
`--decode-device cuda` (conditional) to decode on a GPU.

### 4. Generate cells

```bash
scbgm generate unconditional --model-dir runs/unconditional --ae-artifact-dir runs/ae \
  --output runs/unconditional/generated.h5ad --n-cells 2000

scbgm generate conditional --model-dir runs/conditional --ae-artifact-dir runs/ae \
  --output runs/conditional/generated.h5ad --n-per-label 200
```

For conditional generation, `--labels` chooses the labels (default: all;
`--list-labels` prints them), and the number of cells per label comes from
`--n-per-label`, a JSON mapping (`--counts-json`) or the label counts of a reference
H5AD (`--counts-from-h5ad`, optionally capped with `--max-per-label`). `--seed` fixes
the draws. In the output H5AD, `.X` is log1p(CP10K), the `counts` layer holds the
sampled raw counts and `.obs` holds the generated labels.

## Dataset-specific options

Some settings depend on the dataset. Choose them for your data:

| Command | Options |
| --- | --- |
| `train ae` | `--learning-rate`, `--weight-decay`, `--batch-size`, `--gradient-clip`, `--max-epochs` |
| `train unconditional` | `--res-width` |
| `train conditional` | `--label-key` |

## Downstream analyses

The code for the three downstream analyses is included, from data preparation to
generation:

- **WOT trajectory interpolation:** a continuous-time model generates the withheld
  days D3.5 and D4 between real D3 and D4.5 endpoints.
- **Human Lung rare-cell augmentation:** Ciliated cells are generated from
  down-sampled training data.
- **Disease-state counterfactuals:** separate cell-type and disease-state embeddings
  generate cell type × state combinations that were withheld from training.

Some of these scripts were written for one paper experiment and read their inputs
from path constants at the top of the file; replace the `/path/to/your/...`
placeholders with your own locations (`scbgm doctor` lists how many remain).
See [docs/DOWNSTREAM.md](docs/DOWNSTREAM.md) for commands and a file map.

## Command reference

| Group | Commands |
| --- | --- |
| Setup | `setup`, `doctor`, `list` |
| Data | `data split`, `data hvg`, `data trajectory`, `data counterfactual`, `data rare-manifests`, `data rare-split` |
| Training | `train ae`, `train unconditional`, `train conditional`, `train trajectory`, `train counterfactual` |
| Generation | `generate unconditional`, `generate conditional`, `generate trajectory`, `generate counterfactual` |

`scbgm --dry-run <group> <command> ...` prints the script and `PYTHONPATH` a command
would use without running it.

