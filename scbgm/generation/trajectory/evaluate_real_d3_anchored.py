#!/usr/bin/env python3
"""Generate held-out WOT days between OT-paired real D3/D4.5 anchors.

Each TRAIN-only endpoint pair interpolates its learned Step-2 z, observed
standardized AE residual, and log library size.  The fixed sinusoidal
biological-time embedding supplies continuous time.  D3.5 and D4 remain
unopened until generation is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import tensorflow as tf
import torch


SCRIPT_VERSION = "real_d3_d45_ot_anchored_continuous_time_scbgm_v1"
TARGET_TO_DAY = {"D3.5": 3.5, "D4": 4.0}
TARGET_TO_TIME = {"D3.5": 1.0 / 3.0, "D4": 2.0 / 3.0}
METHOD = "Real-D3/D4.5-OT-anchored continuous-time scBGM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--ae-dir", type=Path, required=True)
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--trajectory-reference-h5ad", type=Path, required=True)
    parser.add_argument("--holdout-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-per-state", type=int, default=500)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--pca-fit-max-cells", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--library-cost-weight", type=float, default=0.25)
    parser.add_argument("--decode-device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def forward_with_time_embedding(
    generator: tf.keras.Model,
    z: np.ndarray,
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z_tensor = tf.convert_to_tensor(z, dtype=tf.float32)
    embedding_tensor = tf.repeat(
        tf.convert_to_tensor(embedding[None, :], dtype=tf.float32), len(z), axis=0
    )
    injection = str(generator.label_injection)
    if injection == "early_concat":
        hidden = generator.input_projection(
            tf.concat([z_tensor, embedding_tensor], axis=-1)
        )
        for block in generator.blocks:
            hidden = block(hidden, training=False)
    elif injection in {"additive_per_block", "concat_per_block"}:
        generator_input = (
            z_tensor + embedding_tensor
            if injection == "additive_per_block"
            else tf.concat([z_tensor, embedding_tensor], axis=-1)
        )
        hidden = generator.input_projection(generator_input)
        if len(generator.blocks) != len(generator.block_label_projections):
            raise RuntimeError(
                "generator block/projection count mismatch: "
                f"{len(generator.blocks)} != {len(generator.block_label_projections)}"
            )
        for block, label_projection in zip(
            generator.blocks, generator.block_label_projections
        ):
            hidden = block(
                hidden + label_projection(embedding_tensor), training=False
            )
    else:
        raise RuntimeError(f"unsupported label injection mode: {injection}")
    hidden = generator.final_norm(hidden)
    mean = generator.mean_layer(hidden)
    variance = tf.nn.softplus(generator.var_layer(hidden)) + tf.cast(
        generator.variance_epsilon, mean.dtype
    )
    return mean.numpy().astype(np.float32), variance.numpy().astype(np.float32)


def plot_umap(combined: ad.AnnData, destination: Path) -> None:
    xy = combined.obsm["X_umap"]
    real = combined.obs["source"].astype(str).eq("real").to_numpy()
    generated = combined.obs["method"].astype(str).eq(METHOD).to_numpy()
    real_period = combined.obs["period"].astype(str)
    real_time_map = {"D3": 0.0, "D3.5": 1.0 / 3.0, "D4": 2.0 / 3.0, "D4.5": 1.0}
    real_focus = real & real_period.isin(real_time_map).to_numpy()
    real_time = real_period.loc[real_focus].map(real_time_map).to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    for ax in axes:
        ax.scatter(
            xy[real, 0], xy[real, 1], s=3, c="lightgray", alpha=0.18,
            rasterized=True,
        )
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    real_points = axes[0].scatter(
        xy[real_focus, 0], xy[real_focus, 1], s=8, c=real_time,
        cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.8, rasterized=True,
    )
    axes[0].set_title("Real WOT: D3, D3.5, D4, D4.5")

    axes[1].scatter(
        xy[generated, 0], xy[generated, 1], s=7,
        c=combined.obs.loc[generated, "flow_time"].to_numpy(dtype=float),
        cmap="viridis", vmin=0.0, vmax=1.0, alpha=0.75, rasterized=True,
    )
    axes[1].set_title("Real-D3/D4.5 OT-anchored continuous-time scBGM")
    fig.colorbar(
        real_points, ax=axes, label="Normalized biological time: D3 to D4.5",
        fraction=0.035, pad=0.03,
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.n_per_state < 20 or args.library_cost_weight < 0:
        raise ValueError("invalid endpoint pairing settings")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    import scbgm.training.trajectory.train_continuous_time_scbgm as ct
    import scbgm.generation.trajectory.generate_conditional as cg
    import scbgm.generation.trajectory.wot_metrics as ev

    model_dir = args.model_dir.expanduser().resolve()
    ae_dir = args.ae_dir.expanduser().resolve()
    config = read_json(model_dir / "config.json")
    label_map = read_json(model_dir / "label_map.json")
    label_names = list(label_map["id_to_label"])
    if any(target in label_names for target in TARGET_TO_DAY):
        raise RuntimeError("held-out D3.5/D4 leaked into continuous scBGM training")
    if config.get("conditioning_type") != "fixed_sinusoidal_biological_time":
        raise RuntimeError("model is not a continuous-time scBGM checkpoint")

    table = np.load(model_dir / "continuous_time_embedding_table.npy", allow_pickle=False)
    ct.CONTINUOUS_TIME_TABLE = np.asarray(table, dtype=np.float32)
    residual = ct.ResidualConfig(
        width=int(config.get("res_width", 256)),
        generator_blocks=int(config.get("res_generator_blocks", 4)),
        encoder_blocks=int(config.get("res_encoder_blocks", 3)),
        expansion=int(config.get("res_expansion", 4)),
        dropout=float(config.get("res_dropout", 0.0)),
        normalization=str(config.get("res_normalization", "layernorm")),
    )
    generator = ct.ConditionalResidualVariationalGenerator(
        z_dim=int(config["bgm_z_dim"]),
        output_dim=int(config["ae_latent_dim"]),
        num_classes=len(label_names),
        label_embed_dim=int(config["label_embed_dim"]),
        variance_epsilon=float(config["generator_variance_eps"]),
        config=residual,
        label_injection=str(config["label_injection"]),
    )
    generator(
        tf.zeros((2, int(config["bgm_z_dim"])), dtype=tf.float32),
        tf.zeros((2,), dtype=tf.int32),
        training=False,
    )
    generator.load_weights(str(model_dir / "generator.weights.h5"))
    generator.trainable = False

    genes = np.load(model_dir / "genes.npy", allow_pickle=False).astype(str)
    train_labels = np.load(model_dir / "train_labels.npy", allow_pickle=False)
    train_z = np.load(model_dir / "data_z.npy", allow_pickle=False).astype(np.float32)
    train_latent = np.load(ae_dir / "train_latent.npy", allow_pickle=False).astype(np.float32)
    train_library = np.load(ae_dir / "train_library_size.npy", allow_pickle=False).astype(np.float32)
    if not (len(train_labels) == len(train_z) == len(train_latent) == len(train_library)):
        raise RuntimeError("TRAIN labels/z/latent/library arrays are not aligned")
    with np.load(model_dir / "latent_scale.npz", allow_pickle=False) as values:
        latent_mean = np.asarray(values["mean"], dtype=np.float32)
        latent_sd = np.asarray(values["sd"], dtype=np.float32)

    device = torch.device(
        "cuda" if args.decode_device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    ae = cg.validate_nb_ae_artifacts(
        ae_dir, genes, config, train_labels, label_names, device
    )[0]

    d3_id = label_names.index("D3")
    d45_id = label_names.index("D4.5")
    log_library = np.log(np.maximum(train_library, 1.0)).astype(np.float32)
    log_library_mean = float(log_library.mean())
    log_library_sd = max(float(log_library.std()), 1e-6)

    # Select equal-sized endpoint pools using TRAIN only and find the minimum
    # cost one-to-one assignment in standardized AE latent + log-library space.
    rng = np.random.default_rng(args.seed)
    endpoint_indices: dict[str, np.ndarray] = {}
    for label, label_id in (("D3", d3_id), ("D4.5", d45_id)):
        pool = np.flatnonzero(train_labels == label_id)
        if len(pool) < args.n_per_state:
            raise RuntimeError(
                f"{label} has {len(pool)} TRAIN cells, fewer than {args.n_per_state}"
            )
        endpoint_indices[label] = np.sort(
            rng.choice(pool, args.n_per_state, replace=False)
        )

    idx0 = endpoint_indices["D3"]
    idx1 = endpoint_indices["D4.5"]
    library_cost_scale = math.sqrt(args.library_cost_weight)
    cost0 = np.concatenate(
        [
            (train_latent[idx0] - latent_mean) / latent_sd,
            (
                library_cost_scale
                * ((log_library[idx0] - log_library_mean) / log_library_sd)
            )[:, None],
        ],
        axis=1,
    )
    cost1 = np.concatenate(
        [
            (train_latent[idx1] - latent_mean) / latent_sd,
            (
                library_cost_scale
                * ((log_library[idx1] - log_library_mean) / log_library_sd)
            )[:, None],
        ],
        axis=1,
    )
    cost = cdist(
        cost0.astype(np.float64), cost1.astype(np.float64), metric="sqeuclidean"
    )
    rows, columns = linear_sum_assignment(cost)
    anchor_indices0 = idx0[rows]
    anchor_indices1 = idx1[columns]
    np.savez_compressed(
        output / "ot_pairing.npz",
        d3_train_indices=anchor_indices0,
        d45_train_indices=anchor_indices1,
        squared_cost=cost[rows, columns].astype(np.float32),
    )

    z0, z1 = train_z[anchor_indices0], train_z[anchor_indices1]
    latent0, latent1 = train_latent[anchor_indices0], train_latent[anchor_indices1]
    latent_std0 = (latent0 - latent_mean) / latent_sd
    latent_std1 = (latent1 - latent_mean) / latent_sd
    log_library0 = log_library[anchor_indices0]
    log_library1 = log_library[anchor_indices1]

    variance_floor = max(float(config["generator_variance_eps"]), 1e-6)
    endpoint_residuals: list[np.ndarray] = []
    endpoint_rmses: list[float] = []
    for day, z_endpoint, latent_endpoint in (
        (3.0, z0, latent_std0),
        (4.5, z1, latent_std1),
    ):
        endpoint_embedding, _ = ct.sinusoidal_time_table(
            [f"D{day:g}"],
            int(config["label_embed_dim"]),
            day_minimum=float(config["continuous_time_day_minimum"]),
            day_maximum=float(config["continuous_time_day_maximum"]),
            max_period=float(config["continuous_time_max_period"]),
            scale=float(config["continuous_time_scale"]),
        )
        endpoint_mean, endpoint_variance = forward_with_time_embedding(
            generator, z_endpoint, endpoint_embedding[0]
        )
        endpoint_scale = np.sqrt(np.maximum(endpoint_variance, variance_floor))
        endpoint_residual = (latent_endpoint - endpoint_mean) / endpoint_scale
        endpoint_reconstruction = endpoint_mean + endpoint_scale * endpoint_residual
        endpoint_residuals.append(endpoint_residual)
        endpoint_rmses.append(
            float(np.sqrt(np.mean((endpoint_reconstruction - latent_endpoint) ** 2)))
        )
    residual0, residual1 = endpoint_residuals

    train_path = args.train_h5ad.expanduser().resolve()
    train_backed = ad.read_h5ad(train_path, backed="r")
    try:
        if not np.array_equal(train_backed.var_names.astype(str), genes):
            raise RuntimeError("TRAIN H5AD gene order differs from model artifacts")
        if train_backed.n_obs != len(train_labels):
            raise RuntimeError("TRAIN H5AD rows differ from model artifacts")
        observed_period = train_backed.obs["period"].astype(str).to_numpy()
        if not np.all(observed_period[anchor_indices0] == "D3"):
            raise RuntimeError("selected start anchors are not D3")
        if not np.all(observed_period[anchor_indices1] == "D4.5"):
            raise RuntimeError("selected end anchors are not D4.5")
        anchor_barcodes0 = train_backed.obs_names[anchor_indices0].astype(str).to_numpy()
        anchor_barcodes1 = train_backed.obs_names[anchor_indices1].astype(str).to_numpy()
        anchor_slice0 = train_backed[anchor_indices0].to_memory()
        anchor_slice1 = train_backed[anchor_indices1].to_memory()
    finally:
        train_backed.file.close()
    anchor_counts0 = (
        anchor_slice0.X.toarray().astype(np.float32)
        if sparse.issparse(anchor_slice0.X)
        else np.asarray(anchor_slice0.X, dtype=np.float32)
    )
    anchor_counts1 = (
        anchor_slice1.X.toarray().astype(np.float32)
        if sparse.issparse(anchor_slice1.X)
        else np.asarray(anchor_slice1.X, dtype=np.float32)
    )
    for label, endpoint_counts in (("D3", anchor_counts0), ("D4.5", anchor_counts1)):
        if np.any(endpoint_counts < 0) or not np.all(np.isfinite(endpoint_counts)):
            raise RuntimeError(f"invalid real-{label} anchor counts")

    days = sorted(set(np.linspace(3.0, 4.5, 11).tolist() + list(TARGET_TO_DAY.values())))
    times = [(day - 3.0) / 1.5 for day in days]
    expression: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    for index, (day, time) in enumerate(zip(days, times)):
        time32 = np.float32(time)
        z = (1.0 - time32) * z0 + time32 * z1
        residual_at_time = (1.0 - time32) * residual0 + time32 * residual1
        embedding, _ = ct.sinusoidal_time_table(
            [f"D{day:g}"],
            int(config["label_embed_dim"]),
            day_minimum=float(config["continuous_time_day_minimum"]),
            day_maximum=float(config["continuous_time_day_maximum"]),
            max_period=float(config["continuous_time_max_period"]),
            scale=float(config["continuous_time_scale"]),
        )
        mean_std, variance_std = forward_with_time_embedding(generator, z, embedding[0])
        # Jointly interpolate Step-2 z and the observed endpoint residuals.
        # The formula reconstructs both observed endpoint AE latents exactly.
        sampled_std = mean_std + np.sqrt(
            np.maximum(variance_std, variance_floor)
        ) * residual_at_time
        latent = sampled_std * latent_sd + latent_mean
        library = np.exp(
            (1.0 - time32) * log_library0 + time32 * log_library1
        ).astype(np.float32)
        if np.isclose(day, 3.0):
            decoded = anchor_counts0.copy()
        elif np.isclose(day, 4.5):
            decoded = anchor_counts1.copy()
        else:
            decoded = cg.decode_sample_counts(
                ae.model,
                latent,
                library,
                batch_size=256,
                device=device,
                seed=args.seed + 1000,
            ).astype(np.float32)
        counts.append(decoded)
        expression.append(cg.normalize_total_log1p_dense(decoded, 10000.0))

    generated = ev.make_generated_adata(expression, counts, times, genes, METHOD)
    generated.obs["biological_day"] = np.repeat(np.asarray(days), args.n_per_state)
    generated.obs["d3_anchor_barcode"] = np.tile(anchor_barcodes0, len(days))
    generated.obs["d45_anchor_barcode"] = np.tile(anchor_barcodes1, len(days))
    generated.obs["anchor_source"] = "TRAIN-only OT-paired real D3/D4.5"
    generated.write_h5ad(output / "continuous_time_scbgm.h5ad", compression="gzip")
    anchor_slice0.write_h5ad(output / "real_d3_anchors.h5ad", compression="gzip")
    anchor_slice1.write_h5ad(output / "real_d45_anchors_ot_paired.h5ad", compression="gzip")
    np.save(output / "real_d3_anchor_z.npy", z0)
    np.save(output / "real_d45_anchor_z.npy", z1)
    np.save(output / "real_d3_anchor_residual.npy", residual0)
    np.save(output / "real_d45_anchor_residual.npy", residual1)
    anchor_diagnostics = {
        "d3_anchor_latent_rmse_standardized": endpoint_rmses[0],
        "d45_anchor_latent_rmse_standardized": endpoint_rmses[1],
        "n_ot_pairs": int(args.n_per_state),
        "library_cost_weight": float(args.library_cost_weight),
        "ot_squared_cost_mean": float(np.mean(cost[rows, columns])),
        "ot_squared_cost_median": float(np.median(cost[rows, columns])),
        "d3_residual_abs_p95": float(np.quantile(np.abs(residual0), 0.95)),
        "d45_residual_abs_p95": float(np.quantile(np.abs(residual1), 0.95)),
        "endpoint_library_exact": True,
    }
    (output / "anchor_diagnostics.json").write_text(
        json.dumps(anchor_diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    hvg_names, expression_scaler, pca = ev.fit_train_only_representation(
        args.train_h5ad.expanduser().resolve(),
        genes,
        args.pca_fit_max_cells,
        args.n_hvg,
        args.seed,
    )
    np.save(output / "train_selected_hvg.npy", hvg_names)
    holdout = ad.read_h5ad(args.holdout_h5ad.expanduser().resolve())
    if not np.array_equal(holdout.var_names.astype(str), genes):
        holdout = holdout[:, genes].copy()
    metric_rows: list[dict[str, Any]] = []
    for target, day in TARGET_TO_DAY.items():
        time = TARGET_TO_TIME[target]
        real = holdout[holdout.obs["period"].astype(str).eq(target).to_numpy()]
        mask = np.isclose(generated.obs["biological_day"].to_numpy(dtype=float), day)
        mmd, ilisi, n_eval = ev.pair_metrics(
            real,
            generated[mask],
            hvg_names,
            expression_scaler,
            pca,
            args.n_per_state,
            args.seed,
        )
        metric_rows.append(
            {
                "method": METHOD,
                "target": target,
                "biological_day": day,
                "normalized_time": time,
                "mmd_pca20": mmd,
                "ilisi_pca20": ilisi,
                "n_per_group": n_eval,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output / "heldout_metrics.csv", index=False)

    reference = ad.read_h5ad(args.trajectory_reference_h5ad.expanduser().resolve())
    if not np.array_equal(reference.var_names.astype(str), genes):
        reference = reference[:, genes].copy()
    combined, dpt_summary = ev.run_dpt(
        reference,
        [generated],
        hvg_names,
        expression_scaler,
        pca,
        args.seed,
    )
    combined.write_h5ad(output / "trajectory_joint_embedding.h5ad", compression="gzip")
    dpt_summary.to_csv(output / "dpt_state_summary.csv", index=False)
    real_dpt = dpt_summary[dpt_summary["method"].eq("real")].set_index("state")
    generated_dpt = dpt_summary[dpt_summary["method"].eq(METHOD)].sort_values("flow_time")
    dpt_rows = []
    for target, time in TARGET_TO_TIME.items():
        row = generated_dpt.iloc[
            int(np.argmin(np.abs(generated_dpt["flow_time"].to_numpy() - time)))
        ]
        dpt_rows.append(
            {
                "target": target,
                "generated_mean_dpt": float(row["mean_dpt"]),
                "real_mean_dpt": float(real_dpt.loc[target, "mean_dpt"]),
                "absolute_dpt_error": abs(
                    float(row["mean_dpt"]) - float(real_dpt.loc[target, "mean_dpt"])
                ),
            }
        )
    pd.DataFrame(dpt_rows).to_csv(output / "dpt_target_metrics.csv", index=False)
    plot_umap(combined, output / "trajectory_umap.png")
    marker_values = ev.marker_table(reference, [generated])
    marker_values.to_csv(output / "marker_expression_by_state.csv", index=False)
    ev.plot_markers(marker_values, output / "marker_curves.png")

    manifest = {
        "schema_version": 1,
        "status": "COMPLETE",
        "script_version": SCRIPT_VERSION,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "method": METHOD,
        "generator": "scBGM G(interpolated endpoint Step-2 z, fixed SinCos128(biological day))",
        "anchoring": "TRAIN-only OT-paired real D3/D4.5 AE latents, residuals, z, and log library sizes",
        "endpoint_output_counts": "actual observed TRAIN D3 and D4.5 counts",
        "d3_anchor_latent_rmse_standardized": endpoint_rmses[0],
        "d45_anchor_latent_rmse_standardized": endpoint_rmses[1],
        "z_source": "linear interpolation of OT-paired endpoint Step-2 per-cell data_z",
        "residual_source": "linear interpolation of inferred observed endpoint standardized residuals",
        "ot_pairing": "Hungarian assignment in standardized TRAIN AE latent plus weighted log-library space",
        "library_cost_weight": args.library_cost_weight,
        "flow_matching_used": False,
        "ode_used": False,
        "heldout_targets": TARGET_TO_DAY,
        "heldout_truth_opened_after_generation": True,
        "heldout_used_for_training_or_checkpoint_selection": False,
        "library_size": "linear interpolation of OT-paired observed endpoint log library sizes",
        "expression_representation": "HVG/scaler/PCA fit on TRAIN only",
        "n_per_state": args.n_per_state,
        "seed": args.seed,
        "generator_checkpoint_sha256": sha256_file(model_dir / "generator.weights.h5"),
    }
    (output / "RUN_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
