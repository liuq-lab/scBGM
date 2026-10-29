"""Factorized label embedding for compositional conditions such as ``celltype||status``.

A condition label is an integer ID into the model's label list. With a factor
specification, each label name is split into factor levels (e.g. cell type and
disease status) and the label embedding is the concatenation of one embedding
per factor. The label -> factor lookup table is a constant, not a weight, so a
trained generator can be rebuilt with extra labels that were never seen during
training (new combinations of seen levels) and load the same weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tensorflow as tf


def load_factor_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).expanduser().read_text())
    factors = list(spec["factors"])
    levels = {name: list(map(str, spec["levels"][name])) for name in factors}
    dims = [int(value) for value in spec["dims"]]
    if len(dims) != len(factors) or any(value <= 0 for value in dims):
        raise ValueError("Factor spec needs one positive embedding size per factor")
    for name in factors:
        if len(levels[name]) < 2 or len(set(levels[name])) != len(levels[name]):
            raise ValueError(f"Factor {name!r} needs at least two unique levels")
    return {"separator": str(spec.get("separator", "||")), "factors": factors,
            "levels": levels, "dims": dims}


def factor_table(label_names: Sequence[str], spec: Mapping[str, Any]) -> np.ndarray:
    """Map each label name to its factor level IDs, shape (n_labels, n_factors)."""
    table = np.empty((len(label_names), len(spec["factors"])), dtype=np.int32)
    index = {name: {level: i for i, level in enumerate(spec["levels"][name])}
             for name in spec["factors"]}
    for row, label in enumerate(map(str, label_names)):
        parts = label.split(spec["separator"])
        if len(parts) != len(spec["factors"]):
            raise ValueError(f"Label {label!r} does not have {len(spec['factors'])} factors")
        for column, (name, level) in enumerate(zip(spec["factors"], parts)):
            if level not in index[name]:
                raise ValueError(f"Label {label!r}: unknown {name} level {level!r}")
            table[row, column] = index[name][level]
    return table


class FactorizedLabelEmbedding(tf.keras.layers.Layer):
    """concat(Embedding_f(level_f(label)) for each factor f)."""

    def __init__(self, table: np.ndarray, spec: Mapping[str, Any], name: str = "label_embedding") -> None:
        super().__init__(name=name)
        table = np.asarray(table, dtype=np.int32)
        if table.ndim != 2 or table.shape[1] != len(spec["factors"]):
            raise ValueError("Factor table shape does not match the factor spec")
        self.num_labels = int(table.shape[0])
        self.output_dim = int(sum(spec["dims"]))
        self.tables = [tf.constant(table[:, column]) for column in range(table.shape[1])]
        self.factor_embeddings = [
            tf.keras.layers.Embedding(len(spec["levels"][factor]), int(dim),
                                      embeddings_initializer="glorot_uniform",
                                      name=f"{factor}_embedding")
            for factor, dim in zip(spec["factors"], spec["dims"])
        ]

    def call(self, labels: tf.Tensor) -> tf.Tensor:
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        return tf.concat([embedding(tf.gather(table, labels))
                          for embedding, table in zip(self.factor_embeddings, self.tables)], axis=-1)


def check_factor_embedding(layer: FactorizedLabelEmbedding, num_classes: int, embed_dim: int) -> None:
    if layer.output_dim != int(embed_dim):
        raise ValueError(f"Factor embedding sizes sum to {layer.output_dim}, expected label_embed_dim={embed_dim}")
    if layer.num_labels != int(num_classes):
        raise ValueError(f"Factor table has {layer.num_labels} labels, expected {num_classes}")
