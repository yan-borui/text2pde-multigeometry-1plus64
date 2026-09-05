"""Portable physical prediction archives shared by all four stride-8 baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PREDICTION_SCHEMA = "cylinderflow.physical_prediction.v2"
RAW_FRAME_INDICES = np.arange(0, 513, 8, dtype=np.int64)
PHYSICAL_TIME = np.arange(65, dtype=np.float64) * 0.08


def writeback_velocity(
    prediction: np.ndarray, initial: np.ndarray, node_type: np.ndarray
) -> np.ndarray:
    """Clamp only inlet/wall UV to the observed first frame after decoding."""
    result = np.asarray(prediction).copy()
    mask = np.isin(np.asarray(node_type).reshape(-1), (4, 6))
    result[1:, mask, :2] = np.asarray(initial)[None, mask, :2]
    return result


def boundary_metrics(
    prediction: np.ndarray,
    pre_boundary: np.ndarray,
    initial: np.ndarray,
    node_type: np.ndarray,
) -> dict[str, float | None]:
    """Unweighted inlet/wall UV error against known, fixed boundary values."""
    mask = np.isin(np.asarray(node_type).reshape(-1), (4, 6))
    result = {}
    for label, values in (("pre", pre_boundary), ("post", prediction)):
        error = np.asarray(values)[1:, mask, :2] - np.asarray(initial)[None, mask, :2]
        result[f"boundary_uv_rmse_{label}_writeback"] = (
            float(np.sqrt(np.mean(error.astype(np.float64) ** 2)))
            if mask.any() and np.isfinite(error).all()
            else None
        )
    return result


def validate_prediction(bundle: Any) -> None:
    """Validate axes and exact evaluation time without discarding failed predictions."""
    required = (
        "prediction",
        "pre_boundary",
        "target",
        "points",
        "cells",
        "node_type",
        "raw_frame_indices",
        "physical_time",
        "trajectory_index",
        "seed",
    )
    for name in required:
        if name not in bundle:
            raise ValueError(f"prediction archive is missing {name}")
    if not np.array_equal(bundle["raw_frame_indices"], RAW_FRAME_INDICES):
        raise ValueError("prediction raw time indices differ from the contract")
    if np.shape(bundle["physical_time"]) != (65,) or not np.allclose(
        bundle["physical_time"], PHYSICAL_TIME, rtol=0, atol=1e-12
    ):
        raise ValueError("prediction physical time differs from dt=0.08")
    nodes = len(bundle["points"])
    for name in ("prediction", "pre_boundary", "target"):
        if np.shape(bundle[name]) != (65, nodes, 3):
            raise ValueError(f"{name} must have shape [65,N,3]")
    if np.shape(bundle["points"]) != (nodes, 2):
        raise ValueError("points must have shape [N,2]")
    if np.shape(bundle["node_type"]) != (nodes,):
        raise ValueError("node_type must have shape [N]")
    cells = np.asarray(bundle["cells"])
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("cells must have shape [C,3]")
    if not np.isfinite(bundle["target"]).all():
        raise ValueError("reference target must be finite")
    for name in ("prediction", "pre_boundary"):
        if not np.array_equal(bundle[name][0], bundle["target"][0]):
            raise ValueError("the observed first frame must be preserved exactly")


def save_prediction(
    file_path: Path,
    *,
    prediction: np.ndarray,
    pre_boundary: np.ndarray,
    target: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
    node_type: np.ndarray,
    trajectory_index: int,
    seed: int,
    provenance: dict[str, Any],
    **diagnostics: Any,
) -> None:
    bundle = {
        "prediction_schema": np.asarray(PREDICTION_SCHEMA),
        "units": np.asarray("physical_uvp"),
        "prediction": np.asarray(prediction),
        "pre_boundary": np.asarray(pre_boundary),
        "target": np.asarray(target),
        "points": np.asarray(points),
        "cells": np.asarray(cells, dtype=np.int64),
        "node_type": np.asarray(node_type, dtype=np.int64).reshape(-1),
        "raw_frame_indices": RAW_FRAME_INDICES,
        "frame_indices": RAW_FRAME_INDICES,
        "physical_time": PHYSICAL_TIME,
        "trajectory_index": np.asarray(trajectory_index, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "provenance": np.asarray(json.dumps(provenance, sort_keys=True)),
    }
    if set(diagnostics) & set(bundle):
        raise ValueError("diagnostics cannot replace common prediction fields")
    bundle.update(diagnostics)
    validate_prediction(bundle)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("xb") as stream:
        np.savez_compressed(stream, **bundle)
