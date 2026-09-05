"""Physical-mesh metrics shared verbatim by the four stride-8 baselines.

Adapted from the project's CylinderFlow physical-mesh evaluator. Spatial
derivatives are piecewise linear on triangles; spectra below are temporal.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def node_area_weights(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Assign one third of every triangle area to each incident node."""

    triangle_points = np.asarray(points, dtype=np.float64)[cells]
    cross = (triangle_points[:, 1, 0] - triangle_points[:, 0, 0]) * (
        triangle_points[:, 2, 1] - triangle_points[:, 0, 1]
    ) - (triangle_points[:, 2, 0] - triangle_points[:, 0, 0]) * (
        triangle_points[:, 1, 1] - triangle_points[:, 0, 1]
    )
    triangle_area = 0.5 * np.abs(cross)
    if np.any(triangle_area <= 0):
        raise ValueError("mesh contains a degenerate triangle")
    weights = np.zeros(points.shape[0], dtype=np.float64)
    for local_vertex in range(3):
        np.add.at(weights, cells[:, local_vertex], triangle_area / 3.0)
    if np.any(weights <= 0):
        raise ValueError("mesh contains a node with zero triangle area")
    return weights


def triangle_vorticity_divergence(
    velocity: np.ndarray, points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return piecewise-linear vorticity, divergence, and triangle area."""

    triangle_points = np.asarray(points, dtype=np.float64)[cells]
    x0, y0 = triangle_points[:, 0, 0], triangle_points[:, 0, 1]
    x1, y1 = triangle_points[:, 1, 0], triangle_points[:, 1, 1]
    x2, y2 = triangle_points[:, 2, 0], triangle_points[:, 2, 1]
    determinant = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if np.any(np.abs(determinant) <= np.finfo(np.float64).eps):
        raise ValueError("mesh contains a degenerate triangle")

    u = np.asarray(velocity, dtype=np.float64)[..., 0][:, cells]
    v = np.asarray(velocity, dtype=np.float64)[..., 1][:, cells]
    du_dx = (
        u[:, :, 0] * (y1 - y2) + u[:, :, 1] * (y2 - y0) + u[:, :, 2] * (y0 - y1)
    ) / determinant
    du_dy = (
        u[:, :, 0] * (x2 - x1) + u[:, :, 1] * (x0 - x2) + u[:, :, 2] * (x1 - x0)
    ) / determinant
    dv_dx = (
        v[:, :, 0] * (y1 - y2) + v[:, :, 1] * (y2 - y0) + v[:, :, 2] * (y0 - y1)
    ) / determinant
    dv_dy = (
        v[:, :, 0] * (x2 - x1) + v[:, :, 1] * (x0 - x2) + v[:, :, 2] * (x1 - x0)
    ) / determinant
    return dv_dx - du_dy, du_dx + dv_dy, 0.5 * np.abs(determinant)


def _relative_rmse(error_energy: float, target_energy: float) -> float:
    if target_energy > 0:
        return math.sqrt(error_energy / target_energy)
    return 0.0 if error_energy == 0 else float("inf")


def _series_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def weighted_correlation(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> float | None:
    repeated_weights = np.broadcast_to(
        weights[None, :, None], prediction.shape
    ).reshape(-1)
    prediction_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    weight_sum = repeated_weights.sum()
    prediction_centered = (
        prediction_flat - np.sum(repeated_weights * prediction_flat) / weight_sum
    )
    target_centered = target_flat - np.sum(repeated_weights * target_flat) / weight_sum
    denominator = math.sqrt(
        np.sum(repeated_weights * prediction_centered**2)
        * np.sum(repeated_weights * target_centered**2)
    )
    if denominator == 0:
        return None
    value = float(
        np.sum(repeated_weights * prediction_centered * target_centered) / denominator
    )
    return value if math.isfinite(value) else None


def best_energy_lag(
    prediction_energy: np.ndarray,
    target_energy: np.ndarray,
    maximum_lag: int = 10,
) -> tuple[int, float | None]:
    best_lag = 0
    best_correlation: float | None = None
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            left, right = prediction_energy[-lag:], target_energy[:lag]
        elif lag > 0:
            left, right = prediction_energy[:-lag], target_energy[lag:]
        else:
            left, right = prediction_energy, target_energy
        correlation = _series_correlation(left, right)
        if correlation is not None and (
            best_correlation is None or correlation > best_correlation
        ):
            best_lag = lag
            best_correlation = correlation
    return best_lag, best_correlation


def _temporal_velocity_spectrum(
    velocity: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    centered = velocity - velocity.mean(axis=0, keepdims=True)
    transform = np.fft.rfft(centered, axis=0)
    power = (
        np.sum(weights[None, :, None] * np.abs(transform) ** 2, axis=(1, 2))
        / weights.sum()
    )
    return power[1:]


def _masked_uv_rmse(error: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(math.sqrt(np.mean(error[:, mask] ** 2)))


def _frame_uv_relative_errors(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> list[float]:
    rows = []
    for frame in range(prediction.shape[0]):
        error_energy = float(
            np.sum(
                weights[:, None]
                * (prediction[frame, :, :2] - target[frame, :, :2]) ** 2
            )
        )
        target_energy = float(np.sum(weights[:, None] * target[frame, :, :2] ** 2))
        rows.append(_relative_rmse(error_energy, target_energy))
    return rows


def _seam_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    left_frame: int,
) -> dict[str, float]:
    right_frame = left_frame + 1
    prediction_step = prediction[right_frame, :, :2] - prediction[left_frame, :, :2]
    target_step = target[right_frame, :, :2] - target[left_frame, :, :2]
    step_error = prediction_step - target_step
    step_error_rmse = math.sqrt(
        np.sum(weights[:, None] * step_error**2) / (weights.sum() * 2)
    )
    prediction_step_rms = math.sqrt(
        np.sum(weights[:, None] * prediction_step**2) / (weights.sum() * 2)
    )
    target_step_rms = math.sqrt(
        np.sum(weights[:, None] * target_step**2) / (weights.sum() * 2)
    )
    prefix = f"seam_{left_frame}_{right_frame}"
    return {
        f"{prefix}_uv_step_rmse": float(step_error_rmse),
        f"{prefix}_predicted_step_rms": float(prediction_step_rms),
        f"{prefix}_target_step_rms": float(target_step_rms),
        f"{prefix}_step_rms_ratio": (
            float(prediction_step_rms / target_step_rms)
            if target_step_rms > 0
            else float("inf")
        ),
    }


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
    node_type: np.ndarray,
    dt: float,
) -> dict[str, Any]:
    """Compute matched raw-grid metrics for one complete 65-frame clip."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching shape [T,N,3]")
    if prediction.shape[0] != 65 or prediction.shape[2] != 3:
        raise ValueError("rollout evaluation requires exactly [65,N,3]")
    if not np.isfinite(target).all():
        raise ValueError("target contains non-finite values")
    if not np.isfinite(prediction).all():
        return {"finite": False}

    node_type = np.asarray(node_type).reshape(-1)
    weights = node_area_weights(points, cells)
    future_prediction = np.asarray(prediction[1:], dtype=np.float64)
    future_target = np.asarray(target[1:], dtype=np.float64)
    velocity_error = future_prediction[..., :2] - future_target[..., :2]
    velocity_error_energy = float(np.sum(weights[None, :, None] * velocity_error**2))
    velocity_target_energy = float(
        np.sum(weights[None, :, None] * future_target[..., :2] ** 2)
    )

    pressure_error = future_prediction[..., 2] - future_target[..., 2]
    pressure_offset = np.sum(weights[None, :] * pressure_error, axis=1) / weights.sum()
    gauge_error = pressure_error - pressure_offset[:, None]
    pressure_denominator = weights.sum() * pressure_error.shape[0]

    target_vorticity, target_divergence, triangle_area = triangle_vorticity_divergence(
        future_target[..., :2], points, cells
    )
    prediction_vorticity, prediction_divergence, _ = triangle_vorticity_divergence(
        future_prediction[..., :2], points, cells
    )
    triangle_denominator = triangle_area.sum() * future_target.shape[0]

    energy_prediction = np.sum(
        weights[None, :] * np.sum(future_prediction[..., :2] ** 2, axis=-1), axis=1
    ) / (2.0 * weights.sum())
    energy_target = np.sum(
        weights[None, :] * np.sum(future_target[..., :2] ** 2, axis=-1), axis=1
    ) / (2.0 * weights.sum())
    enstrophy_prediction = np.sum(
        triangle_area[None, :] * prediction_vorticity**2, axis=1
    ) / (2.0 * triangle_area.sum())
    enstrophy_target = np.sum(triangle_area[None, :] * target_vorticity**2, axis=1) / (
        2.0 * triangle_area.sum()
    )
    prediction_spectrum = _temporal_velocity_spectrum(
        future_prediction[..., :2], weights
    )
    target_spectrum = _temporal_velocity_spectrum(future_target[..., :2], weights)
    lag_frames, lag_correlation = best_energy_lag(energy_prediction, energy_target)

    per_frame_uv = _frame_uv_relative_errors(prediction, target, weights)
    conditioning_error = prediction[0, :, :2] - target[0, :, :2]
    metrics: dict[str, Any] = {
        "finite": True,
        "uv_rmse": math.sqrt(velocity_error_energy / (64 * weights.sum() * 2)),
        "uv_relative_rmse": _relative_rmse(
            velocity_error_energy, velocity_target_energy
        ),
        "pressure_raw_rmse": float(
            math.sqrt(
                np.sum(weights[None, :] * pressure_error**2) / pressure_denominator
            )
        ),
        "pressure_gauge_free_rmse": float(
            math.sqrt(np.sum(weights[None, :] * gauge_error**2) / pressure_denominator)
        ),
        "vorticity_rmse": float(
            math.sqrt(
                np.sum(
                    triangle_area[None, :]
                    * (prediction_vorticity - target_vorticity) ** 2
                )
                / triangle_denominator
            )
        ),
        "divergence_rmse": float(
            math.sqrt(
                np.sum(
                    triangle_area[None, :]
                    * (prediction_divergence - target_divergence) ** 2
                )
                / triangle_denominator
            )
        ),
        "energy_relative_rmse": _relative_rmse(
            float(np.sum((energy_prediction - energy_target) ** 2)),
            float(np.sum(energy_target**2)),
        ),
        "enstrophy_relative_rmse": _relative_rmse(
            float(np.sum((enstrophy_prediction - enstrophy_target) ** 2)),
            float(np.sum(enstrophy_target**2)),
        ),
        "temporal_velocity_spectrum_relative_l2": _relative_rmse(
            float(np.sum((prediction_spectrum - target_spectrum) ** 2)),
            float(np.sum(target_spectrum**2)),
        ),
        "velocity_field_correlation": weighted_correlation(
            future_prediction[..., :2], future_target[..., :2], weights
        ),
        "energy_series_correlation": _series_correlation(
            energy_prediction, energy_target
        ),
        "enstrophy_series_correlation": _series_correlation(
            enstrophy_prediction, enstrophy_target
        ),
        "energy_best_lag_frames": lag_frames,
        "energy_best_lag_time": lag_frames * dt,
        "energy_best_lag_correlation": lag_correlation,
        "conditioning_frame_uv_rmse": float(
            math.sqrt(
                np.sum(weights[:, None] * conditioning_error**2) / (weights.sum() * 2)
            )
        ),
        "boundary_uv_rmse": _masked_uv_rmse(velocity_error, node_type != 0),
        "inlet_uv_rmse": _masked_uv_rmse(velocity_error, node_type == 4),
        "outlet_uv_rmse": _masked_uv_rmse(velocity_error, node_type == 5),
        "wall_uv_rmse": _masked_uv_rmse(velocity_error, node_type == 6),
        "per_frame_uv_relative_rmse": per_frame_uv,
        "frame24_uv_relative_rmse": per_frame_uv[24],
        "frame25_uv_relative_rmse": per_frame_uv[25],
        "frame48_uv_relative_rmse": per_frame_uv[48],
        "frame49_uv_relative_rmse": per_frame_uv[49],
        "frame64_uv_relative_rmse": per_frame_uv[64],
        "per_frame_pressure_raw_rmse": np.sqrt(
            np.sum(weights[None, :] * pressure_error**2, axis=1) / weights.sum()
        ).tolist(),
        "per_frame_pressure_gauge_free_rmse": np.sqrt(
            np.sum(weights[None, :] * gauge_error**2, axis=1) / weights.sum()
        ).tolist(),
        "prediction_energy": energy_prediction.tolist(),
        "target_energy": energy_target.tolist(),
        "prediction_enstrophy": enstrophy_prediction.tolist(),
        "target_enstrophy": enstrophy_target.tolist(),
        "temporal_frequencies_hz": np.fft.rfftfreq(64, d=dt)[1:].tolist(),
        "prediction_temporal_spectrum": prediction_spectrum.tolist(),
        "target_temporal_spectrum": target_spectrum.tolist(),
    }
    metrics["prediction_divergence_rms"] = float(
        math.sqrt(
            np.sum(triangle_area[None, :] * prediction_divergence**2)
            / triangle_denominator
        )
    )
    metrics["target_divergence_rms"] = float(
        math.sqrt(
            np.sum(triangle_area[None, :] * target_divergence**2) / triangle_denominator
        )
    )
    return metrics


def summarize_trajectories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average sampling seeds within trajectory before population statistics."""
    grouped: dict[int, list[dict]] = {}
    identities = set()
    for row in rows:
        identity = (int(row["trajectory_index"]), int(row["seed"]))
        if identity in identities:
            raise ValueError("duplicate trajectory/sample in metric aggregation")
        identities.add(identity)
        grouped.setdefault(identity[0], []).append(row)
    trajectories = []
    for index, samples in sorted(grouped.items()):
        valid = all(row.get("finite", False) for row in samples)
        entry = {
            "trajectory_index": index,
            "finite": valid,
            "sample_count": len(samples),
            "failure_count": sum(not row.get("finite", False) for row in samples),
        }
        if valid:
            for key in samples[0]:
                if key in {"trajectory_index", "seed", "finite", "sample_seed"}:
                    continue
                values = [row.get(key) for row in samples]
                if all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in values
                ):
                    entry[key] = float(np.mean(values))
        trajectories.append(entry)
    summary = aggregate_rows(trajectories)
    failed = sum(not row.get("finite", False) for row in rows)
    primary = [row["uv_relative_rmse"] for row in trajectories if row["finite"]]
    summary.update(
        {
            "clip_count": len(rows),
            "trajectory_count": len(trajectories),
            "failed_clips": failed,
            "failed_trajectories": sum(not row["finite"] for row in trajectories),
            "selection_uv_relative_rmse": float(np.mean(primary)) if primary else None,
            "trajectory_metrics": trajectories,
            "aggregation": "sampling-seed mean within trajectory; then equal trajectory weight",
        }
    )
    return summary


def selection_key(summary: dict, checkpoint_update: int) -> tuple:
    score = summary.get("selection_uv_relative_rmse")
    return (
        summary["failed_clips"],
        float("inf") if score is None else score,
        checkpoint_update,
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize finite scalar values without silently dropping failed cases."""

    excluded = {
        "sample_index",
        "seed",
        "start",
        "stop",
        "trajectory_index",
        "source_local_index",
        "node_count",
        "cell_count",
        "rollout_steps",
    }
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key not in excluded
        }
    )
    aggregate: dict[str, Any] = {
        "rows": len(rows),
        "failure_count": sum(not row.get("finite", False) for row in rows),
    }
    for name in metric_names:
        values = np.asarray(
            [row[name] for row in rows if isinstance(row.get(name), (int, float))],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        aggregate[name] = {
            "finite_count": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "p90": float(np.quantile(values, 0.9)) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }
    return aggregate
