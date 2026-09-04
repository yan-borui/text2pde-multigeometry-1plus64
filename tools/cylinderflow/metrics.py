from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def node_area_weights(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
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
    include_segment_seams: bool = True,
) -> dict[str, Any]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching shape [T,N,3]")
    if prediction.shape[0] != 65 or prediction.shape[2] != 3:
        raise ValueError("rollout evaluation requires exactly [65,N,3]")
    if not np.isfinite(target).all():
        raise ValueError("target contains non-finite values")
    if not np.isfinite(prediction).all():
        return {"finite": False}

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
    }
    if include_segment_seams:
        metrics.update(_seam_metrics(prediction, target, weights, 24))
        metrics.update(_seam_metrics(prediction, target, weights, 48))
    return metrics


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key
            not in {
                "sample_index",
                "seed",
                "start",
                "stop",
                "trajectory_index",
                "global_step",
                "node_count",
                "window_length",
                "frame_stride",
                "frame_dt",
                "raw_frame_dt",
                "temporal_stride",
                "phase_offset",
                "sequence_start",
                "sequence_length",
                "stored_frame_count",
                "raw_frame_start",
                "raw_frame_stop_inclusive",
                "source_local_index",
                "evaluation_seed",
                "sampler_seed",
                "ddim_steps",
            }
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
            "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "p90": float(np.quantile(values, 0.9)) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }
    return aggregate


def _gauge_center_pressure(field: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pressure = field[..., 2]
    means = np.sum(weights[None, :] * pressure, axis=1) / weights.sum()
    return pressure - means[:, None]


def shared_visual_scales(npz_paths: Iterable[Path]) -> dict[str, float]:
    speed_fields = []
    speed_errors = []
    pressure_fields = []
    pressure_errors = []
    for file_path in npz_paths:
        data = np.load(file_path, allow_pickle=False)
        target = data["target"]
        prediction = data["prediction"]
        weights = node_area_weights(data["points"], data["cells"])
        target_speed = np.linalg.norm(target[..., :2], axis=-1)
        prediction_speed = np.linalg.norm(prediction[..., :2], axis=-1)
        target_pressure = _gauge_center_pressure(target, weights)
        prediction_pressure = _gauge_center_pressure(prediction, weights)
        speed_fields.extend((target_speed, prediction_speed))
        speed_errors.append(np.abs(prediction_speed - target_speed))
        pressure_fields.extend((np.abs(target_pressure), np.abs(prediction_pressure)))
        pressure_errors.append(np.abs(prediction_pressure - target_pressure))

    def robust_max(values: list[np.ndarray]) -> float:
        maximum = float(
            np.quantile(np.concatenate([value.ravel() for value in values]), 0.995)
        )
        return max(maximum, np.finfo(np.float64).eps)

    return {
        "speed_max": robust_max(speed_fields),
        "speed_error_max": robust_max(speed_errors),
        "gauge_pressure_abs_max": robust_max(pressure_fields),
        "gauge_pressure_error_max": robust_max(pressure_errors),
    }


def render_comparison_gif(
    npz_path: Path,
    output_path: Path,
    scales: dict[str, float],
    segment_seams: tuple[int, ...] = (24, 48),
) -> None:
    data = np.load(npz_path, allow_pickle=False)
    target = data["target"]
    prediction = data["prediction"]
    points = data["points"]
    cells = data["cells"]
    time_values = data["physical_time"]
    weights = node_area_weights(points, cells)
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    target_speed = np.linalg.norm(target[..., :2], axis=-1)
    prediction_speed = np.linalg.norm(prediction[..., :2], axis=-1)
    speed_error = np.abs(prediction_speed - target_speed)
    target_pressure = _gauge_center_pressure(target, weights)
    prediction_pressure = _gauge_center_pressure(prediction, weights)
    pressure_error = np.abs(prediction_pressure - target_pressure)

    frames = []
    for frame_index in range(target.shape[0]):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
        fields = (
            (
                target_speed[frame_index],
                "truth speed",
                "turbo",
                0.0,
                scales["speed_max"],
            ),
            (
                prediction_speed[frame_index],
                "Text2PDE speed",
                "turbo",
                0.0,
                scales["speed_max"],
            ),
            (
                speed_error[frame_index],
                "absolute speed error",
                "magma",
                0.0,
                scales["speed_error_max"],
            ),
            (
                target_pressure[frame_index],
                "truth gauge pressure",
                "coolwarm",
                -scales["gauge_pressure_abs_max"],
                scales["gauge_pressure_abs_max"],
            ),
            (
                prediction_pressure[frame_index],
                "Text2PDE gauge pressure",
                "coolwarm",
                -scales["gauge_pressure_abs_max"],
                scales["gauge_pressure_abs_max"],
            ),
            (
                pressure_error[frame_index],
                "gauge pressure error",
                "magma",
                0.0,
                scales["gauge_pressure_error_max"],
            ),
        )
        for axis, (field, title, cmap, lower, upper) in zip(axes.flat, fields):
            artist = axis.tripcolor(
                triangulation,
                field,
                shading="gouraud",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.set_axis_off()
            fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
        seam_label = ""
        if any(frame_index in (left, left + 1) for left in segment_seams):
            seam_label = " | rollout seam neighborhood"
        fig.suptitle(
            f"raw frame {int(data['frame_indices'][frame_index])}, "
            f"t={time_values[frame_index]:.2f}{seam_label}"
        )
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
    )
