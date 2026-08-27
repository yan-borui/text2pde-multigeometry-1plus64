from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from dataset.datamodule import FluidsDataModule
from dataset.multigeometry import MultiGeometryWindowDataset
from modules.models.ddpm import LatentDiffusion
from modules.modules.ddim import DDIMSampler
from modules.utils import get_yaml


def node_area_weights(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    triangle_points = points[cells]
    cross = (
        (triangle_points[:, 1, 0] - triangle_points[:, 0, 0])
        * (triangle_points[:, 2, 1] - triangle_points[:, 0, 1])
        - (triangle_points[:, 2, 0] - triangle_points[:, 0, 0])
        * (triangle_points[:, 1, 1] - triangle_points[:, 0, 1])
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


def triangle_vorticity(
    velocity: np.ndarray, points: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangle_points = points[cells]
    x0, y0 = triangle_points[:, 0, 0], triangle_points[:, 0, 1]
    x1, y1 = triangle_points[:, 1, 0], triangle_points[:, 1, 1]
    x2, y2 = triangle_points[:, 2, 0], triangle_points[:, 2, 1]
    determinant = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if np.any(np.abs(determinant) <= np.finfo(np.float64).eps):
        raise ValueError("mesh contains a degenerate triangle")

    u = velocity[..., 0][:, cells]
    v = velocity[..., 1][:, cells]
    du_dy = (
        u[:, :, 0] * (x2 - x1)
        + u[:, :, 1] * (x0 - x2)
        + u[:, :, 2] * (x1 - x0)
    ) / determinant
    dv_dx = (
        v[:, :, 0] * (y1 - y2)
        + v[:, :, 1] * (y2 - y0)
        + v[:, :, 2] * (y0 - y1)
    ) / determinant
    area = 0.5 * np.abs(determinant)
    return dv_dx - du_dy, area


def weighted_correlation(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> float:
    repeated_weights = np.broadcast_to(
        weights[None, :, None], prediction.shape
    ).reshape(-1)
    prediction_flat = prediction.reshape(-1)
    target_flat = target.reshape(-1)
    weight_sum = repeated_weights.sum()
    prediction_mean = np.sum(repeated_weights * prediction_flat) / weight_sum
    target_mean = np.sum(repeated_weights * target_flat) / weight_sum
    prediction_centered = prediction_flat - prediction_mean
    target_centered = target_flat - target_mean
    covariance = np.sum(
        repeated_weights * prediction_centered * target_centered
    )
    denominator = math.sqrt(
        np.sum(repeated_weights * prediction_centered**2)
        * np.sum(repeated_weights * target_centered**2)
    )
    return float(covariance / denominator) if denominator > 0 else float("nan")


def best_energy_lag(
    prediction: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    maximum_lag: int = 10,
) -> tuple[int, float]:
    prediction_energy = np.sum(
        weights[None, :] * np.sum(prediction**2, axis=-1), axis=1
    ) / weights.sum()
    target_energy = np.sum(
        weights[None, :] * np.sum(target**2, axis=-1), axis=1
    ) / weights.sum()
    best_lag = 0
    best_correlation = -np.inf
    for lag in range(-maximum_lag, maximum_lag + 1):
        if lag < 0:
            left, right = prediction_energy[-lag:], target_energy[:lag]
        elif lag > 0:
            left, right = prediction_energy[:-lag], target_energy[lag:]
        else:
            left, right = prediction_energy, target_energy
        if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
            correlation = -np.inf
        else:
            correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > best_correlation:
            best_lag = lag
            best_correlation = correlation
    return best_lag, best_correlation


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
    boundary: np.ndarray,
    dt: float,
) -> dict[str, float | bool]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must both have shape [T,N,3]")
    finite = bool(np.isfinite(prediction).all())
    if not finite:
        return {"finite": False}

    weights = node_area_weights(points, cells)
    future_prediction = prediction[1:]
    future_target = target[1:]
    velocity_error = future_prediction[..., :2] - future_target[..., :2]
    velocity_numerator = np.sum(
        weights[None, :, None] * velocity_error**2
    )
    velocity_denominator = np.sum(
        weights[None, :, None] * future_target[..., :2] ** 2
    )
    velocity_relative_rmse = math.sqrt(velocity_numerator / velocity_denominator)

    pressure_error = future_prediction[..., 2] - future_target[..., 2]
    raw_pressure_rmse = math.sqrt(
        np.sum(weights[None, :] * pressure_error**2)
        / (weights.sum() * pressure_error.shape[0])
    )
    pressure_offset = np.sum(weights[None, :] * pressure_error, axis=1) / weights.sum()
    gauge_error = pressure_error - pressure_offset[:, None]
    gauge_pressure_rmse = math.sqrt(
        np.sum(weights[None, :] * gauge_error**2)
        / (weights.sum() * gauge_error.shape[0])
    )

    target_vorticity, triangle_area = triangle_vorticity(
        future_target[..., :2], points, cells
    )
    prediction_vorticity, _ = triangle_vorticity(
        future_prediction[..., :2], points, cells
    )
    vorticity_rmse = math.sqrt(
        np.sum(
            triangle_area[None, :]
            * (prediction_vorticity - target_vorticity) ** 2
        )
        / (triangle_area.sum() * target_vorticity.shape[0])
    )

    boundary_mask = boundary != 0
    boundary_uv_rmse = math.sqrt(
        np.mean(velocity_error[:, boundary_mask] ** 2)
    )
    inlet_uv_rmse = math.sqrt(np.mean(velocity_error[:, boundary == 2] ** 2))
    outlet_uv_rmse = math.sqrt(np.mean(velocity_error[:, boundary == 3] ** 2))
    wall_uv_rmse = math.sqrt(np.mean(velocity_error[:, boundary == 4] ** 2))
    conditioning_frame_uv_rmse = math.sqrt(
        np.sum(
            weights[:, None]
            * (prediction[0, :, :2] - target[0, :, :2]) ** 2
        )
        / (weights.sum() * 2)
    )
    velocity_correlation = weighted_correlation(
        future_prediction[..., :2], future_target[..., :2], weights
    )
    lag_frames, lag_correlation = best_energy_lag(
        future_prediction[..., :2], future_target[..., :2], weights
    )
    return {
        "finite": True,
        "uv_relative_rmse": velocity_relative_rmse,
        "pressure_raw_rmse": raw_pressure_rmse,
        "pressure_gauge_free_rmse": gauge_pressure_rmse,
        "vorticity_rmse": vorticity_rmse,
        "boundary_uv_rmse": boundary_uv_rmse,
        "inlet_uv_rmse": inlet_uv_rmse,
        "outlet_uv_rmse": outlet_uv_rmse,
        "wall_uv_rmse": wall_uv_rmse,
        "conditioning_frame_uv_rmse": conditioning_frame_uv_rmse,
        "velocity_field_correlation": velocity_correlation,
        "energy_best_lag_frames": lag_frames,
        "energy_best_lag_time": lag_frames * dt,
        "energy_best_lag_correlation": lag_correlation,
    }


def instantiate_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    ae_checkpoint: Path,
    datamodule: FluidsDataModule,
    device: torch.device,
) -> tuple[LatentDiffusion, int]:
    model_config = copy.deepcopy(config["model"])
    model_config["first_stage_config"]["pretrained_path"] = str(ae_checkpoint)
    model = LatentDiffusion(
        **model_config,
        normalizer=datamodule.normalizer,
        use_embed=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    global_step = int(checkpoint.get("global_step", -1))
    model.eval().to(device)
    return model, global_step


@torch.inference_mode()
def ae_ceiling(
    model: LatentDiffusion, target: torch.Tensor, pos: torch.Tensor
) -> torch.Tensor:
    normalized = model.normalizer.normalize(target)
    posterior = model.first_stage_model.encode(
        normalized,
        pos,
        model.first_stage_model.latent_grid,
    )
    reconstruction = model.first_stage_model.decode(
        posterior.mode(),
        model.first_stage_model.latent_grid,
        pos,
    )
    return model.normalizer.denormalize(reconstruction)


@torch.inference_mode()
def sample_prediction(
    model: LatentDiffusion,
    sampler: DDIMSampler,
    target: torch.Tensor,
    pos: torch.Tensor,
    ddim_steps: int,
) -> torch.Tensor:
    normalized = model.normalizer.normalize(target)
    conditioning = model.get_learned_conditioning(
        (normalized[:, :1], pos[:, :1, :, :2], None)
    )
    shape = (
        1,
        model.channels,
        model.image_size[0],
        model.image_size[1],
        model.image_size[2],
    )
    latent, _ = sampler.sample(
        S=ddim_steps,
        batch_size=1,
        shape=shape,
        conditioning=conditioning,
        eta=0.0,
        verbose=False,
    )
    return model.decode_first_stage(latent, pos)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and key not in {"sample_index", "seed", "start", "global_step"}
        }
    )
    aggregate: dict[str, Any] = {
        "rows": len(rows),
        "failure_count": sum(not row.get("finite", False) for row in rows),
    }
    for name in metric_names:
        values = np.asarray([row[name] for row in rows if name in row], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        aggregate[name] = {
            "mean": float(finite_values.mean()) if finite_values.size else None,
            "median": float(np.median(finite_values)) if finite_values.size else None,
            "min": float(finite_values.min()) if finite_values.size else None,
            "max": float(finite_values.max()) if finite_values.size else None,
        }
    return aggregate


def render_comparison_gif(npz_path: Path, output_path: Path) -> None:
    data = np.load(npz_path, allow_pickle=False)
    target = data["target"]
    prediction = data["prediction"]
    points = data["points"]
    cells = data["cells"]
    time_values = data["time"]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], cells)
    target_speed = np.linalg.norm(target[..., :2], axis=-1)
    prediction_speed = np.linalg.norm(prediction[..., :2], axis=-1)
    speed_max = float(np.quantile(np.concatenate((target_speed, prediction_speed)), 0.995))
    speed_error = np.abs(prediction_speed - target_speed)
    speed_error_max = float(np.quantile(speed_error, 0.995))
    pressure_abs = float(
        np.quantile(
            np.abs(np.concatenate((target[..., 2], prediction[..., 2]))), 0.995
        )
    )
    pressure_error = np.abs(prediction[..., 2] - target[..., 2])
    pressure_error_max = float(np.quantile(pressure_error, 0.995))

    frames = []
    for frame_index in range(target.shape[0]):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
        fields = (
            (target_speed[frame_index], "truth speed", "turbo", 0.0, speed_max),
            (prediction_speed[frame_index], "Text2PDE speed", "turbo", 0.0, speed_max),
            (speed_error[frame_index], "absolute speed error", "magma", 0.0, speed_error_max),
            (target[frame_index, :, 2], "truth pressure", "coolwarm", -pressure_abs, pressure_abs),
            (prediction[frame_index, :, 2], "Text2PDE pressure", "coolwarm", -pressure_abs, pressure_abs),
            (pressure_error[frame_index], "absolute pressure error", "magma", 0.0, pressure_error_max),
        )
        for axis, (field, title, cmap, lower, upper) in zip(axes.flat, fields):
            artist = axis.tripcolor(
                triangulation,
                field,
                shading="gouraud",
                cmap=cmap,
                vmin=lower,
                vmax=max(upper, np.finfo(float).eps),
            )
            axis.set_title(title)
            axis.set_aspect("equal")
            axis.set_axis_off()
            fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
        fig.suptitle(f"t*={time_values[frame_index]:.1f}")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
    )


def evaluate_one_checkpoint(
    checkpoint_path: Path,
    ae_checkpoint: Path,
    config: dict[str, Any],
    datamodule: FluidsDataModule,
    dataset: MultiGeometryWindowDataset,
    seeds: list[int],
    ddim_steps: int,
    device: torch.device,
    sample_dir: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, global_step = instantiate_model(
        config, checkpoint_path, ae_checkpoint, datamodule, device
    )
    sampler = DDIMSampler(model=model)
    rows = []
    baseline_rows = []
    ae_rows = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for sample_index in range(len(dataset)):
            sample = dataset.__getitem__(sample_index, eval=True)
            target_tensor = sample["x"].unsqueeze(0).to(device)
            pos_tensor = sample["pos"].unsqueeze(0).to(device)
            target = sample["x"].numpy()
            points = sample["pos"][0, :, :2].numpy()
            cells = sample["cells"].numpy()
            boundary = sample["boundary"].numpy()
            dt = float(sample["time"][1] - sample["time"][0])

            sample_started = time.perf_counter()
            prediction_tensor = sample_prediction(
                model, sampler, target_tensor, pos_tensor, ddim_steps
            )
            torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - sample_started
            prediction = prediction_tensor[0].float().cpu().numpy()
            metrics = compute_metrics(
                prediction, target, points, cells, boundary, dt
            )
            row = {
                **sample["metadata"],
                **metrics,
                "sample_index": sample_index,
                "seed": seed,
                "global_step": global_step,
                "inference_seconds": inference_seconds,
            }
            rows.append(row)

            if seed == seeds[0]:
                persistence = np.repeat(target[:1], target.shape[0], axis=0)
                baseline_rows.append(
                    {
                        **sample["metadata"],
                        **compute_metrics(
                            persistence, target, points, cells, boundary, dt
                        ),
                        "sample_index": sample_index,
                        "method": "persistence",
                    }
                )
                ceiling = ae_ceiling(model, target_tensor, pos_tensor)[0].float().cpu().numpy()
                ae_rows.append(
                    {
                        **sample["metadata"],
                        **compute_metrics(ceiling, target, points, cells, boundary, dt),
                        "sample_index": sample_index,
                        "method": "ae_posterior_mode",
                    }
                )
                if sample_dir is not None:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        sample_dir / f"sample_{sample_index:03d}_{sample['metadata']['case_id']}_start{sample['metadata']['start']}.npz",
                        target=target,
                        prediction=prediction,
                        points=points,
                        cells=cells,
                        boundary=boundary,
                        time=sample["time"].numpy(),
                    )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "checkpoint": str(checkpoint_path),
        "ae_checkpoint": str(ae_checkpoint),
        "global_step": global_step,
        "ddim_steps": ddim_steps,
        "seeds": seeds,
        "aggregate": aggregate_rows(rows),
        "persistence": aggregate_rows(baseline_rows),
        "ae_ceiling": aggregate_rows(ae_rows),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    del sampler, model
    torch.cuda.empty_cache()
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("select", "validation", "test"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-first-seed", action="store_true")
    parser.add_argument("--render-family-gifs", action="store_true")
    args = parser.parse_args()

    if args.mode == "select":
        if args.checkpoint_dir is None or args.checkpoint is not None:
            raise ValueError("select mode requires --checkpoint-dir only")
        checkpoints = sorted(args.checkpoint_dir.glob("ldm-epoch*.ckpt"))
        if len(checkpoints) != 3:
            raise RuntimeError(
                f"expected three LDM cycle checkpoints, found {len(checkpoints)}"
            )
    else:
        if args.checkpoint is None or args.checkpoint_dir is not None:
            raise ValueError(f"{args.mode} mode requires --checkpoint only")
        checkpoints = [args.checkpoint]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = get_yaml(args.config)
    config["data"]["dataset"]["validation_manifest"] = str(args.manifest)
    datamodule = FluidsDataModule(config["data"])
    dataset = MultiGeometryWindowDataset(str(args.manifest), return_metadata=True)
    device = torch.device(args.device)

    candidate_results = []
    rows_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for checkpoint in checkpoints:
        sample_dir = None
        if args.save_first_seed and len(checkpoints) == 1:
            sample_dir = args.output_dir / "samples_seed0"
        result, rows = evaluate_one_checkpoint(
            checkpoint,
            args.ae_checkpoint,
            config,
            datamodule,
            dataset,
            args.seeds,
            args.ddim_steps,
            device,
            sample_dir,
        )
        candidate_results.append(result)
        rows_by_checkpoint[str(checkpoint)] = rows

    if args.mode == "select":
        selected = min(
            candidate_results,
            key=lambda record: (
                record["aggregate"]["uv_relative_rmse"]["mean"],
                record["global_step"],
            ),
        )
        (args.output_dir / "selected_checkpoint.txt").write_text(
            selected["checkpoint"] + "\n", encoding="utf-8"
        )
    else:
        selected = candidate_results[0]

    summary = {
        "schema": "text2pde.multigeometry.ldm_evaluation.v1",
        "mode": args.mode,
        "manifest": str(args.manifest),
        "test_accessed": args.mode == "test",
        "selection_metric": "mean_case_seed_future_area_weighted_uv_relative_rmse",
        "candidates": candidate_results,
        "selected": selected,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (args.output_dir / "rows.json").open("w", encoding="utf-8") as handle:
        json.dump(rows_by_checkpoint, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.render_family_gifs:
        if not args.save_first_seed or len(checkpoints) != 1:
            raise ValueError("GIF rendering requires one checkpoint and --save-first-seed")
        first_seed_rows = [
            row
            for row in rows_by_checkpoint[str(checkpoints[0])]
            if row["seed"] == args.seeds[0]
        ]
        gif_dir = args.output_dir / "gifs"
        gif_dir.mkdir()
        sample_dir = args.output_dir / "samples_seed0"
        for family in sorted({row["geometry_family"] for row in first_seed_rows}):
            family_rows = sorted(
                (row for row in first_seed_rows if row["geometry_family"] == family),
                key=lambda row: row["uv_relative_rmse"],
            )
            selected_row = family_rows[len(family_rows) // 2]
            pattern = (
                f"sample_{selected_row['sample_index']:03d}_"
                f"{selected_row['case_id']}_start{selected_row['start']}.npz"
            )
            render_comparison_gif(sample_dir / pattern, gif_dir / f"{family}.gif")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
