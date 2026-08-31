from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset.cylinderflow import CylinderFlowWindowDataset
from dataset.datamodule import FluidsDataModule
from modules.models.ddpm import LatentDiffusion
from modules.modules.ddim import DDIMSampler
from modules.utils import get_yaml
from tools.cylinderflow.metrics import (
    aggregate_rows,
    compute_metrics,
    render_comparison_gif,
    shared_visual_scales,
)
from tools.cylinderflow.rollout import (
    derive_segment_seed,
    rollout_three_segments,
    sample_model_segment,
    set_segment_seed,
)


EXPECTED_MILESTONES = (48000, 96000, 144000)


def write_json(file_path: Path, value: Any) -> None:
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def checkpoint_identity(file_path: Path) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(file_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"checkpoint has no state_dict: {file_path}")
    return int(checkpoint.get("global_step", -1)), checkpoint


def instantiate_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    ae_checkpoint: Path,
    datamodule: FluidsDataModule,
    device: torch.device,
) -> tuple[LatentDiffusion, int]:
    if not ae_checkpoint.is_file():
        raise FileNotFoundError(ae_checkpoint)
    model_config = copy.deepcopy(config["model"])
    model_config["first_stage_config"]["pretrained_path"] = str(ae_checkpoint.resolve())
    model = LatentDiffusion(
        **model_config,
        normalizer=datamodule.normalizer,
        use_embed=False,
    )
    global_step, checkpoint = checkpoint_identity(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)
    del checkpoint
    return model, global_step


def trajectory_averaged_uv(rows: list[dict[str, Any]]) -> float | None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("uv_relative_rmse")
        if (
            row.get("finite")
            and isinstance(value, (int, float))
            and math.isfinite(value)
        ):
            grouped[int(row["trajectory_index"])].append(float(value))
    if not grouped:
        return None
    trajectory_means = [sum(values) / len(values) for values in grouped.values()]
    return float(sum(trajectory_means) / len(trajectory_means))


def sample_file_name(sample_index: int, trajectory_index: int, start: int) -> str:
    return (
        f"sample_{sample_index:03d}_trajectory_{trajectory_index:03d}_"
        f"start_{start:03d}_seed0.npz"
    )


def save_seed_zero_sample(
    output_path: Path,
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    points: np.ndarray,
    cells: np.ndarray,
    node_type: np.ndarray,
    frame_indices: np.ndarray,
    physical_time: np.ndarray,
    conditioning_frames: np.ndarray,
    segment_seeds: list[int],
    start: int,
) -> None:
    np.savez_compressed(
        output_path,
        target=target.astype(np.float32, copy=False),
        prediction=prediction.astype(np.float32, copy=False),
        points=points.astype(np.float32, copy=False),
        cells=cells.astype(np.int32, copy=False),
        node_type=node_type.astype(np.int32, copy=False),
        frame_indices=frame_indices.astype(np.int32, copy=False),
        physical_time=physical_time.astype(np.float64, copy=False),
        segment_condition_frames=conditioning_frames.astype(np.float32, copy=False),
        segment_seeds=np.asarray(segment_seeds, dtype=np.int64),
        segment_condition_source_kind=np.asarray(
            ("truth", "prediction", "prediction"), dtype="U10"
        ),
        segment_condition_source_global_frame=np.asarray((0, 24, 48), dtype=np.int32),
        segment_condition_source_raw_frame=np.asarray(
            (start, start + 24, start + 48), dtype=np.int32
        ),
        segment_future_raw_frame_ranges=np.asarray(
            (
                (start + 1, start + 24),
                (start + 25, start + 48),
                (start + 49, start + 64),
            ),
            dtype=np.int32,
        ),
    )


def evaluate_one_checkpoint(
    *,
    checkpoint_path: Path,
    ae_checkpoint: Path,
    config: dict[str, Any],
    datamodule: FluidsDataModule,
    dataset: CylinderFlowWindowDataset,
    seeds: list[int],
    ddim_steps: int,
    device: torch.device,
    sample_dir: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, global_step = instantiate_model(
        config, checkpoint_path, ae_checkpoint, datamodule, device
    )
    sampler = DDIMSampler(model=model)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for evaluation_seed in seeds:
        for sample_index in range(len(dataset)):
            sample = dataset.__getitem__(sample_index, eval=True)
            metadata = sample["metadata"]
            trajectory_index = int(metadata["trajectory_index"])
            start = int(metadata["start"])
            target = sample["x"].numpy()
            points = sample["points"].numpy()
            cells = sample["cells"].numpy()
            node_type = sample["node_type"].numpy()
            normalized_points = sample["pos"][0, :, :2]
            segment_seeds = [
                derive_segment_seed(
                    evaluation_seed, trajectory_index, start, segment_index
                )
                for segment_index in range(3)
            ]
            sample_started = time.perf_counter()
            try:
                initial_frame = sample["x"][:1].to(device)

                def run_segment(
                    condition: torch.Tensor, segment_index: int
                ) -> torch.Tensor:
                    set_segment_seed(segment_seeds[segment_index])
                    return sample_model_segment(
                        model,
                        sampler,
                        condition,
                        normalized_points,
                        ddim_steps,
                    )

                rollout = rollout_three_segments(initial_frame, run_segment)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds = time.perf_counter() - sample_started
                prediction = rollout.prediction.float().cpu().numpy()
                conditioning_frames = (
                    torch.cat(rollout.conditioning_frames, dim=0).float().cpu().numpy()
                )
                metrics = compute_metrics(
                    prediction,
                    target,
                    points,
                    cells,
                    node_type,
                    float(metadata["frame_dt"]),
                )
                row: dict[str, Any] = {
                    **metadata,
                    **metrics,
                    "sample_index": sample_index,
                    "seed": evaluation_seed,
                    "segment_seeds": segment_seeds,
                    "segment_condition_sources": [
                        "truth_global_frame_0",
                        "prediction_global_frame_24",
                        "prediction_global_frame_48",
                    ],
                    "global_step": global_step,
                    "inference_seconds": inference_seconds,
                }
                if sample_dir is not None and evaluation_seed == seeds[0]:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    save_seed_zero_sample(
                        sample_dir
                        / sample_file_name(sample_index, trajectory_index, start),
                        target=target,
                        prediction=prediction,
                        points=points,
                        cells=cells,
                        node_type=node_type,
                        frame_indices=sample["frame_indices"].numpy(),
                        physical_time=sample["time"].numpy(),
                        conditioning_frames=conditioning_frames,
                        segment_seeds=segment_seeds,
                        start=start,
                    )
            except FloatingPointError as error:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                row = {
                    **metadata,
                    "finite": False,
                    "failure": str(error),
                    "sample_index": sample_index,
                    "seed": evaluation_seed,
                    "segment_seeds": segment_seeds,
                    "global_step": global_step,
                    "inference_seconds": time.perf_counter() - sample_started,
                }
            rows.append(row)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    aggregate = aggregate_rows(rows)
    aggregate["trajectory_averaged_uv_relative_rmse"] = trajectory_averaged_uv(rows)
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "ae_checkpoint": str(ae_checkpoint.resolve()),
        "global_step": global_step,
        "strict_checkpoint_load": True,
        "ddim_steps_per_segment": ddim_steps,
        "autoregressive_segments": 3,
        "future_frames": 64,
        "seeds": seeds,
        "aggregate": aggregate,
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
    }
    del sampler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, rows


def render_representative_gifs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    first_seed: int,
) -> dict[str, Any]:
    finite_rows = sorted(
        (
            row
            for row in rows
            if row.get("seed") == first_seed
            and row.get("finite")
            and isinstance(row.get("uv_relative_rmse"), (int, float))
        ),
        key=lambda row: row["uv_relative_rmse"],
    )
    if not finite_rows:
        raise RuntimeError("no finite seed-0 samples are available for rendering")
    rank_indices = {
        "median": len(finite_rows) // 2,
        "difficult": int(round(0.9 * (len(finite_rows) - 1))),
        "worst": len(finite_rows) - 1,
    }
    sample_dir = output_dir / "samples_seed0"
    selections: dict[str, Any] = {}
    npz_paths: list[Path] = []
    for label, rank_index in rank_indices.items():
        row = finite_rows[rank_index]
        file_path = sample_dir / sample_file_name(
            int(row["sample_index"]),
            int(row["trajectory_index"]),
            int(row["start"]),
        )
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        npz_paths.append(file_path)
        selections[label] = {
            "sample": str(file_path.resolve()),
            "trajectory_index": row["trajectory_index"],
            "start": row["start"],
            "uv_relative_rmse": row["uv_relative_rmse"],
        }

    scales = shared_visual_scales(npz_paths)
    gif_dir = output_dir / "gifs_shared_scale"
    gif_dir.mkdir()
    for label, file_path in zip(rank_indices, npz_paths):
        render_comparison_gif(file_path, gif_dir / f"{label}.gif", scales)
        selections[label]["gif"] = str((gif_dir / f"{label}.gif").resolve())
    record = {
        "schema": "text2pde.cylinderflow.visual_selection.v1",
        "selection_axis": "seed-0 rollout64 UV relative RMSE",
        "shared_scales": scales,
        "selections": selections,
    }
    write_json(output_dir / "visual_selection.json", record)
    return record


def validate_mode_dataset(
    mode: str, dataset: CylinderFlowWindowDataset, allow_small: bool = False
) -> None:
    if dataset.window_length != 65 or dataset.frame_stride != 1:
        raise ValueError("rollout evaluation requires contiguous 65-frame windows")
    expected_split = "test" if mode == "test" else "validation"
    if dataset.split != expected_split:
        raise ValueError(f"{mode} mode cannot read split {dataset.split}")
    if not allow_small:
        if dataset.trajectory_count != 100 or len(dataset) != 300:
            raise ValueError(
                f"formal {mode} requires 100 trajectories x 3 starts = 300 clips"
            )
        expected_windows = [
            (trajectory_index, start)
            for trajectory_index in range(100)
            for start in (0, 268, 535)
        ]
        observed_windows = [
            dataset.resolve_window(index) for index in range(len(dataset))
        ]
        if observed_windows != expected_windows:
            raise ValueError(
                "formal rollout windows are not trajectory-major 0/268/535"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("select", "validation", "test"), required=True
    )
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
    parser.add_argument("--render-representative-gifs", action="store_true")
    parser.add_argument(
        "--allow-small-protocol",
        action="store_true",
        help="Permit a reduced manifest for code smoke only.",
    )
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("evaluation seeds must be unique")
    if not args.allow_small_protocol:
        if args.seeds != [0, 1, 2]:
            raise ValueError("formal rollout evaluation requires seeds 0/1/2")
        if args.ddim_steps != 20:
            raise ValueError(
                "formal rollout evaluation requires 20 DDIM steps per segment"
            )
    if args.mode == "select":
        if args.checkpoint_dir is None or args.checkpoint is not None:
            raise ValueError("select mode requires --checkpoint-dir only")
        checkpoints = sorted(args.checkpoint_dir.glob("ldm-epoch*.ckpt"))
        if len(checkpoints) != 3:
            raise RuntimeError(
                f"expected three LDM milestone checkpoints, found {len(checkpoints)}"
            )
        steps = sorted(checkpoint_identity(path)[0] for path in checkpoints)
        if tuple(steps) != EXPECTED_MILESTONES:
            raise RuntimeError(
                f"LDM checkpoint steps {steps} != {list(EXPECTED_MILESTONES)}"
            )
    else:
        if args.checkpoint is None or args.checkpoint_dir is not None:
            raise ValueError(f"{args.mode} mode requires --checkpoint only")
        checkpoints = [args.checkpoint]
    if args.render_representative_gifs and (
        not args.save_first_seed or len(checkpoints) != 1
    ):
        raise ValueError(
            "representative GIFs require one checkpoint and --save-first-seed"
        )

    config = get_yaml(args.config)
    config["data"]["dataset"]["validation_manifest"] = str(args.manifest)
    datamodule = FluidsDataModule(config["data"])
    dataset = CylinderFlowWindowDataset(str(args.manifest), return_metadata=True)
    validate_mode_dataset(args.mode, dataset, allow_small=args.allow_small_protocol)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    candidate_results: list[dict[str, Any]] = []
    rows_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for checkpoint in checkpoints:
        sample_dir = (
            args.output_dir / "samples_seed0"
            if args.save_first_seed and len(checkpoints) == 1
            else None
        )
        result, rows = evaluate_one_checkpoint(
            checkpoint_path=checkpoint,
            ae_checkpoint=args.ae_checkpoint,
            config=config,
            datamodule=datamodule,
            dataset=dataset,
            seeds=args.seeds,
            ddim_steps=args.ddim_steps,
            device=device,
            sample_dir=sample_dir,
        )
        candidate_results.append(result)
        rows_by_checkpoint[str(checkpoint.resolve())] = rows

    if args.mode == "select":
        viable = [
            row
            for row in candidate_results
            if row["aggregate"]["trajectory_averaged_uv_relative_rmse"] is not None
        ]
        if not viable:
            raise FloatingPointError("all LDM checkpoint rollouts failed")
        selected = min(
            viable,
            key=lambda row: (
                row["aggregate"]["trajectory_averaged_uv_relative_rmse"],
                row["global_step"],
            ),
        )
        (args.output_dir / "selected_checkpoint.txt").write_text(
            selected["checkpoint"] + "\n", encoding="utf-8"
        )
    else:
        selected = candidate_results[0]

    summary = {
        "schema": "text2pde.cylinderflow.rollout64_evaluation.v1",
        "mode": args.mode,
        "manifest": str(args.manifest.resolve()),
        "test_accessed": args.mode == "test",
        "protocol": {
            "source_frames": 600,
            "source_dt": 0.01,
            "source_frame_stride": 1,
            "segment_protocol": "1 clean condition -> 24 future frames",
            "stitch": "1:25 + 1:25 + 1:17",
            "output_shape": "[65,N,3]",
            "condition_sources": [
                "truth frame 0",
                "predicted frame 24",
                "predicted frame 48",
            ],
        },
        "selection_metric": "trajectory-averaged rollout64 area-weighted UV relative RMSE",
        "candidates": candidate_results,
        "selected": selected,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "rows.json", rows_by_checkpoint)
    if args.render_representative_gifs:
        render_representative_gifs(
            args.output_dir,
            rows_by_checkpoint[str(checkpoints[0].resolve())],
            args.seeds[0],
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
