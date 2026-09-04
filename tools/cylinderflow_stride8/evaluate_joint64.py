from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset
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
from tools.cylinderflow_stride8.joint64 import (
    derive_sample_seed,
    sample_joint64,
    set_sample_seed,
)
from tools.cylinderflow_stride8.protocol import (
    DDIM_STEPS,
    LDM_MILESTONES,
    SAMPLING_SEEDS,
    validate_locked_config,
    validation_monitor_indices,
)


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
    return model, global_step


def sample_file_name(sample_index: int, trajectory_index: int) -> str:
    return f"sample_{sample_index:03d}_trajectory_{trajectory_index:04d}.npz"


def save_seed_zero_sample(
    file_path: Path,
    sample: dict[str, Any],
    prediction: np.ndarray,
    evaluation_seed: int,
    sampler_seed: int,
) -> None:
    np.savez_compressed(
        file_path,
        target=sample["x"].numpy().astype(np.float32, copy=False),
        prediction=prediction.astype(np.float32, copy=False),
        points=sample["mesh_pos"].numpy().astype(np.float32, copy=False),
        cells=sample["cells"].numpy().astype(np.int64, copy=False),
        node_type=sample["node_type"].numpy().astype(np.int64, copy=False),
        frame_indices=sample["frame_indices"].numpy().astype(np.int64, copy=False),
        physical_time=sample["time"].numpy().astype(np.float64, copy=False),
        evaluation_seed=np.asarray(evaluation_seed, dtype=np.int64),
        sampler_seed=np.asarray(sampler_seed, dtype=np.int64),
        trajectory_index=np.asarray(
            sample["metadata"]["trajectory_index"], dtype=np.int64
        ),
        sequence_rule=np.asarray(
            "one DDIM call; clean frame 0 + predicted frames 1:65"
        ),
    )


def evaluate_one_checkpoint(
    checkpoint_path: Path,
    ae_checkpoint: Path,
    config: dict[str, Any],
    datamodule: FluidsDataModule,
    dataset: CylinderFlowStride8TrajectoryDataset,
    indices: tuple[int, ...],
    seeds: tuple[int, ...],
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
        for sample_index in indices:
            sample = dataset.__getitem__(sample_index, eval=True)
            metadata = sample["metadata"]
            trajectory_index = int(metadata["trajectory_index"])
            sampler_seed = derive_sample_seed(evaluation_seed, trajectory_index)
            set_sample_seed(sampler_seed)
            target_tensor = sample["x"].unsqueeze(0).to(device)
            position_tensor = sample["pos"].unsqueeze(0).to(device)
            sample_started = time.perf_counter()
            try:
                prediction_tensor = sample_joint64(
                    model,
                    sampler,
                    target_tensor,
                    position_tensor,
                    ddim_steps=DDIM_STEPS,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                inference_seconds = time.perf_counter() - sample_started
                prediction = prediction_tensor[0].float().cpu().numpy()
                metrics = compute_metrics(
                    prediction,
                    sample["x"].numpy(),
                    sample["mesh_pos"].numpy(),
                    sample["cells"].numpy(),
                    sample["node_type"].numpy(),
                    dt=0.08,
                    include_segment_seams=False,
                )
                row = {
                    **metadata,
                    **metrics,
                    "sample_index": sample_index,
                    "evaluation_seed": evaluation_seed,
                    "sampler_seed": sampler_seed,
                    "global_step": global_step,
                    "ddim_steps": DDIM_STEPS,
                    "inference_seconds": inference_seconds,
                }
                rows.append(row)
                if sample_dir is not None and evaluation_seed == seeds[0]:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    save_seed_zero_sample(
                        sample_dir / sample_file_name(sample_index, trajectory_index),
                        sample,
                        prediction,
                        evaluation_seed,
                        sampler_seed,
                    )
            except FloatingPointError as error:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                rows.append(
                    {
                        **metadata,
                        "sample_index": sample_index,
                        "evaluation_seed": evaluation_seed,
                        "sampler_seed": sampler_seed,
                        "global_step": global_step,
                        "ddim_steps": DDIM_STEPS,
                        "inference_seconds": time.perf_counter() - sample_started,
                        "finite": False,
                        "failure": str(error),
                    }
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    finite_count = sum(bool(row.get("finite")) for row in rows)
    total_inference_seconds = sum(float(row["inference_seconds"]) for row in rows)
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "ae_checkpoint": str(ae_checkpoint.resolve()),
        "global_step": global_step,
        "ddim_steps": DDIM_STEPS,
        "autoregressive_segments": 0,
        "sampling_protocol": "one joint 1 clean + 64 future DDIM call",
        "evaluation_seeds": list(seeds),
        "validation_indices": list(indices),
        "aggregate": aggregate_rows(rows),
        "finite_sequences": finite_count,
        "total_sequences": len(rows),
        "elapsed_seconds": elapsed_seconds,
        "total_inference_seconds": total_inference_seconds,
        "sequences_per_inference_second": (
            len(rows) / total_inference_seconds if total_inference_seconds > 0 else None
        ),
        "future_frames_per_inference_second": (
            finite_count * 64 / total_inference_seconds
            if total_inference_seconds > 0
            else None
        ),
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


def render_representatives(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    finite_seed_zero = sorted(
        (
            row
            for row in rows
            if row["evaluation_seed"] == SAMPLING_SEEDS[0]
            and row.get("finite")
            and math.isfinite(float(row["uv_relative_rmse"]))
        ),
        key=lambda row: float(row["uv_relative_rmse"]),
    )
    if not finite_seed_zero:
        raise RuntimeError("no finite seed-0 Validation samples can be rendered")
    positions = {
        "favorable": 0,
        "median": len(finite_seed_zero) // 2,
        "difficult": min(
            len(finite_seed_zero) - 1,
            int(round(0.9 * (len(finite_seed_zero) - 1))),
        ),
        "worst": len(finite_seed_zero) - 1,
    }
    sample_dir = output_dir / "samples_seed0"
    selections: dict[str, Any] = {}
    paths = []
    for label, position in positions.items():
        row = finite_seed_zero[position]
        file_path = sample_dir / sample_file_name(
            int(row["sample_index"]), int(row["trajectory_index"])
        )
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        paths.append(file_path)
        selections[label] = {
            "sample": str(file_path.resolve()),
            "sample_index": int(row["sample_index"]),
            "trajectory_index": int(row["trajectory_index"]),
            "uv_relative_rmse": float(row["uv_relative_rmse"]),
        }
    scales = shared_visual_scales(paths)
    gif_dir = output_dir / "gifs_shared_scale"
    gif_dir.mkdir()
    for label, record in selections.items():
        gif_path = gif_dir / f"{label}.gif"
        render_comparison_gif(
            Path(record["sample"]), gif_path, scales, segment_seams=()
        )
        record["gif"] = str(gif_path.resolve())
    record = {"shared_scales": scales, "selections": selections}
    write_json(output_dir / "representative_samples.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("select", "validation"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SAMPLING_SEEDS))
    parser.add_argument("--ddim-steps", type=int, default=DDIM_STEPS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    if seeds != SAMPLING_SEEDS:
        raise ValueError("formal selection/evaluation requires sampling seeds 0 1 2")
    if args.ddim_steps != DDIM_STEPS:
        raise ValueError("formal joint64 evaluation requires 20 DDIM steps")
    if args.mode == "select":
        if args.checkpoint_dir is None or args.checkpoint is not None:
            raise ValueError("select mode requires --checkpoint-dir only")
        checkpoints = sorted(args.checkpoint_dir.glob("ldm-epoch*.ckpt"))
        identities = [checkpoint_identity(path)[0] for path in checkpoints]
        if len(checkpoints) != 4 or tuple(sorted(identities)) != LDM_MILESTONES:
            raise RuntimeError(
                f"LDM checkpoint steps {sorted(identities)} != {list(LDM_MILESTONES)}"
            )
        indices = validation_monitor_indices()
    else:
        if args.checkpoint is None or args.checkpoint_dir is not None:
            raise ValueError("validation mode requires --checkpoint only")
        checkpoints = [args.checkpoint]
        indices = tuple(range(100))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = get_yaml(args.config)
    validate_locked_config(config, "ldm")
    datamodule = FluidsDataModule(config["data"])
    dataset = datamodule.val_dataset
    if not isinstance(dataset, CylinderFlowStride8TrajectoryDataset):
        raise TypeError("joint64 evaluation requires the stride-8 dataset")
    if dataset.split != "validation" or len(dataset) != 100:
        raise ValueError("joint64 evaluation requires all 100 Validation trajectories")
    if not args.ae_checkpoint.is_file():
        raise FileNotFoundError(args.ae_checkpoint)
    device = torch.device(args.device)

    candidate_results = []
    rows_by_checkpoint: dict[str, list[dict[str, Any]]] = {}
    for checkpoint in checkpoints:
        sample_dir = (
            args.output_dir / "samples_seed0" if args.mode == "validation" else None
        )
        result, rows = evaluate_one_checkpoint(
            checkpoint,
            args.ae_checkpoint,
            config,
            datamodule,
            dataset,
            indices,
            seeds,
            device,
            sample_dir,
        )
        expected_rows = len(indices) * len(seeds)
        if len(rows) != expected_rows:
            raise AssertionError(
                f"evaluation emitted {len(rows)} != {expected_rows} rows"
            )
        candidate_results.append(result)
        rows_by_checkpoint[str(checkpoint.resolve())] = rows

    if args.mode == "select":
        for result in candidate_results:
            value = result["aggregate"].get("uv_relative_rmse", {}).get("mean")
            if value is None or not math.isfinite(float(value)):
                raise FloatingPointError("candidate has no finite UV selection metric")
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
        if selected["total_sequences"] != 300:
            raise AssertionError(
                "formal Validation must contain exactly 100 x 3 samples"
            )

    summary = {
        "schema": "text2pde.cylinderflow_stride8.joint64_evaluation.v1",
        "mode": args.mode,
        "selection_metric": "mean trajectory-seed area-weighted UV relative RMSE",
        "physical_dt": 0.08,
        "sequence_raw_indices": "0:520:8",
        "condition": "clean frame 0 only",
        "prediction": "frames 1 through 64 from one joint DDIM sample",
        "ddim_steps": DDIM_STEPS,
        "sampling_seeds": list(seeds),
        "test_entry_available": False,
        "test_accessed": False,
        "candidates": candidate_results,
        "selected": selected,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "rows.json", rows_by_checkpoint)
    if args.mode == "validation":
        rows = rows_by_checkpoint[str(checkpoints[0].resolve())]
        render_representatives(args.output_dir, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
