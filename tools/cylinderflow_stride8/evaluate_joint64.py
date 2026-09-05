from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset

if TYPE_CHECKING:
    from dataset.datamodule import FluidsDataModule
    from modules.models.ddpm import LatentDiffusion
from modules.utils import get_yaml
from tools.cylinderflow_stride8.metrics import (
    compute_metrics,
    selection_key,
    summarize_trajectories,
)
from tools.cylinderflow_stride8.predictions import (
    boundary_metrics,
    save_prediction,
    writeback_velocity,
)
from tools.cylinderflow_stride8.evaluation_io import append_json, write_csv, write_json

from tools.cylinderflow_stride8.joint64 import (
    derive_sample_seed,
    sample_joint64,
    set_sample_seed,
)
from tools.cylinderflow_stride8.protocol import (
    DDIM_STEPS,
    LDM_MILESTONES,
    SAMPLING_SEEDS,
    checkpoint_identifier,
    stage_data_contract,
    validate_ae_dependency,
    validate_checkpoint_contract,
    validate_locked_config,
    validation_monitor_indices,
)


def make_sampler(model):
    from modules.modules.ddim import DDIMSampler

    return DDIMSampler(model=model)


def checkpoint_identity(file_path: Path) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(file_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"checkpoint has no state_dict: {file_path}")
    validate_checkpoint_contract(checkpoint, "ldm")
    return int(checkpoint.get("global_step", -1)), checkpoint


def instantiate_model(
    config: dict[str, Any],
    checkpoint_path: Path,
    ae_checkpoint: Path,
    datamodule: FluidsDataModule,
    device: torch.device,
) -> tuple[LatentDiffusion, int]:
    from modules.models.ddpm import LatentDiffusion

    ae_state = torch.load(ae_checkpoint, map_location="cpu")
    global_step, checkpoint = checkpoint_identity(checkpoint_path)
    dependencies = validate_ae_dependency(checkpoint, ae_state)
    del ae_state
    model_config = copy.deepcopy(config["model"])
    model_config["first_stage_config"]["pretrained_path"] = str(ae_checkpoint.resolve())
    model = LatentDiffusion(
        **model_config,
        normalizer=datamodule.normalizer,
        use_embed=False,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model._stride8_dependencies = dependencies
    model._stride8_checkpoint_id = checkpoint_identifier(checkpoint)
    model.eval().to(device)
    return model, global_step


def sample_file_name(trajectory_index: int, evaluation_seed: int) -> str:
    return f"trajectory_{trajectory_index:04d}_seed_{evaluation_seed}.npz"


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
    sampler = make_sampler(model)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    repository = Path(__file__).resolve().parents[2]
    code_result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    provenance = {
        "training_seed": int(config["training"]["seed"]),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_update": global_step,
        "checkpoint_id": getattr(model, "_stride8_checkpoint_id", None),
        "dependencies": getattr(model, "_stride8_dependencies", {}),
        "ae_checkpoint": str(ae_checkpoint.resolve()),
        "normalizer_values": list(dataset.normalizer_values),
        "data_contract": stage_data_contract("ldm"),
        "ae_data_contract": stage_data_contract("ae"),
        "configuration": config,
        "evaluation_code_commit": code_result.stdout.strip(),
    }
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=False)

    for evaluation_seed in seeds:
        for sample_index in indices:
            sample = dataset.__getitem__(sample_index, eval=True)
            metadata = sample["metadata"]
            trajectory_index = int(metadata["trajectory_index"])
            sampler_seed = derive_sample_seed(evaluation_seed, trajectory_index)
            set_sample_seed(sampler_seed)
            target = sample["x"].numpy()
            prediction = np.full_like(target, np.nan)
            prediction[0] = target[0]
            pre_boundary = prediction.copy()
            failure = None
            sample_started = time.perf_counter()
            try:
                target_tensor = sample["x"].unsqueeze(0).to(device)
                position_tensor = sample["pos"].unsqueeze(0).to(device)
                prediction_tensor = sample_joint64(
                    model,
                    sampler,
                    target_tensor,
                    position_tensor,
                    ddim_steps=DDIM_STEPS,
                )
                pre_boundary = prediction_tensor[0].float().cpu().numpy()
                prediction = writeback_velocity(
                    pre_boundary, target[0], sample["node_type"].numpy()
                )
            except (FloatingPointError, torch.cuda.OutOfMemoryError) as error:
                failure = f"{type(error).__name__}: {error}"
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - sample_started
            metrics = compute_metrics(
                prediction,
                target,
                sample["mesh_pos"].numpy(),
                sample["cells"].numpy(),
                sample["node_type"].numpy(),
                dt=0.08,
            )
            metrics.update(
                boundary_metrics(
                    prediction, pre_boundary, target[0], sample["node_type"].numpy()
                )
            )
            primary = metrics.get("uv_relative_rmse")
            metrics["finite"] = bool(
                metrics.get("finite") and primary is not None and np.isfinite(primary)
            )
            row = {
                **metadata,
                **metrics,
                "sample_index": sample_index,
                "seed": evaluation_seed,
                "evaluation_seed": evaluation_seed,
                "sampler_seed": sampler_seed,
                "global_step": global_step,
                "ddim_steps": DDIM_STEPS,
                "inference_seconds": inference_seconds,
            }
            if not row["finite"]:
                row["failure"] = failure or "nonfinite prediction or primary metric"
            if sample_dir is not None:
                file_path = sample_dir / sample_file_name(
                    trajectory_index, evaluation_seed
                )
                save_prediction(
                    file_path,
                    prediction=prediction,
                    pre_boundary=pre_boundary,
                    target=target,
                    points=sample["mesh_pos"].numpy(),
                    cells=sample["cells"].numpy(),
                    node_type=sample["node_type"].numpy(),
                    trajectory_index=trajectory_index,
                    seed=evaluation_seed,
                    provenance={
                        **provenance,
                        "sampler_seed": sampler_seed,
                        "failure": row.get("failure"),
                        "prediction_is_placeholder": failure is not None,
                    },
                )
                row["prediction_file"] = str(file_path.resolve())
                append_json(sample_dir.parent / "case_metrics.jsonl", row)
            rows.append(row)

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
        "checkpoint_id": getattr(model, "_stride8_checkpoint_id", None),
        "dependencies": getattr(model, "_stride8_dependencies", {}),
        "data_contract": stage_data_contract("ldm"),
        "sampling_protocol": "one joint 1 clean + 64 future DDIM call",
        "evaluation_seeds": list(seeds),
        "validation_indices": list(indices),
        "aggregate": summarize_trajectories(rows),
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
    if sample_dir is not None:
        write_csv(sample_dir.parent / "case_metrics.csv", rows)
        write_csv(
            sample_dir.parent / "trajectory_metrics.csv",
            result["aggregate"]["trajectory_metrics"],
        )
        write_json(sample_dir.parent / "summary.json", result)
        write_json(
            sample_dir.parent / "failures.json",
            {"failures": [row for row in rows if not row["finite"]]},
        )
    del sampler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, rows


def render_representatives(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from tools.cylinderflow.metrics import render_comparison_gif, shared_visual_scales

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
        result = {"status": "no finite seed-0 samples", "selections": {}}
        write_json(output_dir / "representative_samples.json", result)
        return result
    positions = {
        "favorable": 0,
        "median": len(finite_seed_zero) // 2,
        "difficult": min(
            len(finite_seed_zero) - 1,
            int(round(0.9 * (len(finite_seed_zero) - 1))),
        ),
        "worst": len(finite_seed_zero) - 1,
    }
    selections: dict[str, Any] = {}
    paths = []
    for label, position in positions.items():
        row = finite_seed_zero[position]
        file_path = Path(row["prediction_file"])
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
    from dataset.datamodule import FluidsDataModule

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
    for number, checkpoint in enumerate(checkpoints):
        sample_dir = args.output_dir / f"candidate_{number:03d}" / "predictions"
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
        selected = min(
            candidate_results,
            key=lambda record: selection_key(
                record["aggregate"], record["global_step"]
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
        "schema": "text2pde.cylinderflow_stride8.joint64_evaluation.v2",
        "data_contract": stage_data_contract("ldm"),
        "evaluator": "cylinderflow.physical_mesh.v1",
        "mode": args.mode,
        "selection_metric": "failed clips, then mean trajectory UV relative RMSE, then earlier update",
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
