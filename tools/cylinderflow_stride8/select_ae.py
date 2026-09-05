from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset
from dataset.datamodule import FluidsDataModule
from modules.models.ae.ae_mesh import AutoencoderKL
from modules.utils import get_yaml
from tools.cylinderflow_stride8.protocol import (
    AE_MILESTONES,
    checkpoint_identifier,
    stage_data_contract,
    validate_checkpoint_contract,
    validate_locked_config,
    validation_monitor_indices,
)


def checkpoint_identity(file_path: Path) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(file_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"checkpoint has no state_dict: {file_path}")
    validate_checkpoint_contract(checkpoint, "ae")
    return int(checkpoint.get("global_step", -1)), checkpoint


def evaluate_checkpoint(
    file_path: Path,
    config: dict[str, Any],
    datamodule: FluidsDataModule,
    dataset: CylinderFlowStride8TrajectoryDataset,
    indices: tuple[int, ...],
    device: torch.device,
) -> dict[str, Any]:
    global_step, checkpoint = checkpoint_identity(file_path)
    model = AutoencoderKL(
        config["model"]["aeconfig"],
        config["model"]["lossconfig"],
        config["training"],
        normalizer=datamodule.normalizer,
        batch_size=1,
        accumulation_steps=config["training"]["accumulate_grad_batches"],
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().to(device)
    normalized_l1 = []
    physical_l1 = []
    normalized_channel_l1 = []
    physical_channel_l1 = []
    channel_variance = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index in indices:
            sample = dataset.__getitem__(index, eval=True)
            physical = sample["x"].unsqueeze(0).to(device)
            position = sample["pos"].unsqueeze(0).to(device)
            normalized = datamodule.normalizer.normalize(physical)
            reconstruction, posterior = model(
                normalized,
                position,
                model.latent_grid,
                sample_posterior=False,
            )
            if not (
                torch.isfinite(reconstruction).all()
                and torch.isfinite(posterior.mean).all()
                and torch.isfinite(posterior.logvar).all()
            ):
                raise FloatingPointError(
                    f"non-finite AE output for Validation index {index}"
                )
            physical_reconstruction = datamodule.normalizer.denormalize(reconstruction)
            normalized_error = torch.abs(reconstruction - normalized)
            physical_error = torch.abs(physical_reconstruction - physical)
            normalized_l1.append(float(normalized_error.mean().cpu()))
            physical_l1.append(float(physical_error.mean().cpu()))
            normalized_channel_l1.append(
                normalized_error.mean(dim=(0, 1, 2)).float().cpu()
            )
            physical_channel_l1.append(physical_error.mean(dim=(0, 1, 2)).float().cpu())
            channel_variance.append(
                reconstruction.var(dim=(0, 1, 2), unbiased=False).float().cpu()
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    mean_variance = torch.stack(channel_variance).mean(dim=0)
    if torch.any(mean_variance <= 1.0e-12):
        raise FloatingPointError(
            f"degenerate AE reconstruction at global_step={global_step}"
        )
    result = {
        "checkpoint": str(file_path.resolve()),
        "checkpoint_id": checkpoint_identifier(checkpoint),
        "global_step": global_step,
        "data_contract": stage_data_contract("ae"),
        "validation_indices": list(indices),
        "trajectory_count": len(indices),
        "mean_normalized_uvp_l1": sum(normalized_l1) / len(normalized_l1),
        "mean_physical_uvp_l1": sum(physical_l1) / len(physical_l1),
        "normalized_channel_l1_uvp": torch.stack(normalized_channel_l1)
        .mean(dim=0)
        .tolist(),
        "physical_channel_l1_uvp": torch.stack(physical_channel_l1)
        .mean(dim=0)
        .tolist(),
        "normalized_reconstruction_variance_uvp": mean_variance.tolist(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "finite": True,
        "strict_load": True,
        "posterior_mode": True,
    }
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    config = get_yaml(args.config)
    validate_locked_config(config, "ae")
    datamodule = FluidsDataModule(config["data"])
    dataset = datamodule.val_dataset
    if not isinstance(dataset, CylinderFlowStride8TrajectoryDataset):
        raise TypeError("AE selection requires the stride-8 trajectory dataset")
    if dataset.split != "validation" or len(dataset) != 100:
        raise ValueError("AE selection requires all 100 Validation trajectories")
    indices = validation_monitor_indices()
    device = torch.device(args.device)

    candidates = sorted(args.checkpoint_dir.glob("ae-epoch*.ckpt"))
    identities = [checkpoint_identity(candidate)[0] for candidate in candidates]
    if len(candidates) != 4 or tuple(sorted(identities)) != AE_MILESTONES:
        raise RuntimeError(
            f"AE checkpoint steps {sorted(identities)} != {list(AE_MILESTONES)}"
        )
    results = [
        evaluate_checkpoint(candidate, config, datamodule, dataset, indices, device)
        for candidate in candidates
    ]
    if not all(math.isfinite(row["mean_normalized_uvp_l1"]) for row in results):
        raise FloatingPointError("AE selection contains a non-finite aggregate")
    selected = min(
        results,
        key=lambda row: (row["mean_normalized_uvp_l1"], row["global_step"]),
    )
    summary = {
        "schema": "text2pde.cylinderflow_stride8.ae_selection.v2",
        "data_contract": stage_data_contract("ae"),
        "selection_metric": "mean_normalized_uvp_l1",
        "monitor_rule": "24 uniformly covering Validation trajectory indices",
        "monitor_indices": list(indices),
        "sequence_raw_indices": "0:600:8",
        "test_accessed": False,
        "candidates": results,
        "selected": selected,
    }
    with (args.output_dir / "selection.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (args.output_dir / "selected_checkpoint.txt").write_text(
        selected["checkpoint"] + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
