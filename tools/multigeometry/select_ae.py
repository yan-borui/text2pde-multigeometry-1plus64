from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from dataset.datamodule import FluidsDataModule
from dataset.multigeometry import MultiGeometryWindowDataset
from modules.models.ae.ae_mesh import AutoencoderKL
from modules.utils import get_yaml


def checkpoint_identity(file_path: Path) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(file_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"checkpoint has no state_dict: {file_path}")
    return int(checkpoint.get("global_step", -1)), checkpoint


def evaluate_checkpoint(
    file_path: Path,
    config: dict[str, Any],
    datamodule: FluidsDataModule,
    dataset: MultiGeometryWindowDataset,
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

    window_l1 = []
    normalized_channel_l1 = []
    physical_channel_l1 = []
    channel_variance = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset.__getitem__(index, eval=True)
            physical = sample["x"].unsqueeze(0).to(device)
            pos = sample["pos"].unsqueeze(0).to(device)
            normalized = datamodule.normalizer.normalize(physical)
            reconstruction, posterior = model(
                normalized,
                pos,
                model.latent_grid,
                sample_posterior=False,
            )
            if not torch.isfinite(reconstruction).all():
                raise FloatingPointError(
                    f"non-finite AE reconstruction for {sample['metadata']['case_id']}"
                )
            if not torch.isfinite(posterior.mean).all() or not torch.isfinite(
                posterior.logvar
            ).all():
                raise FloatingPointError(
                    f"non-finite AE posterior for {sample['metadata']['case_id']}"
                )
            physical_reconstruction = datamodule.normalizer.denormalize(reconstruction)
            normalized_error = torch.abs(reconstruction - normalized)
            physical_error = torch.abs(physical_reconstruction - physical)
            window_l1.append(float(normalized_error.mean().cpu()))
            normalized_channel_l1.append(
                normalized_error.mean(dim=(0, 1, 2)).float().cpu()
            )
            physical_channel_l1.append(
                physical_error.mean(dim=(0, 1, 2)).float().cpu()
            )
            channel_variance.append(
                reconstruction.var(dim=(0, 1, 2), unbiased=False).float().cpu()
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    mean_variance = torch.stack(channel_variance).mean(dim=0)
    if torch.any(mean_variance <= 1.0e-12):
        raise FloatingPointError(
            f"degenerate AE channel variance at global_step={global_step}: "
            f"{mean_variance.tolist()}"
        )

    result = {
        "checkpoint": str(file_path),
        "global_step": global_step,
        "window_count": len(dataset),
        "mean_window_normalized_l1": sum(window_l1) / len(window_l1),
        "normalized_channel_l1_uvp": torch.stack(normalized_channel_l1)
        .mean(dim=0)
        .tolist(),
        "physical_channel_l1_uvp": torch.stack(physical_channel_l1)
        .mean(dim=0)
        .tolist(),
        "normalized_reconstruction_variance_uvp": mean_variance.tolist(),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "finite": True,
        "strict_load": True,
        "posterior_mode": True,
    }
    del model, checkpoint
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = get_yaml(args.config)
    config["data"]["dataset"]["validation_manifest"] = str(args.manifest)
    datamodule = FluidsDataModule(config["data"])
    dataset = MultiGeometryWindowDataset(str(args.manifest), return_metadata=True)
    device = torch.device(args.device)

    candidates = sorted(args.checkpoint_dir.glob("ae-epoch*.ckpt"))
    if len(candidates) != 3:
        raise RuntimeError(f"expected three AE cycle checkpoints, found {len(candidates)}")

    results = []
    for candidate in candidates:
        result = evaluate_checkpoint(candidate, config, datamodule, dataset, device)
        if not math.isfinite(result["mean_window_normalized_l1"]):
            raise FloatingPointError(
                f"non-finite aggregate AE validation loss for {candidate}"
            )
        results.append(result)

    selected = min(
        results,
        key=lambda record: (
            record["mean_window_normalized_l1"],
            record["global_step"],
        ),
    )
    summary = {
        "schema": "text2pde.multigeometry.ae_selection.v1",
        "selection_metric": "mean_window_normalized_uvp_l1",
        "monitor_manifest": str(args.manifest),
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
