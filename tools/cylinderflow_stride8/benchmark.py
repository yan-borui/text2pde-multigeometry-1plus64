"""Benchmark a selected Text2PDE joint64 checkpoint with the shared cost protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset
from modules.utils import get_yaml
from tools.cylinderflow_stride8.evaluate_joint64 import instantiate_model, make_sampler
from tools.cylinderflow_stride8.joint64 import sample_initial64
from tools.cylinderflow_stride8.performance import (
    benchmark,
    configure,
    sync,
    write_json,
)
from tools.cylinderflow_stride8.predictions import writeback_velocity
from tools.cylinderflow_stride8.protocol import (
    validate_locked_config,
    validation_monitor_indices,
)

DATA_REPOSITORY = "DingDong1921/mgn-cylinderflow-stride8-75frames"
DATA_REVISION = "8eae2c7a697e7d01f3b98f4d642ea476784df84a"


def forecast_initial(model, sampler, sample: dict) -> np.ndarray:
    """Pack static coordinates and return physical CPU fields inside the timer."""
    points = torch.from_numpy(sample["points"])
    minima = points.amin(dim=0)
    extents = points.amax(dim=0) - minima
    spatial = ((points - minima) / extents)[None].expand(65, -1, -1)
    local_time = torch.linspace(0.0, 1.0, 65, dtype=torch.float32)
    coordinates = torch.cat(
        (spatial, local_time[:, None, None].expand(-1, len(points), 1)), dim=-1
    )[None]
    initial = torch.from_numpy(sample["initial"])[None, None]
    prediction = (
        sample_initial64(model, sampler, initial, coordinates, check_finite=False)[0]
        .float()
        .cpu()
        .numpy()
    )
    return writeback_velocity(prediction, sample["initial"], sample["node_type"])


def benchmark_checkpoint(
    config: dict,
    checkpoint_file: Path,
    ae_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    threads: int = 2,
) -> dict:
    from dataset.datamodule import FluidsDataModule

    validate_locked_config(config, "ldm")
    configure(device, threads)
    datamodule = FluidsDataModule(config["data"])
    dataset = datamodule.val_dataset
    if (
        not isinstance(dataset, CylinderFlowStride8TrajectoryDataset)
        or dataset.split != "validation"
        or len(dataset) != 100
    ):
        raise ValueError("benchmark requires the formal stride-8 Validation dataset")
    started = time.perf_counter()
    model, update = instantiate_model(
        config, checkpoint_file, ae_checkpoint, datamodule, device
    )
    model.eval().float()
    sampler = make_sampler(model)
    sync(device)
    model_load_seconds = time.perf_counter() - started
    try:
        return benchmark(
            method="text2pde",
            indices=validation_monitor_indices(),
            load_case=dataset.initial,
            predict=lambda sample: forecast_initial(model, sampler, sample),
            device=device,
            output_dir=output_dir,
            data_identity={
                "repository": DATA_REPOSITORY,
                "revision": DATA_REVISION,
                "split": "validation",
                "phase_offset": 0,
                "raw_frame_indices": list(range(0, 513, 8)),
            },
            provenance={
                "checkpoint": str(checkpoint_file.resolve()),
                "checkpoint_id": model._stride8_checkpoint_id,
                "checkpoint_update": update,
                "training_seed": config["training"]["seed"],
                "configuration": config,
                "dependencies": model._stride8_dependencies,
                "normalization": list(dataset.normalizer_values),
            },
            models=[model],
            model_load_seconds=model_load_seconds,
        )
    finally:
        datamodule.train_dataset.close()
        dataset.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    try:
        result = benchmark_checkpoint(
            get_yaml(args.config),
            args.checkpoint,
            args.ae_checkpoint,
            args.output_dir,
            torch.device(args.device),
            args.threads,
        )
    except Exception as error:
        write_json(
            args.output_dir / "exit.json",
            {"exit_code": 1, "error": f"{type(error).__name__}: {error}"},
        )
        raise
    code = 0 if result["status"] == "complete" else 1
    write_json(args.output_dir / "exit.json", {"exit_code": code})
    print(json.dumps(result, indent=2))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
