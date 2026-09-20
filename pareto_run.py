"""Run one selected baseline's matched timing and fresh Validation100."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

METHOD = "text2pde"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        parser.error("formal handoff requires a CUDA GPU")
    if METHOD in ("aroma", "text2pde") and args.ae_checkpoint is None:
        parser.error("this method requires --ae-checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if METHOD == "text2pde":
        from tools.cylinderflow_stride8.performance import (
            configure,
            runtime_identity,
            write_json,
        )
        from tools.cylinderflow_stride8.benchmark import benchmark_checkpoint
        from tools.cylinderflow_stride8.evaluate_joint64 import evaluate_one_checkpoint
        from modules.utils import get_yaml

        config = get_yaml(args.config)
    else:
        from cylinderflow.performance import configure, runtime_identity, write_json
        from cylinderflow.benchmark import benchmark_checkpoint
        from cylinderflow.engine import config_from_file, evaluate
        from cylinderflow.data import Dataset

        if any(value is None for value in (args.dataset, args.manifest, args.prepared)):
            parser.error("--dataset, --manifest and --prepared are required")
        config = config_from_file(args.config)
        if config["method"] != METHOD:
            parser.error("configuration method does not match this repository")
    configure(device, args.threads)
    environment = runtime_identity(device)
    try:
        if METHOD == "text2pde":
            timing = benchmark_checkpoint(
                config,
                args.checkpoint,
                args.ae_checkpoint,
                args.output_dir / "timing",
                device,
                args.threads,
            )
        else:
            dataset = Dataset(args.dataset, args.manifest)
            timing = benchmark_checkpoint(
                dataset,
                config,
                args.prepared,
                args.checkpoint,
                args.output_dir / "timing",
                device,
                args.ae_checkpoint,
                args.threads,
            )
        if timing["status"] != "complete":
            raise RuntimeError("timing contains failures")
        if METHOD != "text2pde":
            config = timing["provenance"]["configuration"]
        # Fresh physical evaluation, outside the timing region. Preserve each
        # baseline's official single-draw (or three scored draws) protocol.
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            if METHOD == "text2pde":
                from dataset.datamodule import FluidsDataModule

                datamodule = FluidsDataModule(config["data"])
                try:
                    quality, _ = evaluate_one_checkpoint(
                        args.checkpoint,
                        args.ae_checkpoint,
                        config,
                        datamodule,
                        datamodule.val_dataset,
                        tuple(range(100)),
                        (0, 1, 2),
                        device,
                        args.output_dir / "quality" / "predictions",
                    )
                finally:
                    datamodule.train_dataset.close()
                    datamodule.val_dataset.close()
                summary = quality["aggregate"]
                quality_id = quality["checkpoint_id"]
                quality_file = "quality/summary.json"
            else:
                result = evaluate(
                    dataset,
                    config,
                    args.prepared,
                    [args.checkpoint],
                    args.output_dir / "quality",
                    device,
                    "validation",
                    args.ae_checkpoint,
                )
                selected = result["selected"]
                summary = selected["summary"]
                quality_id = selected["checkpoint_id"]
                quality_file = "quality/candidate_000/summary.json"
        if quality_id != timing["provenance"]["checkpoint_id"]:
            raise ValueError("quality and timing checkpoint identities differ")
        if summary["trajectory_count"] != 100 or summary["failed_clips"]:
            raise RuntimeError("Validation100 incomplete; preserve failure artifacts")
        if runtime_identity(device) != environment:
            raise ValueError("execution settings changed during the run")
        point = {
            "schema": "cylinderflow.pareto_point.v1",
            "method": METHOD,
            "campaign_id": args.campaign_id,
            "environment": environment,
            "data_identity": timing["data_identity"],
            "provenance": timing["provenance"],
            "sampling_steps": {"aroma": 4, "text2pde": 20}.get(METHOD),
            "ensemble_size": 1,
            "quality_semantics": "official_single_sample_score_mean",
            "timing_file": "timing/summary.json",
            "quality_file": quality_file,
            "evaluator": "cylinderflow.physical_mesh.v1",
            "test_accessed": False,
        }
        write_json(args.output_dir / "point.json", point)
        write_json(args.output_dir / "exit.json", {"exit_code": 0})
    except Exception as error:
        write_json(
            args.output_dir / "exit.json",
            {"exit_code": 1, "error": f"{type(error).__name__}: {error}"},
        )
        raise


if __name__ == "__main__":
    main()
