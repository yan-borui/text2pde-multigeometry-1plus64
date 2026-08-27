from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from dataset.datamodule import FluidsDataModule
from modules.models.ae.ae_mesh import AutoencoderKL
from modules.models.ddpm import LatentDiffusion
from modules.utils import get_yaml


class FiniteAuditCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.losses: list[float] = []
        self.gradient_norms: list[float] = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if isinstance(loss, torch.Tensor):
            value = float(loss.detach().float().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite training loss at batch {batch_idx}")
            self.losses.append(value)

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        squared_norm = torch.zeros((), device=pl_module.device, dtype=torch.float32)
        gradient_count = 0
        for parameter in pl_module.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    f"non-finite gradient before optimizer step {trainer.global_step}"
                )
            squared_norm += gradient.square().sum()
            gradient_count += gradient.numel()
        if gradient_count == 0:
            raise RuntimeError("no gradients before optimizer step")
        norm = float(torch.sqrt(squared_norm).cpu())
        if not math.isfinite(norm):
            raise FloatingPointError("non-finite gradient norm")
        self.gradient_norms.append(norm)


def configure_data(config: dict[str, Any], prepared_dir: Path) -> None:
    dataset = config["data"]["dataset"]
    dataset["train_manifest"] = str(prepared_dir / "largest_train_smoke_windows.json")
    dataset["validation_manifest"] = str(
        prepared_dir / "largest_validation_smoke_window.json"
    )
    config["data"]["normalizer"]["stat_path"] = str(
        prepared_dir / "train_normal_stats.pkl"
    )


def configure_memory_options(
    config: dict[str, Any],
    query_chunk_size: int | None,
    gradient_checkpointing: bool,
) -> None:
    if "aeconfig" in config["model"]:
        aeconfig = config["model"]["aeconfig"]
    else:
        aeconfig = config["model"]["first_stage_config"]["aeconfig"]
    aeconfig["encoder"]["query_chunk_size"] = query_chunk_size
    aeconfig["decoder"]["query_chunk_size"] = query_chunk_size
    if "model_config" in config["model"]:
        config["model"]["model_config"][
            "gradient_checkpointing"
        ] = gradient_checkpointing


def make_ae(config: dict[str, Any], datamodule: FluidsDataModule) -> AutoencoderKL:
    return AutoencoderKL(
        config["model"]["aeconfig"],
        config["model"]["lossconfig"],
        config["training"],
        normalizer=datamodule.normalizer,
        batch_size=config["data"]["batch_size"],
        accumulation_steps=config["training"]["accumulate_grad_batches"],
    )


def make_ldm(config: dict[str, Any], datamodule: FluidsDataModule) -> LatentDiffusion:
    model_config = copy.deepcopy(config["model"])
    training = config["training"]
    scheduler = model_config["scheduler_config"]
    scheduler["batch_size"] = config["data"]["batch_size"]
    scheduler["accumulate_grad_batches"] = training["accumulate_grad_batches"]
    scheduler["dataset_size"] = 100
    scheduler["max_epochs"] = 2
    scheduler["max_steps"] = training["max_steps"]
    return LatentDiffusion(
        **model_config,
        normalizer=datamodule.normalizer,
        use_embed=config["data"]["dataset"]["use_embed"],
    )


def run_stage(
    stage_name: str,
    config: dict[str, Any],
    output_dir: Path,
    model_factory: Callable[[dict[str, Any], FluidsDataModule], L.LightningModule],
    microbatches: int,
    precision: str,
) -> dict[str, Any]:
    accumulation = int(config["training"]["accumulate_grad_batches"])
    if microbatches % accumulation != 0:
        raise ValueError("microbatches must be divisible by gradient accumulation")
    optimizer_steps = microbatches // accumulation
    config["training"]["max_steps"] = optimizer_steps
    config["training"]["max_epochs"] = 2

    stage_dir = output_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=False)
    datamodule = FluidsDataModule(config["data"])
    model = model_factory(config, datamodule)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    audit = FiniteAuditCallback()
    checkpoint = ModelCheckpoint(
        dirpath=str(stage_dir / "checkpoints"),
        every_n_train_steps=optimizer_steps,
        save_top_k=0,
        save_last=True,
    )
    logger = CSVLogger(save_dir=str(stage_dir), name="lightning", version="initial")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision=precision,
        max_steps=optimizer_steps,
        max_epochs=2,
        accumulate_grad_batches=accumulation,
        check_val_every_n_epoch=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
        callbacks=[checkpoint, audit],
        logger=logger,
        enable_progress_bar=False,
        deterministic=False,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model=model, datamodule=datamodule)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    if trainer.global_step != optimizer_steps:
        raise RuntimeError(
            f"{stage_name} reached global_step={trainer.global_step}, expected {optimizer_steps}"
        )
    if len(audit.losses) != microbatches:
        raise RuntimeError(
            f"{stage_name} recorded {len(audit.losses)} losses, expected {microbatches}"
        )
    if len(audit.gradient_norms) != optimizer_steps:
        raise RuntimeError(
            f"{stage_name} recorded {len(audit.gradient_norms)} optimizer gradients, "
            f"expected {optimizer_steps}"
        )

    last_checkpoint = checkpoint.last_model_path
    if not last_checkpoint or not Path(last_checkpoint).is_file():
        raise FileNotFoundError(f"{stage_name} last checkpoint was not written")

    del trainer, model, datamodule
    torch.cuda.empty_cache()
    resume_datamodule = FluidsDataModule(config["data"])
    resumed_model = model_factory(config, resume_datamodule)
    resume_trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision=precision,
        max_steps=optimizer_steps + 1,
        max_epochs=2,
        accumulate_grad_batches=accumulation,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=CSVLogger(save_dir=str(stage_dir), name="lightning", version="resume"),
        enable_checkpointing=False,
        enable_progress_bar=False,
        deterministic=False,
    )
    resume_trainer.fit(
        model=resumed_model,
        datamodule=resume_datamodule,
        ckpt_path=last_checkpoint,
    )
    if resume_trainer.global_step != optimizer_steps + 1:
        raise RuntimeError(
            f"{stage_name} resume reached global_step={resume_trainer.global_step}, "
            f"expected {optimizer_steps + 1}"
        )

    result = {
        "stage": stage_name,
        "microbatches": microbatches,
        "optimizer_steps": optimizer_steps,
        "resume_global_step": resume_trainer.global_step,
        "precision": precision,
        "elapsed_seconds": elapsed,
        "microbatches_per_second": microbatches / elapsed,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "loss_first": audit.losses[0],
        "loss_last": audit.losses[-1],
        "loss_min": min(audit.losses),
        "loss_max": max(audit.losses),
        "gradient_norm_first": audit.gradient_norms[0],
        "gradient_norm_last": audit.gradient_norms[-1],
        "gradient_norm_min": min(audit.gradient_norms),
        "gradient_norm_max": max(audit.gradient_norms),
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "checkpoint": last_checkpoint,
        "finite": True,
        "resume_verified": True,
    }
    with (stage_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    del resume_trainer, resumed_model, resume_datamodule
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-config", required=True)
    parser.add_argument("--ldm-config", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--released-ae", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--microbatches", type=int, default=100)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--accumulate-grad-batches", type=int, default=None)
    parser.add_argument("--query-chunk-size", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--stage", choices=("ae", "ldm", "both"), default="both")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    ae_config = get_yaml(args.ae_config)
    ldm_config = get_yaml(args.ldm_config)
    if args.accumulate_grad_batches is not None:
        if args.accumulate_grad_batches <= 0:
            raise ValueError("accumulate-grad-batches must be positive")
        ae_config["training"]["accumulate_grad_batches"] = (
            args.accumulate_grad_batches
        )
        ldm_config["training"]["accumulate_grad_batches"] = (
            args.accumulate_grad_batches
        )
    configure_data(ae_config, args.prepared_dir)
    configure_data(ldm_config, args.prepared_dir)
    configure_memory_options(
        ae_config,
        args.query_chunk_size,
        args.gradient_checkpointing,
    )
    configure_memory_options(
        ldm_config,
        args.query_chunk_size,
        args.gradient_checkpointing,
    )
    ldm_config["model"]["first_stage_config"]["pretrained_path"] = str(
        args.released_ae
    )

    results = []
    if args.stage in ("ae", "both"):
        L.seed_everything(int(ae_config["training"]["seed"]), workers=True)
        result = run_stage(
            "ae",
            ae_config,
            args.output_dir,
            make_ae,
            args.microbatches,
            args.precision,
        )
        result.update(
            {
                "accumulate_grad_batches": ae_config["training"][
                    "accumulate_grad_batches"
                ],
                "query_chunk_size": args.query_chunk_size,
                "gradient_checkpointing": args.gradient_checkpointing,
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
            }
        )
        results.append(result)
    if args.stage in ("ldm", "both"):
        L.seed_everything(int(ldm_config["training"]["seed"]), workers=True)
        result = run_stage(
            "ldm",
            ldm_config,
            args.output_dir,
            make_ldm,
            args.microbatches,
            args.precision,
        )
        result.update(
            {
                "accumulate_grad_batches": ldm_config["training"][
                    "accumulate_grad_batches"
                ],
                "query_chunk_size": args.query_chunk_size,
                "gradient_checkpointing": args.gradient_checkpointing,
                "torch_version": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
            }
        )
        results.append(result)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
