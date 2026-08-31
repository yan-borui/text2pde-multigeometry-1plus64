from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from dataset.cylinderflow import CylinderFlowWindowDataset
from dataset.datamodule import FluidsDataModule
from modules.models.ae.ae_mesh import AutoencoderKL
from modules.models.ddpm import LatentDiffusion
from modules.modules.reproducible_resume import ExactResumeCallback
from modules.utils import get_yaml
from tools.cylinderflow.evaluate_rollout import evaluate_one_checkpoint


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
        del trainer, pl_module, batch
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if isinstance(loss, torch.Tensor):
            value = float(loss.detach().float().cpu())
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"non-finite training loss at batch {batch_idx}"
                )
            self.losses.append(value)

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        del optimizer
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
    dataset["train_manifest"] = str(prepared_dir / "largest_train_smoke_windows_4.json")
    dataset["validation_manifest"] = str(
        prepared_dir / "validation_monitor_windows_25.json"
    )
    config["data"]["normalizer"]["stat_path"] = str(
        prepared_dir / "train_normal_stats.pkl"
    )
    config["data"]["train_start_offset"] = 0
    config["training"]["max_steps"] = 1
    config["training"]["max_epochs"] = 1
    config["training"]["limit_val_batches"] = 1
    config["training"]["accumulate_grad_batches"] = 4


def make_ae(config: dict[str, Any], datamodule: FluidsDataModule) -> AutoencoderKL:
    return AutoencoderKL(
        config["model"]["aeconfig"],
        config["model"]["lossconfig"],
        config["training"],
        normalizer=datamodule.normalizer,
        batch_size=1,
        accumulation_steps=4,
    )


def make_ldm(config: dict[str, Any], datamodule: FluidsDataModule) -> LatentDiffusion:
    model_config = copy.deepcopy(config["model"])
    scheduler = model_config["scheduler_config"]
    scheduler["batch_size"] = 1
    scheduler["accumulate_grad_batches"] = 4
    scheduler["dataset_size"] = 4
    scheduler["max_epochs"] = 1
    scheduler["max_steps"] = 1
    return LatentDiffusion(
        **model_config,
        normalizer=datamodule.normalizer,
        use_embed=False,
    )


def run_update_smoke(
    *,
    stage: str,
    config: dict[str, Any],
    output_dir: Path,
    factory: Callable[[dict[str, Any], FluidsDataModule], L.LightningModule],
    precision: str,
) -> dict[str, Any]:
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    datamodule = FluidsDataModule(config["data"])
    model = factory(config, datamodule)
    audit = FiniteAuditCallback()
    checkpoint = ModelCheckpoint(
        dirpath=str(stage_dir / "checkpoints"),
        every_n_train_steps=1,
        save_top_k=0,
        save_last=True,
    )
    callbacks = [ExactResumeCallback(0), audit, checkpoint]
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision=precision,
        max_steps=1,
        max_epochs=1,
        accumulate_grad_batches=4,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        deterministic=False,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model=model, datamodule=datamodule)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if trainer.global_step != 1:
        raise RuntimeError(f"{stage} smoke did not complete one optimizer step")
    if len(audit.losses) != 4 or len(audit.gradient_norms) != 1:
        raise RuntimeError(
            f"{stage} smoke observed {len(audit.losses)} losses and "
            f"{len(audit.gradient_norms)} gradient norms"
        )
    if not checkpoint.last_model_path or not Path(checkpoint.last_model_path).is_file():
        raise FileNotFoundError(f"{stage} smoke did not save last.ckpt")
    result = {
        "stage": stage,
        "microbatch": 1,
        "accumulate_grad_batches": 4,
        "optimizer_steps": 1,
        "losses": audit.losses,
        "gradient_norm": audit.gradient_norms[0],
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "checkpoint": checkpoint.last_model_path,
        "finite": True,
    }
    with (stage_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    del trainer, model, datamodule
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-config", required=True)
    parser.add_argument("--ldm-config", required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--released-ae", type=Path, required=True)
    parser.add_argument("--released-ldm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--ddim-steps", type=int, default=20)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    ae_config = get_yaml(args.ae_config)
    ldm_config = get_yaml(args.ldm_config)
    configure_data(ae_config, args.prepared_dir)
    configure_data(ldm_config, args.prepared_dir)
    ldm_config["model"]["first_stage_config"]["pretrained_path"] = str(
        args.released_ae.resolve()
    )

    L.seed_everything(int(ae_config["training"]["seed"]), workers=True)
    results = [
        run_update_smoke(
            stage="ae_update",
            config=ae_config,
            output_dir=args.output_dir,
            factory=make_ae,
            precision=args.precision,
        )
    ]
    L.seed_everything(int(ldm_config["training"]["seed"]), workers=True)
    results.append(
        run_update_smoke(
            stage="ldm_update",
            config=ldm_config,
            output_dir=args.output_dir,
            factory=make_ldm,
            precision=args.precision,
        )
    )

    rollout_config = get_yaml(args.ldm_config)
    configure_data(rollout_config, args.prepared_dir)
    rollout_manifest = args.prepared_dir / "validation_monitor_rollout64.json"
    rollout_config["data"]["dataset"]["validation_manifest"] = str(rollout_manifest)
    rollout_datamodule = FluidsDataModule(rollout_config["data"])
    rollout_dataset = CylinderFlowWindowDataset(
        str(rollout_manifest), return_metadata=True
    )
    rollout_result, rollout_rows = evaluate_one_checkpoint(
        checkpoint_path=args.released_ldm,
        ae_checkpoint=args.released_ae,
        config=rollout_config,
        datamodule=rollout_datamodule,
        dataset=rollout_dataset,
        seeds=[0],
        ddim_steps=args.ddim_steps,
        device=torch.device("cuda:0"),
        sample_dir=args.output_dir / "rollout64" / "samples_seed0",
    )
    if rollout_result["aggregate"]["failure_count"] != 0:
        raise FloatingPointError("three-segment rollout smoke failed")
    rollout_dir = args.output_dir / "rollout64"
    rollout_dir.mkdir(exist_ok=True)
    with (rollout_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"result": rollout_result, "rows": rollout_rows},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    summary = {
        "schema": "text2pde.cylinderflow.gpu_smoke.v1",
        "prepared_dir": str(args.prepared_dir.resolve()),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "update_stages": results,
        "three_segment_rollout": rollout_result,
        "formal_training_started": False,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
