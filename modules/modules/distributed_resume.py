"""Fixed-world Lightning sampling and checkpoint state for the four-GPU recipe."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import traceback
import time
import uuid

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from .reproducible_resume import RESUME_KEY, infer_batch_size

SCHEMA = "text2pde.cylinderflow.resume.ddp.v2"


def training_contract(config: dict) -> dict:
    result = deepcopy(config)
    result["training"].pop("checkpoint", None)
    result["data"].pop("train_examples_seen", None)
    return result


def capture_local_rng(device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None,
    }


def restore_local_rng(state, device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


def freeze_source(run: Path, resume: bool) -> None:
    root = Path(__file__).resolve().parents[2]
    files = [root / "train_AE.py", root / "train_ldm.py"]
    for directory in ("modules", "dataset", "tools/cylinderflow_stride8"):
        files.extend(sorted((root / directory).rglob("*.py")))
    for original in files:
        destination = run / "source" / original.relative_to(root)
        if resume or destination.exists():
            if (
                not destination.is_file()
                or destination.read_bytes() != original.read_bytes()
            ):
                raise ValueError(f"resume source changed: {original.relative_to(root)}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)


class DistributedExactResumeCallback(L.Callback):
    def __init__(
        self,
        start_examples_seen=0,
        data_contract=None,
        dependencies=None,
        *,
        contract,
        seed,
        stop_after_updates=None,
    ):
        self.examples_seen = start_examples_seen
        self.data_contract, self.dependencies = data_contract, dependencies
        self.contract, self.seed = deepcopy(contract), seed
        self.stop_after_updates = stop_after_updates
        self.pending_rng = None
        self.first_batch = True
        self.resuming = False
        self.last_train_rng = None
        self.validation_rng = None
        self.elapsed_prior = 0.0
        self.validation_seconds = 0.0
        self.started = time.perf_counter()
        self.initial_parameters = None
        self.finalized = False
        self.memory_segment = None
        self.memory_start_update = None
        self.memory_started = None

    def _local_memory(self, trainer: L.Trainer, pl_module: L.LightningModule) -> dict:
        """Read this process's cumulative CUDA peaks without resetting them."""
        record = {"rank": trainer.global_rank, "device": str(pl_module.device)}
        if pl_module.device.type == "cuda":
            device = pl_module.device
            allocated = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
            properties = torch.cuda.get_device_properties(device)
            record.update(
                gpu_name=torch.cuda.get_device_name(device),
                gpu_total_bytes=int(properties.total_memory),
                peak_allocated_bytes=allocated,
                peak_reserved_bytes=reserved,
                allocated_gib=allocated / 2**30,
                reserved_gib=reserved / 2**30,
            )
        return record

    def _record_metrics(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        event: str,
        checkpoint_id: str | None = None,
    ) -> dict | None:
        """Collect all ranks at a shared hook; preserve the existing metrics log."""
        if self.memory_segment is None:
            return None
        ranks = [None] * trainer.world_size
        dist.all_gather_object(ranks, self._local_memory(trainer, pl_module))
        elapsed = self.elapsed_prior + time.perf_counter() - self.started
        cuda_ranks = [row for row in ranks if "peak_allocated_bytes" in row]
        record = {
            "schema": "text2pde.training_memory.v1",
            "event": event,
            "stage": (self.data_contract or {}).get("stage", "unknown"),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "update": int(trainer.global_step),
            "epoch": int(trainer.current_epoch),
            "checkpoint_id": checkpoint_id,
            "examples_seen": self.examples_seen,
            "world_size": trainer.world_size,
            "gpu_count": len(cuda_ranks),
            "precision": str(trainer.precision),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "lightning_version": L.__version__,
            "rank_metrics": ranks,
            "max_allocated_bytes": max(
                (row["peak_allocated_bytes"] for row in cuda_ranks), default=None
            ),
            "max_reserved_bytes": max(
                (row["peak_reserved_bytes"] for row in cuda_ranks), default=None
            ),
            "memory_segment": self.memory_segment,
            "memory_start_update": self.memory_start_update,
            "memory_elapsed_seconds": time.perf_counter() - self.memory_started,
            "memory_scope": "since_train_start_including_in_training_validation",
            "sanity_validation_included": False,
            "historical_memory_restored": False,
            "elapsed_seconds": elapsed,
            "validation_seconds": self.validation_seconds,
            "allocated_gpu_hours": trainer.world_size * elapsed / 3600
            if pl_module.device.type == "cuda"
            else None,
        }
        if trainer.is_global_zero:
            folder = Path(trainer.default_root_dir)
            with (folder / "distributed_metrics.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(record) + "\n")
            # Keep each process segment separately when a run resumes.
            summary_dir = folder / "training_memory"
            summary_dir.mkdir(exist_ok=True)
            destination = summary_dir / f"{self.memory_segment}.json"
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
        return record

    @property
    def state_key(self):
        return "DistributedExactResumeCallback"

    def on_fit_start(self, trainer, pl_module):
        if trainer.world_size != self.contract["training"]["devices"]:
            raise ValueError("runtime world differs from frozen training configuration")
        result = [None]
        if trainer.is_global_zero:
            try:
                freeze_source(Path(trainer.default_root_dir), self.resuming)
                result[0] = None
            except BaseException:
                result[0] = traceback.format_exc()
        dist.broadcast_object_list(result, src=0)
        if result[0]:
            raise RuntimeError(result[0])

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.first_batch:
            if self.pending_rng is None:
                seed = self.seed + trainer.global_rank * 1000003
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.manual_seed(seed)
                if pl_module.device.type == "cuda":
                    torch.cuda.manual_seed(seed)
            else:
                restore_local_rng(self.pending_rng, pl_module.device)
            self.first_batch = False
            self.pending_rng = None

    def on_train_start(self, trainer, pl_module):
        self.started = time.perf_counter()
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        segment = [str(uuid.uuid4()) if trainer.is_global_zero else None]
        dist.broadcast_object_list(segment, src=0)
        self.memory_segment = segment[0]
        self.memory_start_update = int(trainer.global_step)
        self.memory_started = time.perf_counter()
        self._record_metrics(trainer, pl_module, "train_start")
        if self.stop_after_updates is not None:
            self.initial_parameters = {
                name: parameter.detach().cpu().clone()
                for name, parameter in pl_module.named_parameters()
                if parameter.requires_grad
            }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.examples_seen += infer_batch_size(batch) * trainer.world_size
        self.last_train_rng = capture_local_rng(pl_module.device)
        every = self.contract["training"].get("log_every_n_steps", 20)
        if trainer.global_step == 1 or trainer.global_step % every == 0:
            self._record_metrics(trainer, pl_module, "train_batch_end")
        if (
            self.stop_after_updates is not None
            and trainer.global_step >= self.stop_after_updates
        ):
            trainer.should_stop = True

    def on_validation_start(self, trainer, pl_module):
        self.validation_started = time.perf_counter()
        self.validation_rng = self.last_train_rng or capture_local_rng(pl_module.device)

    def on_validation_end(self, trainer, pl_module):
        if self.validation_rng is not None:
            restore_local_rng(self.validation_rng, pl_module.device)
            self.validation_rng = None
        self.validation_seconds += time.perf_counter() - self.validation_started
        if not trainer.sanity_checking:
            self._record_metrics(trainer, pl_module, "validation_end")

    def on_train_epoch_end(self, trainer, pl_module):
        self._record_metrics(trainer, pl_module, "train_epoch_end")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        loop = checkpoint.get("loops", {}).get("fit_loop", {})
        progress = loop.get("epoch_loop.batch_progress", {})
        for scope in ("total", "current"):
            if scope in progress and "ready" in progress[scope]:
                progress[scope]["completed"] = progress[scope]["ready"]
        state = loop.get("epoch_loop.state_dict", {})
        if "_batches_that_stepped" in state:
            state["_batches_that_stepped"] = int(trainer.global_step)
        local = {
            "rank": trainer.global_rank,
            "examples_seen": self.examples_seen,
            "rng": capture_local_rng(pl_module.device),
        }
        ranks = [None] * trainer.world_size
        dist.all_gather_object(ranks, local)
        if any(row["examples_seen"] != self.examples_seen for row in ranks):
            raise ValueError("ranks disagree about global samples consumed")
        identifier = [str(uuid.uuid4()) if trainer.is_global_zero else None]
        dist.broadcast_object_list(identifier, src=0)
        checkpoint[RESUME_KEY] = {
            "schema": SCHEMA,
            "checkpoint_id": identifier[0],
            "data_contract": self.data_contract,
            "dependencies": self.dependencies,
            "training_contract": self.contract,
            "world_size": trainer.world_size,
            "examples_seen": self.examples_seen,
            "global_step": int(trainer.global_step),
            "current_epoch": int(trainer.current_epoch),
            "rank_states": ranks,
            "elapsed_seconds": self.elapsed_prior + time.perf_counter() - self.started,
            "validation_seconds": self.validation_seconds,
        }
        checkpoint["training_memory"] = self._record_metrics(
            trainer, pl_module, "checkpoint_save", checkpoint_id=identifier[0]
        )

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        record = checkpoint.get(RESUME_KEY, {})
        if (
            record.get("schema") != SCHEMA
            or record.get("training_contract") != self.contract
            or record.get("data_contract") != self.data_contract
            or record.get("dependencies") != self.dependencies
            or record.get("world_size") != trainer.world_size
            or record.get("examples_seen") != self.examples_seen
        ):
            raise ValueError("distributed resume config/data/world/cursor mismatch")
        self.pending_rng = record["rank_states"][trainer.global_rank]["rng"]
        self.first_batch, self.resuming = True, True
        self.elapsed_prior = record.get("elapsed_seconds", 0.0)
        self.validation_seconds = record.get("validation_seconds", 0.0)

    def on_train_end(self, trainer, pl_module):
        expected = self.stop_after_updates or self.contract["training"]["max_steps"]
        if trainer.global_step != expected:
            raise RuntimeError(
                f"training ended at {trainer.global_step}, expected {expected}"
            )
        if self.initial_parameters is not None:
            changes = [
                (parameter.detach().cpu() - self.initial_parameters[name]).abs().max()
                for name, parameter in pl_module.named_parameters()
                if parameter.requires_grad
            ]
            if not all(torch.isfinite(value) for value in changes) or not any(
                value > 0 for value in changes
            ):
                raise RuntimeError(
                    "bounded preflight did not update parameters finitely"
                )
        # All ranks participate in checkpoint construction; Lightning writes only on rank zero.
        trainer.save_checkpoint(
            str(Path(trainer.default_root_dir) / "checkpoints" / "last.ckpt")
        )
        # Lightning tears down the GPU strategy before on_fit_end.
        self._record_metrics(trainer, pl_module, "train_end")
        if trainer.is_global_zero:
            from tools.cylinderflow_stride8.evaluation_io import write_json

            write_json(
                Path(trainer.default_root_dir) / "training_completion.json",
                {
                    "state": "preflight_complete"
                    if self.stop_after_updates
                    else "training_complete",
                    "update": trainer.global_step,
                    "examples_seen": self.examples_seen,
                    "world_size": trainer.world_size,
                    "test_accessed": False,
                },
            )

        self.finalized = True

    def on_fit_end(self, trainer, pl_module):
        # At a restored budget endpoint Lightning skips the train loop entirely.
        # The checkpoint has already passed on_load_checkpoint and source checks.
        if self.finalized:
            return
        expected = self.stop_after_updates or self.contract["training"]["max_steps"]
        if not self.resuming or trainer.global_step != expected:
            raise RuntimeError(
                "fit ended without completing the allocated update budget"
            )
        if trainer.is_global_zero:
            from tools.cylinderflow_stride8.evaluation_io import write_json

            write_json(
                Path(trainer.default_root_dir) / "training_completion.json",
                {
                    "state": "preflight_complete"
                    if self.stop_after_updates
                    else "training_complete",
                    "update": trainer.global_step,
                    "examples_seen": self.examples_seen,
                    "world_size": trainer.world_size,
                    "test_accessed": False,
                    "completed_from_restored_budget": True,
                    "training_diagnostic_replayed": False,
                    "standalone_physical_selection_pending": True,
                },
            )
        self.finalized = True

    def on_exception(self, trainer, pl_module, exception):
        folder = Path(trainer.default_root_dir)
        folder.mkdir(parents=True, exist_ok=True)
        record = {
            "rank": trainer.global_rank,
            "update": trainer.global_step,
            "examples_seen": self.examples_seen,
            "error": repr(exception),
            "traceback": traceback.format_exc(),
        }
        if self.memory_segment is not None:
            # A failed rank may be alone; no collective is safe in this hook.
            try:
                record["training_memory_local"] = {
                    "memory_segment": self.memory_segment,
                    "memory_start_update": self.memory_start_update,
                    "epoch": int(trainer.current_epoch),
                    "rank_metrics": self._local_memory(trainer, pl_module),
                }
            except RuntimeError:
                record["training_memory_local"] = {"status": "cuda_unavailable"}
        (folder / f"rank_{trainer.global_rank:03d}_failure.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
