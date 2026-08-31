from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Iterator, Sized

import lightning as L
import numpy as np
import torch
from torch.utils.data import Sampler


RESUME_SCHEMA = "text2pde.cylinderflow.resume.v1"
RESUME_KEY = "cylinderflow_resume_state"


class DeterministicPermutationSampler(Sampler[int]):
    """One fixed permutation with a resumable example offset."""

    def __init__(self, data_source: Sized, seed: int, start_offset: int = 0) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.start_offset = int(start_offset)
        if self.start_offset < 0 or self.start_offset > len(self.data_source):
            raise ValueError(
                f"start_offset={self.start_offset} is outside "
                f"[0,{len(self.data_source)}]"
            )

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        permutation = torch.randperm(
            len(self.data_source), generator=generator, dtype=torch.int64
        )
        yield from permutation[self.start_offset :].tolist()

    def __len__(self) -> int:
        return len(self.data_source) - self.start_offset


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def load_resume_record(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    record = checkpoint.get(RESUME_KEY)
    if not isinstance(record, dict) or record.get("schema") != RESUME_SCHEMA:
        raise ValueError(f"{checkpoint_path} has no exact CylinderFlow resume record")
    examples_seen = int(record["examples_seen"])
    if examples_seen < 0:
        raise ValueError("checkpoint examples_seen must be non-negative")
    return record


def infer_batch_size(batch: Any) -> int:
    if isinstance(batch, dict):
        candidate = batch.get("x")
    elif isinstance(batch, (tuple, list)) and batch:
        candidate = batch[0]
    else:
        candidate = batch
    if not isinstance(candidate, torch.Tensor) or candidate.ndim == 0:
        raise TypeError("unable to infer training microbatch size")
    return int(candidate.shape[0])


class ExactResumeCallback(L.Callback):
    """Persist the sample cursor and restore RNG after loader setup on resume."""

    def __init__(self, start_examples_seen: int = 0) -> None:
        super().__init__()
        self.examples_seen = int(start_examples_seen)
        self._pending_rng_state: dict[str, Any] | None = None
        self._rng_restored = False

    @property
    def state_key(self) -> str:
        return "ExactResumeCallback"

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, outputs, batch_idx
        self.examples_seen += infer_batch_size(batch)

    def on_save_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del pl_module
        # Lightning snapshots callback checkpoints from on_train_batch_end before
        # it marks that batch completed. At an optimizer boundary this otherwise
        # restores one stale accumulation microbatch and steps too early.
        fit_loop = checkpoint.get("loops", {}).get("fit_loop", {})
        batch_progress = fit_loop.get("epoch_loop.batch_progress", {})
        for scope in ("total", "current"):
            progress = batch_progress.get(scope)
            if isinstance(progress, dict) and "ready" in progress:
                progress["completed"] = progress["ready"]
        epoch_loop_state = fit_loop.get("epoch_loop.state_dict", {})
        if "_batches_that_stepped" in epoch_loop_state:
            epoch_loop_state["_batches_that_stepped"] = int(trainer.global_step)
        checkpoint[RESUME_KEY] = {
            "schema": RESUME_SCHEMA,
            "examples_seen": self.examples_seen,
            "global_step": int(trainer.global_step),
            "current_epoch": int(trainer.current_epoch),
            "rng_state": capture_rng_state(),
        }

    def on_load_checkpoint(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        del trainer, pl_module
        record = checkpoint.get(RESUME_KEY)
        if not isinstance(record, dict) or record.get("schema") != RESUME_SCHEMA:
            raise ValueError("checkpoint cannot provide exact CylinderFlow resume")
        checkpoint_examples_seen = int(record["examples_seen"])
        if checkpoint_examples_seen != self.examples_seen:
            raise ValueError(
                "configured data cursor does not match checkpoint: "
                f"{self.examples_seen} != {checkpoint_examples_seen}"
            )
        self._pending_rng_state = record["rng_state"]
        self._rng_restored = False

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if self._pending_rng_state is not None and not self._rng_restored:
            # DataLoader construction consumes torch RNG. Restore immediately before
            # the first resumed stochastic forward pass so diffusion noise and the
            # AE posterior continue bit-for-bit.
            restore_rng_state(self._pending_rng_state)
            self._rng_restored = True
            self._pending_rng_state = None
