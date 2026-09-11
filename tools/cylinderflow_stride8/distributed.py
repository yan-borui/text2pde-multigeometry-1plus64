"""Small fixed-world distributed runtime; no elastic restarts or implicit GPU claims."""

from __future__ import annotations

from datetime import timedelta
import os
import random
import traceback

import numpy as np
import torch
import torch.distributed as dist


def capture_rng(device: torch.device) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


class Context:
    def __init__(self, expected_world: int, device_name: str):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.world != expected_world:
            raise ValueError(
                f"configured world_size={expected_world}, launched={self.world}"
            )
        self.device = torch.device(
            f"cuda:{self.local_rank}"
            if device_name.startswith("cuda") and self.world > 1
            else device_name
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.owns_group = self.world > 1 and not dist.is_initialized()
        if self.owns_group:
            dist.init_process_group(
                "nccl" if self.device.type == "cuda" else "gloo",
                timeout=timedelta(minutes=30),
            )

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def gather(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world
        dist.all_gather_object(values, value)
        return values

    def broadcast(self, value):
        values = [value if self.primary else None]
        if self.world > 1:
            dist.broadcast_object_list(values, src=0)
        return values[0]

    def primary_call(self, function):
        result = None
        if self.primary:
            try:
                result = (True, function())
            except BaseException:
                result = (False, traceback.format_exc())
        ok, value = self.broadcast(result)
        if not ok:
            raise RuntimeError(value)
        return value

    def all_call(self, function):
        try:
            result = (True, function())
        except BaseException:
            result = (False, traceback.format_exc())
        results = self.gather(result)
        failures = [
            f"rank {rank}: {value}"
            for rank, (ok, value) in enumerate(results)
            if not ok
        ]
        if failures:
            raise RuntimeError("\n".join(failures))
        return [value for _, value in results]

    def mean(self, value: torch.Tensor) -> torch.Tensor:
        value = value.detach().clone()
        if self.world > 1:
            dist.all_reduce(value)
            value /= self.world
        return value

    def close(self) -> None:
        if self.owns_group and dist.is_initialized():
            dist.destroy_process_group()
