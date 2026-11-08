"""Torch-only deterministic global-cursor sampling for distributed trajectories."""

from __future__ import annotations

from torch.utils.data import Sampler
import torch


class DistributedEpochPermutationSampler(Sampler[int]):
    def __init__(self, data_source, seed, start_examples_seen=0, *, rank, world_size):
        size = len(data_source)
        if (
            size < 1
            or world_size < 1
            or not 0 <= rank < world_size
            or start_examples_seen < 0
            or start_examples_seen % world_size
        ):
            raise ValueError(
                "distributed epoch/cursor must partition exactly across ranks"
            )
        self.seed, self.size = seed, size
        self.rank, self.world_size = rank, world_size
        self.epoch_size = ((size + world_size - 1) // world_size) * world_size
        self.next_cursor = start_examples_seen

    def __iter__(self):
        start = self.next_cursor
        stop = (start // self.epoch_size + 1) * self.epoch_size
        self.next_cursor = stop
        previous_epoch, permutation = None, None
        # Continue into the next deterministic permutation instead of padding or
        # dropping the two trajectories at a 782-case/four-rank boundary.
        for position in range(start + self.rank, stop, self.world_size):
            epoch, offset = divmod(position, self.size)
            if epoch != previous_epoch:
                permutation = torch.randperm(
                    self.size,
                    generator=torch.Generator().manual_seed(self.seed + epoch),
                )
                previous_epoch = epoch
            yield int(permutation[offset])

    def __len__(self):
        # Lightning restores the consumed batch count and adds it to the fetcher.
        # Report the full epoch so that the resumed suffix is not truncated twice.
        return self.epoch_size // self.world_size


class FixedValidationSampler(Sampler[int]):
    def __init__(self, indices, rank, world_size):
        self.indices = tuple(indices)[rank::world_size]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
