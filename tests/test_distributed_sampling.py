"""Distributed sampling must conserve global windows and exact resumed order."""

import unittest

import torch

from modules.modules.distributed_sampling import (
    DistributedEpochPermutationSampler,
    FixedValidationSampler,
)


class DistributedSamplingTest(unittest.TestCase):
    def test_global_order_and_mid_epoch_resume(self):
        source = range(20)
        samplers = [
            DistributedEpochPermutationSampler(source, 42, rank=rank, world_size=4)
            for rank in range(4)
        ]
        first = [list(sampler) for sampler in samplers]
        observed = [value for group in zip(*first) for value in group]
        expected = torch.randperm(
            20, generator=torch.Generator().manual_seed(42)
        ).tolist()
        self.assertEqual(observed, expected)
        second = [list(sampler) for sampler in samplers]
        resumed = [
            DistributedEpochPermutationSampler(source, 42, 28, rank=rank, world_size=4)
            for rank in range(4)
        ]
        remaining = [list(sampler) for sampler in resumed]
        self.assertEqual(remaining, [values[2:] for values in second])
        self.assertEqual(
            [list(sampler) for sampler in resumed],
            [list(sampler) for sampler in samplers],
        )

    def test_global_validation_has_no_duplicates(self):
        shards = [list(FixedValidationSampler(range(24), rank, 4)) for rank in range(4)]
        self.assertEqual([len(shard) for shard in shards], [6] * 4)
        self.assertEqual(
            sorted(value for shard in shards for value in shard), list(range(24))
        )


if __name__ == "__main__":
    unittest.main()
