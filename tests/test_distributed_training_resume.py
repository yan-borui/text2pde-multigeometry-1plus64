"""Exercise real Lightning DDP across a mid-epoch resume and epoch boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

HAS_LIGHTNING = importlib.util.find_spec("lightning") is not None
if HAS_LIGHTNING:
    import lightning as L

    from modules.modules.distributed_resume import DistributedExactResumeCallback
    from modules.modules.distributed_sampling import DistributedEpochPermutationSampler
    from modules.modules.reproducible_resume import RESUME_KEY, load_resume_record


class IndexDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {"x": torch.tensor([float(index)])}


if HAS_LIGHTNING:

    class StochasticModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def training_step(self, batch, batch_idx):
            target = batch["x"].mean() * 0.01 + 0.02 * torch.randn(
                (), device=self.device
            )
            return (self.weight - target).square()

        def validation_step(self, batch, batch_idx):
            torch.randn(19, device=self.device)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.03)

    class DataModule(L.LightningDataModule):
        def __init__(self, examples):
            super().__init__()
            self.examples = examples

        def train_dataloader(self):
            dataset = IndexDataset()
            sampler = DistributedEpochPermutationSampler(
                dataset,
                42,
                self.examples,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
            )
            return DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0)

        def val_dataloader(self):
            return DataLoader(IndexDataset(), batch_size=1, num_workers=0)


@unittest.skipUnless(HAS_LIGHTNING, "requires the target Lightning environment")
class DistributedTrainingResumeTest(unittest.TestCase):
    def run_training(self, root, *, checkpoint=None, stop=None, validation=True):
        examples = load_resume_record(checkpoint)["examples_seen"] if checkpoint else 0
        contract = {
            "training": {"devices": 2, "max_steps": 8, "log_every_n_steps": 2},
            "data": {"fixture": "twelve deterministic indices"},
        }
        callback = DistributedExactResumeCallback(
            examples,
            data_contract={"fixture": "twelve deterministic indices"},
            contract=contract,
            seed=42,
            stop_after_updates=stop,
        )
        L.seed_everything(991 if checkpoint else 42, workers=True)
        trainer = L.Trainer(
            accelerator="cpu",
            devices=2,
            strategy="ddp_spawn",
            max_steps=stop or 8,
            max_epochs=5,
            accumulate_grad_batches=1,
            callbacks=[callback],
            default_root_dir=str(root),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            val_check_interval=2,
            check_val_every_n_epoch=None,
            limit_val_batches=1 if validation else 0,
            use_distributed_sampler=False,
            deterministic=True,
        )
        trainer.fit(
            StochasticModel(),
            datamodule=DataModule(examples),
            ckpt_path=str(checkpoint) if checkpoint else None,
        )
        return torch.load(
            root / "checkpoints/last.ckpt", map_location="cpu", weights_only=False
        )

    def test_resume_across_epoch_and_validation_rng_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uninterrupted = self.run_training(root / "continuous")
            split = root / "resumed"
            prefix = self.run_training(split, stop=2)
            self.assertEqual(prefix[RESUME_KEY]["examples_seen"], 4)
            resumed = self.run_training(
                split, checkpoint=split / "checkpoints/last.ckpt"
            )
            terminal = self.run_training(
                split, checkpoint=split / "checkpoints/last.ckpt"
            )
            without_validation = self.run_training(
                root / "no_validation", validation=False
            )
            for actual in (resumed, terminal, without_validation):
                self.assertEqual(actual[RESUME_KEY]["examples_seen"], 16)
                self.assertEqual(actual["global_step"], 8)
                torch.testing.assert_close(
                    actual["state_dict"]["weight"],
                    uninterrupted["state_dict"]["weight"],
                    rtol=0,
                    atol=0,
                )
                expected_adam = uninterrupted["optimizer_states"][0]["state"][0]
                actual_adam = actual["optimizer_states"][0]["state"][0]
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        actual_adam[key], expected_adam[key], rtol=0, atol=0
                    )
                for observed, expected in zip(
                    actual[RESUME_KEY]["rank_states"],
                    uninterrupted[RESUME_KEY]["rank_states"],
                ):
                    torch.testing.assert_close(
                        observed["rng"]["torch_cpu"],
                        expected["rng"]["torch_cpu"],
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
