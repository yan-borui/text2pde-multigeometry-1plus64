from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from modules.modules.reproducible_resume import (
    DeterministicPermutationSampler,
    ExactResumeCallback,
    load_resume_record,
)


class _IndexDataset(Dataset):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": torch.tensor([float(index)])}


class _StochasticModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.index_trace: list[int] = []
        self.noise_trace: list[float] = []
        self.loss_trace: list[float] = []
        self.lr_trace: list[float] = []

    def training_step(self, batch, batch_idx):
        del batch_idx
        index = int(batch["x"][0, 0])
        noise = torch.randn((), device=self.device)
        target = batch["x"].float().mean() * 0.01 + noise * 0.02
        loss = torch.square(self.weight - target)
        self.index_trace.append(index)
        self.noise_trace.append(float(noise.detach().cpu()))
        self.loss_trace.append(float(loss.detach().cpu()))
        self.lr_trace.append(float(self.optimizers().param_groups[0]["lr"]))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 / (step + 1.0)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def make_loader(offset: int) -> DataLoader:
    dataset = _IndexDataset()
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=DeterministicPermutationSampler(dataset, seed=91, start_offset=offset),
    )


def make_trainer(
    root: Path,
    max_steps: int,
    callbacks: list[L.Callback],
) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        max_epochs=1,
        accumulate_grad_batches=2,
        callbacks=callbacks,
        default_root_dir=str(root),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        deterministic=True,
    )


class CylinderFlowResumeTest(unittest.TestCase):
    def test_sampler_prefix_plus_resumed_suffix_matches_full_order(self) -> None:
        dataset = _IndexDataset()
        full = list(DeterministicPermutationSampler(dataset, 7, 0))
        prefix = full[:5]
        suffix = list(DeterministicPermutationSampler(dataset, 7, 5))
        self.assertEqual(prefix + suffix, full)

    def test_interrupted_run_matches_uninterrupted_order_rng_lr_loss_and_weights(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            L.seed_everything(123, workers=True)
            uninterrupted = _StochasticModel()
            uninterrupted_trainer = make_trainer(
                root / "uninterrupted", 3, [ExactResumeCallback(0)]
            )
            uninterrupted_trainer.fit(uninterrupted, train_dataloaders=make_loader(0))

            L.seed_everything(123, workers=True)
            interrupted = _StochasticModel()
            checkpoint = ModelCheckpoint(
                dirpath=str(root / "interrupted" / "checkpoints"),
                every_n_train_steps=2,
                save_top_k=0,
                save_last=True,
            )
            interrupted_trainer = make_trainer(
                root / "interrupted",
                2,
                [ExactResumeCallback(0), checkpoint],
            )
            interrupted_trainer.fit(interrupted, train_dataloaders=make_loader(0))
            record = load_resume_record(checkpoint.last_model_path)
            self.assertEqual(record["examples_seen"], 4)
            self.assertEqual(record["global_step"], 2)

            # A different launch seed demonstrates that checkpoint RNG controls
            # the continued posterior/diffusion draw sequence.
            L.seed_everything(999, workers=True)
            resumed = _StochasticModel()
            resumed_trainer = make_trainer(
                root / "resumed", 3, [ExactResumeCallback(4)]
            )
            resumed_trainer.fit(
                resumed,
                train_dataloaders=make_loader(4),
                ckpt_path=checkpoint.last_model_path,
            )

            combined_indices = interrupted.index_trace + resumed.index_trace
            combined_noise = interrupted.noise_trace + resumed.noise_trace
            combined_losses = interrupted.loss_trace + resumed.loss_trace
            combined_lrs = interrupted.lr_trace + resumed.lr_trace
            self.assertEqual(combined_indices, uninterrupted.index_trace)
            self.assertEqual(combined_noise, uninterrupted.noise_trace)
            self.assertEqual(combined_losses, uninterrupted.loss_trace)
            self.assertEqual(combined_lrs, uninterrupted.lr_trace)
            torch.testing.assert_close(
                resumed.weight, uninterrupted.weight, rtol=0, atol=0
            )
            self.assertEqual(resumed_trainer.global_step, 3)
            self.assertEqual(
                resumed_trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                uninterrupted_trainer.lr_scheduler_configs[0].scheduler.state_dict(),
            )


if __name__ == "__main__":
    unittest.main()
