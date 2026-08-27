from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset


class TinyScheduledModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.seen_microbatches = 0

    def training_step(self, batch, batch_idx):
        del batch_idx
        self.seen_microbatches += 1
        target = batch[0].float().mean()
        return torch.square(self.weight - target)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0 / (step + 1.0)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def make_trainer(
    root: Path,
    max_steps: int,
    callbacks: list[L.Callback] | None = None,
) -> L.Trainer:
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        max_steps=max_steps,
        accumulate_grad_batches=4,
        callbacks=callbacks,
        default_root_dir=str(root),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )


class TrainingControlTest(unittest.TestCase):
    def test_accumulation_scheduler_and_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            loader = DataLoader(
                TensorDataset(torch.arange(12, dtype=torch.float32)),
                batch_size=1,
                shuffle=False,
            )
            checkpoint = ModelCheckpoint(
                dirpath=str(root / "checkpoints"),
                every_n_train_steps=2,
                save_top_k=0,
                save_last=True,
            )
            model = TinyScheduledModel()
            trainer = make_trainer(root, max_steps=2, callbacks=[checkpoint])
            trainer.fit(model, train_dataloaders=loader)

            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(model.seen_microbatches, 8)
            scheduler = trainer.lr_scheduler_configs[0].scheduler
            self.assertEqual(scheduler.last_epoch, 2)
            self.assertTrue(Path(checkpoint.last_model_path).is_file())

            resumed = TinyScheduledModel()
            resume_trainer = make_trainer(root / "resume", max_steps=3)
            resume_trainer.fit(
                resumed,
                train_dataloaders=loader,
                ckpt_path=checkpoint.last_model_path,
            )
            self.assertEqual(resume_trainer.global_step, 3)
            self.assertGreaterEqual(resumed.seen_microbatches, 1)
            self.assertLessEqual(resumed.seen_microbatches, 4)
            resumed_scheduler = resume_trainer.lr_scheduler_configs[0].scheduler
            self.assertEqual(resumed_scheduler.last_epoch, 3)


if __name__ == "__main__":
    unittest.main()
