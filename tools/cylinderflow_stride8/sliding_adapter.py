"""Bind long-horizon inference to the native Text2PDE joint64 sampler."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset.cylinderflow_stride8 import CylinderFlowStride8TrajectoryDataset
from modules.utils import get_yaml
from .evaluate_joint64 import instantiate_model, make_sampler
from .joint64 import derive_sample_seed, sample_initial64, set_sample_seed
from .predictions import writeback_velocity
from .protocol import validate_locked_config

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGES = ("tools", "dataset", "modules", "mgn_data")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)


class DataView:
    """Expose global trajectory IDs while preserving native dataset positions."""

    def __init__(self, dataset: CylinderFlowStride8TrajectoryDataset) -> None:
        self.dataset = dataset
        self.splits = {dataset.split: dataset.trajectory_indices}
        self.positions = {
            index: i for i, index in enumerate(dataset.trajectory_indices)
        }

    def evaluation(self, index: int) -> dict:
        sample = self.dataset.__getitem__(self.positions[index], eval=True)
        return {"field": sample["x"].numpy()}


class Predictor:
    """Condition on physical frame zero with local decoder time reset per window."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        from dataset.datamodule import FluidsDataModule

        self.device = device
        config = get_yaml(args.config)
        validate_locked_config(config, "ldm")
        datamodule = FluidsDataModule(config["data"])
        training = datamodule.val_dataset
        if not isinstance(training, CylinderFlowStride8TrajectoryDataset):
            raise TypeError("requires the native stride8 dataset")
        if training.split != "validation" or len(training) != 100:
            raise ValueError("requires the complete Validation100 data contract")
        dataset = training
        if args.scope == "test":
            dataset = CylinderFlowStride8TrajectoryDataset(
                args.test_data_dir / "manifest.json",
                split="test",
                return_metadata=True,
                stage="ldm",
            )
            if dataset.normalizer_values != training.normalizer_values:
                raise ValueError("Test normalization differs from training")
        self.model, update = instantiate_model(
            config,
            args.checkpoint,
            args.ae_checkpoint,
            datamodule,
            device,
        )
        self.model.eval().float().requires_grad_(False)
        self.sampler = make_sampler(self.model)
        self.data = DataView(dataset)
        self.ensemble_size = args.ensemble_size
        self.identity = {
            "method": "Text2PDE",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_id": self.model._stride8_checkpoint_id,
            "update": update,
            "ae_checkpoint": str(args.ae_checkpoint.resolve()),
            "dependencies": self.model._stride8_dependencies,
            "configuration": config,
            "normalization": training.normalizer_values,
            "training_manifest": training.manifest,
            "evaluation_manifest": dataset.manifest,
            "sampling_steps": 20,
            "model_calls_per_member_window": 20,
            "seed_rule": "(derive_sample_seed(member+label*K,index)+start*104729) modulo (2**31-1)",
        }

    def load_case(self, index: int) -> dict:
        return self.data.dataset.initial(self.data.positions[index])

    def draw_seed(self, index: int, start: int, member: int, label: int) -> int:
        native = derive_sample_seed(member + label * self.ensemble_size, index)
        return (native + start * 104729) % (2**31 - 1)

    def predict(
        self, sample: dict, seed: int, *, diagnostics: bool = False
    ) -> tuple[np.ndarray, dict]:
        set_sample_seed(seed)
        points = torch.from_numpy(sample["points"])
        spatial = (points - points.amin(dim=0)) / (
            points.amax(dim=0) - points.amin(dim=0)
        )
        local_time = torch.linspace(0.0, 1.0, 65, dtype=torch.float32)
        position = torch.cat(
            (
                spatial[None].expand(65, -1, -1),
                local_time[:, None, None].expand(-1, len(points), 1),
            ),
            dim=-1,
        )[None]
        initial = torch.from_numpy(sample["initial"])[None, None]
        result = sample_initial64(self.model, self.sampler, initial, position)
        prediction = writeback_velocity(
            result[0].float().cpu().numpy(),
            sample["initial"],
            sample["node_type"],
        )
        return prediction, {}
