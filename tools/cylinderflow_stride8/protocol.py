from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


AE_MILESTONES = (62_500, 125_000, 187_500, 250_000)
LDM_MILESTONES = AE_MILESTONES
SAMPLING_SEEDS = (0, 1, 2)
DDIM_STEPS = 20
VALIDATION_TRAJECTORY_COUNT = 100
MONITOR_TRAJECTORY_COUNT = 24


def validation_monitor_indices(
    trajectory_count: int = VALIDATION_TRAJECTORY_COUNT,
    monitor_count: int = MONITOR_TRAJECTORY_COUNT,
) -> tuple[int, ...]:
    if trajectory_count <= 0 or monitor_count <= 0 or monitor_count > trajectory_count:
        raise ValueError("invalid Validation monitor dimensions")
    indices = np.rint(np.linspace(0, trajectory_count - 1, monitor_count)).astype(
        np.int64
    )
    if len(np.unique(indices)) != monitor_count:
        raise AssertionError("uniform Validation monitor indices are not unique")
    return tuple(int(value) for value in indices)


def validate_locked_config(config: dict[str, Any], stage: str) -> None:
    if stage not in ("ae", "ldm"):
        raise ValueError(f"unsupported training stage: {stage}")
    data = config["data"]
    training = config["training"]
    expected_data = {
        "mode": "cylinderflow_stride8",
        "batch_size": 1,
        "num_workers": 0,
        "sampler_seed": 42,
    }
    for name, expected in expected_data.items():
        if data.get(name) != expected:
            raise ValueError(f"data.{name}={data.get(name)!r}, expected {expected!r}")
    if data["dataset"].get("strict_formal_counts") is not True:
        raise ValueError("formal config must enforce 1000/100 trajectory counts")
    expected_training = {
        "seed": 42,
        "precision": "16-mixed",
        "accumulate_grad_batches": 4,
        "dataset_size": 1000,
        "max_epochs": 1000,
        "max_steps": 250_000,
        "last_every_n_steps": 5_000,
        "milestone_every_n_steps": 62_500,
    }
    for name, expected in expected_training.items():
        if training.get(name) != expected:
            raise ValueError(
                f"training.{name}={training.get(name)!r}, expected {expected!r}"
            )
    if stage == "ae":
        ae = config["model"]["aeconfig"]
    else:
        model = config["model"]
        if model.get("timesteps") != 1000:
            raise ValueError("LDM diffusion schedule must contain 1000 steps")
        architecture = model["model_config"]
        expected_architecture = {
            "hidden_size": 512,
            "depth": 24,
            "num_heads": 16,
            "in_channels": 16,
        }
        for name, expected in expected_architecture.items():
            if architecture.get(name) != expected:
                raise ValueError(f"LDM {name} differs from DiTSmall-FF")
        ae = model["first_stage_config"]["aeconfig"]
    if ae.get("latent_grid_size") != 64:
        raise ValueError("GINO AE latent grid must be 64^3")
    if ae["encoder"].get("z_channels") != 16:
        raise ValueError("GINO AE latent must contain 16 channels")
    if not np.isclose(float(ae["encoder"].get("gno_radius")), 0.0425):
        raise ValueError("GINO encoder radius must remain 0.0425")


def require_existing_file(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is not configured")
    path = Path(value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path
