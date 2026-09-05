from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from modules.models.ddpm import LatentDiffusion
    from modules.modules.ddim import DDIMSampler


SEQUENCE_LENGTH = 65
FUTURE_FRAME_COUNT = 64
DDIM_STEPS = 20


def derive_sample_seed(evaluation_seed: int, trajectory_index: int) -> int:
    """Keep a trajectory's stochastic draw stable across monitor/full evaluation."""

    modulus = 2**31 - 1
    return int(
        ((int(evaluation_seed) + 1) * 1_000_003 + int(trajectory_index) * 9_176)
        % modulus
    )


def set_sample_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_joint64_inputs(
    physical_sequence: torch.Tensor, position: torch.Tensor
) -> None:
    if physical_sequence.ndim != 4 or physical_sequence.shape[0] != 1:
        raise ValueError("physical_sequence must have shape [1,65,N,3]")
    if tuple(physical_sequence.shape[:2]) != (1, SEQUENCE_LENGTH):
        raise ValueError("joint64 requires exactly 65 physical frames")
    if physical_sequence.shape[-1] != 3:
        raise ValueError("physical_sequence must contain UVP channels")
    if position.ndim != 4 or tuple(position.shape[:2]) != (1, SEQUENCE_LENGTH):
        raise ValueError("position must have shape [1,65,N,3]")
    if position.shape[2] != physical_sequence.shape[2] or position.shape[-1] != 3:
        raise ValueError("position and physical_sequence node axes must match")


def build_conditioning_input(
    model: LatentDiffusion,
    physical_sequence: torch.Tensor,
    position: torch.Tensor,
) -> object:
    """Build conditioning from clean frame 0 only; future truth is inaccessible."""

    validate_joint64_inputs(physical_sequence, position)
    clean_frame = physical_sequence[:, :1]
    clean_position = position[:, :1, :, :2]
    normalized_clean_frame = model.normalizer.normalize(clean_frame)
    return model.get_learned_conditioning(
        (normalized_clean_frame, clean_position, None)
    )


@torch.inference_mode()
def sample_joint64(
    model: LatentDiffusion,
    sampler: DDIMSampler,
    physical_sequence: torch.Tensor,
    position: torch.Tensor,
    ddim_steps: int = DDIM_STEPS,
) -> torch.Tensor:
    """Generate all 64 future frames in one DDIM call.

    The decoder's local frame 0 is discarded and replaced by the exact clean
    conditioning frame. Frames 1--64 all come from the same sampled latent.
    """

    if int(ddim_steps) != DDIM_STEPS:
        raise ValueError("formal joint64 evaluation requires exactly 20 DDIM steps")
    validate_joint64_inputs(physical_sequence, position)
    device = model.device
    sequence = physical_sequence.to(device=device, dtype=torch.float32)
    coordinates = position.to(device=device, dtype=torch.float32)
    conditioning = build_conditioning_input(model, sequence, coordinates)
    latent_shape = (
        1,
        model.channels,
        model.image_size[0],
        model.image_size[1],
        model.image_size[2],
    )
    latent, _ = sampler.sample(
        S=DDIM_STEPS,
        batch_size=1,
        shape=latent_shape,
        conditioning=conditioning,
        eta=0.0,
        verbose=False,
    )
    decoded = model.decode_first_stage(latent, coordinates)
    expected_shape = tuple(sequence.shape)
    if tuple(decoded.shape) != expected_shape:
        raise ValueError(
            f"joint64 decoder returned {tuple(decoded.shape)}, expected {expected_shape}"
        )
    prediction = torch.cat((sequence[:, :1], decoded[:, 1:]), dim=1)
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("joint64 prediction is non-finite")
    return prediction
