from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from modules.models.ddpm import LatentDiffusion
from modules.modules.ddim import DDIMSampler


SEGMENT_LENGTH = 25
ROLLOUT_FUTURE_FRAMES = 64


@dataclass
class RolloutResult:
    prediction: torch.Tensor
    segments: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    conditioning_frames: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def derive_segment_seed(
    evaluation_seed: int,
    trajectory_index: int,
    start: int,
    segment_index: int,
) -> int:
    if segment_index not in (0, 1, 2):
        raise ValueError("segment_index must be 0, 1, or 2")
    modulus = 2**31 - 1
    return int(
        (
            (int(evaluation_seed) + 1) * 1_000_003
            + int(trajectory_index) * 9_176
            + int(start) * 131
            + int(segment_index) * 53
        )
        % modulus
    )


def set_segment_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rollout_three_segments(
    initial_frame: torch.Tensor,
    sample_segment: Callable[[torch.Tensor, int], torch.Tensor],
) -> RolloutResult:
    """Generate 64 future frames without exposing future truth to the sampler."""

    if initial_frame.ndim != 3 or initial_frame.shape[0] != 1:
        raise ValueError("initial_frame must have shape [1,N,3]")
    if initial_frame.shape[-1] != 3:
        raise ValueError("initial_frame must contain UVP channels")

    conditions: list[torch.Tensor] = []
    segments: list[torch.Tensor] = []
    condition = initial_frame
    expected_shape = (SEGMENT_LENGTH, initial_frame.shape[1], 3)
    for segment_index in range(3):
        conditions.append(condition.detach().clone())
        segment = sample_segment(condition, segment_index)
        if tuple(segment.shape) != expected_shape:
            raise ValueError(
                f"segment {segment_index} has shape {tuple(segment.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(segment).all():
            raise FloatingPointError(f"segment {segment_index} is non-finite")
        segments.append(segment)
        # Local frame 24 is global prediction frame 24 or 48. It is the only
        # field passed into the next autoregressive Text2PDE call.
        condition = segment[24:25].detach()

    prediction = torch.cat(
        (
            initial_frame,
            segments[0][1:25],
            segments[1][1:25],
            segments[2][1:17],
        ),
        dim=0,
    )
    expected_rollout_shape = (65, initial_frame.shape[1], 3)
    if tuple(prediction.shape) != expected_rollout_shape:
        raise AssertionError(
            f"rollout stitching produced {tuple(prediction.shape)}, "
            f"expected {expected_rollout_shape}"
        )
    return RolloutResult(
        prediction=prediction,
        segments=(segments[0], segments[1], segments[2]),
        conditioning_frames=(conditions[0], conditions[1], conditions[2]),
    )


def local_position_tensor(
    normalized_points: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if normalized_points.ndim != 2 or normalized_points.shape[1] != 2:
        raise ValueError("normalized_points must have shape [N,2]")
    points = normalized_points.to(device=device, dtype=torch.float32)
    spatial = points[None, None].expand(1, SEGMENT_LENGTH, -1, -1)
    local_time = torch.linspace(
        0.0, 1.0, SEGMENT_LENGTH, device=device, dtype=torch.float32
    )
    time_channel = local_time[None, :, None, None].expand(1, -1, points.shape[0], 1)
    return torch.cat((spatial, time_channel), dim=-1)


@torch.inference_mode()
def sample_model_segment(
    model: LatentDiffusion,
    sampler: DDIMSampler,
    physical_condition: torch.Tensor,
    normalized_points: torch.Tensor,
    ddim_steps: int,
) -> torch.Tensor:
    """Sample one local 1->24 window from one physical UVP condition frame."""

    if physical_condition.ndim != 3 or physical_condition.shape[0] != 1:
        raise ValueError("physical_condition must have shape [1,N,3]")
    device = model.device
    condition_batch = physical_condition.unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    position = local_position_tensor(normalized_points, device)
    normalized_condition = model.normalizer.normalize(condition_batch)
    conditioning = model.get_learned_conditioning(
        (normalized_condition, position[:, :1, :, :2], None)
    )
    latent_shape = (
        1,
        model.channels,
        model.image_size[0],
        model.image_size[1],
        model.image_size[2],
    )
    latent, _ = sampler.sample(
        S=ddim_steps,
        batch_size=1,
        shape=latent_shape,
        conditioning=conditioning,
        eta=0.0,
        verbose=False,
    )
    decoded = model.decode_first_stage(latent, position)
    if tuple(decoded.shape[:2]) != (1, SEGMENT_LENGTH):
        raise ValueError(f"decoder returned unexpected shape {tuple(decoded.shape)}")
    return decoded[0]
