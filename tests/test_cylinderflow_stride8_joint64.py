from __future__ import annotations

import unittest

import torch

from tools.cylinderflow_stride8.joint64 import sample_initial64, sample_joint64
from tools.cylinderflow_stride8.benchmark import forecast_initial


class _IdentityNormalizer:
    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return value.clone()


class _FakeModel:
    def __init__(self) -> None:
        self.normalizer = _IdentityNormalizer()
        self.device = torch.device("cpu")
        self.channels = 16
        self.image_size = (16, 16, 16)
        self.conditioning_calls: list[tuple[torch.Tensor, torch.Tensor, object]] = []

    def get_learned_conditioning(self, value):
        self.conditioning_calls.append(value)
        return value[0]

    def decode_first_stage(self, latent, position):
        del latent
        return torch.full(
            (position.shape[0], position.shape[1], position.shape[2], 3),
            7.0,
            dtype=position.dtype,
        )


class _FakeSampler:
    def __init__(self) -> None:
        self.calls = []

    def sample(self, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(kwargs["shape"]), None


class Joint64InformationFlowTest(unittest.TestCase):
    def test_initial_only_and_benchmark_paths_preserve_joint64_output(self) -> None:
        model = _FakeModel()
        sampler = _FakeSampler()
        sequence = torch.arange(65 * 5 * 3, dtype=torch.float32).reshape(1, 65, 5, 3)
        position = torch.zeros((1, 65, 5, 3), dtype=torch.float32)
        expected = sample_joint64(model, sampler, sequence, position)
        actual = sample_initial64(model, sampler, sequence[:, :1], position)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        sample = {
            "initial": sequence[0, 0].numpy(),
            "points": torch.tensor(
                [[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=torch.float32
            ).numpy(),
            "node_type": torch.tensor([4, 6, 0, 5, 0]).numpy(),
        }
        prediction = forecast_initial(model, sampler, sample)
        torch.testing.assert_close(torch.from_numpy(prediction[0]), sequence[0, 0])
        torch.testing.assert_close(
            torch.from_numpy(prediction[1:, :2, :2]),
            sequence[0, 0, :2, :2][None].expand(64, -1, -1),
        )
        self.assertTrue((prediction[1:, :, 2] == 7).all())
        self.assertTrue((prediction[1:, 3, :2] == 7).all())

    def test_one_call_conditions_only_on_clean_frame_and_predicts_64_future_frames(
        self,
    ) -> None:
        model = _FakeModel()
        sampler = _FakeSampler()
        sequence = torch.arange(65 * 5 * 3, dtype=torch.float32).reshape(1, 65, 5, 3)
        position = torch.zeros((1, 65, 5, 3), dtype=torch.float32)
        prediction = sample_joint64(model, sampler, sequence, position, ddim_steps=20)

        self.assertEqual(len(sampler.calls), 1)
        self.assertEqual(sampler.calls[0]["S"], 20)
        self.assertEqual(tuple(model.conditioning_calls[0][0].shape), (1, 1, 5, 3))
        torch.testing.assert_close(model.conditioning_calls[0][0], sequence[:, :1])
        torch.testing.assert_close(prediction[:, :1], sequence[:, :1])
        torch.testing.assert_close(
            prediction[:, 1:], torch.full_like(prediction[:, 1:], 7.0)
        )

        altered_future = sequence.clone()
        altered_future[:, 1:] = -999.0
        second_prediction = sample_joint64(
            model, sampler, altered_future, position, ddim_steps=20
        )
        torch.testing.assert_close(second_prediction, prediction)

    def test_rejects_nonformal_sampling_step_count(self) -> None:
        model = _FakeModel()
        sampler = _FakeSampler()
        sequence = torch.zeros((1, 65, 5, 3))
        position = torch.zeros((1, 65, 5, 3))
        with self.assertRaisesRegex(ValueError, "exactly 20"):
            sample_joint64(model, sampler, sequence, position, ddim_steps=19)


if __name__ == "__main__":
    unittest.main()
