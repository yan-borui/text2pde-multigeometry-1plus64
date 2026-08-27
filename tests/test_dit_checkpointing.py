from __future__ import annotations

import unittest

import torch

from modules.models.transformer import DiT


class DitCheckpointingTest(unittest.TestCase):
    def test_checkpointed_blocks_preserve_outputs_and_gradients(self) -> None:
        torch.manual_seed(11)
        kwargs = {
            "input_size": [4, 4, 4],
            "patch_size": [2, 2, 2],
            "in_channels": 4,
            "hidden_size": 32,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "learn_sigma": False,
            "use_cross_attn": True,
            "context_dim": 16,
            "dim": 3,
        }
        reference = DiT(**kwargs, gradient_checkpointing=False)
        checkpointed = DiT(**kwargs, gradient_checkpointing=True)
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(mean=0.0, std=0.02)
        checkpointed.load_state_dict(reference.state_dict())
        reference.train()
        checkpointed.train()

        reference_input = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        checkpointed_input = reference_input.detach().clone().requires_grad_(True)
        timestep = torch.tensor([17])
        reference_context = torch.randn(1, 5, 16, requires_grad=True)
        checkpointed_context = reference_context.detach().clone().requires_grad_(True)

        reference_output = reference(reference_input, timestep, reference_context)
        checkpointed_output = checkpointed(
            checkpointed_input, timestep, checkpointed_context
        )
        torch.testing.assert_close(
            checkpointed_output, reference_output, rtol=1e-5, atol=1e-6
        )

        reference_output.square().sum().backward()
        checkpointed_output.square().sum().backward()
        torch.testing.assert_close(
            checkpointed_input.grad, reference_input.grad, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            checkpointed_context.grad, reference_context.grad, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
