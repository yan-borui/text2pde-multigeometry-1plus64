from __future__ import annotations

import unittest

import torch

from modules.models.ae.gino_ae import GINO_Decoder, GINO_Encoder


class GinoChunkingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        axis = torch.linspace(0.0, 1.0, 4)
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
        self.latent_queries = grid.unsqueeze(0)

    def test_encoder_chunking_preserves_outputs_and_gradients(self) -> None:
        kwargs = {
            "in_channels": 3,
            "projection_channels": 4,
            "gno_coord_dim": 3,
            "gno_coord_embed_dim": None,
            "gno_radius": 2.0,
            "gno_mlp_hidden_layers": [8],
            "gno_transform_type": "linear",
            "gno_use_torch_scatter": False,
            "use_open3d": False,
        }
        reference = GINO_Encoder(**kwargs, query_chunk_size=None)
        chunked = GINO_Encoder(**kwargs, query_chunk_size=7)
        chunked.load_state_dict(reference.state_dict())

        geometry = torch.rand(1, 12, 3)
        reference_input = torch.rand(1, 12, 3, requires_grad=True)
        chunked_input = reference_input.detach().clone().requires_grad_(True)
        reference_output = reference(reference_input, geometry, self.latent_queries)
        chunked_output = chunked(chunked_input, geometry, self.latent_queries)
        torch.testing.assert_close(chunked_output, reference_output, rtol=1e-5, atol=1e-6)

        reference_output.square().sum().backward()
        chunked_output.square().sum().backward()
        torch.testing.assert_close(
            chunked_input.grad, reference_input.grad, rtol=1e-5, atol=1e-6
        )

    def test_decoder_chunking_preserves_outputs_and_gradients(self) -> None:
        kwargs = {
            "out_channels": 3,
            "projection_channels": 4,
            "gno_coord_dim": 3,
            "gno_coord_embed_dim": None,
            "gno_radius": 2.0,
            "gno_mlp_hidden_layers": [8],
            "gno_transform_type": "linear",
            "gno_use_torch_scatter": False,
            "use_open3d": False,
        }
        reference = GINO_Decoder(**kwargs, query_chunk_size=None)
        chunked = GINO_Decoder(**kwargs, query_chunk_size=6)
        chunked.load_state_dict(reference.state_dict())

        output_queries = torch.rand(1, 20, 3)
        reference_latent = torch.rand(1, 4, 4, 4, 4, requires_grad=True)
        chunked_latent = reference_latent.detach().clone().requires_grad_(True)
        reference_output = reference(
            reference_latent, self.latent_queries, output_queries
        )
        chunked_output = chunked(chunked_latent, self.latent_queries, output_queries)
        torch.testing.assert_close(chunked_output, reference_output, rtol=1e-5, atol=1e-6)

        reference_output.square().sum().backward()
        chunked_output.square().sum().backward()
        torch.testing.assert_close(
            chunked_latent.grad, reference_latent.grad, rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
