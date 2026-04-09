import torch
import torch.nn as nn

from .attention import LightAttention
from .types import ModelConfig, VOCAB_DIM


class ThermoMPNNHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        if config.use_light_attention:
            self.light_attention: LightAttention | None = LightAttention(
                embeddings_dim=config.input_dim,
                kernel_size=config.kernel_size,
                conv_dropout=config.attention_dropout,
            )
        else:
            self.light_attention = None

        layer_sizes = [config.input_dim, *config.hidden_dims, VOCAB_DIM]
        layers: list[nn.Module] = []
        for put, out in zip(layer_sizes, layer_sizes[1:]):
            # Match the original ThermoMPNN "both_out" ordering: ReLU -> Linear.
            layers.append(nn.ReLU())
            layers.append(nn.Linear(put, out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        squeeze_batch = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True
        if x.ndim != 2:
            raise ValueError(f"Expected [N, C] tensor, got shape {tuple(x.shape)}.")

        if self.light_attention is not None:
            x = self.light_attention(x.unsqueeze(-1), mask=mask).squeeze(-1)

        aa_scores = self.mlp(x)
        if squeeze_batch:
            return aa_scores.squeeze(0)
        return aa_scores
