from typing import cast
import torch
import torch.nn as nn
from .types import HeadConfig, input_dim, use_light_attention, layer
from .attention import LightAttention, LightAttentionConfig, LightAttentionInit


class ThermoMPNNHead(nn.Module):
    config: HeadConfig
    light_attention: LightAttention | None

    def __init__(self, head_config: HeadConfig):
        super().__init__()
        self.config = head_config

        if use_light_attention(head_config):
            light_attention_config = LightAttentionConfig(
                embeddings_dim=input_dim(head_config),
                kernel_size=9,
                conv_dropout=0.25,
            )
            self.light_attention = LightAttentionInit(light_attention_config)
        else:
            self.light_attention = None
        layer_sizes: list[int] = layer(head_config)
        self.both_out = nn.Sequential()
        for put, out in zip(layer_sizes, layer_sizes[1:]):
            self.both_out.append(nn.ReLU())
            self.both_out.append(nn.Linear(put, out))
        self.ddg_out = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _forward(self, x)


def _forward(thermo: ThermoMPNNHead, x: torch.Tensor) -> torch.Tensor:
    if thermo.light_attention is not None:
        x = x.unsqueeze(-1).unsqueeze(0)
        x = thermo.light_attention(x)
        aa_scores: torch.Tensor = thermo.both_out(x)

    return cast(torch.Tensor, thermo.ddg_out(aa_scores.unsqueeze(-1))).squeeze(-1)
