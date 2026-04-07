import pydantic
import torch
import torch.nn as nn


class LightAttentionConfig(pydantic.BaseModel):
    embeddings_dim: int = pydantic.Field(
        default=384, description="Dimension of the input embeddings."
    )
    kernel_size: int = pydantic.Field(
        default=9, description="Kernel size for the convolutional layers."
    )
    conv_dropout: float = pydantic.Field(
        default=0.25, description="Dropout rate for the convolutional layers."
    )
    pass


class LightAttention(nn.Module):
    def __init__(self, config: LightAttentionConfig) -> None:
        super().__init__()
        embeddings_dim = config.embeddings_dim
        self.feature_convolution = nn.Conv1d(
            in_channels=embeddings_dim,
            out_channels=embeddings_dim,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            in_channels=embeddings_dim,
            out_channels=embeddings_dim,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.kernel_size // 2,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(config.conv_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_convolution(x)
        features = self.dropout(features)

        attention_logits = self.attention_convolution(x)
        attended = features * self.softmax(attention_logits)
        return torch.squeeze(attended)


def LightAttentionInit(config: LightAttentionConfig) -> LightAttention:
    return LightAttention(config)
