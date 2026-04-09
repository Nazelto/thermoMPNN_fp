import torch
import torch.nn as nn


class LightAttention(nn.Module):
    """Light attention over [batch, channels, length] tensors."""

    def __init__(
        self,
        embeddings_dim: int,
        kernel_size: int = 9,
        conv_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.feature_convolution = nn.Conv1d(
            in_channels=embeddings_dim,
            out_channels=embeddings_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            in_channels=embeddings_dim,
            out_channels=embeddings_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, C, L] tensor, got shape {tuple(x.shape)}.")

        features = self.feature_convolution(x)
        features = self.dropout(features)
        attention_logits = self.attention_convolution(x)

        if mask is not None:
            if mask.ndim != 2:
                raise ValueError("Mask must have shape [B, L].")
            attention_logits = attention_logits.masked_fill(
                ~mask.unsqueeze(1).bool(), torch.finfo(attention_logits.dtype).min
            )

        attended = features * self.softmax(attention_logits)
        if mask is not None:
            attended = attended * mask.unsqueeze(1)
        return attended
