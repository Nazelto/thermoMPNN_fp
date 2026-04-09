from __future__ import annotations

import warnings

import torch
import torch.nn as nn

from .protein_mpnn_utils import (
    ProteinMPNN,
    checkpoint_state_dict,
    normalize_proteinmpnn_state_dict_keys,
)
from .types import ALPHABET, BackboneInput, BackboneOutput, ModelConfig


class ProteinMPNNBackbone(nn.Module):
    def __init__(self, config: ModelConfig, checkpoint_path: str | None = None):
        super().__init__()
        if config.embedding_dim != config.hidden_dim:
            raise ValueError(
                "The original ProteinMPNN backbone expects embedding_dim == hidden_dim. "
                f"Got embedding_dim={config.embedding_dim} and hidden_dim={config.hidden_dim}."
            )

        self.config = config
        self.mpnn = ProteinMPNN(
            num_letters=len(ALPHABET),
            node_features=config.hidden_dim,
            edge_features=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            vocab=len(ALPHABET),
            k_neighbors=config.num_neighbors,
            augment_eps=0.0,
            dropout=0.1,
        )

        if checkpoint_path and config.load_pretrained:
            self.load_checkpoint(checkpoint_path, strict=config.strict_backbone_weights)

        if config.freeze_backbone:
            self.freeze()

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False

    def unfreeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = normalize_proteinmpnn_state_dict_keys(
            checkpoint_state_dict(checkpoint)
        )
        incompatible = self.mpnn.load_state_dict(state_dict, strict=strict)
        if not strict:
            missing = list(incompatible.missing_keys)
            unexpected = list(incompatible.unexpected_keys)
            if missing or unexpected:
                warnings.warn(
                    "Loaded ProteinMPNN backbone in non-strict mode. "
                    f"Missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}; "
                    f"unexpected keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}"
                )

    def forward(self, backbone_input: BackboneInput) -> BackboneOutput:
        x = backbone_input.atom_coords[:, :4, :].unsqueeze(0)
        s = backbone_input.sequence_tensor.unsqueeze(0)
        mask = backbone_input.mask.to(dtype=x.dtype).unsqueeze(0)
        chain_m = mask.clone()
        residue_idx = backbone_input.residue_idx.unsqueeze(0)
        chain_encoding_all = torch.ones_like(residue_idx)
        randn = torch.zeros_like(mask)

        decoder_hidden_states, sequence_embedding, _ = self.mpnn(
            x,
            s,
            mask,
            chain_m,
            residue_idx,
            chain_encoding_all,
            randn=randn,
        )

        squeezed_hidden = [hidden.squeeze(0) for hidden in decoder_hidden_states]
        squeezed_sequence = sequence_embedding.squeeze(0)
        return BackboneOutput(
            decoder_hidden_states=squeezed_hidden,
            sequence_embedding=squeezed_sequence,
            node_embedding=squeezed_hidden[0],
        )
