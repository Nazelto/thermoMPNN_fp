import torch

import torch.nn as nn

from .head import ThermoMPNNHead
from .protein_mpnn_utils import normalize_proteinmpnn_state_dict_keys
from .proteinmpnn_backbone import ProteinMPNNBackbone
from .types import (
    BackboneInput,
    BackboneOutput,
    BatchPrediction,
    ModelConfig,
    MutationRecord,
    PredictionResult,
    ddg_from_scores,
)


def extract_site_feature_vector(
    backbone_output: BackboneOutput,
    mutation: MutationRecord,
    config: ModelConfig,
) -> torch.Tensor:
    if mutation.position >= backbone_output.sequence_embedding.shape[0]:
        raise IndexError(
            f"Mutation position {mutation.position} is outside the sequence length."
        )

    selected_hidden = backbone_output.decoder_hidden_states[: config.num_final_layers]
    if len(selected_hidden) < config.num_final_layers:
        raise ValueError(
            f"Backbone returned {len(selected_hidden)} decoder states but "
            f"{config.num_final_layers} are required."
        )

    hidden_vector = torch.cat(
        [hidden_state[mutation.position] for hidden_state in selected_hidden], dim=-1
    )
    seq_embedding = backbone_output.sequence_embedding[mutation.position]
    return torch.cat([hidden_vector, seq_embedding], dim=-1)


class ThermoMPNNModel(nn.Module):
    def __init__(self, config: ModelConfig, checkpoint_path: str | None = None):
        super().__init__()
        self.config = config
        self.backbone = ProteinMPNNBackbone(config, checkpoint_path=checkpoint_path)
        self.head = ThermoMPNNHead(config)

    def backbone_features(self, backbone_input: BackboneInput) -> BackboneOutput:
        return self.backbone(backbone_input)

    def score_mutations(
        self,
        backbone_input: BackboneInput,
        mutations: list[MutationRecord],
    ) -> BatchPrediction:
        backbone_output = self.backbone_features(backbone_input)
        site_features = torch.stack(
            [
                extract_site_feature_vector(backbone_output, mutation, self.config)
                for mutation in mutations
            ],
            dim=0,
        )
        aa_scores = self.head(site_features)
        predictions = [
            PredictionResult(
                mutation=mutation,
                aa_scores=aa_scores[index],
                ddg=ddg_from_scores(aa_scores[index], mutation),
            )
            for index, mutation in enumerate(mutations)
        ]
        return BatchPrediction(
            protein_id=backbone_input.protein_id,
            predictions=predictions,
        )

    def forward(
        self,
        backbone_input: BackboneInput,
        mutations: list[MutationRecord],
    ) -> BatchPrediction:
        return self.score_mutations(backbone_input, mutations)


def _checkpoint_state_dict(payload: dict) -> dict:
    if "model_state_dict" in payload:
        return payload["model_state_dict"]
    if "state_dict" in payload:
        return payload["state_dict"]
    return payload


def _is_original_thermompnn_checkpoint(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("model.prot_mpnn.") for key in state_dict)


def convert_original_thermompnn_state_dict(
    state_dict: dict[str, torch.Tensor],
    model: ThermoMPNNModel,
) -> dict[str, torch.Tensor]:
    converted = model.state_dict()
    prot_mpnn_prefix = "model.prot_mpnn."
    light_attention_prefix = "model.light_attention."
    both_out_prefix = "model.both_out."
    ddg_out_weight = state_dict.get("model.ddg_out.weight")
    ddg_out_bias = state_dict.get("model.ddg_out.bias")

    for key, value in state_dict.items():
        if key.startswith(prot_mpnn_prefix):
            stripped = key[len(prot_mpnn_prefix) :]
            normalized = normalize_proteinmpnn_state_dict_keys({stripped: value})
            for normalized_key, normalized_value in normalized.items():
                target_key = f"backbone.mpnn.{normalized_key}"
                if target_key in converted:
                    converted[target_key] = normalized_value
            continue

        if key.startswith(light_attention_prefix):
            target_key = f"head.light_attention.{key[len(light_attention_prefix):]}"
            if target_key in converted:
                converted[target_key] = value
            continue

        if key.startswith(both_out_prefix):
            target_key = f"head.mlp.{key[len(both_out_prefix):]}"
            if target_key in converted:
                converted[target_key] = value

    if ddg_out_weight is not None:
        final_linear_index = max(
            index
            for index, module in enumerate(model.head.mlp)
            if isinstance(module, nn.Linear)
        )
        weight_key = f"head.mlp.{final_linear_index}.weight"
        bias_key = f"head.mlp.{final_linear_index}.bias"
        converted[weight_key] = converted[weight_key] * ddg_out_weight.reshape(1, 1)
        converted[bias_key] = converted[bias_key] * ddg_out_weight.reshape(1)
        if ddg_out_bias is not None:
            converted[bias_key] = converted[bias_key] + ddg_out_bias.reshape(1)

    return converted


def load_compatible_state_dict(
    model: ThermoMPNNModel,
    checkpoint_path: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)
    if _is_original_thermompnn_checkpoint(state_dict):
        state_dict = convert_original_thermompnn_state_dict(state_dict, model)
    model.load_state_dict(state_dict, strict=False)


def load_model(
    config: ModelConfig,
    checkpoint_path: str | None = None,
    model_weights_path: str | None = None,
    device: str | torch.device | None = None,
) -> ThermoMPNNModel:
    model = ThermoMPNNModel(config, checkpoint_path=model_weights_path)
    if checkpoint_path:
        load_compatible_state_dict(model, checkpoint_path)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def predict_mutations(
    model: ThermoMPNNModel,
    backbone_input: BackboneInput,
    mutations: list[MutationRecord],
) -> BatchPrediction:
    with torch.no_grad():
        return model(backbone_input, mutations)
